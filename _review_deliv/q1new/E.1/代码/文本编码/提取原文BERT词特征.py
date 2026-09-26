"""Encode the official Excel `text` column into one 768-d BERT vector per word.

Word order is the exact official-word order used by step 4. Whisper text is never
read. All official words receive a text vector, including words without a valid
speech timestamp. The last BERT encoder states of a word's WordPieces are averaged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "运行环境" / "text_packages"))

import numpy as np
import onnxruntime as ort
import openpyxl
from tokenizers import Tokenizer


SOURCE_XLSX = ROOT / "输入" / "原始标签" / "label-100.xlsx"
MANIFEST = ROOT / "输入" / "样本清单" / "样本清单.json"
STEP4_WORDS = ROOT / "日志" / "第4步文本一致性检查" / "原文词级时间_MFA候选及最终采用.csv"
MODEL_DIR = ROOT / "模型" / "BERT_base_uncased"
ENCODER = MODEL_DIR / "model_encoder_768.onnx"
TOKENIZER = MODEL_DIR / "tokenizer.json"
MODEL_DETAILS = MODEL_DIR / "model_encoder_768.json"
OUTPUT = ROOT / "结果" / "05_BERT_base_uncased"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")


def read_official_texts() -> dict[str, str]:
    workbook = openpyxl.load_workbook(SOURCE_XLSX, read_only=True, data_only=True)
    try:
        sheet = workbook["label"]
        rows = sheet.iter_rows(values_only=True)
        header = next(rows)
        columns = {str(value).strip(): i for i, value in enumerate(header) if value is not None}
        required = {"video_id", "clip_id", "text"}
        if not required.issubset(columns):
            raise ValueError(f"Missing official workbook columns: {required - set(columns)}")
        result: dict[str, str] = {}
        for row in rows:
            video_id = str(row[columns["video_id"]] or "").strip()
            clip_value = row[columns["clip_id"]]
            clip_id = str(int(clip_value)) if isinstance(clip_value, float) and clip_value.is_integer() else str(clip_value).strip()
            if not video_id or not clip_id:
                continue
            sample_id = f"{video_id}$_${clip_id}"
            if sample_id in result:
                raise ValueError(f"Duplicate sample in workbook: {sample_id}")
            text = row[columns["text"]]
            result[sample_id] = "" if text is None else str(text)
        return result
    finally:
        workbook.close()


def read_step4_words() -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with STEP4_WORDS.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[row["sample_id"]].append(row)
    return grouped


def main() -> None:
    official = read_official_texts()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    word_rows = read_step4_words()
    details = json.loads(MODEL_DETAILS.read_text(encoding="utf-8"))
    if len(official) != 100 or len(manifest) != 100 or set(official) != set(word_rows):
        raise ValueError("Expected the same 100 samples in Excel, manifest, and step 4")
    if sha256(ENCODER) != details["encoder_sha256"]:
        raise ValueError("BERT encoder checksum mismatch")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(ENCODER), sess_options=options, providers=["CPUExecutionProvider"])
    output_info = session.get_outputs()
    if len(output_info) != 1 or output_info[0].shape[-1] != 768:
        raise ValueError("BERT encoder must have one 768-dimensional output")

    features: list[np.ndarray] = []
    index: list[dict[str, object]] = []
    sample_summary: list[dict[str, object]] = []
    time_valid_count = 0
    for item in manifest:
        sample_id = item["sample_id"]
        text = official[sample_id]
        if text != item["text"]:
            raise ValueError(f"Manifest text differs from official Excel text: {sample_id}")
        normalized = normalize(text)
        words = list(WORD_RE.finditer(normalized))
        prior = word_rows[sample_id]
        if len(words) != len(prior):
            raise ValueError(f"Step 4 word count differs from Excel text: {sample_id}")
        for position, (word, row) in enumerate(zip(words, prior), 1):
            if int(row["official_word_index"]) != position or word.group().casefold() != normalize(row["official_word"]).casefold():
                raise ValueError(f"Step 4 word identity differs from Excel text: {sample_id} word {position}")

        encoded = tokenizer.encode(normalized)
        if len(encoded.ids) > 512:
            raise ValueError(f"BERT input exceeds 512 tokens; implement context windows: {sample_id}")
        word_token_positions: list[list[int]] = []
        for word in words:
            positions = [i for i, ((start, end), special) in enumerate(zip(encoded.offsets, encoded.special_tokens_mask))
                         if not special and end > start and start >= word.start() and end <= word.end()]
            if not positions:
                raise ValueError(f"No BERT subword mapped to {sample_id} word {word.group()!r}")
            word_token_positions.append(positions)

        inputs = {
            "input_ids": np.asarray([encoded.ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(encoded.ids)), dtype=np.int64),
            "token_type_ids": np.zeros((1, len(encoded.ids)), dtype=np.int64),
        }
        hidden = session.run([output_info[0].name], inputs)[0][0]
        if hidden.shape != (len(encoded.ids), 768) or not np.isfinite(hidden).all():
            raise ValueError(f"Invalid BERT hidden states: {sample_id}")
        sample_start = len(features)
        for position, (word, token_positions, row) in enumerate(zip(words, word_token_positions, prior), 1):
            vector = hidden[token_positions].mean(axis=0, dtype=np.float32).astype(np.float32)
            if vector.shape != (768,) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid word vector: {sample_id} word {position}")
            features.append(vector)
            valid_time = row["final_primary_valid_mask"].strip().lower() == "true"
            time_valid_count += valid_time
            index.append({
                "feature_row_0based": len(features) - 1,
                "sample_id": sample_id,
                "official_word_index": position,
                "official_word": word.group(),
                "text_char_start_0based": word.start(),
                "text_char_end_exclusive": word.end(),
                "bert_token_positions_0based": ";".join(map(str, token_positions)),
                "bert_wordpieces": " ".join(encoded.tokens[i] for i in token_positions),
                "bert_subword_count": len(token_positions),
                "text_feature_valid_mask": 1,
                "final_primary_start_time_wav_s": row["final_primary_start_time_wav_s"],
                "final_primary_end_time_wav_s": row["final_primary_end_time_wav_s"],
                "final_primary_valid_mask": int(valid_time),
                "timestamp_status": row["timestamp_status"],
            })
        sample_summary.append({
            "sample_id": sample_id,
            "feature_row_start_0based": sample_start,
            "word_count": len(words),
            "bert_input_token_count": len(encoded.ids),
            "valid_word_time_count": sum(r["final_primary_valid_mask"].strip().lower() == "true" for r in prior),
            "text_sha256_utf8": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
        print(f"{len(sample_summary):03d}/100 {sample_id}: {len(words)} words, {len(encoded.ids)} BERT tokens", flush=True)

    matrix = np.stack(features).astype(np.float32, copy=False)
    if matrix.shape != (1934, 768) or not np.isfinite(matrix).all():
        raise ValueError(f"Unexpected feature matrix shape: {matrix.shape}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT / "原文逐词BERT特征.npy", matrix, allow_pickle=False)
    for filename, rows in (("原文逐词索引.csv", index), ("逐样本统计.csv", sample_summary)):
        with (OUTPUT / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "status": "completed",
        "text_source": "label-100.xlsx sheet=label column=text",
        "whisper_text_used": False,
        "sample_count": len(sample_summary),
        "official_word_count": len(index),
        "feature_matrix_shape": list(matrix.shape),
        "feature_dtype": str(matrix.dtype),
        "feature_valid_word_count": len(index),
        "valid_timestamp_word_count": time_valid_count,
        "missing_timestamp_word_count": len(index) - time_valid_count,
        "encoder": "google-bert/bert-base-uncased",
        "revision": details["source_revision"],
        "source_model_sha256": details["source_sha256"],
        "encoder_model_sha256": details["encoder_sha256"],
        "tokenizer_sha256": sha256(TOKENIZER),
        "workbook_sha256": sha256(SOURCE_XLSX),
        "pooling": "arithmetic mean of final-layer BERT hidden states over WordPieces whose offsets lie within each official word",
        "word_order": "Step 4 official_word_index, checked against Excel text",
        "text_normalization": "NFKC and curly apostrophe to ASCII apostrophe before word tokenization and BERT encoding",
        "max_bert_input_tokens": max(r["bert_input_token_count"] for r in sample_summary),
        "onnxruntime_version": ort.__version__,
        "tokenizers_version": __import__("tokenizers").__version__,
        "numpy_version": np.__version__,
        "feature_file_sha256": sha256(OUTPUT / "原文逐词BERT特征.npy"),
    }
    (OUTPUT / "运行报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
