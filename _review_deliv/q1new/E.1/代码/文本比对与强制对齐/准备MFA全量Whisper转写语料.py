"""Prepare 98 audio + Whisper-transcript pairs for MFA alignment in E.1."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "输入" / "样本清单" / "样本清单.json"
WAV_DIR = ROOT / "输入" / "音频WAV"
WHISPER_DIR = ROOT / "结果" / "03_Whisper_small_en" / "逐样本"
CORPUS_DIR = ROOT / "MFA" / "corpus_whisper" / "audio"
LOG_DIR = ROOT / "日志" / "第4步文本一致性检查"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    empty_ids: list[str] = []

    for row in manifest:
        sample_id = str(row["sample_id"])
        result_path = WHISPER_DIR / f"{sample_id}_Whisper_small.en.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        text = str(result.get("transcription_text", "") or "").strip()
        if not text:
            empty_ids.append(sample_id)
            continue

        safe_id = f"sample_{len(rows) + 1:04d}"
        source_wav = WAV_DIR / f"{sample_id}.wav"
        corpus_wav = CORPUS_DIR / f"{safe_id}.wav"
        lab_path = CORPUS_DIR / f"{safe_id}.lab"
        if not source_wav.is_file():
            raise FileNotFoundError(source_wav)
        digest = sha256(source_wav)
        if corpus_wav.exists() and sha256(corpus_wav) != digest:
            raise FileExistsError(f"MFA ASR语料中的音频与源WAV不同：{corpus_wav}")
        if not corpus_wav.exists():
            shutil.copy2(source_wav, corpus_wav)
        if lab_path.exists() and lab_path.read_text(encoding="utf-8").strip() != text:
            raise FileExistsError(f"MFA ASR语料文本已存在且不同：{lab_path}")
        if not lab_path.exists():
            lab_path.write_text(text + "\n", encoding="utf-8")
        with wave.open(str(corpus_wav), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
        rows.append({
            "mfa_utterance_id": safe_id,
            "sample_id": sample_id,
            "audio_path": str(corpus_wav),
            "lab_path": str(lab_path),
            "whisper_transcript": text,
            "whisper_lexical_word_count": int(result.get("word_count", 0)),
            "audio_duration_s": duration,
            "wav_sha256": digest,
            "source_audio_start_s": float(result.get("source_audio_start_s", 0.0) or 0.0),
        })

    # Keep this corpus isolated and complete; extra files could silently enter MFA.
    expected_names = {f"sample_{i:04d}.{suffix}" for i in range(1, len(rows) + 1) for suffix in ("wav", "lab")}
    existing_names = {path.name for path in CORPUS_DIR.iterdir() if path.is_file()}
    extras = existing_names - expected_names
    if extras:
        raise ValueError(f"MFA ASR语料目录中有非本次清单文件：{sorted(extras)[:10]}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    mapping_json = LOG_DIR / "MFA全量Whisper转写语料映射.json"
    mapping_csv = LOG_DIR / "MFA全量Whisper转写语料映射.csv"
    mapping_json.write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_count": len(rows),
        "empty_asr_count": len(empty_ids),
        "empty_asr_sample_ids": empty_ids,
        "corpus_directory": str(CORPUS_DIR),
        "transcript_source": "逐样本 Whisper small.en transcription_text;题目原文未直接强制对齐",
        "mapping": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with mapping_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "mfa_corpus_count": len(rows),
        "empty_asr_count": len(empty_ids),
        "corpus_directory": str(CORPUS_DIR),
        "mapping_json": str(mapping_json),
        "mapping_csv": str(mapping_csv),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
