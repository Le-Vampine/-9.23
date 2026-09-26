"""Recover safe per-word mappings for text-mismatch samples without listening.

Only exact tokens that must pair to one unique Whisper token in every optimal
LCS mapping inherit the ASR word interval. Unmatched/ambiguous tokens stay null.
Whisper intervals remain estimates; this script does not claim gold boundaries.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openpyxl


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import 自动比对原始文本与Whisper as compare  # noqa: E402

MANIFEST = ROOT / "输入" / "样本清单" / "样本清单.json"
SOURCE_XLSX = ROOT / "输入" / "原始标签" / "label-100.xlsx"
WHISPER_DIR = ROOT / "结果" / "03_Whisper_small_en" / "逐样本"
OUTPUT_DIR = ROOT / "日志" / "第4步文本一致性检查"
OUTPUT_CSV = OUTPUT_DIR / "原文词级时间_逐词保守恢复.csv"
OUTPUT_JSON = OUTPUT_DIR / "逐词保守恢复汇总.json"
SAMPLE_CSV = OUTPUT_DIR / "逐词恢复逐样本统计.csv"


def load_official_texts() -> dict[str, str]:
    workbook = openpyxl.load_workbook(SOURCE_XLSX, read_only=True, data_only=True)
    try:
        sheet = workbook["label"]
        iterator = sheet.iter_rows(values_only=True)
        header = next(iterator)
        pos = {str(value).strip(): i for i, value in enumerate(header) if value is not None}
        texts: dict[str, str] = {}
        for row in iterator:
            video_id = str(row[pos["video_id"]] or "").strip()
            clip_value = row[pos["clip_id"]]
            clip_id = str(clip_value).strip()
            if isinstance(clip_value, float) and clip_value.is_integer():
                clip_id = str(int(clip_value))
            if video_id and clip_id:
                sample_id = f"{video_id}$_${clip_id}"
                value = row[pos["text"]]
                texts[sample_id] = "" if value is None else str(value)
        return texts
    finally:
        workbook.close()


def lcs_unique_mandatory_map(a: list[str], b: list[str]) -> dict[int, int]:
    """Map 0-based a indices only when every optimal LCS pairs to one b index."""
    n, m = len(a), len(b)
    prefix = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                prefix[i + 1][j + 1] = prefix[i][j] + 1
            else:
                prefix[i + 1][j + 1] = max(prefix[i][j + 1], prefix[i + 1][j])
    suffix = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                suffix[i][j] = suffix[i + 1][j + 1] + 1
            else:
                suffix[i][j] = max(suffix[i + 1][j], suffix[i][j + 1])
    optimum = prefix[n][m]
    mapping: dict[int, int] = {}
    if not optimum:
        return mapping

    for i in range(n):
        # If omitting a[i] still reaches optimum, some optimal LCS may skip it.
        best_without_i = 0
        for j in range(m + 1):
            best_without_i = max(best_without_i, prefix[i][j] + suffix[i + 1][j])
        if best_without_i == optimum:
            continue
        possible_js = [
            j for j in range(m)
            if a[i] == b[j] and prefix[i][j] + 1 + suffix[i + 1][j + 1] == optimum
        ]
        if len(possible_js) == 1:
            mapping[i] = possible_js[0]
    return mapping


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    official_texts = load_official_texts()
    rows: list[dict[str, Any]] = []
    sample_counts: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    duration_by_sample: dict[str, float] = {}
    source_offset_by_sample: dict[str, float] = {}

    for manifest_row in manifest:
        sample_id = str(manifest_row["sample_id"])
        official_text = official_texts[sample_id]
        official = compare.tokenize(official_text)
        result_path = WHISPER_DIR / f"{sample_id}_Whisper_small.en.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        duration_by_sample[sample_id] = float(result["audio_duration_s"])
        source_offset_by_sample[sample_id] = float(result.get("source_audio_start_s", 0.0) or 0.0)
        whisper_text = str(result.get("transcription_text", "") or "")
        whisper = compare.tokenize(whisper_text)
        timed = compare.load_timed_tokens(result)
        lexical_timed = [item for item in timed if not item.get("nonlexical")]
        time_sequence_ok = [x["token"] for x in lexical_timed] == [x["token"] for x in whisper]
        audit_rows, _ = compare.audit_word_times(sample_id, timed, result.get("audio_duration_s"))
        audit_by_position = {
            (row["whisper_word_position"], row["token_piece_position"]): row
            for row in audit_rows
        }
        a = [item["token"] for item in official]
        b = [item["token"] for item in whisper]
        exact = bool(a) and a == b
        mapping = {i: i for i in range(len(a))} if exact else lcs_unique_mandatory_map(a, b)
        sample_recovered = 0
        sample_unambiguous_matches = len(mapping)
        sample_flagged = 0

        for i, official_token in enumerate(official):
            whisper_index = mapping.get(i)
            item = lexical_timed[whisper_index] if (
                whisper_index is not None and time_sequence_ok and whisper_index < len(lexical_timed)
            ) else None
            audit = audit_by_position.get((item.get("word_position"), item.get("piece_position")), {}) if item else {}
            flags = str(audit.get("automatic_flags", "") or "")
            if not whisper_text.strip():
                status = "asr_empty_no_original_time"
            elif not time_sequence_ok:
                status = "whisper_text_time_sequence_mismatch"
                flags = "whisper_text_word_sequence_mismatch"
            elif whisper_index is None:
                status = "official_word_unmatched_or_ambiguous"
            elif flags:
                status = "whisper_time_flagged_null"
                sample_flagged += 1
            elif item is None:
                status = "whisper_word_time_unmapped"
                flags = "whisper_word_time_unmapped"
            else:
                status = "exact_text_whisper_estimate_auto_qc_pass" if exact else "partial_whisper_estimate_auto_qc_pass"
                sample_recovered += 1

            keep_time = status in {
                "exact_text_whisper_estimate_auto_qc_pass",
                "partial_whisper_estimate_auto_qc_pass",
            }
            row = {
                "sample_id": sample_id,
                "official_word_index": i + 1,
                "official_word": official_token["raw"],
                "whisper_token_index": (whisper_index + 1) if whisper_index is not None else "",
                "whisper_word_index": item["word_position"] if item and whisper_index is not None else "",
                "whisper_word": item["raw"] if item and whisper_index is not None else "",
                "start_time_wav_s": item["start_wav"] if keep_time and item else "",
                "end_time_wav_s": item["end_wav"] if keep_time and item else "",
                "start_time_source_s": item["start_source"] if keep_time and item else "",
                "end_time_source_s": item["end_source"] if keep_time and item else "",
                "word_probability": item["probability"] if item and whisper_index is not None else "",
                "mapping_method": "exact_normalized_sequence" if exact else "mandatory_unique_pair_in_all_optimal_LCS",
                "timestamp_source": "Whisper small.en word timestamp estimate" if keep_time else "",
                "timestamp_status": status,
                "word_time_valid_mask": keep_time,
                "automatic_flags": flags,
            }
            rows.append(row)
            status_counts[status] += 1

        sample_counts.append({
            "sample_id": sample_id,
            "official_word_count": len(official),
            "whisper_word_count": len(whisper),
            "exact_normalized_text_match": exact,
            "unique_mandatory_token_mappings": sample_unambiguous_matches,
            "recovered_word_times": sample_recovered,
            "mapped_but_time_flagged": sample_flagged,
            "unmatched_or_ambiguous_words": len(official) - sample_unambiguous_matches,
            "time_sequence_matches_whisper_text": time_sequence_ok,
        })

    keys = [(row["sample_id"], row["official_word_index"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("恢复结果中出现重复的 sample_id + official_word_index")
    rows_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_sample[row["sample_id"]].append(row)
    for sample_id, sample_rows in rows_by_sample.items():
        previous_start = -1.0
        for row in sample_rows:
            if not row["word_time_valid_mask"]:
                if any(row[name] != "" for name in (
                    "start_time_wav_s", "end_time_wav_s", "start_time_source_s", "end_time_source_s"
                )):
                    raise ValueError(f"缺失掩码为 false 但时间不为空：{sample_id}/{row['official_word_index']}")
                continue
            start = float(row["start_time_wav_s"])
            end = float(row["end_time_wav_s"])
            source_start = float(row["start_time_source_s"])
            source_end = float(row["end_time_source_s"])
            if start < 0 or end <= start or end > duration_by_sample[sample_id] + compare.TIME_TOLERANCE_S:
                raise ValueError(f"保留的 Whisper 时间超出边界规则：{sample_id}/{row['official_word_index']}")
            if abs(source_start - (start + source_offset_by_sample[sample_id])) > 1e-6:
                raise ValueError(f"源媒体起点映射不一致：{sample_id}/{row['official_word_index']}")
            if abs(source_end - (end + source_offset_by_sample[sample_id])) > 1e-6:
                raise ValueError(f"源媒体终点映射不一致：{sample_id}/{row['official_word_index']}")
            if start < previous_start - 1e-9:
                raise ValueError(f"原文词时间顺序倒退：{sample_id}/{row['official_word_index']}")
            previous_start = start

    fields = list(rows[0]) if rows else []
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    sample_fields = list(sample_counts[0]) if sample_counts else []
    with SAMPLE_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(sample_counts)

    recovered = sum(bool(row["word_time_valid_mask"]) for row in rows)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "仅对原文与Whisper之间在所有最优LCS匹配中都必须匹配到唯一同词位置的原文词继承Whisper词时间；未匹配、歧义或时间质检有标记的词留空。",
        "important_limit": "Whisper词时间是估计值，不等于MFA强制对齐，也不构成边界精度证明。",
        "mfa_status": "MFA 3.4.2 单条 align_one 试跑超过 7 分钟未生成结果，已停止；当前未接受任何 MFA 时间戳。",
        "automated_integrity_checks": {
            "unique_sample_word_keys": True,
            "false_mask_has_null_times": True,
            "retained_time_bounds_and_positive_duration": True,
            "source_time_mapping_consistent": True,
            "retained_word_start_order_per_sample": True,
        },
        "sample_count": len(sample_counts),
        "official_word_count": len(rows),
        "retained_word_time_count": recovered,
        "null_word_time_count": len(rows) - recovered,
        "retained_rate": recovered / len(rows) if rows else None,
        "status_counts": dict(sorted(status_counts.items())),
        "files": {"word_time_csv": str(OUTPUT_CSV), "per_sample_csv": str(SAMPLE_CSV)},
        "samples": sample_counts,
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        key: report[key] for key in (
            "sample_count", "official_word_count", "retained_word_time_count",
            "null_word_time_count", "retained_rate", "status_counts",
        )
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
