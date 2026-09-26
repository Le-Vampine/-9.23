"""Map MFA word alignments of Whisper transcripts back to official words.

Use an MFA interval as the primary estimate only for a uniquely matched word
that passes the interval checks and, when comparable, agrees with Whisper's
estimate within 0.5 seconds on both boundaries. Keep transcript mismatch as an
explicit evidence tier; this is not a claim of gold-standard accuracy.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import 自动比对原始文本与Whisper as compare  # noqa: E402
from 恢复原文逐词时间 import lcs_unique_mandatory_map, load_official_texts  # noqa: E402

MANIFEST = ROOT / "输入" / "样本清单" / "样本清单.json"
WHISPER_DIR = ROOT / "结果" / "03_Whisper_small_en" / "逐样本"
MFA_JSON_DIR = ROOT / "MFA" / "aligned_whisper_json"
MFA_MAP = ROOT / "日志" / "第4步文本一致性检查" / "MFA全量Whisper转写语料映射.json"
OUTPUT_DIR = ROOT / "日志" / "第4步文本一致性检查"
OUTPUT_CSV = OUTPUT_DIR / "原文词级时间_MFA候选及最终采用.csv"
SAMPLE_CSV = OUTPUT_DIR / "MFA原文时间映射逐样本统计.csv"
REPORT_JSON = OUTPUT_DIR / "MFA原文时间映射汇总.json"
MFA_BAD_LABELS = {"<unk>", "<eps>", "spn", "sil", "silence", "#0"}
REVIEW_SHORT_S = 0.02
REVIEW_LONG_S = 1.5
WHISPER_MFA_EDGE_REVIEW_S = 0.5


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def parse_mfa_words(path: Path) -> tuple[list[dict[str, Any]], float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    duration = float(data["end"])
    words: list[dict[str, Any]] = []
    for tier_index, entry in enumerate(data.get("tiers", {}).get("words", {}).get("entries", []), start=1):
        start, end, label = entry
        label = str(label).strip()
        if label.casefold() in MFA_BAD_LABELS:
            continue
        for piece_index, piece in enumerate(compare.tokenize(label), start=1):
            words.append({
                "token": piece["token"],
                "raw": label,
                "start": float(start),
                "end": float(end),
                "tier_index": tier_index,
                "piece_index": piece_index,
            })
    return words, float(data["start"]), duration


def interval_flags(start: float, end: float, duration: float, previous_start: float | None,
                   previous_end: float | None) -> list[str]:
    flags: list[str] = []
    if not (math.isfinite(start) and math.isfinite(end)):
        return ["time_missing_or_nonfinite"]
    if start < 0:
        flags.append("negative_start")
    if end <= start:
        flags.append("nonpositive_duration")
    if end > duration + compare.TIME_TOLERANCE_S:
        flags.append("beyond_audio_duration")
    if previous_start is not None and start < previous_start - 1e-9:
        flags.append("out_of_order")
    if previous_end is not None and start < previous_end - compare.TIME_TOLERANCE_S:
        flags.append("overlaps_previous_word")
    if end > start:
        if end - start < REVIEW_SHORT_S:
            flags.append("very_short_duration_review")
        if end - start > REVIEW_LONG_S:
            flags.append("very_long_duration_review")
    return flags


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    official_texts = load_official_texts()
    mfa_map = json.loads(MFA_MAP.read_text(encoding="utf-8"))
    corpus_by_sample = {row["sample_id"]: row for row in mfa_map["mapping"]}
    empty_asr = set(mfa_map.get("empty_asr_sample_ids", []))
    rows: list[dict[str, Any]] = []
    sample_stats: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    group_stats: dict[str, Counter[str]] = defaultdict(Counter)
    duration_by_sample: dict[str, float] = {}
    offset_by_sample: dict[str, float] = {}

    for manifest_row in manifest:
        sample_id = str(manifest_row["sample_id"])
        official = compare.tokenize(official_texts[sample_id])
        official_tokens = [item["token"] for item in official]
        whisper_path = WHISPER_DIR / f"{sample_id}_Whisper_small.en.json"
        whisper_result = json.loads(whisper_path.read_text(encoding="utf-8"))
        whisper_text = str(whisper_result.get("transcription_text", "") or "")
        whisper = compare.tokenize(whisper_text)
        whisper_tokens = [item["token"] for item in whisper]
        whisper_timed = [
            item for item in compare.load_timed_tokens(whisper_result)
            if not item.get("nonlexical")
        ]
        whisper_time_sequence_ok = [item["token"] for item in whisper_timed] == whisper_tokens
        exact = bool(official_tokens) and official_tokens == whisper_tokens
        duration = float(whisper_result["audio_duration_s"])
        offset = float(whisper_result.get("source_audio_start_s", 0.0) or 0.0)
        duration_by_sample[sample_id] = duration
        offset_by_sample[sample_id] = offset
        group = "empty_asr" if sample_id in empty_asr else ("exact_text" if exact else "text_mismatch")

        official_to_whisper = (
            {i: i for i in range(len(official_tokens))}
            if exact else lcs_unique_mandatory_map(official_tokens, whisper_tokens)
        )
        mfa_record = corpus_by_sample.get(sample_id)
        mfa_path = MFA_JSON_DIR / f"{mfa_record['mfa_utterance_id']}.json" if mfa_record else None
        mfa_words: list[dict[str, Any]] = []
        mfa_start = 0.0
        mfa_duration = duration
        missing_mfa = not (mfa_path and mfa_path.is_file())
        if not missing_mfa and mfa_path is not None:
            mfa_words, mfa_start, mfa_duration = parse_mfa_words(mfa_path)
        mfa_tokens = [item["token"] for item in mfa_words]
        whisper_to_mfa = lcs_unique_mandatory_map(whisper_tokens, mfa_tokens) if mfa_words else {}

        sample_candidate = 0
        sample_selected = 0
        sample_unmapped = 0
        sample_flagged = 0
        sample_discrepancy_flagged = 0
        sample_diffs: list[float] = []
        previous_start: float | None = None
        previous_end: float | None = None
        per_sample_rows: list[dict[str, Any]] = []

        for official_index, official_word in enumerate(official):
            whisper_index = official_to_whisper.get(official_index)
            mfa_index = whisper_to_mfa.get(whisper_index) if whisper_index is not None else None
            mfa_word = mfa_words[mfa_index] if mfa_index is not None and mfa_index < len(mfa_words) else None
            flags: list[str] = []
            candidate_start: float | str = ""
            candidate_end: float | str = ""
            candidate_source_start: float | str = ""
            candidate_source_end: float | str = ""
            whisper_start: float | str = ""
            whisper_end: float | str = ""
            edge_diff: float | str = ""

            if sample_id in empty_asr:
                status = "asr_empty_no_timestamp"
            elif missing_mfa:
                status = "mfa_alignment_output_missing"
            elif whisper_index is None:
                status = "official_word_unmatched_or_ambiguous_in_asr"
            elif mfa_word is None:
                status = "asr_word_unmatched_or_ambiguous_in_mfa"
            else:
                start = float(mfa_word["start"])
                end = float(mfa_word["end"])
                flags = interval_flags(start, end, min(duration, mfa_duration), previous_start, previous_end)
                previous_start, previous_end = start, end
                if flags:
                    status = "mfa_interval_flagged_null"
                    sample_flagged += 1
                else:
                    candidate_start = start
                    candidate_end = end
                    candidate_source_start = start + offset
                    candidate_source_end = end + offset
                    sample_candidate += 1
                    if whisper_time_sequence_ok and whisper_index < len(whisper_timed):
                        whisper_start = float(whisper_timed[whisper_index]["start_wav"])
                        whisper_end = float(whisper_timed[whisper_index]["end_wav"])
                        edge_diff = max(abs(start - whisper_start), abs(end - whisper_end))
                        sample_diffs.append(float(edge_diff))
                    if edge_diff != "" and float(edge_diff) > WHISPER_MFA_EDGE_REVIEW_S:
                        status = "mfa_whisper_discrepancy_over_0_5s_null_primary"
                        sample_discrepancy_flagged += 1
                    else:
                        status = "mfa_primary_exact_transcript_auto_qc_pass" if exact else "mfa_primary_unique_word_overlap_auto_qc_pass"

            selected = status in {
                "mfa_primary_exact_transcript_auto_qc_pass",
                "mfa_primary_unique_word_overlap_auto_qc_pass",
            }
            if edge_diff != "" and float(edge_diff) > WHISPER_MFA_EDGE_REVIEW_S:
                flags.append("mfa_whisper_edge_difference_over_0_5s")
            if selected:
                sample_selected += 1
            if status in {"official_word_unmatched_or_ambiguous_in_asr", "asr_word_unmatched_or_ambiguous_in_mfa"}:
                sample_unmapped += 1
            statuses[status] += 1
            group_stats[group][status] += 1
            row = {
                "sample_id": sample_id,
                "transcript_group": group,
                "official_word_index": official_index + 1,
                "official_word": official_word["raw"],
                "whisper_token_index": whisper_index + 1 if whisper_index is not None else "",
                "mfa_token_index": mfa_index + 1 if mfa_index is not None else "",
                "mfa_word": mfa_word["raw"] if mfa_word else "",
                "mfa_tier_word_index": mfa_word["tier_index"] if mfa_word else "",
                "mfa_candidate_start_time_wav_s": candidate_start,
                "mfa_candidate_end_time_wav_s": candidate_end,
                "mfa_candidate_start_time_source_s": candidate_source_start,
                "mfa_candidate_end_time_source_s": candidate_source_end,
                "whisper_start_time_wav_s": whisper_start,
                "whisper_end_time_wav_s": whisper_end,
                "mfa_whisper_max_edge_difference_s": edge_diff,
                "mfa_mapping_method": "mandatory_unique_exact_token_pair_in_all_optimal_LCS_alignments",
                "timestamp_status": status,
                "timestamp_evidence_tier": (
                    "A_exact_transcript" if candidate_start != "" and exact else
                    "B_unique_word_overlap_in_mismatched_transcript" if candidate_start != "" else ""
                ),
                "timestamp_source": "MFA 3.4.2 forced alignment of Whisper transcript" if candidate_start != "" else "",
                "mfa_whisper_discrepancy_over_0_5s": bool(edge_diff != "" and float(edge_diff) > WHISPER_MFA_EDGE_REVIEW_S),
                "final_primary_start_time_wav_s": candidate_start if selected else "",
                "final_primary_end_time_wav_s": candidate_end if selected else "",
                "final_primary_start_time_source_s": candidate_source_start if selected else "",
                "final_primary_end_time_source_s": candidate_source_end if selected else "",
                "final_primary_valid_mask": bool(selected),
                "candidate_valid_mask": candidate_start != "",
                "automatic_flags": ";".join(flags),
                "mfa_alignment_file": mfa_path.name if mfa_path and mfa_path.is_file() else "",
            }
            rows.append(row)
            per_sample_rows.append(row)

        sample_stats.append({
            "sample_id": sample_id,
            "transcript_group": group,
            "official_word_count": len(official),
            "whisper_word_count": len(whisper),
            "official_whisper_exact_match": exact,
            "mfa_alignment_output_present": not missing_mfa,
            "official_to_asr_unique_mappings": len(official_to_whisper),
            "asr_to_mfa_unique_mappings": len(whisper_to_mfa),
            "mfa_candidate_word_times": sample_candidate,
            "selected_primary_word_times": sample_selected,
            "unmapped_words": sample_unmapped,
            "flagged_mfa_intervals": sample_flagged,
            "flagged_whisper_mfa_discrepancies": sample_discrepancy_flagged,
            "median_mfa_whisper_edge_difference_s": sorted(sample_diffs)[len(sample_diffs) // 2] if sample_diffs else None,
        })

    keys = [(row["sample_id"], row["official_word_index"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("MFA映射结果出现重复的 sample_id + official_word_index")
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)
        if not row["candidate_valid_mask"] and any(row[name] != "" for name in (
            "mfa_candidate_start_time_wav_s", "mfa_candidate_end_time_wav_s",
            "mfa_candidate_start_time_source_s", "mfa_candidate_end_time_source_s",
        )):
            raise ValueError(f"候选掩码为 false 但时间非空：{row['sample_id']}/{row['official_word_index']}")
        if not row["final_primary_valid_mask"] and any(row[name] != "" for name in (
            "final_primary_start_time_wav_s", "final_primary_end_time_wav_s",
            "final_primary_start_time_source_s", "final_primary_end_time_source_s",
        )):
            raise ValueError(f"最终掩码为 false 但时间非空：{row['sample_id']}/{row['official_word_index']}")

    for sample_id, sample_rows in by_sample.items():
        previous_start = -1.0
        for row in sample_rows:
            if not row["candidate_valid_mask"]:
                continue
            start = float(row["mfa_candidate_start_time_wav_s"])
            end = float(row["mfa_candidate_end_time_wav_s"])
            if start < 0 or end <= start or end > duration_by_sample[sample_id] + compare.TIME_TOLERANCE_S:
                raise ValueError(f"保留的MFA候选时间越界：{sample_id}/{row['official_word_index']}")
            if abs(float(row["mfa_candidate_start_time_source_s"]) - (start + offset_by_sample[sample_id])) > 1e-6:
                raise ValueError(f"源媒体起点偏移错误：{sample_id}/{row['official_word_index']}")
            if abs(float(row["mfa_candidate_end_time_source_s"]) - (end + offset_by_sample[sample_id])) > 1e-6:
                raise ValueError(f"源媒体终点偏移错误：{sample_id}/{row['official_word_index']}")
            if start < previous_start - 1e-9:
                raise ValueError(f"MFA映射后的时间顺序倒退：{sample_id}/{row['official_word_index']}")
            previous_start = start

    write_csv(OUTPUT_CSV, rows)
    write_csv(SAMPLE_CSV, sample_stats)
    candidates = sum(bool(row["candidate_valid_mask"]) for row in rows)
    primary = sum(bool(row["final_primary_valid_mask"]) for row in rows)
    missing_alignment_ids = [
        row["mfa_utterance_id"] for row in mfa_map["mapping"]
        if not (MFA_JSON_DIR / f"{row['mfa_utterance_id']}.json").is_file()
    ]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mfa_run": {
            "version": "3.4.2",
            "dictionary": "english_us_arpa",
            "acoustic_model": "english_us_arpa",
            "corpus_transcript": "Whisper small.en transcription_text",
            "aligned_utterances": len(list(MFA_JSON_DIR.glob("*.json"))),
            "corpus_utterances": len(mfa_map["mapping"]),
            "alignment_errors": missing_alignment_ids,
            "run_log": str(OUTPUT_DIR / "MFA全量Whisper文本强制对齐控制台.json"),
        },
        "why_mfa_previously_hung": "Kalpy懒加载librosa时Numba默认缓存目录初始化停滞；将NUMBA_CACHE_DIR重定向到E.1内后，librosa导入及MFA运行正常。另有OpenFST帮助检测误判，已由E.1兼容入口绕过。",
        "timestamp_policy": {
            "primary": "仅当官方词到Whisper词及Whisper词到MFA词均为所有最优LCS中唯一必选的精确同词匹配、MFA区间通过自动边界规则，且可比对时MFA与Whisper任一边界差不超过0.5秒，才选为主时间戳。",
            "mismatch_samples": "不一致样本中符合上述逐词条件的时间由MFA对Whisper转写强制对齐产生，纳入主时间轴但标记为B层；整句文本完全一致的词标记为A层。两层均是自动估计，不是人工真值。",
            "empty_asr_or_unmatched_or_flagged_or_missing_mfa": "start/end留空并置valid_mask=false。",
            "limit": "MFA强制对齐是声学模型在给定转写约束下的估计；自动筛查不能证明词确实发出或时间边界为人工真值。",
            "interval_checks": ["非负且正时长", "不超WAV时长+0.05秒", "时间顺序", "相邻重叠不超过0.05秒", "0.02秒以下或1.5秒以上标记复核", "MFA与Whisper可比词的任一边界差超过0.5秒时主时间留空"],
        },
        "sample_count": len(sample_stats),
        "official_word_count": len(rows),
        "mfa_candidate_word_time_count": candidates,
        "mfa_candidate_coverage": candidates / len(rows) if rows else None,
        "final_primary_word_time_count": primary,
        "final_primary_coverage": primary / len(rows) if rows else None,
        "final_primary_null_word_count": len(rows) - primary,
        "status_counts": dict(sorted(statuses.items())),
        "group_counts": {group: dict(sorted(counts.items())) for group, counts in sorted(group_stats.items())},
        "evidence_tier_counts": {
            "A_exact_transcript": sum(row["final_primary_valid_mask"] and row["timestamp_evidence_tier"] == "A_exact_transcript" for row in rows),
            "B_unique_word_overlap_in_mismatched_transcript": sum(row["final_primary_valid_mask"] and row["timestamp_evidence_tier"] == "B_unique_word_overlap_in_mismatched_transcript" for row in rows),
        },
        "mfa_whisper_discrepancy_threshold_s": WHISPER_MFA_EDGE_REVIEW_S,
        "integrity_checks": {
            "unique_sample_word_keys": True,
            "candidate_false_mask_has_null_times": True,
            "primary_false_mask_has_null_times": True,
            "retained_candidate_bounds_positive_and_within_duration": True,
            "source_time_offset_mapping_consistent": True,
            "candidate_word_start_order_per_sample": True,
        },
        "files": {
            "word_time_csv": str(OUTPUT_CSV),
            "per_sample_csv": str(SAMPLE_CSV),
            "summary_json": str(REPORT_JSON),
        },
        "samples": sample_stats,
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "sample_count", "official_word_count", "mfa_candidate_word_time_count",
        "mfa_candidate_coverage", "final_primary_word_time_count", "final_primary_coverage",
        "final_primary_null_word_count", "status_counts", "group_counts",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
