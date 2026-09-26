"""Automatically compare official transcripts with Whisper and audit word times.

The script does not listen to audio or ask for manual labels. It uses a
conservative exact-token gate to identify candidates for a later forced-alignment
run. A candidate status is an automated screen, not proof that the transcript is
correct.
"""
from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openpyxl


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "输入" / "样本清单" / "样本清单.json"
SOURCE_XLSX = ROOT / "输入" / "原始标签" / "label-100.xlsx"
WHISPER_DIR = ROOT / "结果" / "03_Whisper_small_en" / "逐样本"
OUTPUT_DIR = ROOT / "日志" / "第4步文本一致性检查"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")
NEGATIONS = {
    "no", "not", "never", "none", "neither", "nor", "cannot", "can't",
    "won't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't",
    "weren't", "hasn't", "haven't", "hadn't", "shouldn't", "wouldn't",
    "couldn't", "mustn't", "needn't", "without", "n't",
}
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "first", "second",
    "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
    "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
}
TIME_TOLERANCE_S = 0.05
VERY_SHORT_WORD_S = 0.02
VERY_LONG_WORD_S = 1.5


def canonical(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    return value.replace("’", "'").replace("‘", "'").casefold()


def tokenize(value: str) -> list[dict[str, str]]:
    value = unicodedata.normalize("NFKC", value or "").replace("’", "'").replace("‘", "'")
    return [{"raw": match.group(0), "token": match.group(0).casefold()}
            for match in TOKEN_RE.finditer(value)]


def is_critical(token: str) -> bool:
    token = canonical(token)
    return token in NEGATIONS or bool(re.search(r"\d", token)) or token in NUMBER_WORDS


def align_tokens(official: list[dict[str, str]], whisper: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Levenshtein backtrace, preferring a substitution on tied edit paths."""
    a = [item["token"] for item in official]
    b = [item["token"] for item in whisper]
    rows, cols = len(a) + 1, len(b) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = i
    for j in range(1, cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            )

    operations: list[dict[str, Any]] = []
    i, j = len(a), len(b)
    while i or j:
        if i and j and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            operations.append({"op": "match", "official_index": i, "whisper_index": j})
            i -= 1
            j -= 1
            continue
        if i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            operations.append({"op": "substitution", "official_index": i, "whisper_index": j})
            i -= 1
            j -= 1
            continue
        if i and dp[i][j] == dp[i - 1][j] + 1:
            operations.append({"op": "deletion_official", "official_index": i, "whisper_index": None})
            i -= 1
            continue
        operations.append({"op": "insertion_whisper", "official_index": None, "whisper_index": j})
        j -= 1

    operations.reverse()
    for item in operations:
        oi, wi = item["official_index"], item["whisper_index"]
        item["official_token"] = official[oi - 1]["raw"] if oi else ""
        item["whisper_token"] = whisper[wi - 1]["raw"] if wi else ""
        item["critical_mismatch"] = bool(
            item["op"] != "match" and
            ((oi and is_critical(official[oi - 1]["token"])) or
             (wi and is_critical(whisper[wi - 1]["token"])))
        )
    return operations


def load_official_texts() -> tuple[dict[str, str], dict[str, Any]]:
    workbook = openpyxl.load_workbook(SOURCE_XLSX, read_only=True, data_only=True)
    try:
        sheet = workbook["label"]
        iterator = sheet.iter_rows(values_only=True)
        header = next(iterator)
        positions = {str(value).strip(): index for index, value in enumerate(header) if value is not None}
        required = {"video_id", "clip_id", "text"}
        if not required.issubset(positions):
            raise ValueError(f"原始标签表缺少列：{sorted(required - set(positions))}")
        texts: dict[str, str] = {}
        for row_no, row in enumerate(iterator, start=2):
            video_id = str(row[positions["video_id"]] or "").strip()
            clip_value = row[positions["clip_id"]]
            clip_id = str(clip_value).strip()
            if isinstance(clip_value, float) and clip_value.is_integer():
                clip_id = str(int(clip_value))
            if not video_id or not clip_id:
                continue
            sample_id = f"{video_id}$_${clip_id}"
            if sample_id in texts:
                raise ValueError(f"原始标签工作簿第 {row_no} 行出现重复编号：{sample_id}")
            value = row[positions["text"]]
            texts[sample_id] = "" if value is None else str(value)
        return texts, {"sheet": sheet.title, "data_rows": len(texts), "columns": list(positions)}
    finally:
        workbook.close()


def load_timed_tokens(result: dict[str, Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for word_position, word in enumerate(result.get("word_timestamps", []), start=1):
        surface = str(word.get("word", ""))
        pieces = tokenize(surface)
        if not pieces:
            flattened.append({
                "token": "",
                "raw": surface,
                "word_position": word_position,
                "piece_position": 0,
                "start_wav": word.get("start_time_wav_s"),
                "end_wav": word.get("end_time_wav_s"),
                "start_source": word.get("start_time_source_s"),
                "end_source": word.get("end_time_source_s"),
                "probability": word.get("probability"),
                "nonlexical": True,
            })
        for piece_position, piece in enumerate(pieces, start=1):
            flattened.append({
                "token": piece["token"],
                "raw": piece["raw"],
                "word_position": word_position,
                "piece_position": piece_position,
                "start_wav": word.get("start_time_wav_s"),
                "end_wav": word.get("end_time_wav_s"),
                "start_source": word.get("start_time_source_s"),
                "end_source": word.get("end_time_source_s"),
                "probability": word.get("probability"),
                "nonlexical": False,
            })
    return flattened


def audit_word_times(sample_id: str, timed_tokens: list[dict[str, Any]], duration: Any) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    sample_issues: list[str] = []
    previous_start: float | None = None
    previous_end: float | None = None
    duration_value = float(duration) if isinstance(duration, (int, float)) else None
    for position, item in enumerate(timed_tokens, start=1):
        flags: list[str] = []
        if item.get("nonlexical"):
            rows.append({
                "sample_id": sample_id,
                "whisper_word_position": item["word_position"],
                "token_piece_position": item["piece_position"],
                "whisper_token": item["raw"],
                "start_time_wav_s": item["start_wav"],
                "end_time_wav_s": item["end_wav"],
                "start_time_source_s": item["start_source"],
                "end_time_source_s": item["end_source"],
                "duration_s": None,
                "probability": item["probability"],
                "automatic_flags": "nonlexical_punctuation_item",
            })
            continue
        start, end = item["start_wav"], item["end_wav"]
        valid_numbers = isinstance(start, (int, float)) and isinstance(end, (int, float))
        if not valid_numbers or not math.isfinite(float(start)) or not math.isfinite(float(end)):
            flags.append("time_missing_or_nonfinite")
        else:
            start_f, end_f = float(start), float(end)
            if start_f < 0:
                flags.append("negative_start")
            if end_f <= start_f:
                flags.append("nonpositive_duration")
            if duration_value is not None and end_f > duration_value + TIME_TOLERANCE_S:
                flags.append("beyond_audio_duration")
            if previous_start is not None and start_f < previous_start - 1e-9:
                flags.append("out_of_order")
            if previous_end is not None and start_f < previous_end - TIME_TOLERANCE_S:
                flags.append("overlaps_previous_word")
            if end_f > start_f:
                word_duration = end_f - start_f
                if word_duration < VERY_SHORT_WORD_S:
                    flags.append("very_short_duration_review")
                if word_duration > VERY_LONG_WORD_S:
                    flags.append("very_long_duration_review")
            previous_start, previous_end = start_f, end_f
        if flags:
            sample_issues.extend(flags)
        rows.append({
            "sample_id": sample_id,
            "whisper_word_position": item["word_position"],
            "token_piece_position": item["piece_position"],
            "whisper_token": item["raw"],
            "start_time_wav_s": item["start_wav"],
            "end_time_wav_s": item["end_wav"],
            "start_time_source_s": item["start_source"],
            "end_time_source_s": item["end_source"],
            "duration_s": (float(end) - float(start)) if valid_numbers and math.isfinite(float(end)) and math.isfinite(float(start)) else None,
            "probability": item["probability"],
            "automatic_flags": ";".join(dict.fromkeys(flags)),
        })
    return rows, list(dict.fromkeys(sample_issues))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    official_texts, workbook_info = load_official_texts()
    sample_ids = [str(row["sample_id"]) for row in manifest]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("样本清单中存在重复 sample_id")
    if set(sample_ids) != set(official_texts):
        missing = sorted(set(sample_ids) - set(official_texts))
        extra = sorted(set(official_texts) - set(sample_ids))
        raise ValueError(f"清单与原始标签工作簿编号不一致：缺少={missing[:5]}，多出={extra[:5]}")

    summary_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    official_time_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    for manifest_row in manifest:
        sample_id = str(manifest_row["sample_id"])
        official_text = official_texts[sample_id]
        if str(manifest_row.get("text", "")) != official_text:
            raise ValueError(f"样本清单里的原文与工作簿不一致：{sample_id}")
        result_path = WHISPER_DIR / f"{sample_id}_Whisper_small.en.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        whisper_text = str(result.get("transcription_text", "") or "")
        official_tokens = tokenize(official_text)
        whisper_tokens = tokenize(whisper_text)
        operations = align_tokens(official_tokens, whisper_tokens)
        substitutions = sum(item["op"] == "substitution" for item in operations)
        deletions = sum(item["op"] == "deletion_official" for item in operations)
        insertions = sum(item["op"] == "insertion_whisper" for item in operations)
        critical_mismatches = sum(item["critical_mismatch"] for item in operations)
        edit_count = substitutions + deletions + insertions
        wer = edit_count / len(official_tokens) if official_tokens else None
        exact_match = bool(official_tokens) and [x["token"] for x in official_tokens] == [x["token"] for x in whisper_tokens]

        if not official_tokens:
            gate = "source_text_missing_no_align"
            reason = "原始文本为空"
        elif not whisper_text.strip():
            gate = "asr_empty_no_align"
            reason = "Whisper未输出文本"
        elif exact_match:
            gate = "candidate_for_mfa_auto_screen"
            reason = "忽略大小写和标点后词序列完全一致"
        else:
            gate = "text_mismatch_no_align"
            reason = "归一化词序列存在插入、漏词或替换；不对原文强制对齐"

        timed_tokens = load_timed_tokens(result)
        lexical_timed_tokens = [x for x in timed_tokens if not x.get("nonlexical")]
        nonlexical_count = len(timed_tokens) - len(lexical_timed_tokens)
        timed_text_matches = [x["token"] for x in lexical_timed_tokens] == [x["token"] for x in whisper_tokens]
        audited_times, time_issues = audit_word_times(sample_id, timed_tokens, result.get("audio_duration_s"))
        time_rows.extend(audited_times)
        time_issue_count = sum(bool(row["automatic_flags"]) for row in audited_times)
        timed_by_index = lexical_timed_tokens if timed_text_matches else []
        audited_by_position = {
            (row["whisper_word_position"], row["token_piece_position"]): row
            for row in audited_times
        }
        for operation in operations:
            wi = operation["whisper_index"]
            time_item = timed_by_index[wi - 1] if wi and timed_by_index and wi <= len(timed_by_index) else {}
            flags = ""
            if time_item:
                matched_time_row = next((r for r in audited_times if r["whisper_word_position"] == time_item["word_position"] and r["token_piece_position"] == time_item["piece_position"]), None)
                flags = matched_time_row["automatic_flags"] if matched_time_row else ""
            token_rows.append({
                "sample_id": sample_id,
                "comparison_operation": operation["op"],
                "official_token_index": operation["official_index"] or "",
                "official_token": operation["official_token"],
                "whisper_token_index": wi or "",
                "whisper_token": operation["whisper_token"],
                "critical_token_difference": operation["critical_mismatch"],
                "whisper_start_time_wav_s": time_item.get("start_wav", ""),
                "whisper_end_time_wav_s": time_item.get("end_wav", ""),
                "whisper_start_time_source_s": time_item.get("start_source", ""),
                "whisper_end_time_source_s": time_item.get("end_source", ""),
                "whisper_time_automatic_flags": flags,
                "timestamp_comparison_note": "Whisper估计时间；原文词时间待MFA步骤" if wi else "原文词时间待MFA步骤",
            })

        for official_index, official_token in enumerate(official_tokens, start=1):
            mapped = bool(exact_match and timed_text_matches and official_index <= len(timed_by_index))
            time_item = timed_by_index[official_index - 1] if mapped else {}
            audit_item = audited_by_position.get((time_item.get("word_position"), time_item.get("piece_position")), {}) if mapped else {}
            flags = str(audit_item.get("automatic_flags", "") or "")
            if not exact_match:
                status = "text_mismatch_no_original_time" if whisper_text.strip() else "asr_empty_no_original_time"
                start_wav = end_wav = start_source = end_source = ""
                whisper_token = ""
                whisper_index = ""
                valid = False
            elif not mapped:
                status = "whisper_word_time_unmapped"
                start_wav = end_wav = start_source = end_source = ""
                whisper_token = ""
                whisper_index = ""
                valid = False
                flags = "whisper_text_word_sequence_mismatch"
            elif flags:
                status = "whisper_time_flagged_null"
                start_wav = end_wav = start_source = end_source = ""
                whisper_token = time_item["raw"]
                whisper_index = time_item["word_position"]
                valid = False
            else:
                status = "whisper_estimate_auto_qc_pass"
                start_wav = time_item["start_wav"]
                end_wav = time_item["end_wav"]
                start_source = time_item["start_source"]
                end_source = time_item["end_source"]
                whisper_token = time_item["raw"]
                whisper_index = time_item["word_position"]
                valid = True
            official_time_rows.append({
                "sample_id": sample_id,
                "official_word_index": official_index,
                "official_word": official_token["raw"],
                "whisper_word_index": whisper_index,
                "whisper_word": whisper_token,
                "start_time_wav_s": start_wav,
                "end_time_wav_s": end_wav,
                "start_time_source_s": start_source,
                "end_time_source_s": end_source,
                "timestamp_source": "Whisper small.en word timestamp estimate" if valid else "",
                "timestamp_status": status,
                "word_time_valid_mask": valid,
                "automatic_flags": flags,
            })

        summary_rows.append({
            "sample_id": sample_id,
            "official_transcript": official_text,
            "whisper_transcript": whisper_text,
            "official_token_count": len(official_tokens),
            "whisper_token_count": len(whisper_tokens),
            "substitutions": substitutions,
            "official_words_missing_from_whisper": deletions,
            "extra_whisper_words": insertions,
            "word_error_rate": round(wer, 6) if wer is not None else "",
            "critical_token_mismatch_count": critical_mismatches,
            "normalized_sequence_exact": exact_match,
            "whisper_word_timestamp_count": len(timed_tokens),
            "whisper_nonlexical_timestamp_item_count": nonlexical_count,
            "whisper_text_matches_timed_word_sequence": timed_text_matches,
            "whisper_word_timestamp_flag_count": time_issue_count,
            "whisper_word_timestamp_flags": ";".join(time_issues),
            "mfa_gate": gate,
            "automatic_reason": reason,
            "source_audio_start_s": result.get("source_audio_start_s"),
            "audio_duration_s": result.get("audio_duration_s"),
            "language_probability": result.get("language_probability"),
            "mean_word_probability": result.get("mean_word_probability"),
        })
        report_rows.append({
            "sample_id": sample_id,
            "gate": gate,
            "exact_match": exact_match,
            "wer": wer,
            "edit_count": edit_count,
            "critical_mismatch_count": critical_mismatches,
            "whisper_time_flag_count": time_issue_count,
            "timed_word_sequence_matches_asr_text": timed_text_matches,
        })

    summary_path = OUTPUT_DIR / "100条原始文本与Whisper比对汇总.csv"
    token_path = OUTPUT_DIR / "逐词对照与Whisper初步时间.csv"
    times_path = OUTPUT_DIR / "Whisper词时间戳自动核查.csv"
    official_times_path = OUTPUT_DIR / "原文词级时间候选_Whisper估计.csv"
    write_csv(summary_path, summary_rows, list(summary_rows[0]))
    write_csv(token_path, token_rows, list(token_rows[0]))
    write_csv(times_path, time_rows, list(time_rows[0]) if time_rows else [
        "sample_id", "whisper_word_position", "token_piece_position", "whisper_token",
        "start_time_wav_s", "end_time_wav_s", "start_time_source_s", "end_time_source_s",
        "duration_s", "probability", "automatic_flags",
    ])
    write_csv(official_times_path, official_time_rows, list(official_time_rows[0]) if official_time_rows else [
        "sample_id", "official_word_index", "official_word", "whisper_word_index", "whisper_word",
        "start_time_wav_s", "end_time_wav_s", "start_time_source_s", "end_time_source_s",
        "timestamp_source", "timestamp_status", "word_time_valid_mask", "automatic_flags",
    ])

    gate_counts: dict[str, int] = {}
    for row in summary_rows:
        gate_counts[row["mfa_gate"]] = gate_counts.get(row["mfa_gate"], 0) + 1
    candidate_ids = [row["sample_id"] for row in summary_rows if row["mfa_gate"] == "candidate_for_mfa_auto_screen"]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": "第4步：原始文本与Whisper自动比对、Whisper词时间戳规则核查",
        "sample_count": len(summary_rows),
        "gate_counts": gate_counts,
        "mfa_candidate_count": len(candidate_ids),
        "mfa_candidate_sample_ids": candidate_ids,
        "source_workbook": str(SOURCE_XLSX),
        "source_workbook_info": workbook_info,
        "normalization": {
            "case_insensitive": True,
            "ignores_punctuation": True,
            "normalizes_unicode_apostrophes": True,
            "splits_hyphenated_words": True,
            "expands_contractions": False,
            "number_spellings_are_equated": False,
        },
        "mfa_gate_rule": "仅当原文与Whisper文本在上述保守归一化后词序列完全一致时，标记为MFA自动筛查候选。其余样本不对题目原文强制对齐，词级原文时间应保留为null。候选状态不证明原文或时间戳正确。",
        "whisper_timestamp_rules": {
            "invalid_flags": ["missing/nonfinite", "negative start", "nonpositive duration", "beyond WAV duration by more than 0.05 s", "out of order", "overlap by more than 0.05 s"],
            "review_only_flags": [f"duration below {VERY_SHORT_WORD_S} s", f"duration above {VERY_LONG_WORD_S} s"],
            "note": "Whisper边界是估计值。规则标记用于机器筛查，不是精度证明；时间差异不取平均。",
        },
        "files": {
            "summary_csv": str(summary_path),
            "token_comparison_csv": str(token_path),
            "whisper_time_audit_csv": str(times_path),
            "provisional_official_word_times_csv": str(official_times_path),
        },
        "mfa_execution_status": "validation_interrupted_before_completion; no MFA timestamps were accepted",
        "provisional_timestamp_policy": "For exact normalized text matches, retain Whisper estimates only when all automatic boundary checks pass. Flagged times are null. Original transcript times are null for text mismatch and empty ASR cases. These estimates are not gold boundaries and no numeric accuracy guarantee is made.",
        "provisional_word_time_counts": {
            "total_official_words": len(official_time_rows),
            "whisper_estimates_retained": sum(row["word_time_valid_mask"] for row in official_time_rows),
            "null_or_unmapped": sum(not row["word_time_valid_mask"] for row in official_time_rows),
            "by_status": {key: sum(row["timestamp_status"] == key for row in official_time_rows)
                          for key in sorted({row["timestamp_status"] for row in official_time_rows})},
        },
        "rows": report_rows,
    }
    report_path = OUTPUT_DIR / "第4步自动文本比对与Whisper时间戳核查报告.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "sample_count": len(summary_rows),
        "gate_counts": gate_counts,
        "token_rows": len(token_rows),
        "whisper_time_rows": len(time_rows),
        "summary_csv": str(summary_path),
        "token_comparison_csv": str(token_path),
        "whisper_time_audit_csv": str(times_path),
        "provisional_official_word_times_csv": str(official_times_path),
        "report_json": str(report_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
