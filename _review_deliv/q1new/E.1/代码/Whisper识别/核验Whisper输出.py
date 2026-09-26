"""Check E.1 Whisper outputs and word-time bounds without loading official text."""
from __future__ import annotations

import array
import json
import math
import wave
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "输入" / "样本清单" / "样本清单.json"
WAV_DIR = ROOT / "输入" / "音频WAV"
OUTPUT_DIR = ROOT / "结果" / "03_Whisper_small_en" / "逐样本"
STEP2_REPORT = ROOT / "日志" / "第2步视频检查与音频分离" / "第2步视频与音频分离核验.json"
REPORT_DIR = ROOT / "日志" / "第3步Whisper_small.en"
TIME_TOLERANCE_S = 0.05


def wave_stats(path: Path) -> dict:
    with wave.open(str(path), "rb") as audio:
        rate = audio.getframerate()
        count = audio.getnframes()
        samples = array.array("h")
        while True:
            block = audio.readframes(131_072)
            if not block:
                break
            samples.frombytes(block)
    n = len(samples)
    peak = max((abs(value) for value in samples), default=0)
    rms = math.sqrt(sum(value * value for value in samples) / max(n, 1))
    return {
        "duration_s": count / rate if rate else 0.0,
        "rms_dbfs": 20 * math.log10(max(rms / 32768, 1e-12)),
        "peak_pcm16": peak,
        "nonzero_sample_percent": 100 * sum(value != 0 for value in samples) / max(n, 1),
    }


def main() -> int:
    sample_ids = [str(row["sample_id"]) for row in json.loads(MANIFEST.read_text(encoding="utf-8"))]
    offsets_report = json.loads(STEP2_REPORT.read_text(encoding="utf-8"))
    offsets = {row["sample_id"]: float(row.get("wav_zero_maps_to_source_time_s") or 0.0)
               for row in offsets_report.get("rows", [])}
    expected_files = {f"{sid}_Whisper_small.en.json" for sid in sample_ids}
    actual_files = {path.name for path in OUTPUT_DIR.glob("*.json")}
    rows = []
    structural_issues = []

    for sid in sample_ids:
        result_path = OUTPUT_DIR / f"{sid}_Whisper_small.en.json"
        wav_path = WAV_DIR / f"{sid}.wav"
        if not result_path.is_file() or not wav_path.is_file():
            structural_issues.append({"sample_id": sid, "issue": "缺少识别结果或 WAV"})
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        stats = wave_stats(wav_path)
        issues = []
        if result.get("sample_id") != sid:
            issues.append("结果文件 sample_id 与文件名不一致")
        if result.get("official_transcript_loaded") is not False:
            issues.append("第3步输出不应载入题目原始转写文本")
        offset = offsets.get(sid, 0.0)
        if abs(float(result.get("source_audio_start_s") or 0.0) - offset) > 1e-8:
            issues.append("源音轨起点偏移与第2步报告不一致")
        word_rows = result.get("word_timestamps") or []
        out_of_bounds = []
        for word in word_rows:
            start = float(word["start_time_wav_s"])
            end = float(word["end_time_wav_s"])
            if start < -TIME_TOLERANCE_S or end < start or end > stats["duration_s"] + TIME_TOLERANCE_S:
                out_of_bounds.append({"word_index": word.get("word_index"), "word": word.get("word"),
                                      "start_time_wav_s": start, "end_time_wav_s": end})
            expected_source_start = offset + start
            expected_source_end = offset + end
            if (abs(float(word.get("start_time_source_s", expected_source_start)) - expected_source_start) > 1e-7 or
                    abs(float(word.get("end_time_source_s", expected_source_end)) - expected_source_end) > 1e-7):
                issues.append(f"第 {word.get('word_index')} 个词的源媒体时间映射不一致")
        if out_of_bounds:
            issues.append(f"{len(out_of_bounds)} 个词级时间超出 WAV 范围")
        if issues:
            structural_issues.append({"sample_id": sid, "issues": issues, "out_of_bounds": out_of_bounds})

        empty = not bool((result.get("transcription_text") or "").strip())
        rows.append({
            "sample_id": sid,
            "status": result.get("status"),
            "transcription_empty": empty,
            "word_count": len(word_rows),
            "segment_count": len(result.get("segments") or []),
            "audio_duration_s": stats["duration_s"],
            "audio_rms_dbfs": stats["rms_dbfs"],
            "audio_peak_pcm16": stats["peak_pcm16"],
            "audio_nonzero_sample_percent": stats["nonzero_sample_percent"],
            "empty_asr_non_silent_review": empty and stats["rms_dbfs"] > -55.0,
            "source_audio_start_s": offset,
        })

    extra_files = sorted(actual_files - expected_files)
    missing_files = sorted(expected_files - actual_files)
    if extra_files:
        structural_issues.append({"issue": "存在非本清单的 ASR JSON 文件", "files": extra_files})
    if missing_files:
        structural_issues.append({"issue": "缺少 ASR JSON 文件", "files": missing_files})

    empty_rows = [row for row in rows if row["transcription_empty"]]
    non_silent_empty = [row for row in rows if row["empty_asr_non_silent_review"]]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": "第3步 Whisper 输出结构和词时间范围核验",
        "sample_count": len(sample_ids),
        "result_file_count": len(actual_files),
        "structural_pass": not structural_issues and len(rows) == len(sample_ids),
        "structural_issue_count": len(structural_issues),
        "structural_issues": structural_issues,
        "empty_transcript_count": len(empty_rows),
        "empty_transcript_samples": [row["sample_id"] for row in empty_rows],
        "non_silent_empty_transcript_count": len(non_silent_empty),
        "non_silent_empty_transcript_samples": non_silent_empty,
        "word_timestamp_bounds": f"检查每词 WAV 相对起止时间；容许范围 -{TIME_TOLERANCE_S:.2f}s 至 WAV 时长 +{TIME_TOLERANCE_S:.2f}s",
        "official_transcript_loaded": False,
        "note": "空转写保留为空；有音频信号的空结果进入第4步人工/文本核对，不自动补词。",
        "rows": rows,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "第3步Whisper输出结构与时间戳核验.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "sample_count", "result_file_count", "structural_pass", "structural_issue_count",
        "empty_transcript_count", "non_silent_empty_transcript_count", "non_silent_empty_transcript_samples",
    )}, ensure_ascii=False, indent=2))
    return 0 if report["structural_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
