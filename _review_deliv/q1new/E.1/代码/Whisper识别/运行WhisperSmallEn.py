"""Run local Whisper small.en on E.1 WAV files and save segment/word timestamps.

This stage reads sample IDs and WAVs only; it does not load or compare the
official transcripts. Word times are Whisper estimates, not forced alignment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "输入" / "样本清单" / "样本清单.json"
WAV_DIR = ROOT / "输入" / "音频WAV"
MODEL_DIR = ROOT / "模型" / "Whisper_small_en"
STEP2_REPORT = ROOT / "日志" / "第2步视频检查与音频分离" / "第2步视频与音频分离核验.json"
OUTPUT_DIR = ROOT / "结果" / "03_Whisper_small_en" / "逐样本"
REPORT_DIR = ROOT / "日志" / "第3步Whisper_small.en"
SAMPLE_RATE = 16_000
EXPECTED_MODEL_SHA256 = "62b2a45b05ee59acb4a5341b33ee35e041395d378d418a18acfe4c9e768ee37a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_hashes() -> dict[str, str]:
    return {p.relative_to(MODEL_DIR).as_posix(): sha256(p)
            for p in sorted(MODEL_DIR.rglob("*")) if p.is_file()}


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        if audio.getframerate() != SAMPLE_RATE or audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError(f"WAV 不是 16 kHz 单声道 PCM16：{path}")
        if audio.getnframes() <= 0:
            raise ValueError(f"WAV 为空：{path}")
        return audio.getnframes() / audio.getframerate()


def read_step2_offsets() -> dict[str, float]:
    if not STEP2_REPORT.is_file():
        return {}
    report = json.loads(STEP2_REPORT.read_text(encoding="utf-8"))
    return {row["sample_id"]: float(row.get("wav_zero_maps_to_source_time_s") or 0.0)
            for row in report.get("rows", [])}


def transcribe_one(sample_id: str, model: WhisperModel, source_start_s: float) -> dict[str, Any]:
    wav_path = WAV_DIR / f"{sample_id}.wav"
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)
    duration = audio_duration(wav_path)
    segments, info = model.transcribe(
        str(wav_path),
        language="en",
        task="transcribe",
        beam_size=5,
        word_timestamps=True,
        vad_filter=False,
    )
    segment_rows: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    issues: list[str] = []
    for segment in segments:
        segment_rows.append({
            "segment_index": len(segment_rows) + 1,
            "start_time_wav_s": float(segment.start),
            "end_time_wav_s": float(segment.end),
            "start_time_source_s": source_start_s + float(segment.start),
            "end_time_source_s": source_start_s + float(segment.end),
            "text": segment.text,
            "no_speech_probability": float(segment.no_speech_prob),
            "average_log_probability": float(segment.avg_logprob),
        })
        for word in segment.words or []:
            start = float(word.start)
            end = float(word.end)
            if start < -0.05 or end < start or end > duration + 0.05:
                issues.append(f"词级时间越出 WAV 范围：{word.word!r} {start:.4f}–{end:.4f}s")
            words.append({
                "word_index": len(words) + 1,
                "word": word.word.strip(),
                "surface": word.word,
                "start_time_wav_s": start,
                "end_time_wav_s": end,
                "start_time_source_s": source_start_s + start,
                "end_time_source_s": source_start_s + end,
                "probability": float(word.probability),
            })

    transcript = "".join(segment["text"] for segment in segment_rows).strip()
    word_probabilities = [word["probability"] for word in words]
    if not transcript:
        issues.append("Whisper 未输出文本")
    payload = {
        "sample_id": sample_id,
        "status": "已完成" if transcript else "已运行但无文本输出",
        "audio_path": str(wav_path.resolve()),
        "audio_sha256": sha256(wav_path),
        "audio_duration_s": duration,
        "source_audio_start_s": source_start_s,
        "source_time_mapping": "source_time_s = wav_time_s + source_audio_start_s",
        "tool": "faster-whisper",
        "model": "Systran/faster-whisper-small.en",
        "model_sha256": EXPECTED_MODEL_SHA256,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "timestamp_method": "Whisper word_timestamps; approximate ASR timestamps, not forced alignment",
        "parameters": {
            "task": "transcribe",
            "language": "en",
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": False,
            "device": "cpu",
            "compute_type": "int8",
        },
        "transcription_text": transcript,
        "segments": segment_rows,
        "word_timestamps": words,
        "word_count": len(words),
        "mean_word_probability": statistics.mean(word_probabilities) if word_probabilities else None,
        "issues": issues,
        "official_transcript_loaded": False,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="允许重写本步骤已生成的逐样本 JSON")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sample_ids = [str(item["sample_id"]) for item in manifest]
    if len(sample_ids) != 100 or len(set(sample_ids)) != 100:
        raise ValueError(f"样本清单须有 100 个唯一编号，实际 {len(sample_ids)} 条")
    missing_audio = [sid for sid in sample_ids if not (WAV_DIR / f"{sid}.wav").is_file()]
    if missing_audio:
        raise FileNotFoundError(f"缺少 {len(missing_audio)} 个 WAV：{missing_audio[:10]}")

    model_bin = MODEL_DIR / "model.bin"
    if not model_bin.is_file() or sha256(model_bin) != EXPECTED_MODEL_SHA256:
        raise ValueError("本地 small.en model.bin 哈希与锁定值不一致")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    existing_outputs = [OUTPUT_DIR / f"{sid}_Whisper_small.en.json" for sid in sample_ids]
    if not args.overwrite and any(path.exists() for path in existing_outputs):
        raise FileExistsError("发现已有逐样本 ASR 输出；如要重跑，请明确添加 --overwrite")

    offsets = read_step2_offsets()
    model = WhisperModel(
        str(MODEL_DIR), device="cpu", compute_type="int8", cpu_threads=args.cpu_threads,
        local_files_only=True,
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, sample_id in enumerate(sample_ids, start=1):
        try:
            result = transcribe_one(sample_id, model, offsets.get(sample_id, 0.0))
            out_path = OUTPUT_DIR / f"{sample_id}_Whisper_small.en.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append({
                "sample_id": sample_id,
                "status": result["status"],
                "transcription_text": result["transcription_text"],
                "segment_count": len(result["segments"]),
                "word_count": result["word_count"],
                "audio_duration_s": result["audio_duration_s"],
                "source_audio_start_s": result["source_audio_start_s"],
                "language_probability": result["language_probability"],
                "mean_word_probability": result["mean_word_probability"],
                "timestamp_issue_count": len(result["issues"]),
                "result_path": str(out_path.resolve()),
            })
        except Exception as exc:  # retain per-sample failures and finish the batch
            failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
            rows.append({"sample_id": sample_id, "status": "失败", "transcription_text": "",
                         "segment_count": 0, "word_count": 0, "audio_duration_s": None,
                         "source_audio_start_s": offsets.get(sample_id, 0.0),
                         "language_probability": None, "mean_word_probability": None,
                         "timestamp_issue_count": 1, "result_path": ""})
        print(f"[{index:03d}/100] {sample_id}: {rows[-1]['status']}，{rows[-1]['word_count']} 词", flush=True)

    csv_path = REPORT_DIR / "100条Whisper转写与词时间戳汇总.csv"
    fields = ["sample_id", "status", "transcription_text", "segment_count", "word_count",
              "audio_duration_s", "source_audio_start_s", "language_probability",
              "mean_word_probability", "timestamp_issue_count", "result_path"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": "第3步：Whisper small.en 英文 ASR 与词级时间戳",
        "sample_count": len(sample_ids),
        "success_count": sum(row["status"] == "已完成" for row in rows),
        "empty_transcript_count": sum(row["status"] == "已运行但无文本输出" for row in rows),
        "failure_count": len(failures),
        "total_word_count": sum(row["word_count"] for row in rows),
        "timestamp_issue_count": sum(row["timestamp_issue_count"] for row in rows),
        "model": "Systran/faster-whisper-small.en",
        "model_sha256": EXPECTED_MODEL_SHA256,
        "model_file_sha256": model_hashes(),
        "runtime": {
            "python": sys.version,
            "faster_whisper": __import__("importlib.metadata", fromlist=["version"]).version("faster-whisper"),
            "ctranslate2": __import__("importlib.metadata", fromlist=["version"]).version("ctranslate2"),
        },
        "parameters": {
            "device": "cpu", "compute_type": "int8", "cpu_threads": args.cpu_threads,
            "language": "en", "task": "transcribe", "beam_size": 5,
            "word_timestamps": True, "vad_filter": False,
            "official_transcript_used": False,
        },
        "timestamp_note": "词级时间由 Whisper 识别器估计；不是强制对齐结果。时间同时保存为 WAV 相对时间和源媒体时间。",
        "output_directory": str(OUTPUT_DIR.resolve()),
        "summary_csv": str(csv_path.resolve()),
        "failures": failures,
        "rows": rows,
    }
    report_path = REPORT_DIR / "第3步Whisper_small.en运行报告.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "sample_count": report["sample_count"],
        "success_count": report["success_count"],
        "empty_transcript_count": report["empty_transcript_count"],
        "failure_count": report["failure_count"],
        "total_word_count": report["total_word_count"],
        "timestamp_issue_count": report["timestamp_issue_count"],
        "report": str(report_path.resolve()),
        "summary_csv": str(csv_path.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
