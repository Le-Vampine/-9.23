"""检查样本 MP4 并核验完整 WAV 音轨的分离结果。

WAV 的 t=0 对应源音轨首个呈现时间戳。源音轨相对视频的起点偏移和原始时长
写入报告；不裁剪、不补齐，现有有效 WAV 不覆盖。缺失 WAV 才会由源 MP4 补出。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "输入" / "样本清单" / "样本清单.json"
WAV_DIR = ROOT / "输入" / "音频WAV"
REPORT_DIR = ROOT / "日志" / "第2步视频检查与音频分离"
SAMPLE_RATE = 16_000
PCM_WIDTH = 2
DURATION_TOLERANCE_S = 0.04


def number(value: Any) -> float | None:
    try:
        if value in (None, "", "N/A"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def probe(path: Path, ffprobe: str) -> dict[str, Any]:
    entries = (
        "format=start_time,duration:"
        "stream=index,codec_type,codec_name,time_base,start_time,duration,"
        "sample_rate,channels,channel_layout,width,height,avg_frame_rate,r_frame_rate"
    )
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", entries, "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        timeout=60,
    )
    return json.loads(result.stdout)


def stream_of(data: dict[str, Any], kind: str) -> dict[str, Any] | None:
    return next((s for s in data.get("streams", []) if s.get("codec_type") == kind), None)


def pcm_sha256(path: Path) -> tuple[str, int, int, int, float]:
    digest = hashlib.sha256()
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        rate = audio.getframerate()
        width = audio.getsampwidth()
        frames = audio.getnframes()
        while True:
            block = audio.readframes(131_072)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest(), channels, rate, width, frames / rate if rate else 0.0


def decode_all_streams(path: Path, ffmpeg: str) -> None:
    subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(path),
         "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        timeout=180,
    )


def extract_full_audio(source: Path, target: Path, ffmpeg: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-nostdin", "-y", "-v", "error", "-i", str(source),
         "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-c:a", "pcm_s16le", str(target)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        timeout=180,
    )


def check_record(record: dict[str, Any], temp_dir: Path, ffmpeg: str, ffprobe: str) -> dict[str, Any]:
    sid = str(record["sample_id"])
    source = Path(record["video_path"])
    target = WAV_DIR / f"{sid}.wav"
    row: dict[str, Any] = {
        "sample_id": sid,
        "video_path": str(source),
        "wav_path": str(target),
        "status": "通过",
        "issues": [],
    }
    if not source.is_file():
        row["status"] = "失败"
        row["issues"].append("原始视频不存在")
        return row

    row["video_size_bytes"] = source.stat().st_size
    expected_size = record.get("file_size_bytes")
    if expected_size is not None and row["video_size_bytes"] != int(expected_size):
        row["issues"].append("视频文件大小与样本清单不一致")

    try:
        media = probe(source, ffprobe)
        video = stream_of(media, "video")
        audio = stream_of(media, "audio")
        fmt = media.get("format", {})
        row.update({
            "video_codec": video.get("codec_name") if video else None,
            "video_width": int(video["width"]) if video and video.get("width") else None,
            "video_height": int(video["height"]) if video and video.get("height") else None,
            "video_frame_rate": (video.get("avg_frame_rate") or video.get("r_frame_rate")) if video else None,
            "video_start_s": number(video.get("start_time")) if video else None,
            "video_duration_s": number(video.get("duration")) if video else number(fmt.get("duration")),
            "audio_codec": audio.get("codec_name") if audio else None,
            "source_audio_rate_hz": int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
            "source_audio_channels": int(audio["channels"]) if audio and audio.get("channels") else None,
            "audio_start_s": number(audio.get("start_time")) if audio else None,
            "source_audio_duration_s": number(audio.get("duration")) if audio else None,
            "container_start_s": number(fmt.get("start_time")),
        })
        if video is None:
            row["issues"].append("缺少视频流")
        if audio is None:
            row["issues"].append("缺少音频流")
        if video is None or audio is None:
            row["status"] = "失败"
            return row
        if row["audio_start_s"] is None:
            row["audio_start_s"] = row["container_start_s"]
            row["audio_start_basis"] = "源音轨 start_time 缺失，回退至容器 start_time"
        else:
            row["audio_start_basis"] = "源音轨 start_time"
        video_start = row["video_start_s"]
        if video_start is None:
            video_start = row["container_start_s"] or 0.0
        audio_start = row["audio_start_s"] or 0.0
        row["audio_start_offset_from_video_s"] = audio_start - video_start
        row["wav_zero_maps_to_source_time_s"] = audio_start
        row["time_axis_rule"] = "wav_time + audio_start_s = source_media_time_s"

        decode_all_streams(source, ffmpeg)
        row["full_video_audio_decode"] = "通过"

        fresh = temp_dir / f"{sid}.wav"
        extract_full_audio(source, fresh, ffmpeg)
        fresh_hash, fresh_channels, fresh_rate, fresh_width, fresh_duration = pcm_sha256(fresh)
        row["fresh_extraction_pcm_sha256"] = fresh_hash
        row["fresh_extraction_duration_s"] = fresh_duration
        row["extraction_parameters"] = "完整音轨；无 -ss、-t、atrim 或补齐；单声道、16 kHz、PCM16"

        if target.is_file():
            current_hash, channels, rate, width, duration = pcm_sha256(target)
            row.update({
                "existing_wav_bytes": target.stat().st_size,
                "existing_wav_pcm_sha256": current_hash,
                "wav_channels": channels,
                "wav_sample_rate_hz": rate,
                "wav_sample_width_bytes": width,
                "wav_duration_s": duration,
                "wav_matches_fresh_pcm": current_hash == fresh_hash,
            })
            if (channels, rate, width) != (1, SAMPLE_RATE, PCM_WIDTH):
                row["issues"].append("现有 WAV 格式不是 16 kHz 单声道 PCM16")
            if current_hash != fresh_hash:
                row["issues"].append("现有 WAV 与从源 MP4 新提取的 PCM 不一致，已保留原文件未覆盖")
        else:
            # Create only a missing output; never replace an existing WAV.
            WAV_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fresh, target)
            current_hash, channels, rate, width, duration = pcm_sha256(target)
            row.update({
                "existing_wav_bytes": target.stat().st_size,
                "existing_wav_pcm_sha256": current_hash,
                "wav_channels": channels,
                "wav_sample_rate_hz": rate,
                "wav_sample_width_bytes": width,
                "wav_duration_s": duration,
                "wav_matches_fresh_pcm": current_hash == fresh_hash,
                "wav_action": "从源 MP4 补提缺失音频",
            })

        if fresh_rate != SAMPLE_RATE or fresh_channels != 1 or fresh_width != PCM_WIDTH:
            row["issues"].append("临时提取音频格式不符合预期")
        source_audio_duration = row["source_audio_duration_s"]
        if source_audio_duration is not None:
            row["fresh_vs_source_audio_duration_diff_s"] = abs(fresh_duration - source_audio_duration)
            if row["fresh_vs_source_audio_duration_diff_s"] > DURATION_TOLERANCE_S:
                row["issues"].append("分离音频与源音轨时长差超过容许阈值")
        row["source_video_audio_duration_gap_s"] = (
            row["video_duration_s"] - row["source_audio_duration_s"]
            if row["video_duration_s"] is not None and row["source_audio_duration_s"] is not None
            else None
        )
    except (OSError, KeyError, ValueError, wave.Error, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        row["issues"].append(f"媒体核验错误：{type(exc).__name__}: {exc}")

    if row["issues"]:
        row["status"] = "需复核"
    return row


def main() -> int:
    global WAV_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--wav-dir", type=Path, default=WAV_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.ffmpeg or not args.ffprobe:
        parser.error("请通过 --ffmpeg 与 --ffprobe 指定 FFmpeg 工具路径")

    WAV_DIR = args.wav_dir.resolve()
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    ids = [str(item["sample_id"]) for item in records]
    if len(records) != 100 or len(set(ids)) != 100:
        raise ValueError(f"预期 100 个不重复样本，实际 {len(records)} 条、{len(set(ids))} 个唯一编号")
    expected_wavs = {f"{sid}.wav" for sid in ids}
    existing_wavs_before = {path.name for path in WAV_DIR.glob("*.wav")} if WAV_DIR.is_dir() else set()
    missing_wavs_before = sorted(expected_wavs - existing_wavs_before)
    extra_wavs_before = sorted(existing_wavs_before - expected_wavs)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="step2_audio_verify_") as temp_name:
        temp_dir = Path(temp_name)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(check_record, item, temp_dir, args.ffmpeg, args.ffprobe) for item in records]
            rows = [future.result() for future in futures]

    success = sum(row["status"] == "通过" for row in rows)
    review = len(rows) - success
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": "第2步：检查视频并分离音频",
        "sample_count": len(rows),
        "passed_count": success,
        "review_count": review + len(extra_wavs_before),
        "existing_wav_count_before_check": len(existing_wavs_before),
        "missing_wavs_before_check": missing_wavs_before,
        "extra_wavs_before_check": extra_wavs_before,
        "ffmpeg": args.ffmpeg,
        "ffprobe": args.ffprobe,
        "input_manifest": str(args.manifest.resolve()),
        "wav_directory": str(WAV_DIR),
        "video_check": "逐条读取 MP4 流信息，并用 FFmpeg 对视频流和音频流做完整解码检查",
        "timeline_policy": {
            "wav_zero": "源音轨首个呈现时间戳",
            "mapping": "wav_time + audio_start_s = source_media_time_s",
            "audio_video_offset": "逐条保存 audio_start_offset_from_video_s",
            "no_trim_or_padding": True,
            "different_video_audio_durations": "保留真实时长并报告差值；不裁切或补齐",
        },
        "extraction_parameters": {
            "stream": "0:a:0",
            "channels": 1,
            "sample_rate_hz": SAMPLE_RATE,
            "sample_format": "pcm_s16le",
            "trimmed": False,
            "padded": False,
        },
        "duration_tolerance_s": DURATION_TOLERANCE_S,
        "rows": rows,
    }
    json_path = args.report_dir / "第2步视频与音频分离核验.json"
    csv_path = args.report_dir / "第2步视频与音频分离核验.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted({key for row in rows for key in row if key != "issues"}) + ["issues"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})

    print(json.dumps({
        "sample_count": len(rows), "passed_count": success, "review_count": report["review_count"],
        "existing_wav_count_before_check": len(existing_wavs_before),
        "missing_wav_filled_count": len(missing_wavs_before),
        "extra_wav_count": len(extra_wavs_before),
        "pcm_identical_count": sum(row.get("wav_matches_fresh_pcm") is True for row in rows),
        "json_report": str(json_path.resolve()), "csv_report": str(csv_path.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if review == 0 and not extra_wavs_before else 1


if __name__ == "__main__":
    raise SystemExit(main())
