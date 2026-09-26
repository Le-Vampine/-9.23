"""Extract 52 Face Landmarker blendshapes and validity masks from E.1 MP4s.

The program selects original decoded frames at roughly 10 fps and stores their
actual presentation timestamps. An invalid face produces 52 placeholder zeros
plus face_valid_mask=0; zero is never interpreted as a neutral expression.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "运行环境" / "matplotlib_cache_vision"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import av
import cv2
import mediapipe as mp
import numpy as np


MANIFEST = ROOT / "输入" / "样本清单" / "样本清单.json"
MEDIA_REPORT = ROOT / "日志" / "第2步视频检查与音频分离" / "第2步视频与音频分离核验.csv"
VIDEO_DIR = ROOT / "输入" / "原始视频"
MODEL = ROOT / "模型" / "FaceLandmarker" / "face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
OUTPUT = ROOT / "结果" / "07_FaceLandmarker52"
TARGET_FPS = 10.0
MAX_IMAGE_WIDTH = 640
MAX_FACES = 3
MIN_FACE_AREA = 0.01

BLENDSHAPES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]
if len(BLENDSHAPES) != 52:
    raise AssertionError("Expected the standard 52 Face Landmarker blendshapes")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_media() -> dict[str, dict[str, str]]:
    with MEDIA_REPORT.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 100 or len({row["sample_id"] for row in rows}) != 100:
        raise ValueError("Step 2 media report must contain 100 unique samples")
    return {row["sample_id"]: row for row in rows}


def frame_pts_s(frame: av.VideoFrame) -> float:
    if frame.pts is None or frame.time_base is None:
        raise ValueError("Decoded frame lacks presentation timestamp")
    return float(frame.pts * frame.time_base)


def frame_rgb(frame: av.VideoFrame) -> np.ndarray:
    rgb = frame.to_ndarray(format="rgb24")
    height, width = rgb.shape[:2]
    if width > MAX_IMAGE_WIDTH:
        scaled_height = max(1, round(height * MAX_IMAGE_WIDTH / width))
        rgb = cv2.resize(rgb, (MAX_IMAGE_WIDTH, scaled_height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


def face_box_area(landmarks) -> float:
    xs = [point.x for point in landmarks]
    ys = [point.y for point in landmarks]
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def classify_result(result) -> tuple[np.ndarray, int, int, int, float, int]:
    """Return vector, detected, multi-face, valid, selected area, count."""
    face_count = len(result.face_landmarks)
    zeros = np.zeros(52, dtype=np.float32)
    if face_count == 0:
        return zeros, 0, 0, 0, 0.0, 0
    areas = [face_box_area(landmarks) for landmarks in result.face_landmarks]
    selected = int(np.argmax(areas))
    area = float(areas[selected])
    multi = int(face_count > 1)
    valid = int(face_count == 1 and area >= MIN_FACE_AREA)
    if not valid:
        return zeros, 1, multi, 0, area, face_count
    if len(result.face_blendshapes) != face_count:
        raise ValueError("Face blendshape count does not match landmark count")
    values = {category.category_name: float(category.score)
              for category in result.face_blendshapes[selected]}
    if set(values) != set(BLENDSHAPES):
        raise ValueError(f"Unexpected blendshape categories: {set(values) ^ set(BLENDSHAPES)}")
    vector = np.asarray([values[name] for name in BLENDSHAPES], dtype=np.float32)
    if not np.isfinite(vector).all() or (vector < 0).any() or (vector > 1).any():
        raise ValueError("Invalid Face Landmarker coefficient")
    return vector, 1, multi, 1, area, face_count


def process_sample(item: dict, media: dict[str, str], model_bytes: bytes) -> dict:
    sample_id = item["sample_id"]
    video = VIDEO_DIR / Path(item["video_relative_path"])
    if not video.is_file() or sha256(video) != item["sha256"]:
        raise ValueError(f"Source video missing or changed: {sample_id}")
    started = time.perf_counter()
    audio_start = float(media["audio_start_s"])
    video_start = float(media["video_start_s"])
    video_duration = float(media["video_duration_s"])
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=MAX_FACES,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    actual_source_times: list[float] = []
    actual_wav_times: list[float] = []
    source_frame_indices: list[int] = []
    features: list[np.ndarray] = []
    detected: list[int] = []
    multi: list[int] = []
    valid: list[int] = []
    box_areas: list[float] = []
    counts: list[int] = []
    decoded_count = 0
    next_grid_s = None
    last_source_time = None
    last_model_ms = -1
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        with av.open(str(video)) as container:
            streams = [stream for stream in container.streams if stream.type == "video"]
            if len(streams) != 1:
                raise ValueError(f"Expected one video stream: {sample_id}")
            for frame in container.decode(video=streams[0].index):
                source_time = frame_pts_s(frame)
                if last_source_time is not None and source_time <= last_source_time:
                    raise ValueError(f"Non-increasing decoded video PTS: {sample_id}")
                last_source_time = source_time
                source_index = decoded_count
                decoded_count += 1
                if next_grid_s is None:
                    next_grid_s = source_time
                if source_time + 1e-8 < next_grid_s:
                    continue
                while next_grid_s <= source_time + 1e-8:
                    next_grid_s += 1 / TARGET_FPS
                model_ms = round((source_time - video_start) * 1000)
                if model_ms <= last_model_ms:
                    raise ValueError(f"MediaPipe millisecond timestamp not increasing: {sample_id}")
                last_model_ms = model_ms
                rgb = frame_rgb(frame)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), model_ms
                )
                vector, is_detected, is_multi, is_valid, area, count = classify_result(result)
                actual_source_times.append(source_time)
                actual_wav_times.append(source_time - audio_start)
                source_frame_indices.append(source_index)
                features.append(vector)
                detected.append(is_detected)
                multi.append(is_multi)
                valid.append(is_valid)
                box_areas.append(area)
                counts.append(count)
    if not features or decoded_count == 0:
        raise ValueError(f"No decoded/sampled video frames: {sample_id}")
    matrix = np.stack(features).astype(np.float32)
    source_times = np.asarray(actual_source_times, dtype=np.float64)
    wav_times = np.asarray(actual_wav_times, dtype=np.float64)
    masks = [np.asarray(values, dtype=np.uint8) for values in (detected, multi, valid)]
    if matrix.shape[1] != 52 or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid visual feature matrix: {sample_id}")
    if (np.diff(source_times) <= 0).any() or source_times[0] < video_start - 0.05:
        raise ValueError(f"Invalid source video timestamps: {sample_id}")
    # Container/stream duration metadata can end before the final decoded PTS.
    # Preserve the decoded presentation time and report the discrepancy.
    reported_video_end = video_start + video_duration
    after_reported_end_count = int((source_times > reported_video_end + 1e-3).sum())
    if not np.allclose(source_times - audio_start, wav_times, atol=1e-9):
        raise ValueError(f"Video/WAV timeline mapping failed: {sample_id}")
    if not np.all(matrix[masks[2] == 0] == 0):
        raise ValueError(f"Invalid face rows must be zero placeholders: {sample_id}")
    dest_dir = OUTPUT / "逐样本"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{sample_id}.npz"
    temporary = dest.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, features=matrix, frame_time_source_s=source_times,
            frame_time_wav_s=wav_times,
            source_decoded_frame_index_0based=np.asarray(source_frame_indices, dtype=np.int32),
            face_detected_mask=masks[0], multi_face_mask=masks[1], face_valid_mask=masks[2],
            face_count=np.asarray(counts, dtype=np.uint8),
            selected_face_bbox_area=np.asarray(box_areas, dtype=np.float32),
        )
    temporary.replace(dest)
    return {
        "sample_id": sample_id,
        "status": "passed",
        "decoded_frame_count": decoded_count,
        "sampled_frame_count": len(source_times),
        "face_detected_frame_count": int(masks[0].sum()),
        "multi_face_frame_count": int(masks[1].sum()),
        "face_valid_frame_count": int(masks[2].sum()),
        "no_face_frame_count": int((masks[0] == 0).sum()),
        "small_single_face_frame_count": int(((masks[0] == 1) & (masks[1] == 0) & (masks[2] == 0)).sum()),
        "first_frame_time_source_s": float(source_times[0]),
        "last_frame_time_source_s": float(source_times[-1]),
        "first_frame_time_wav_s": float(wav_times[0]),
        "last_frame_time_wav_s": float(wav_times[-1]),
        "max_sampled_frame_gap_s": float(np.diff(source_times).max()) if len(source_times) > 1 else 0.0,
        "frames_after_reported_video_end_count": after_reported_end_count,
        "last_pts_minus_reported_video_end_s": float(source_times[-1] - reported_video_end),
        "video_start_s": video_start,
        "audio_start_s": audio_start,
        "feature_file": str(dest.relative_to(ROOT)).replace("\\", "/"),
        "feature_file_sha256": sha256(dest),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", help="Run one sample; default processes all 100")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    media = read_media()
    if len(manifest) != 100 or set(item["sample_id"] for item in manifest) != set(media):
        raise ValueError("Manifest/Step 2 media report mismatch")
    if sha256(MODEL) != MODEL_SHA256:
        raise ValueError("Face Landmarker model missing or SHA-256 mismatch")
    model_bytes = MODEL.read_bytes()
    selected = [item for item in manifest if args.sample_id is None or item["sample_id"] == args.sample_id]
    if not selected:
        raise ValueError(f"Unknown sample ID: {args.sample_id}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for i, item in enumerate(selected, 1):
        try:
            record = process_sample(item, media[item["sample_id"]], model_bytes)
        except Exception as exc:
            record = {"sample_id": item["sample_id"], "status": "failed", "error": repr(exc)}
        records.append(record)
        print(f"{i:03d}/{len(selected):03d} {item['sample_id']}: {record['status']}, "
              f"frames={record.get('sampled_frame_count', 0)}, valid_faces={record.get('face_valid_frame_count', 0)}", flush=True)
    if args.sample_id is not None:
        return 0 if records[0]["status"] == "passed" else 1
    if any(record["status"] != "passed" for record in records):
        (OUTPUT / "失败记录.json").write_text(json.dumps(
            [record for record in records if record["status"] != "passed"], ensure_ascii=False, indent=2
        ) + "\n", encoding="utf-8")
        return 1
    with (OUTPUT / "逐样本统计.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    schema = {
        "feature_names_in_column_order": BLENDSHAPES,
        "dimensions": 52,
        "sampling": "approximately 10 fps, choosing actual decoded video frames; retain their original PTS",
        "target_fps": TARGET_FPS,
        "max_input_image_width": MAX_IMAGE_WIDTH,
        "face_landmarker_running_mode": "VIDEO",
        "maximum_detected_faces": MAX_FACES,
        "valid_face_rule": f"exactly one detected face with normalized landmark bounding-box area >= {MIN_FACE_AREA}",
        "invalid_face_vector": "52 placeholder zeros with face_valid_mask=0; never interpreted as neutral expression",
        "frame_time_wav_s": "original frame presentation time minus Step 2 audio stream start",
        "frame_time_source_s": "original frame presentation timestamp",
        "model_url": MODEL_URL,
        "model_sha256": MODEL_SHA256,
    }
    (OUTPUT / "特征定义.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "completed",
        "sample_count": len(records),
        "decoded_frame_count": sum(row["decoded_frame_count"] for row in records),
        "sampled_frame_count": sum(row["sampled_frame_count"] for row in records),
        "face_detected_frame_count": sum(row["face_detected_frame_count"] for row in records),
        "face_valid_frame_count": sum(row["face_valid_frame_count"] for row in records),
        "multi_face_frame_count": sum(row["multi_face_frame_count"] for row in records),
        "no_face_frame_count": sum(row["no_face_frame_count"] for row in records),
        "small_single_face_frame_count": sum(row["small_single_face_frame_count"] for row in records),
        "samples_with_no_valid_face": [row["sample_id"] for row in records if row["face_valid_frame_count"] == 0],
        "samples_with_frames_after_reported_video_end": [row["sample_id"] for row in records if row["frames_after_reported_video_end_count"] > 0],
        "feature_dimensions": 52,
        "all_feature_values_finite": True,
        "all_timestamps_original_pts_and_strictly_increasing": True,
        "model_sha256": MODEL_SHA256,
        "mediapipe_version": mp.__version__,
        "pyav_version": av.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "manifest_sha256": sha256(MANIFEST),
        "step2_media_report_sha256": sha256(MEDIA_REPORT),
    }
    (OUTPUT / "运行报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
