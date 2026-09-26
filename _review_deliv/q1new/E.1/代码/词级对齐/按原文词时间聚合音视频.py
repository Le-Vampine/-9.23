"""Align Step 5 text, Step 6 audio and Step 7 visual features to official words.

Only the screened Step 4 primary word intervals define the main alignment.
Audio and visual frames are pooled when their actual centers/PTS fall in
[word_start, word_end). A nearest visual frame within 50 ms is recorded as a
separate candidate, never silently inserted into the primary visual matrix.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "输入" / "样本清单" / "样本清单.json"
MEDIA_CSV = ROOT / "日志" / "第2步视频检查与音频分离" / "第2步视频与音频分离核验.csv"
WORD_TIMES_CSV = ROOT / "日志" / "第4步文本一致性检查" / "原文词级时间_MFA候选及最终采用.csv"
TEXT_MATRIX = ROOT / "结果" / "05_BERT_base_uncased" / "原文逐词BERT特征.npy"
TEXT_INDEX_CSV = ROOT / "结果" / "05_BERT_base_uncased" / "原文逐词索引.csv"
AUDIO_DIR = ROOT / "结果" / "06_Librosa74" / "逐样本"
VISUAL_DIR = ROOT / "结果" / "07_FaceLandmarker52" / "逐样本"
OUTPUT = ROOT / "结果" / "08_原文词级跨模态对齐"
NEAREST_VISUAL_MAX_DISTANCE_S = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def grouped_words(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["sample_id"]].append(row)
    return groups


def masked_time(row: dict[str, str], name: str) -> float:
    value = row[name]
    if not value:
        raise ValueError(f"Masked time is absent: {name}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Non-finite masked time: {name}")
    return result


def nearest_valid_visual(
    frame_times: np.ndarray, frame_vectors: np.ndarray, frame_valid: np.ndarray,
    start: float, end: float,
) -> tuple[np.ndarray, int, float, int]:
    valid_indices = np.flatnonzero(frame_valid)
    if len(valid_indices) == 0:
        return np.zeros(52, dtype=np.float32), 0, np.nan, -1
    valid_times = frame_times[valid_indices]
    distances = np.where(valid_times < start, start - valid_times,
                         np.where(valid_times >= end, valid_times - end, 0.0))
    choice = int(np.argmin(distances))
    distance = float(distances[choice])
    if distance > NEAREST_VISUAL_MAX_DISTANCE_S + 1e-9:
        return np.zeros(52, dtype=np.float32), 0, distance, int(valid_indices[choice])
    index = int(valid_indices[choice])
    return frame_vectors[index].astype(np.float32), 1, distance, index


def align_one(
    item: dict, words: list[dict[str, str]], text_rows: list[dict[str, str]],
    text_matrix: np.ndarray, media: dict[str, str], output_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    sample_id = item["sample_id"]
    word_count = len(words)
    if word_count != len(text_rows) or word_count == 0:
        raise ValueError(f"Step 4 / Step 5 word count mismatch: {sample_id}")
    audio_file = AUDIO_DIR / f"{sample_id}.npz"
    visual_file = VISUAL_DIR / f"{sample_id}.npz"
    with np.load(audio_file) as audio_data, np.load(visual_file) as visual_data:
        audio_frames = audio_data["features"]
        audio_times = audio_data["frame_time_wav_s"]
        audio_time_source = audio_data["frame_time_source_s"]
        audio_frame_valid = audio_data["frame_valid_mask"].astype(bool)
        f0_voiced = audio_data["f0_voiced_mask"].astype(bool)
        visual_frames = visual_data["features"]
        visual_times = visual_data["frame_time_wav_s"]
        visual_time_source = visual_data["frame_time_source_s"]
        face_valid = visual_data["face_valid_mask"].astype(bool)
    offset = float(media["audio_start_s"])
    if audio_frames.shape[1] != 74 or visual_frames.shape[1] != 52:
        raise ValueError(f"Input modality dimensions incorrect: {sample_id}")
    if any(np.diff(times).min() <= 0 for times in (audio_times, visual_times) if len(times) > 1):
        raise ValueError(f"Input frame times must strictly increase: {sample_id}")
    if not np.allclose(audio_time_source - audio_times, offset, atol=1e-6):
        raise ValueError(f"Audio source/WAV timeline mismatch: {sample_id}")
    if not np.allclose(visual_time_source - visual_times, offset, atol=1e-6):
        raise ValueError(f"Video source/WAV timeline mismatch: {sample_id}")
    if not np.isfinite(audio_frames).all() or not np.isfinite(visual_frames).all():
        raise ValueError(f"Non-finite input feature: {sample_id}")

    text = np.zeros((word_count, 768), dtype=np.float32)
    aligned_audio = np.zeros((word_count, 74), dtype=np.float32)
    aligned_visual = np.zeros((word_count, 52), dtype=np.float32)
    nearest_visual = np.zeros((word_count, 52), dtype=np.float32)
    word_start = np.full(word_count, np.nan, dtype=np.float64)
    word_end = np.full(word_count, np.nan, dtype=np.float64)
    time_mask = np.zeros(word_count, dtype=np.uint8)
    audio_mask = np.zeros(word_count, dtype=np.uint8)
    audio_f0_mask = np.zeros(word_count, dtype=np.uint8)
    visual_mask = np.zeros(word_count, dtype=np.uint8)
    nearest_visual_mask = np.zeros(word_count, dtype=np.uint8)
    audio_frame_counts = np.zeros(word_count, dtype=np.int32)
    f0_frame_counts = np.zeros(word_count, dtype=np.int32)
    visual_frame_counts = np.zeros(word_count, dtype=np.int32)
    valid_face_frame_counts = np.zeros(word_count, dtype=np.int32)
    indices: list[dict[str, object]] = []

    for i, (word, text_row) in enumerate(zip(words, text_rows)):
        position = i + 1
        if (int(word["official_word_index"]) != position
                or int(text_row["official_word_index"]) != position
                or word["official_word"] != text_row["official_word"]
                or word["sample_id"] != text_row["sample_id"]):
            raise ValueError(f"Original word identity/order mismatch: {sample_id}, {position}")
        text_index = int(text_row["feature_row_0based"])
        if text_row["text_feature_valid_mask"] != "1":
            raise ValueError(f"Missing Step 5 text vector: {sample_id}, {position}")
        text[i] = text_matrix[text_index]
        has_time = word["final_primary_valid_mask"].strip().lower() == "true"
        if (text_row["final_primary_valid_mask"] == "1") != has_time:
            raise ValueError(f"Step 4 / Step 5 time mask mismatch: {sample_id}, {position}")
        audio_left = audio_right = visual_left = visual_right = -1
        nearest_distance = np.nan
        nearest_index = -1
        status = "no_primary_word_time"
        if has_time:
            start = masked_time(word, "final_primary_start_time_wav_s")
            end = masked_time(word, "final_primary_end_time_wav_s")
            source_start = masked_time(word, "final_primary_start_time_source_s")
            source_end = masked_time(word, "final_primary_end_time_source_s")
            if not (0 <= start < end) or abs(source_start - start - offset) > 1e-3 or abs(source_end - end - offset) > 1e-3:
                raise ValueError(f"Invalid primary word interval or source mapping: {sample_id}, {position}")
            word_start[i], word_end[i], time_mask[i] = start, end, 1
            audio_left = int(np.searchsorted(audio_times, start, side="left"))
            audio_right = int(np.searchsorted(audio_times, end, side="left"))
            selected_audio_indices = np.flatnonzero(audio_frame_valid[audio_left:audio_right]) + audio_left
            audio_frame_counts[i] = len(selected_audio_indices)
            if len(selected_audio_indices):
                pooled = audio_frames[selected_audio_indices].mean(axis=0, dtype=np.float32)
                voiced_indices = selected_audio_indices[f0_voiced[selected_audio_indices]]
                f0_frame_counts[i] = len(voiced_indices)
                if len(voiced_indices):
                    pooled[72] = audio_frames[voiced_indices, 72].mean(dtype=np.float32)
                    audio_f0_mask[i] = 1
                else:
                    pooled[72] = 0.0
                aligned_audio[i] = pooled
                audio_mask[i] = 1
            visual_left = int(np.searchsorted(visual_times, start, side="left"))
            visual_right = int(np.searchsorted(visual_times, end, side="left"))
            visual_frame_counts[i] = visual_right - visual_left
            valid_indices = np.flatnonzero(face_valid[visual_left:visual_right]) + visual_left
            valid_face_frame_counts[i] = len(valid_indices)
            if len(valid_indices):
                aligned_visual[i] = visual_frames[valid_indices].mean(axis=0, dtype=np.float32)
                visual_mask[i] = 1
                status = "in_interval_valid_face"
            else:
                status = "in_interval_frames_no_valid_face" if visual_frame_counts[i] else "no_visual_frame_in_interval"
                candidate, candidate_mask, nearest_distance, nearest_index = nearest_valid_visual(
                    visual_times, visual_frames, face_valid, start, end,
                )
                if candidate_mask:
                    nearest_visual[i] = candidate
                    nearest_visual_mask[i] = 1
        indices.append({
            "sample_id": sample_id,
            "official_word_index": position,
            "official_word": word["official_word"],
            "text_feature_row_0based": text_index,
            "word_start_wav_s": "" if not has_time else word_start[i],
            "word_end_wav_s": "" if not has_time else word_end[i],
            "word_start_source_s": "" if not has_time else word_start[i] + offset,
            "word_end_source_s": "" if not has_time else word_end[i] + offset,
            "timestamp_evidence_tier": word["timestamp_evidence_tier"],
            "timestamp_status": word["timestamp_status"],
            "text_valid_mask": 1,
            "word_time_valid_mask": int(time_mask[i]),
            "audio_valid_mask": int(audio_mask[i]),
            "audio_f0_valid_mask": int(audio_f0_mask[i]),
            "visual_valid_mask": int(visual_mask[i]),
            "visual_nearest_candidate_mask": int(nearest_visual_mask[i]),
            "audio_frame_start_index_0based": audio_left,
            "audio_frame_end_index_exclusive": audio_right,
            "audio_frame_count": int(audio_frame_counts[i]),
            "f0_voiced_frame_count": int(f0_frame_counts[i]),
            "visual_frame_start_index_0based": visual_left,
            "visual_frame_end_index_exclusive": visual_right,
            "visual_frame_count": int(visual_frame_counts[i]),
            "visual_valid_face_frame_count": int(valid_face_frame_counts[i]),
            "visual_status": status,
            "nearest_visual_frame_index_0based": nearest_index,
            "nearest_visual_distance_to_interval_s": "" if not np.isfinite(nearest_distance) else nearest_distance,
        })

    if not np.isfinite(text).all() or not np.isfinite(aligned_audio).all() or not np.isfinite(aligned_visual).all():
        raise ValueError(f"Non-finite aligned features: {sample_id}")
    if not np.all(aligned_audio[audio_mask == 0] == 0) or not np.all(aligned_visual[visual_mask == 0] == 0):
        raise ValueError(f"Invalid positions must remain zero placeholders: {sample_id}")
    dest = output_dir / f"{sample_id}.npz"
    temporary = dest.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            text=text,
            audio=aligned_audio,
            visual=aligned_visual,
            visual_nearest_candidate=nearest_visual,
            word_start_wav_s=word_start,
            word_end_wav_s=word_end,
            word_start_source_s=word_start + offset,
            word_end_source_s=word_end + offset,
            text_valid_mask=np.ones(word_count, dtype=np.uint8),
            word_time_valid_mask=time_mask,
            audio_valid_mask=audio_mask,
            audio_f0_valid_mask=audio_f0_mask,
            visual_valid_mask=visual_mask,
            visual_nearest_candidate_mask=nearest_visual_mask,
            audio_frame_count=audio_frame_counts,
            f0_voiced_frame_count=f0_frame_counts,
            visual_frame_count=visual_frame_counts,
            visual_valid_face_frame_count=valid_face_frame_counts,
            valid_length=np.asarray(word_count, dtype=np.int32),
        )
    temporary.replace(dest)
    summary = {
        "sample_id": sample_id,
        "word_count": word_count,
        "primary_time_word_count": int(time_mask.sum()),
        "audio_valid_word_count": int(audio_mask.sum()),
        "audio_f0_valid_word_count": int(audio_f0_mask.sum()),
        "visual_valid_word_count": int(visual_mask.sum()),
        "visual_nearest_candidate_word_count": int(nearest_visual_mask.sum()),
        "audio_visual_both_valid_word_count": int((audio_mask & visual_mask).sum()),
        "words_without_primary_time": int((time_mask == 0).sum()),
        "timed_words_with_no_audio_frame": int(((time_mask == 1) & (audio_mask == 0)).sum()),
        "timed_words_with_no_visual_frame": int(((time_mask == 1) & (visual_frame_counts == 0)).sum()),
        "timed_words_with_frames_but_no_valid_face": int(((time_mask == 1) & (visual_frame_counts > 0) & (visual_mask == 0)).sum()),
        "feature_file": str(dest.relative_to(ROOT)).replace("\\", "/"),
        "feature_file_sha256": sha256(dest),
    }
    return summary, indices


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    media_rows = read_csv(MEDIA_CSV)
    media = {row["sample_id"]: row for row in media_rows}
    step4 = grouped_words(read_csv(WORD_TIMES_CSV))
    text_rows = grouped_words(read_csv(TEXT_INDEX_CSV))
    text_matrix = np.load(TEXT_MATRIX, mmap_mode="r", allow_pickle=False)
    ids = {item["sample_id"] for item in manifest}
    if len(manifest) != 100 or len(ids) != 100 or ids != set(media) or ids != set(step4) or ids != set(text_rows):
        raise ValueError("Input sample IDs do not match across Steps 1, 2, 4, and 5")
    if text_matrix.shape != (1934, 768):
        raise ValueError(f"Unexpected Step 5 matrix shape: {text_matrix.shape}")
    output_dir = OUTPUT / "逐样本"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    all_indices: list[dict[str, object]] = []
    for position, item in enumerate(manifest, 1):
        sample_id = item["sample_id"]
        summary, indices = align_one(item, step4[sample_id], text_rows[sample_id],
                                     text_matrix, media[sample_id], output_dir)
        summaries.append(summary)
        all_indices.extend(indices)
        print(f"{position:03d}/100 {sample_id}: time={summary['primary_time_word_count']}, "
              f"audio={summary['audio_valid_word_count']}, visual={summary['visual_valid_word_count']}", flush=True)
    if len(all_indices) != 1934:
        raise ValueError("Aligned word count must be 1,934")
    for filename, rows in (("逐词对齐索引.csv", all_indices), ("逐样本对齐统计.csv", summaries)):
        with (OUTPUT / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    policy = {
        "primary_word_time_source": "Step 4 final_primary_start_time_wav_s / final_primary_end_time_wav_s only",
        "word_interval": "[start, end); frame centers at end go to following interval",
        "audio_pooling": "mean of valid 10 ms frame centers in interval for 74 dimensions; F0 column 72 uses voiced frames only",
        "visual_pooling": "mean of face_valid_mask=1 original-frame PTS within interval, no interpolation",
        "visual_nearest_candidate": "separate optional vector for nearest valid face frame within 0.05 s of an interval; never part of primary visual",
        "missing_feature_value": "zero placeholder with corresponding valid mask=0",
        "missing_word_time_value": "NaN with word_time_valid_mask=0",
        "timestamp_tiers_preserved": ["A_exact_transcript", "B_unique_word_overlap_in_mismatched_transcript"],
        "no_whisper_word_time_substitution": True,
        "no_time_from_feature_presence": True,
    }
    (OUTPUT / "对齐规则.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "completed",
        "sample_count": len(summaries),
        "word_count": len(all_indices),
        "text_valid_words": len(all_indices),
        "primary_time_words": sum(row["primary_time_word_count"] for row in summaries),
        "audio_valid_words": sum(row["audio_valid_word_count"] for row in summaries),
        "audio_f0_valid_words": sum(row["audio_f0_valid_word_count"] for row in summaries),
        "visual_valid_words": sum(row["visual_valid_word_count"] for row in summaries),
        "visual_nearest_candidate_words_not_in_primary": sum(row["visual_nearest_candidate_word_count"] for row in summaries),
        "audio_visual_both_valid_words": sum(row["audio_visual_both_valid_word_count"] for row in summaries),
        "words_without_primary_time": sum(row["words_without_primary_time"] for row in summaries),
        "timed_words_with_no_audio_frame": sum(row["timed_words_with_no_audio_frame"] for row in summaries),
        "timed_words_with_no_visual_frame": sum(row["timed_words_with_no_visual_frame"] for row in summaries),
        "timed_words_with_frames_but_no_valid_face": sum(row["timed_words_with_frames_but_no_valid_face"] for row in summaries),
        "samples_with_no_primary_time": [row["sample_id"] for row in summaries if row["primary_time_word_count"] == 0],
        "samples_with_no_primary_visual": [row["sample_id"] for row in summaries if row["visual_valid_word_count"] == 0],
        "input_hashes": {
            "word_time_csv_sha256": sha256(WORD_TIMES_CSV),
            "text_matrix_sha256": sha256(TEXT_MATRIX),
            "text_index_csv_sha256": sha256(TEXT_INDEX_CSV),
            "audio_step_report_sha256": sha256(ROOT / "结果" / "06_Librosa74" / "运行报告.json"),
            "visual_step_report_sha256": sha256(ROOT / "结果" / "07_FaceLandmarker52" / "运行报告.json"),
        },
    }
    (OUTPUT / "对齐汇总.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
