"""Extract 74 framewise acoustic features from the 100 verified full-length WAVs.

All arrays use a 10 ms frame-center grid on the original WAV timeline. Frame
windows may overlap a word boundary; downstream word aggregation must use the
stored centers and the Step 4 word intervals. Unvoiced F0 is stored as zero
with a separate mask, not interpreted as a measured 0 Hz pitch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MFA_ENV = Path(r"C:\Users\Administrator\miniforge3\envs\mfa")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "运行环境" / "numba_cache_librosa"))
os.environ["PATH"] = str(MFA_ENV / "Library" / "bin") + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    _dll_directory = os.add_dll_directory(str(MFA_ENV / "Library" / "bin"))

import librosa
import numpy as np


MANIFEST = ROOT / "输入" / "样本清单" / "样本清单.json"
MEDIA_REPORT = ROOT / "日志" / "第2步视频检查与音频分离" / "第2步视频与音频分离核验.csv"
WAV_DIR = ROOT / "输入" / "音频WAV"
OUTPUT = ROOT / "结果" / "06_Librosa74"
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 160
N_MFCC = 20
N_MELS = 128
PITCH_FMIN_HZ = 50.0
PITCH_FMAX_HZ = 600.0

FEATURE_NAMES = (
    [f"mfcc_{i:02d}" for i in range(1, 21)]
    + [f"mfcc_delta_{i:02d}" for i in range(1, 21)]
    + [f"mfcc_delta2_{i:02d}" for i in range(1, 21)]
    + ["rms", "zero_crossing_rate", "spectral_centroid_hz", "spectral_bandwidth_hz",
       "spectral_rolloff_hz", "spectral_flatness"]
    + [f"spectral_contrast_band_{i}" for i in range(1, 7)]
    + ["f0_hz", "voiced_probability"]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        rate = stream.getframerate()
        width = stream.getsampwidth()
        count = stream.getnframes()
        data = stream.readframes(count)
    if (channels, rate, width) != (1, SAMPLE_RATE, 2):
        raise ValueError(f"Unexpected WAV format in {path.name}: {channels=} {rate=} {width=}")
    audio = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if len(audio) != count or not np.isfinite(audio).all():
        raise ValueError(f"Corrupt WAV samples: {path.name}")
    return audio, rate


def media_rows() -> dict[str, dict[str, str]]:
    with MEDIA_REPORT.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {row["sample_id"]: row for row in rows}
    if len(rows) != 100 or len(result) != 100:
        raise ValueError("Expected 100 unique samples in Step 2 media report")
    return result


def framewise_features(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # center=True makes frame k centered at k * hop_length / sample_rate.
    magnitude = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True))
    power = magnitude ** 2
    mel = librosa.feature.melspectrogram(S=power, sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=1.0), n_mfcc=N_MFCC)
    delta1 = librosa.feature.delta(mfcc, width=9, order=1, mode="interp")
    delta2 = librosa.feature.delta(mfcc, width=9, order=2, mode="interp")
    rms = librosa.feature.rms(S=magnitude, frame_length=N_FFT)
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=N_FFT, hop_length=HOP_LENGTH, center=True)
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=SAMPLE_RATE)
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=SAMPLE_RATE, centroid=centroid)
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=SAMPLE_RATE, roll_percent=0.85)
    flatness = librosa.feature.spectral_flatness(S=power)
    # librosa returns n_bands + 1 rows; n_bands=5 yields six spectral bands.
    contrast = librosa.feature.spectral_contrast(S=magnitude, sr=SAMPLE_RATE, fmin=200.0, n_bands=5)
    f0, voiced_flag, voiced_probability = librosa.pyin(
        audio, sr=SAMPLE_RATE, frame_length=N_FFT, hop_length=HOP_LENGTH,
        fmin=PITCH_FMIN_HZ, fmax=PITCH_FMAX_HZ, center=True, fill_na=np.nan,
    )

    count = math.ceil(len(audio) / HOP_LENGTH)
    pieces = [mfcc, delta1, delta2, rms, zcr, centroid, bandwidth, rolloff, flatness, contrast]
    sizes = [part.shape[1] for part in pieces] + [len(f0), len(voiced_flag), len(voiced_probability)]
    if min(sizes) < count or max(sizes) - min(sizes) > 1:
        raise ValueError(f"Librosa feature grids differ: {sizes}; expected at least {count}")
    time_wav_s = np.arange(count, dtype=np.float64) * (HOP_LENGTH / SAMPLE_RATE)
    if len(time_wav_s) and time_wav_s[-1] >= len(audio) / SAMPLE_RATE:
        raise ValueError("A frame center falls beyond WAV duration")
    voiced_mask = np.asarray(voiced_flag[:count], dtype=bool) & np.isfinite(f0[:count])
    f0_clean = np.where(voiced_mask, f0[:count], 0.0).astype(np.float32)
    probability = np.nan_to_num(voiced_probability[:count], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    probability = np.clip(probability, 0.0, 1.0)
    matrix = np.concatenate(
        [part[:, :count] for part in pieces] + [f0_clean[np.newaxis, :], probability[np.newaxis, :]],
        axis=0,
    ).T.astype(np.float32)
    if matrix.shape != (count, 74) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid acoustic feature shape or values: {matrix.shape}")
    return matrix, time_wav_s, voiced_mask, probability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", help="Recompute one named sample; default is all 100")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    media = media_rows()
    if len(manifest) != 100 or set(item["sample_id"] for item in manifest) != set(media):
        raise ValueError("Manifest and Step 2 media report do not contain the same 100 samples")
    if len(FEATURE_NAMES) != 74:
        raise AssertionError("Feature name count must be 74")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples_dir = OUTPUT / "逐样本"
    samples_dir.mkdir(exist_ok=True)
    selected = [item for item in manifest if not args.sample_id or item["sample_id"] == args.sample_id]
    if not selected:
        raise ValueError(f"Unknown sample ID: {args.sample_id}")
    summary: list[dict[str, object]] = []
    for position, item in enumerate(selected, 1):
        sample_id = item["sample_id"]
        source = WAV_DIR / f"{sample_id}.wav"
        if not source.is_file():
            raise FileNotFoundError(source)
        audio, sample_rate = read_wav(source)
        duration_s = len(audio) / sample_rate
        reported_duration = float(media[sample_id]["wav_duration_s"])
        if abs(duration_s - reported_duration) > 1 / sample_rate:
            raise ValueError(f"Step 2 WAV duration mismatch: {sample_id}")
        offset_s = float(media[sample_id]["wav_zero_maps_to_source_time_s"])
        matrix, times_wav, voiced_mask, probabilities = framewise_features(audio)
        times_source = times_wav + offset_s
        dest = samples_dir / f"{sample_id}.npz"
        np.savez_compressed(
            dest, features=matrix, frame_time_wav_s=times_wav,
            frame_time_source_s=times_source,
            f0_voiced_mask=voiced_mask.astype(np.uint8),
            frame_valid_mask=np.ones(len(times_wav), dtype=np.uint8),
        )
        summary.append({
            "sample_id": sample_id,
            "wav_duration_s": duration_s,
            "wav_sample_count": len(audio),
            "wav_sha256": sha256(source),
            "audio_start_offset_s": offset_s,
            "frame_count": len(times_wav),
            "feature_dimensions": matrix.shape[1],
            "frame_hop_s": HOP_LENGTH / SAMPLE_RATE,
            "last_frame_center_wav_s": float(times_wav[-1]),
            "f0_voiced_frame_count": int(voiced_mask.sum()),
            "f0_voiced_frame_fraction": float(voiced_mask.mean()),
            "nonzero_voicing_probability_frame_count": int((probabilities > 0).sum()),
            "sample_rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
            "feature_file": str(dest.relative_to(ROOT)).replace("\\", "/"),
            "feature_file_sha256": sha256(dest),
            "status": "passed",
        })
        print(f"{position:03d}/{len(selected):03d} {sample_id}: {len(times_wav)} frames, {int(voiced_mask.sum())} voiced", flush=True)

    if not args.sample_id:
        with (OUTPUT / "逐样本统计.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        schema = {
            "feature_names_in_column_order": FEATURE_NAMES,
            "dimensions": len(FEATURE_NAMES),
            "sample_rate_hz": SAMPLE_RATE,
            "n_fft": N_FFT,
            "hop_length_samples": HOP_LENGTH,
            "frame_center_spacing_s": HOP_LENGTH / SAMPLE_RATE,
            "centered_window": True,
            "mfcc": {"count": N_MFCC, "n_mels": N_MELS, "input": "power_to_db(mel power spectrogram, ref=1.0)"},
            "mfcc_deltas": {"width_frames": 9, "orders": [1, 2], "mode": "interp"},
            "spectral_rolloff_fraction": 0.85,
            "spectral_contrast": {"fmin_hz": 200, "n_bands_parameter": 5, "output_bands": 6},
            "pitch": {"method": "librosa.pyin", "fmin_hz": PITCH_FMIN_HZ, "fmax_hz": PITCH_FMAX_HZ,
                      "unvoiced_f0_value": 0.0, "unvoiced_mask_field": "f0_voiced_mask"},
            "timeline": "frame_time_wav_s is the analysis window center; frame_time_source_s = frame_time_wav_s + Step 2 audio start offset",
            "normalization": "raw per-frame values; no corpus-wide standardization at extraction",
        }
        (OUTPUT / "特征定义.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report = {
            "status": "completed",
            "sample_count": len(summary),
            "total_frame_count": sum(int(row["frame_count"]) for row in summary),
            "feature_dimensions": 74,
            "all_finite": True,
            "all_frame_centers_within_wav": True,
            "wav_sample_rate_hz": SAMPLE_RATE,
            "librosa_version": librosa.__version__,
            "numpy_version": np.__version__,
            "input_manifest_sha256": sha256(MANIFEST),
            "step2_media_report_sha256": sha256(MEDIA_REPORT),
            "total_f0_voiced_frames": sum(int(row["f0_voiced_frame_count"]) for row in summary),
            "voicing_is_pitch_model_output_not_speech_activity_ground_truth": True,
        }
        (OUTPUT / "运行报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
