"""从 E.1 内的 label-100.xlsx 与原始视频生成并核验样本清单。"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import openpyxl
except ImportError as exc:
    raise SystemExit("本脚本需要 openpyxl；请使用 Codex bundled Python runtime。") from exc


ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = ROOT / "输入" / "原始视频"
LABEL_FILE = ROOT / "输入" / "原始标签" / "label-100.xlsx"
OUTPUT_DIR = ROOT / "输入" / "样本清单"
TOKEN = re.compile(r"\.0$")


def text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(video_id: str, clip_id: str) -> str:
    return f"{video_id}$_${clip_id}"


def main() -> int:
    if not LABEL_FILE.is_file():
        raise FileNotFoundError(LABEL_FILE)
    if not VIDEO_ROOT.is_dir():
        raise FileNotFoundError(VIDEO_ROOT)

    workbook = openpyxl.load_workbook(LABEL_FILE, read_only=True, data_only=True)
    sheet = workbook["label"] if "label" in workbook.sheetnames else workbook[workbook.sheetnames[0]]
    iterator = sheet.iter_rows(values_only=True)
    headers = [text(value) for value in next(iterator)]
    required = {"video_id", "clip_id", "text", "label", "annotation"}
    if not required.issubset(headers):
        raise ValueError(f"Excel 缺少必要列：{sorted(required - set(headers))}")

    source_rows: list[dict[str, Any]] = []
    for excel_row, values in enumerate(iterator, start=2):
        item = dict(zip(headers, values))
        video_id = text(item["video_id"])
        clip_id = TOKEN.sub("", text(item["clip_id"]))
        if not video_id or not clip_id:
            raise ValueError(f"label-100.xlsx 第 {excel_row} 行缺少 video_id 或 clip_id")
        item.update({"video_id": video_id, "clip_id": clip_id, "source_excel_row": excel_row})
        source_rows.append(item)

    actual_files = sorted(VIDEO_ROOT.rglob("*.mp4"))
    actual_by_key: dict[str, list[Path]] = {}
    for path in actual_files:
        relative = path.relative_to(VIDEO_ROOT)
        sample_key = key(relative.parent.name, path.stem)
        actual_by_key.setdefault(sample_key, []).append(path)

    expected_by_key: dict[str, dict[str, Any]] = {}
    duplicate_excel_keys: list[str] = []
    for item in source_rows:
        sample_key = key(item["video_id"], item["clip_id"])
        if sample_key in expected_by_key:
            duplicate_excel_keys.append(sample_key)
        expected_by_key[sample_key] = item

    missing_videos = sorted(set(expected_by_key) - set(actual_by_key))
    extra_videos = sorted(set(actual_by_key) - set(expected_by_key))
    duplicate_video_keys = sorted(sample_key for sample_key, paths in actual_by_key.items() if len(paths) != 1)

    records: list[dict[str, Any]] = []
    value_mismatches: list[dict[str, str]] = []
    copy_integrity_mismatches: list[dict[str, str]] = []
    source_video_root = (
        ROOT.parent / "第二十三届中国研究生数学建模竞赛 - 中文题目" / "中文题目" /
        "E题" / "E题资料" / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条"
    )
    original_label_file = source_video_root / "label-100.xlsx"
    label_sha256 = sha256(LABEL_FILE)
    original_label_sha256 = sha256(original_label_file) if original_label_file.is_file() else None
    label_copy_matches_source = label_sha256 == original_label_sha256

    for item in source_rows:
        sid = key(item["video_id"], item["clip_id"])
        matches = actual_by_key.get(sid, [])
        video_path = matches[0] if len(matches) == 1 else VIDEO_ROOT / item["video_id"] / f"{item['clip_id']}.mp4"
        original_video_path = source_video_root / item["video_id"] / f"{item['clip_id']}.mp4"
        exists = video_path.is_file()
        label_raw = text(item["label"])
        try:
            label_value: float | str = float(label_raw)
        except ValueError:
            label_value = label_raw

        record = {
            "sample_id": sid,
            "video_id": item["video_id"],
            "clip_id": item["clip_id"],
            "source_excel_row": item["source_excel_row"],
            "video_path": str(video_path.resolve()),
            "video_relative_path": video_path.relative_to(VIDEO_ROOT).as_posix() if exists else "",
            "source_video_path": str(original_video_path),
            "text": text(item["text"]),
            "label": label_value,
            "label_raw": label_raw,
            "annotation": text(item["annotation"]),
            "video_exists": exists,
            "file_size_bytes": video_path.stat().st_size if exists else None,
            "sha256": sha256(video_path) if exists else "",
            "inventory_status": "编号、路径、文本和标签核对通过" if exists else "缺少视频",
            "feature_status": "未开始",
            "issues": "",
        }
        if exists and original_video_path.is_file():
            copied_hash = record["sha256"]
            original_hash = sha256(original_video_path)
            if copied_hash != original_hash:
                copy_integrity_mismatches.append({"sample_id": sid, "video_path": str(video_path)})
                record["inventory_status"] = "复制完整性不一致"
                record["issues"] = "E.1 视频副本与原始附件 SHA-256 不一致"
        else:
            record["issues"] = "视频副本或原始附件缺失"

        if len(matches) == 1:
            relative = matches[0].relative_to(VIDEO_ROOT).as_posix()
            if relative != f"{item['video_id']}/{item['clip_id']}.mp4":
                value_mismatches.append({"sample_id": sid, "field": "video_relative_path"})
        records.append(record)

    annotation_counts = dict(Counter(record["annotation"] for record in records))
    expected_ids = set(expected_by_key)
    actual_ids = set(actual_by_key)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": "第1步：建立样本清单，核对编号、路径、文本和标签",
        "label_file": str(LABEL_FILE.resolve()),
        "video_directory": str(VIDEO_ROOT.resolve()),
        "original_attachment_video_directory": str(source_video_root),
        "label_sha256": label_sha256,
        "original_label_sha256": original_label_sha256,
        "label_copy_matches_original_attachment": label_copy_matches_source,
        "label_row_count": len(source_rows),
        "sample_count": len(records),
        "unique_sample_count": len(expected_ids),
        "physical_mp4_count": len(actual_files),
        "missing_video_count": len(missing_videos),
        "extra_video_count": len(extra_videos),
        "duplicate_excel_sample_keys": duplicate_excel_keys,
        "duplicate_video_sample_keys": duplicate_video_keys,
        "missing_videos": missing_videos,
        "extra_videos": extra_videos,
        "copy_integrity_mismatch_count": len(copy_integrity_mismatches),
        "copy_integrity_mismatches": copy_integrity_mismatches,
        "value_mismatches": value_mismatches,
        "annotation_counts": annotation_counts,
        "inventory_pass": (
            len(source_rows) == 100 and len(records) == 100 and len(expected_ids) == 100 and
            len(actual_files) == 100 and not missing_videos and not extra_videos and
            not duplicate_excel_keys and not duplicate_video_keys and not value_mismatches and
            not copy_integrity_mismatches and label_copy_matches_source
        ),
        "rows": records,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "样本清单.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "核查报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(records[0]) if records else []
    with (OUTPUT_DIR / "样本清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({key: report[key] for key in (
        "sample_count", "unique_sample_count", "physical_mp4_count", "missing_video_count",
        "extra_video_count", "copy_integrity_mismatch_count", "annotation_counts", "inventory_pass",
    )}, ensure_ascii=False, indent=2))
    return 0 if report["inventory_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
