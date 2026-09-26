"""Create a compact, auditable candidate attachment for Problem 1.

Original videos, WAVs, model weights and local Python environments are excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "交付候选"
DEFAULT_ZIP = OUT / "问题一核心附件.zip"
LIMIT = 50_000_000  # 更严格地按十进制50 MB核算


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def portable_manifest() -> bytes:
    original = BASE / "输入" / "样本清单" / "样本清单.csv"
    with original.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    wanted = [
        "sample_id", "video_id", "clip_id", "source_excel_row",
        "video_relative_path", "text", "label", "annotation", "sha256",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=wanted, lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: row[k] for k in wanted} for row in rows)
    return stream.getvalue().encode("utf-8-sig")


def included_files() -> list[Path]:
    files: list[Path] = []
    for folder in ("代码", "配置", "结果", "日志"):
        for path in (BASE / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files.append(path)
    for path in (
        BASE / "README.md",
        BASE / "论文" / "复现与交付说明.md",
        BASE / "论文" / "环境与跨机复现说明.md",
        BASE / "论文" / "问题一提交核对清单.md",
        BASE / "论文" / "问题一结构与证据清单.md",
        OUT / "100条交付索引.csv",
        OUT / "提交前全量核验.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path)
    files.extend((BASE / "论文" / "公式").glob("*.png"))
    return sorted(files, key=lambda p: p.relative_to(BASE).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description="打包问题一可复核提交组件")
    parser.add_argument("--output", type=Path, default=DEFAULT_ZIP)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.parent != OUT.resolve() or target.suffix.lower() != ".zip":
        raise ValueError("输出ZIP必须直接位于E.1/交付候选目录")
    OUT.mkdir(exist_ok=True)
    audit = json.loads((OUT / "提交前全量核验.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("提交前全量核验未通过，不得打包")
    manifest: list[dict[str, object]] = []
    files = included_files()
    with ZipFile(target, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            rel = path.relative_to(BASE).as_posix()
            data = path.read_bytes()
            archive.writestr("E.1/" + rel, data)
            manifest.append({"path": "E.1/" + rel, "bytes": len(data), "sha256": sha256(data)})
        data = portable_manifest()
        rel = "E.1/输入/样本清单/可移植样本清单.csv"
        archive.writestr(rel, data)
        manifest.append({"path": rel, "bytes": len(data), "sha256": sha256(data)})
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        archive.writestr("E.1/交付文件哈希.json", manifest_bytes)

    with ZipFile(target) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad_member}")
        names = archive.namelist()
        count_by_group = {
            "主对齐NPZ": "E.1/结果/08_原文词级跨模态对齐/逐样本/",
            "原始声学NPZ": "E.1/结果/06_Librosa74/逐样本/",
            "原始视觉NPZ": "E.1/结果/07_FaceLandmarker52/逐样本/",
            "Whisper逐样本JSON": "E.1/结果/03_Whisper_small_en/逐样本/",
        }
        counts = {}
        for label, prefix in count_by_group.items():
            suffix = ".json" if label.startswith("Whisper") else ".npz"
            counts[label] = sum(n.startswith(prefix) and n.endswith(suffix) for n in names)
            if counts[label] != 100:
                raise RuntimeError(f"{label}预期100份，实际{counts[label]}份")
        required = (
            "E.1/结果/05_BERT_base_uncased/原文逐词BERT特征.npy",
            "E.1/结果/08_原文词级跨模态对齐/逐词对齐索引.csv",
            "E.1/交付候选/100条交付索引.csv",
            "E.1/交付候选/提交前全量核验.json",
            "E.1/论文/环境与跨机复现说明.md",
            "E.1/论文/问题一提交核对清单.md",
        )
        if not all(name in names for name in required):
            raise RuntimeError("提交组件缺少BERT矩阵、逐词/逐样本索引或复现说明")
        if any(n.lower().endswith((".mp4", ".wav", ".xlsx", ".onnx", ".task")) for n in names):
            raise RuntimeError("附件中混入原始媒体、标签工作簿或大型模型权重")
    size = target.stat().st_size
    report = {
        "zip": target.name,
        "size_bytes": size,
        "size_MB_decimal": round(size / 1_000_000, 3),
        "size_mib": round(size / 2**20, 3),
        "under_50_MB_decimal": size <= LIMIT,
        "zip_sha256": sha256(target.read_bytes()),
        "member_count_excluding_manifest": len(manifest),
        "per_sample_file_counts": counts,
        "audit_status": audit["status"],
        "excluded": ["原始 MP4", "WAV", "模型权重", "Python 环境", "原始 Excel"],
        "note": "问题一提交组件；正式竞赛提交应与问题二、三附件一起按十进制50 MB核算。",
    }
    (OUT / "打包核查报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if size > LIMIT:
        raise RuntimeError(f"问题一组件超过50 MB: {size / 1_000_000:.2f} MB")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
