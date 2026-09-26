"""把附件1原视频与第一问核心产物整理成队友接入包。"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


E1 = Path(__file__).resolve().parents[2]
OUT = E1 / "给第二问队友_交接包"
RAW_NAME = "01_附件1_100条原视频与标签.zip"
FEATURE_NAME = "02_问题一特征代码配置日志.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_raw_pack(samples: list[dict]) -> dict:
    raw_zip = OUT / RAW_NAME
    video_root = E1 / "输入" / "原始视频"
    label = E1 / "输入" / "原始标签" / "label-100.xlsx"
    assert label.is_file()
    assert len(samples) == len({x["sample_id"] for x in samples}) == 100

    compact = []
    seen_paths = set()
    with zipfile.ZipFile(raw_zip, "w", allowZip64=True) as archive:
        for row in samples:
            relative = Path(row["video_relative_path"])
            path = video_root / relative
            assert path.is_file(), path
            assert relative.as_posix() not in seen_paths
            seen_paths.add(relative.as_posix())
            actual_hash = sha256(path)
            assert actual_hash == row["sha256"], row["sample_id"]
            archive.write(
                path,
                arcname=f"原始视频/{relative.as_posix()}",
                compress_type=zipfile.ZIP_STORED,
            )
            compact.append(
                {
                    "sample_id": row["sample_id"],
                    "video_relative_path": relative.as_posix(),
                    "sha256": actual_hash,
                    "text": row["text"],
                    "label": row["label"],
                    "annotation": row["annotation"],
                }
            )
        archive.write(label, arcname="原始标签/label-100.xlsx")
        manifest = {
            "sample_count": 100,
            "video_count": 100,
            "label_sha256": sha256(label),
            "samples": compact,
        }
        manifest_entry = zipfile.ZipInfo("样本映射与SHA256.json", date_time=(2026, 9, 26, 0, 0, 0))
        manifest_entry.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(manifest_entry, json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(raw_zip) as archive:
        assert archive.testzip() is None
        assert sum(name.lower().endswith(".mp4") for name in archive.namelist()) == 100
        assert "原始标签/label-100.xlsx" in archive.namelist()
    return {
        "file": RAW_NAME,
        "bytes": raw_zip.stat().st_size,
        "sha256": sha256(raw_zip),
        "video_count": 100,
        "includes_label": True,
    }


def copy_feature_pack() -> dict:
    source = E1 / "交付候选" / "问题一核心附件.zip"
    report = json.loads((E1 / "交付候选" / "打包核查报告.json").read_text(encoding="utf-8"))
    assert sha256(source) == report["zip_sha256"]
    target = OUT / FEATURE_NAME
    shutil.copy2(source, target)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        for section in ("05_BERT_base_uncased", "06_Librosa74", "07_FaceLandmarker52", "08_原文词级跨模态对齐"):
            assert any(section in name for name in names), section
        assert sum("08_原文词级跨模态对齐/逐样本/" in name and name.endswith(".npz") for name in names) == 100
    return {
        "file": FEATURE_NAME,
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
        "word_aligned_npz_count": 100,
        "includes_raw_video": False,
        "includes_model_weights": False,
    }


def write_readme(raw_info: dict, feature_info: dict) -> Path:
    guide = OUT / "00_先读我_给第二问队友.md"
    guide.write_text(
        f"""# 第一问100条视频接入第二问模型：交接说明

请一起发送本目录的两个ZIP、这份说明和 `03_接入核查与要求清单.md`。先前单独发送的 `08_原文词级跨模态对齐` 已完整包含在第二个ZIP内，无须再找散文件。

## 两个包是什么

1. `{RAW_NAME}`：赛事附件1的100条原MP4、`label-100.xlsx` 和按样本编号记录的SHA-256映射。解压后视频路径为 `原始视频/<video_id>/<clip_id>.mp4`。该包是团队内部交接素材，**不要放入比赛≤50MB的最终附件**。
2. `{FEATURE_NAME}`：E.1 已生成的100份词级三模态NPZ，以及逐帧声学/视觉特征、原文BERT特征、逐词索引、代码、配置、关键日志、质量报告和读取说明。解压后从 `E.1/README.md` 与 `E.1/论文/复现与交付说明.md` 开始读。该包不含原MP4、WAV、预训练模型权重或第二问模型权重。

若需要在队友电脑上重跑E.1的提取程序，把第一个包中的 `原始视频/` 和 `原始标签/` 两个目录放到第二个包解出的 `E.1/输入/` 下，再按README配置本机模型与Python环境。仅进行第二问原生推理时，直接把原视频包交给第二问原有的预处理入口即可。

## 请第二问队友优先执行的实验

在第二问**已训练且冻结**的模型环境内，用第二问训练时同一套特征提取、50步对齐、35维视觉生成、74维音频定义、标准化及填充/缺失掩码规则处理第一个包的100条原MP4，然后逐条推理。请输出100行结果：`sample_id`、预测极性、预测情感强度、使用的模型版本/权重哈希、输入版本及预处理状态；若某条输入失败，记录原因。不要用这100条的标签重新选择权重、阈值或标准化参数。

第二个包可用于核对第一问的输入和时间证据；若希望**直接**让第二问模型消费这些E.1特征，须另做经验证的接口适配。E.1为原文词级变长T×768/74/52，第二问 `aligned_50` 为50×768/74/35。52维视觉不可截取35列；同为74维也不证明音频特征列含义相同。513个原文词没有可信词时间，音视频零值必须结合掩码理解；不能以零值直接推断填充或真实缺失。

## 评价分组

附件1的100条中，18条也在附件2标签表：11条属第二问训练集，7条属其测试集；其余82条不在附件2。请对82条非重合样本报告主要外部结果，7条测试重合和11条训练重合分别列出，不能把100条整体指标称为独立泛化成绩。完整编号及核查依据见 `03_接入核查与要求清单.md`。

## 完整性与大小

- 原视频包：{raw_info['bytes']:,} 字节，SHA-256 `{raw_info['sha256']}`；内含100个MP4、1个原始标签表。
- 特征包：{feature_info['bytes']:,} 字节，SHA-256 `{feature_info['sha256']}`；内含100份主对齐NPZ。
- 其余文件哈希见 `SHA256SUMS.txt`。ZIP已完成条目CRC检查。
""",
        encoding="utf-8",
    )
    return guide


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples = json.loads((E1 / "输入" / "样本清单" / "样本清单.json").read_text(encoding="utf-8"))
    raw_info = make_raw_pack(samples)
    feature_info = copy_feature_pack()
    guide = write_readme(raw_info, feature_info)
    audit = OUT / "03_接入核查与要求清单.md"
    shutil.copy2(E1 / "论文" / "问题一要求与第二问模型接入核查.md", audit)
    files = [guide, OUT / RAW_NAME, OUT / FEATURE_NAME, audit]
    sums = OUT / "SHA256SUMS.txt"
    sums.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files), encoding="utf-8")
    report = {
        "raw_pack": raw_info,
        "feature_pack": feature_info,
        "other_files": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in (guide, audit, sums)],
        "note": "团队内部交接包，不等于赛事最终≤50MB附件",
    }
    (OUT / "交接包核验.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
