# -*- coding: utf-8 -*-
r"""
附件路径自动发现（附件目录名含中文，避免在命令行/批处理里写中文路径）。

约定目录结构（实测）：
  附件3-模态缺失特征样本\{对齐版本|未对齐版本}\附件3_XX.pkl        （30 个分文件）
  附件4-可解释专项视频样本与特征文件\附件4-...\{对齐版本|未对齐版本}\XX.pkl （20 个分文件 + videos/）
"""
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_ALIGNED = "对齐版本"
SUB_UNALIGNED = "未对齐版本"


def subdir(version):
    return SUB_ALIGNED if version == "aligned" else SUB_UNALIGNED


def discover_att3(version="aligned"):
    pats = [os.path.join(ROOT, "附件3*", subdir(version), "*.pkl"),
            os.path.join(ROOT, "*附件3*", subdir(version), "*.pkl"),
            os.path.join(ROOT, "附件3*", "**", subdir(version), "*.pkl")]
    return sorted({p for pat in pats for p in glob.glob(pat, recursive=True)})


def discover_att4(version="aligned"):
    pats = [os.path.join(ROOT, "附件4*", "**", subdir(version), "*.pkl"),
            os.path.join(ROOT, "*附件4*", "**", subdir(version), "*.pkl")]
    return sorted({p for pat in pats for p in glob.glob(pat, recursive=True)})


def discover_att4_videos(version="aligned"):
    pats = [os.path.join(ROOT, "附件4*", "**", subdir(version), "videos", "*"),
            os.path.join(ROOT, "*附件4*", "**", subdir(version), "videos", "*")]
    return sorted({p for pat in pats for p in glob.glob(pat, recursive=True)})


def discover(kind, version="aligned"):
    if kind == "att3":
        return discover_att3(version)
    if kind == "att4":
        return discover_att4(version)
    raise ValueError("kind 必须是 att3 或 att4")


def data_dir_of(pkl_paths):
    """取公共目录（用于报告输出）。"""
    if not pkl_paths:
        return None
    return os.path.dirname(pkl_paths[0])


if __name__ == "__main__":
    for kind in ("att3", "att4"):
        for ver in ("aligned", "unaligned"):
            fs = discover(kind, ver)
            print("%s / %-9s : %d 个 pkl  %s" % (kind, ver, len(fs),
                                                 os.path.dirname(fs[0]) if fs else "-"))
    vs = discover_att4_videos()
    print("附件4 videos: %d 个文件 %s" % (len(vs), os.path.dirname(vs[0]) if vs else "-"))
