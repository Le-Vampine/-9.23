# -*- coding: utf-8 -*-
r"""S0：附件3/4 元数据侧车 + 文本有效长度口径修正（决策 D1=A）。

背景（实测，见 `解决方案/11_问题3实施方案与执行计划.md` §1.3）：
  附件2 的 `text` 尾部填充行是 BERT 对 `[PAD]` 的隐状态，**非零**，唯一可靠的长度来源是
  `text_bert` 的注意力掩码；而 `data_att/att3_aligned.pkl`、`att4_aligned.pkl` 生成时未写入
  `text_bert`，导致 `build_split` 退化为"尾部全零推断" → 文本 50 步全部被当作有效内容。
  实测代价（附件2 验证集，同模型对照）：MAE 0.5866 → 0.8514、ACC 0.6223 → 0.3310。

本脚本做的事（不改动原始附件，只改工程内的缓存 pkl）：
  1. 从**原始**附件3/4 pkl 抽取 `text_bert`（文本有效长度）与 `raw_text`（证据回看用）；
  2. 备份原有缓存 pkl 到 `data_att/_backup_preD1/`；
  3. 重写 `data_att/{att3,att4}_aligned.pkl`：数值数组**原样保留**，仅补 `text_bert`/`text_lengths`/`raw_text`；
  4. 生成侧车 `data_att/att_meta.json`（逐样本文本长度、语音/视觉有效长度、视频路径与时长）；
  5. 打印核查报告（长度分布、视频时长、口径断言）。

用法：
  python att_meta.py --kind both
  python att_meta.py --kind att4 --no_write      # 只出报告，不改文件
"""
import argparse
import json
import os
import pickle
import re
import shutil
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
sys.path.append(CODE)
sys.path.append(os.path.join(CODE, "02_model"))
import att_paths                                     # noqa: E402
from data_utils import load_pickle, infer_length_from_zeros   # noqa: E402

DATA_ATT = os.path.join(ROOT, "data_att")
BACKUP = os.path.join(DATA_ATT, "_backup_preD1")
MODS = ("text", "audio", "vision")


# --------------------------------------------------------------------------
def raw_of(d):
    """兼容 {'test': {...}} 与顶层即字段 两种组织方式。"""
    if isinstance(d, dict):
        for k in ("test", "data", "train", "valid"):
            if k in d and isinstance(d[k], dict):
                return d[k]
    return d


def as_bert(tb, L=50):
    """规范成 (1,3,L)。"""
    a = np.asarray(tb)
    if a.ndim == 3:
        a = a[0] if a.shape[0] == 1 else a[0]
    if a.ndim == 2:
        if a.shape[0] == 3:
            return a[None]
        if a.shape[1] == 3:
            return np.transpose(a, (1, 0))[None]
        return a[None]
    if a.ndim == 1:
        return a.reshape(1, 1, -1)
    raise ValueError("无法识别的 text_bert 形状 %s" % (a.shape,))


def mask_len(tb_row):
    """注意力掩码 -> 有效长度。"""
    m = np.asarray(tb_row)
    nz = np.nonzero(m)[0]
    return int(nz[-1] + 1) if len(nz) else 1


def mp4_duration(path):
    """纯 Python 解析 mp4 `mvhd`，返回时长（秒）；失败返回 None。

    注意：本批 mp4 的 `moov` 原子位于文件**末尾**（非 faststart），只读头部会解析失败，
    故整体读入（单文件 ≤4 MB，20 个视频总开销可忽略）。
    """
    try:
        with open(path, "rb") as f:
            head = f.read()
    except OSError:
        return None
    i = head.find(b"mvhd")
    if i < 0:
        return None
    p = i + 4
    ver = head[p]
    try:
        if ver == 0:
            ts, dur = struct.unpack(">II", head[p + 12:p + 20])
        else:
            ts, dur = struct.unpack(">IQ", head[p + 20:p + 32])
    except struct.error:
        return None
    return float(dur) / float(ts) if ts else None


def stem_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def idx_of(name):
    m = re.search(r"(\d+)\s*$", os.path.splitext(name)[0])
    return int(m.group(1)) if m else -1


# --------------------------------------------------------------------------
def collect(kind, version="aligned"):
    """返回 (records, arrays_or_None, src_files)"""
    files = att_paths.discover(kind, version)
    if not files:
        raise SystemExit("未找到 %s/%s 的原始 pkl" % (kind, version))

    # raw_text 的来源：附件3 对齐版没有 raw_text，从未对齐版按编号取
    text_src = {}
    if kind == "att3":
        for p in att_paths.discover("att3", "unaligned"):
            r = raw_of(load_pickle(p))
            if "raw_text" in r:
                rt = np.asarray(r["raw_text"]).reshape(-1)
                text_src[idx_of(p)] = str(rt[0])
    # 视频
    vids = {}
    if kind == "att4":
        for v in att_paths.discover_att4_videos(version):
            if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                vids[idx_of(v)] = v

    recs = []
    for p in files:
        r = raw_of(load_pickle(p))
        i = idx_of(p)
        rec = dict(id=stem_id(p), idx=i, src=p)
        tb = r.get("text_bert")
        if tb is not None:
            tb = as_bert(tb)
            rec["text_len"] = mask_len(tb[0, 1])
            rec["text_bert"] = tb[0].astype(np.int64)
        else:
            rec["text_len"] = None
        for m in ("audio", "vision"):
            if m in r:
                X = np.asarray(r[m], dtype=np.float32)
                X = X[None] if X.ndim == 2 else X
                rec["%s_len" % m] = int(infer_length_from_zeros(X)[0])
        if kind == "att4":
            rt = r.get("raw_text")
            if rt is not None:
                rec["raw_text"] = str(np.asarray(rt).reshape(-1)[0])
            elif i in text_src:
                rec["raw_text"] = text_src[i]
        else:
            rec["raw_text"] = text_src.get(i, "")
        if i in vids:
            rec["video"] = vids[i]
            rec["duration_s"] = mp4_duration(vids[i])
        recs.append(rec)
    return recs, files


def rebuild_cache(kind, recs, write=True, version="aligned"):
    """把 text_bert / text_lengths / raw_text 写回 data_att 缓存 pkl。"""
    cache = os.path.join(DATA_ATT, "%s_%s.pkl" % (kind, version))
    if not os.path.isfile(cache):
        raise SystemExit("缓存不存在：%s" % cache)
    obj = load_pickle(cache)
    n = len(obj["text"])
    if n != len(recs):
        raise SystemExit("样本数不一致：缓存 %d vs 原始 %d" % (n, len(recs)))

    order = [r["id"] for r in recs]
    cache_ids = [str(x) for x in np.asarray(obj["id"]).reshape(-1)]
    print("  [ID ] 缓存 id 顺序与原始文件名顺序一致：%s" % (cache_ids == order,))
    if cache_ids != order:
        print("       缓存 %s" % cache_ids[:4])
        print("       原始 %s" % order[:4])

    new = dict(obj)
    tb_all = np.stack([r["text_bert"] for r in recs], axis=0) if all(
        r.get("text_bert") is not None for r in recs) else None
    if tb_all is not None:
        new["text_bert"] = tb_all
        new["text_lengths"] = np.array([r["text_len"] for r in recs], dtype=np.int64)
    new["raw_text"] = np.array([r.get("raw_text", "") for r in recs], dtype=object)
    if all(r.get("duration_s") for r in recs):
        new["video_duration_s"] = np.array([r["duration_s"] for r in recs], dtype=np.float32)

    if write:
        os.makedirs(BACKUP, exist_ok=True)
        bak = os.path.join(BACKUP, os.path.basename(cache))
        if not os.path.isfile(bak):
            shutil.copy2(cache, bak)
            print("  [BAK] %s" % bak)
        # 校验：数值数组必须逐位未变
        for m in MODS:
            a = np.asarray(obj[m], dtype=np.float32)
            b = np.asarray(new[m], dtype=np.float32)
            assert a.shape == b.shape and np.array_equal(a, b), "数组 %s 被改动了！" % m
        with open(cache, "wb") as f:
            pickle.dump(new, f, protocol=4)
        print("  [SAV] %s（补 text_bert/text_lengths/raw_text，数值数组逐位未变）" % cache)
    return cache


def report(kind, recs):
    print("=== %s ===" % kind)
    tl = np.array([r["text_len"] for r in recs if r.get("text_len")])
    print("  样本数 %d；文本有效长度 min/mean/max = %d/%.1f/%d（修正前恒为 50）"
          % (len(recs), tl.min(), tl.mean(), tl.max()))
    for m in ("audio", "vision"):
        v = np.array([r["%s_len" % m] for r in recs if r.get("%s_len" % m)])
        deg = int((v <= 1).sum())
        print("  %-6s 有效长度 min/mean/max = %d/%.1f/%d（其中整模态缺失 %d 条）"
              % (m, v.min(), v.mean(), v.max(), deg))
    miss_rt = [i for i, r in enumerate(recs) if not r.get("raw_text")]
    print("  raw_text 缺失条数 = %d %s" % (len(miss_rt), miss_rt[:5]))
    if any(r.get("duration_s") for r in recs):
        d = np.array([r["duration_s"] for r in recs if r.get("duration_s")])
        print("  视频时长 min/mean/max = %.3f/%.3f/%.3f s（%d 个视频）"
              % (d.min(), d.mean(), d.max(), len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="both", choices=["att3", "att4", "both"])
    ap.add_argument("--version", default="aligned")
    ap.add_argument("--no_write", action="store_true", help="只出报告，不改缓存文件")
    a = ap.parse_args()

    meta = {}
    kinds = ["att3", "att4"] if a.kind == "both" else [a.kind]
    for kind in kinds:
        recs, files = collect(kind, a.version)
        report(kind, recs)
        rebuild_cache(kind, recs, write=not a.no_write, version=a.version)
        meta[kind] = [{k: v for k, v in r.items()
                       if k not in ("text_bert", "src")} for r in recs]
        meta["%s_src_files" % kind] = files

    out = os.path.join(DATA_ATT, "att_meta.json")
    if not a.no_write:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        print("\n[SAVE] %s" % out)
    print("ATT_META_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
