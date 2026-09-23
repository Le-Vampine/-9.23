# -*- coding: utf-8 -*-
r"""
附件3/附件4 多文件结构探查（这两个附件是"一文件夹多个 pkl"的组织方式）。

用法：
  python inspect_att34.py --dir "..\附件3-模态缺失特征样本\对齐版本"
  python inspect_att34.py --dir "..\附件4-可解释专项视频样本与特征文件\...\对齐版本" --full
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_model"))
from data_utils import load_pickle, missing_regions          # noqa: E402
import att_paths                                             # noqa: E402

MODALITIES = ("text", "audio", "vision")


def describe(obj, name, depth=0, max_depth=3):
    ind = "  " * depth
    if isinstance(obj, dict):
        print("%s%s: dict(%d) keys=%s" % (ind, name, len(obj), list(obj.keys())[:20]))
        if depth < max_depth:
            for k, v in list(obj.items())[:20]:
                describe(v, k, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        print("%s%s: %s(len=%d)" % (ind, name, type(obj).__name__, len(obj)))
        if len(obj) and depth < max_depth:
            describe(obj[0], name + "[0]", depth + 1, max_depth)
    elif isinstance(obj, np.ndarray):
        print("%s%s: ndarray %s %s" % (ind, name, obj.shape, obj.dtype))
    else:
        s = str(obj)
        print("%s%s: %s = %s" % (ind, name, type(obj).__name__, s[:100]))


def arr_of(v):
    """把任意结构转成 (N,L,d) 或 None。"""
    try:
        a = np.asarray(v)
        if a.dtype == object:
            return None
        return a
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="不填则按 --kind/--version 自动发现")
    ap.add_argument("--kind", default="att3", choices=["att3", "att4"])
    ap.add_argument("--version", default="aligned", choices=["aligned", "unaligned"])
    ap.add_argument("--full", action="store_true", help="逐文件详细统计")
    ap.add_argument("--max_files", type=int, default=0)
    a = ap.parse_args()

    if a.dir:
        files_all = sorted(glob.glob(os.path.join(a.dir, "*.pkl")))
    else:
        files_all = att_paths.discover(a.kind, a.version)
    files = files_all[:a.max_files] if a.max_files else files_all
    if not files:
        raise SystemExit("未找到 pkl：dir=%s kind=%s version=%s" % (a.dir, a.kind, a.version))
    a.dir = a.dir or os.path.dirname(files[0])
    print("=" * 78)
    print("[DIR] %s" % a.dir)
    print("[FILES] %d 个 pkl，合计 %.1f MB"
          % (len(files), sum(os.path.getsize(f) for f in files) / 1e6))
    if not files:
        return

    print("\n--- 第 1 个文件结构 ---")
    print("[FILE] %s (%.2f MB)" % (os.path.basename(files[0]),
                                   os.path.getsize(files[0]) / 1e6))
    d0 = load_pickle(files[0])
    describe(d0, "root")

    print("\n--- 逐文件统计 ---")
    tot_n = 0
    records = []
    for f in files:
        d = load_pickle(f)
        # 定位"字段字典"：可能是 d 本身，也可能是 d[split]
        raw = d
        if isinstance(d, dict):
            for k in ("test", "data", "train", "valid"):
                if k in d and isinstance(d[k], dict):
                    raw = d[k]
                    break
        keys = list(raw.keys()) if isinstance(raw, dict) else []
        info = {"file": os.path.basename(f), "size_mb": round(os.path.getsize(f) / 1e6, 2),
                "keys": keys}
        for m in MODALITIES:
            if m in raw:
                A = arr_of(raw[m])
                if A is not None:
                    info[m + "_shape"] = list(A.shape)
        for k in ("id", "ids", "regression_labels", "classification_labels", "annotations",
                  "audio_lengths", "vision_lengths"):
            if k in raw:
                v = raw[k]
                info[k] = len(v) if hasattr(v, "__len__") else str(v)
        if "id" in raw:
            ids = np.asarray(raw["id"]).reshape(-1)
            tot_n += len(ids)
            info["n"] = len(ids)
            info["id_head"] = str(ids[0])[:60]
        elif "text" in raw:
            A = arr_of(raw["text"])
            info["n"] = int(A.shape[0]) if A is not None else 0
            tot_n += info["n"]
        has_lab = "regression_labels" in raw
        info["has_labels"] = has_lab
        records.append(info)
        if a.full or len(files) <= 3:
            print("  %s" % json.dumps(info, ensure_ascii=False))

    print("\n[汇总] 样本总数 = %d" % tot_n)
    if records:
        print("[字段] 任一文件含: %s" % records[0]["keys"])
        print("[标签] %s" % ("有回归/分类标签" if records[0].get("has_labels") else "**无标签**"))

    print("\n" + "=" * 78)
    print("[缺失分布统计] （按'有效区间内连续全零'判定，需先推断有效长度）")
    from data_utils import infer_length_from_zeros

    agg = {m: {"n_samples": 0, "n_with": 0, "ratios": [], "ilens": [], "starts": []}
           for m in MODALITIES}
    combo = {}
    for f in files:
        d = load_pickle(f)
        raw = d
        if isinstance(d, dict):
            for k in ("test", "data", "train", "valid"):
                if k in d and isinstance(d[k], dict):
                    raw = d[k]
                    break
        if not all(m in raw for m in MODALITIES):
            continue
        X = {m: arr_of(raw[m]) for m in MODALITIES}
        if any(v is None for v in X.values()):
            continue
        ln = {m: (np.asarray(raw["%s_lengths" % m], dtype=np.int64)
                  if ("%s_lengths" % m) in raw else infer_length_from_zeros(X[m]))
              for m in MODALITIES}
        N = X["text"].shape[0]
        ids = np.asarray(raw["id"]).reshape(-1) if "id" in raw else np.arange(N)
        for i in range(N):
            mods = []
            for m in MODALITIES:
                L = X[m].shape[1]
                l = int(min(max(ln[m][i], 1), L))
                pad = np.zeros((L,), bool)
                pad[:l] = False
                pad[l:] = True
                zero = (np.abs(X[m][i]).sum(axis=-1) < 1e-8)
                regs = []
                j = 0
                row = zero[:l]
                while j < len(row):
                    if row[j]:
                        k = j
                        while k < len(row) and row[k]:
                            k += 1
                        if k - j >= 2:
                            regs.append((j, k))
                        j = k
                    else:
                        j += 1
                a_ = agg[m]
                a_["n_samples"] += 1
                if regs:
                    a_["n_with"] += 1
                    mods.append(m)
                a_["ratios"].append(sum(e - s for s, e in regs) / max(l, 1))
                for s, e in regs:
                    a_["ilens"].append(e - s)
                    a_["starts"].append(s / max(l, 1))
            key = "none" if not mods else "+".join(mods)
            combo[key] = combo.get(key, 0) + 1

    out = {"dir": a.dir, "n_files": len(files), "n_samples": tot_n, "combo": combo}
    for m in MODALITIES:
        a_ = agg[m]
        if not a_["ratios"]:
            continue
        r = np.asarray(a_["ratios"])
        rec = dict(n_samples=a_["n_samples"], n_with_missing=a_["n_with"],
                   frac_with_missing=round(a_["n_with"] / max(a_["n_samples"], 1), 4),
                   mean_ratio=round(float(r.mean()), 4),
                   p50=round(float(np.percentile(r, 50)), 4),
                   p95=round(float(np.percentile(r, 95)), 4),
                   max_ratio=round(float(r.max()), 4))
        if a_["ilens"]:
            il = np.asarray(a_["ilens"])
            st = np.asarray(a_["starts"])
            rec.update(interval_len_min=int(il.min()), interval_len_mean=round(float(il.mean()), 1),
                       interval_len_max=int(il.max()),
                       head=round(float((st < 0.2).mean()), 3),
                       middle=round(float(((st >= 0.2) & (st <= 0.8)).mean()), 3),
                       tail=round(float((st > 0.8).mean()), 3))
        out[m] = rec
        print("  %-6s 含缺失 %d/%d (%.1f%%)  缺失率 mean=%.3f p50=%.3f p95=%.3f max=%.3f"
              % (m, a_["n_with"], a_["n_samples"],
                 100 * a_["n_with"] / max(a_["n_samples"], 1),
                 r.mean(), np.percentile(r, 50), np.percentile(r, 95), r.max()))
        if a_["ilens"]:
            print("         区间长(步) min/mean/max=%d/%.1f/%d  位置 头/中/尾=%.2f/%.2f/%.2f"
                  % (rec["interval_len_min"], rec["interval_len_mean"], rec["interval_len_max"],
                     rec["head"], rec["middle"], rec["tail"]))
    print("  [缺失模态组合] %s" % {k: v for k, v in combo.items() if v})

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs",
                     "att34_report_%s.json" % os.path.basename(a.dir))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[SAVE] %s" % os.path.normpath(p))


if __name__ == "__main__":
    main()
