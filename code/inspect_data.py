# -*- coding: utf-8 -*-
"""
附件2 / 附件3 / 附件4 特征文件结构核查（写进论文"数据说明与预处理"小节）。

用法：
  python inspect_data.py --pkl "E:\\数学建模\\附件2\\aligned_50.pkl"
  python inspect_data.py --pkl xxx.pkl --save report.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_model"))
from data_utils import load_pickle, build_split, describe, missing_regions  # noqa: E402


def shape_of(v):
    if isinstance(v, (list, tuple)):
        try:
            a = np.asarray(v)
            if a.dtype != object:
                return "%s %s (list)" % (a.shape, a.dtype)
        except Exception:
            pass
        lens = [len(np.asarray(x)) for x in v[:200]]
        d0 = np.asarray(v[0]).shape if len(v) else None
        return "ragged list: n=%d, elem_shape=%s, len min/max=%s/%s" % (
            len(v), d0, min(lens) if lens else "-", max(lens) if lens else "-")
    if isinstance(v, np.ndarray):
        return "%s %s" % (v.shape, v.dtype)
    s = str(v)
    return "%s: %s" % (type(v).__name__, s[:120])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=None, help="不填则自动在工作区中搜索 <version>_50.pkl")
    ap.add_argument("--version", default="aligned", choices=["aligned", "unaligned"])
    ap.add_argument("--save", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.pkl is None:
        from config import Config
        cfg = Config(version=a.version)
        a.pkl = cfg.pkl_path
        print("[AUTO] %s" % a.pkl)

    rep = {"pkl": a.pkl, "size_mb": round(os.path.getsize(a.pkl) / 1e6, 2)}
    print("=" * 78)
    print("[FILE] %s  (%.2f MB)" % (a.pkl, rep["size_mb"]))
    d = load_pickle(a.pkl)
    print("[TOP-LEVEL KEYS] %s" % list(d.keys()))
    rep["top_keys"] = [str(k) for k in d.keys()]

    split_names = [k for k in ("train", "valid", "test") if k in d]
    fields = {}
    if split_names:
        print("\n--- 顶层为 train/valid/test 结构 ---")
        for s in split_names:
            print("\n[SPLIT %s] fields: %s" % (s, list(d[s].keys())))
            for k, v in d[s].items():
                print("   %-22s %s" % (k, shape_of(v)))
            if s not in fields:
                fields[s] = {k: shape_of(v) for k, v in d[s].items()}
    else:
        print("\n--- 顶层为字段结构（单 split，附件3/4 常见）---")
        for k, v in d.items():
            print("   %-22s %s" % (k, shape_of(v)))
    rep["fields"] = fields

    # ---- 用 build_split 做一次真实解析 ----
    print("\n" + "=" * 78)
    print("[PARSE] 按 data_utils 的规则解析")
    targets = split_names if split_names else ["__single__"]
    parsed = {}
    for s in targets:
        raw = d[s] if s != "__single__" else d
        try:
            sp = build_split(raw, limit=a.limit)
        except Exception as e:
            print("[FAIL] %s: %r" % (s, e))
            continue
        name = s if s != "__single__" else "single"
        print("\n" + describe(sp, name))
        # 缺失区间统计（附件3 关键）
        regs = []
        for i in range(sp["N"]):
            r = []
            for j, m in enumerate(("text", "audio", "vision")):
                r += [(m,) + tuple(x) for x in missing_regions(sp["miss"][m][i])]
            regs.append(r)
        nz = sum(1 for r in regs if r)
        print("  含缺失样本数 = %d / %d" % (nz, sp["N"]))
        if nz:
            L = [len(r) for r in regs if r]
            print("  每样本缺失区间数 min/mean/max = %d/%.2f/%d" % (min(L), np.mean(L), max(L)))
            print("  示例: %s" % regs[[i for i, r in enumerate(regs) if r][0]])
        parsed[name] = dict(N=sp["N"], D=sp["D"], L=sp["L"],
                            has_labels=bool(sp["has_labels"]),
                            n_with_missing=nz,
                            ids_sample=[str(x) for x in sp["ids"][:3]])
        if sp.get("raw_text") is not None:
            print("  raw_text 示例: %s" % str(sp["raw_text"][0])[:100])
        print("  id 示例: %s" % str(sp["ids"][0]))
    rep["parsed"] = parsed

    if a.save:
        with open(a.save, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print("\n[SAVE] %s" % a.save)


if __name__ == "__main__":
    main()
