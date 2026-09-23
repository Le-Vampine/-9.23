# -*- coding: utf-8 -*-
r"""
附件3/附件4 深度探查：逐文件 N、字段、形状、缺失分布、id 来源。

用法：python probe_att34.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "02_model"))
import att_paths                                   # noqa: E402
from data_utils import load_pickle, infer_length_from_zeros   # noqa: E402

MODALITIES = ("text", "audio", "vision")
REPORT = {}


def zeros_regions(row, min_len=2):
    regs, j = [], 0
    while j < len(row):
        if row[j]:
            k = j
            while k < len(row) and row[k]:
                k += 1
            if k - j >= min_len:
                regs.append((j, k))
            j = k
        else:
            j += 1
    return regs


def probe(kind, version):
    files = att_paths.discover(kind, version)
    if not files:
        print("[%s/%s] 未找到文件" % (kind, version))
        return None
    print("\n" + "=" * 78)
    print("[%s / %s]  %d 个 pkl，合计 %.2f MB" % (kind, version, len(files),
                                                 sum(os.path.getsize(f) for f in files) / 1e6))
    allkeys, per_file_n, shapes = set(), [], {}
    miss = {m: dict(n=0, with_missing=0, ratios=[], ilens=[], starts=[]) for m in MODALITIES}
    n_text_bert = 0
    for f in files:
        d = load_pickle(f)
        raw = d
        if isinstance(d, dict):
            for k in ("test", "data", "train", "valid"):
                if k in d and isinstance(d[k], dict):
                    raw = d[k]
                    break
        allkeys |= set(raw.keys())
        n = None
        for m in MODALITIES:
            if m in raw:
                A = np.asarray(raw[m])
                if A.dtype != object:
                    shapes.setdefault(m, set()).add(tuple(A.shape[1:]))
                    n = A.shape[0]
                    break
        if n is None:
            for k in ("text_bert",):
                if k in raw:
                    A = np.asarray(raw[k])
                    shapes.setdefault(k, set()).add(tuple(A.shape[1:]))
                    n = A.shape[0]
        per_file_n.append(n or 0)
        if "text_bert" in raw:
            n_text_bert += 1

        # ---- 缺失统计（有效长度：有 lengths 用 lengths，否则尾部全零推断）----
        for m in MODALITIES:
            if m not in raw:
                continue
            X = np.asarray(raw[m], dtype=np.float32)
            if X.dtype == object:
                continue
            L = X.shape[1]
            ln_k = "%s_lengths" % m
            if ln_k in raw:
                ln = np.asarray(raw[ln_k], dtype=np.int64)
            elif m == "text" and "text_bert" in raw:
                am = np.asarray(raw["text_bert"])[:, 1, :].astype(np.int64)
                ln = np.array([(np.nonzero(am[i])[0][-1] + 1) if np.any(am[i]) else 1
                               for i in range(am.shape[0])])
            else:
                ln = infer_length_from_zeros(X)
            for i in range(X.shape[0]):
                l = int(min(max(int(ln[i]), 1), L))
                zero = (np.abs(X[i]).sum(axis=-1) < 1e-8)[:l]
                regs = zeros_regions(zero)
                rec = miss[m]
                rec["n"] += 1
                if regs:
                    rec["with_missing"] += 1
                rec["ratios"].append(sum(e - s for s, e in regs) / max(l, 1))
                for s, e in regs:
                    rec["ilens"].append(e - s)
                    rec["starts"].append(s / max(l, 1))

    print("[字段] %s" % sorted(allkeys))
    print("[形状] %s" % {k: sorted(list(v)) for k, v in shapes.items()})
    print("[样本数] 总=%d  每文件 min/max=%d/%d  有 text_bert 的文件=%d/%d"
          % (sum(per_file_n), min(per_file_n), max(per_file_n), n_text_bert, len(files)))
    print("[是否含预计算 text(768)] %s" % ("是" if "text" in allkeys else "**否 —— 只有 text_bert**"))
    print("[是否含 id] %s" % ("是" if ("id" in allkeys) else "**否 —— 只能以文件名作为样本编号**"))
    rep = dict(kind=kind, version=version, n_files=len(files), n_samples=sum(per_file_n),
               keys=sorted(allkeys), shapes={k: sorted(list(v)) for k, v in shapes.items()})
    for m in MODALITIES:
        rec = miss[m]
        if rec["n"] == 0:
            continue
        r = np.asarray(rec["ratios"])
        d = dict(n=rec["n"], with_missing=rec["with_missing"],
                 frac=round(rec["with_missing"] / rec["n"], 4),
                 mean_ratio=round(float(r.mean()), 4),
                 p50=round(float(np.percentile(r, 50)), 4),
                 p95=round(float(np.percentile(r, 95)), 4),
                 max_ratio=round(float(r.max()), 4))
        if rec["ilens"]:
            il, st = np.asarray(rec["ilens"]), np.asarray(rec["starts"])
            d.update(len_min=int(il.min()), len_mean=round(float(il.mean()), 1), len_max=int(il.max()),
                     head=round(float((st < 0.2).mean()), 3),
                     middle=round(float(((st >= 0.2) & (st <= 0.8)).mean()), 3),
                     tail=round(float((st > 0.8).mean()), 3))
        rep[m] = d
        print("  %-6s 含缺失 %d/%d (%.1f%%)  缺失率 mean=%.3f p50=%.3f p95=%.3f max=%.3f%s"
              % (m, d["with_missing"], d["n"], 100 * d["frac"], d["mean_ratio"], d["p50"],
                 d["p95"], d["max_ratio"],
                 ("  区间长 %.0f步 位置头/中/尾=%.2f/%.2f/%.2f"
                  % (d["len_mean"], d["head"], d["middle"], d["tail"])) if rec["ilens"] else ""))
    return rep


def main():
    for kind, ver in (("att3", "aligned"), ("att3", "unaligned"),
                      ("att4", "aligned"), ("att4", "unaligned")):
        r = probe(kind, ver)
        if r:
            REPORT["%s_%s" % (kind, ver)] = r
    vs = att_paths.discover_att4_videos()
    print("\n[附件4 视频] %d 个文件，示例: %s" % (len(vs), [os.path.basename(v) for v in vs[:5]]))
    p = os.path.join(HERE, "..", "runs", "att34_probe.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(REPORT, f, ensure_ascii=False, indent=2)
    print("[SAVE] %s" % os.path.normpath(p))


if __name__ == "__main__":
    main()
