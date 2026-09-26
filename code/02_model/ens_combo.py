"""集成组合对比：哪种 ckpt 组合在 valid 上最优（赛题：模型选择只在验证集上做）。

动机：E1 配方三种子的 valid Macro-F1 差异很大（0.6203 / 0.6051 / 0.5787），
      全量集成反而被弱种子拉低。这里枚举所有子集，看是否"少而精"的集成更好。

用法：python ens_combo.py --ckpt_dir <含 student_s4*.pt 的目录> --seeds 42,43,44
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(errors="replace")

from data_utils import load_all, MoseiDataset, collate     # noqa: E402
from config import Config                                   # noqa: E402
from runtime import load_ensemble, find_ckpts, predict_split  # noqa: E402


def metrics(y_reg, y_cls, score, prob, theta):
    pred_head = prob.argmax(-1)
    pred_th = np.ones_like(y_cls)
    pred_th[score < -theta] = 0
    pred_th[score > theta] = 2

    def f1(p, c):
        tp = float(((y_cls == c) & (p == c)).sum())
        fp = float(((y_cls != c) & (p == c)).sum())
        fn = float(((y_cls == c) & (p != c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        return 2 * pr * rc / (pr + rc) if pr + rc else 0.0

    return dict(
        mae=float(np.abs(score - y_reg).mean()),
        acc_head=float((pred_head == y_cls).mean()),
        f1_head=float(np.mean([f1(pred_head, c) for c in range(3)])),
        neu_f1_head=f1(pred_head, 1),
        acc_th=float((pred_th == y_cls).mean()),
        f1_th=float(np.mean([f1(pred_th, c) for c in range(3)])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = Config(); cfg.version = "aligned"
    tr, va, te = load_all(cfg)

    cache, thetas = {}, {}
    for s in seeds:
        p = os.path.join(args.ckpt_dir, "student_s%d.pt" % s)
        if not os.path.isfile(p):
            print("**缺失** %s" % p); continue
        models = load_ensemble([p], verbose=False)
        thetas[s] = float(models[0][1]["theta"])
        for name, sp in (("valid", va), ("test", te)):
            r = predict_split(models, sp)
            cache[(name, s)] = dict(score=r["score"], prob=r["prob"],
                                    y_reg=r["y_reg"], y_cls=r["y_cls"])

    rows = []
    for k in range(1, len(seeds) + 1):
        for sub in itertools.combinations(seeds, k):
            if not all((("valid", s) in cache) for s in sub):
                continue
            r = {}
            for name in ("valid", "test"):
                sc = np.mean([cache[(name, s)]["score"] for s in sub], axis=0)
                pb = np.mean([cache[(name, s)]["prob"] for s in sub], axis=0)
                y_r = cache[(name, sub[0])]["y_reg"]
                y_c = cache[(name, sub[0])]["y_cls"]
                th = float(np.mean([thetas[s] for s in sub]))
                r[name] = metrics(y_r, y_c, sc, pb, th)
            rows.append(dict(subset="+".join(str(s) for s in sub), n=len(sub), **r))

    rows.sort(key=lambda x: -x["valid"]["f1_head"])
    print("\n%-10s %-4s %-9s %-9s %-9s %-9s | %-9s %-9s %-9s %-9s" %
          ("ckpt组合", "n", "v.MAE", "v.ACC", "v.MacroF1", "v.neuF1",
           "t.MAE", "t.ACC", "t.MacroF1", "t.neuF1"))
    for r in rows:
        print("%-10s %-4d %-9.4f %-9.4f %-9.4f %-9.4f | %-9.4f %-9.4f %-9.4f %-9.4f" %
              (r["subset"], r["n"], r["valid"]["mae"], r["valid"]["acc_head"],
               r["valid"]["f1_head"], r["valid"]["neu_f1_head"],
               r["test"]["mae"], r["test"]["acc_head"], r["test"]["f1_head"],
               r["test"]["neu_f1_head"]))
    best = rows[0]
    print("\n[按 valid MacroF1 选择] 最优组合 = %s" % best["subset"])
    print("  → 该组合 test: MAE %.4f ACC %.4f MacroF1 %.4f" %
          (best["test"]["mae"], best["test"]["acc_head"], best["test"]["f1_head"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(dict(seeds=seeds, theta=thetas, rows=rows), f,
                      ensure_ascii=False, indent=2)
        print("[SAVE] %s" % args.out)


if __name__ == "__main__":
    main()
