# -*- coding: utf-8 -*-
r"""独立缺失退化扫描：对**任意** checkpoint 目录跑完整 (类型 × 位置 × 缺失率) 网格。

存在的理由
----------
`train.py --sweep 1` 的扫描是训练流程的一部分，只能扫"刚训完的那个模型"，且网格固定
（ρ≤0.5、无 ρ=1.0 整模态）。但论文第 4 节必须给出：
    ① **常规固定权重融合模型**（A0/A1 朴素基线）与 **本文模型** 在同一网格下的退化曲线对照；
    ② 整模态缺失（ρ=1.0）的极端点 —— 题目所述"性能急剧下降"正是在这里出现。
本脚本直接加载已有 ckpt 做推理，**无需重训**，网格可配置。

用法：
  python sweep_degradation.py --ckpt_dir <dir> --tag a0_plain
  python sweep_degradation.py --ckpt_dir <dir> --tag ours --rhos 0.1,0.2,0.3,0.4,0.5,0.7,1.0
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.append(HERE)

from config import Config                                  # noqa: E402
from data_utils import load_all                            # noqa: E402
from runtime import load_ensemble, find_ckpts, predict_split   # noqa: E402

MODS = ("text", "audio", "vision")


def metrics(pred):
    """MAE / Pearson / ACC(分类头) / Macro-F1(分类头)。"""
    y, s, p = pred["y_reg"], pred["score"], pred["prob"]
    yc = pred["y_cls"]
    mae = float(np.abs(s - y).mean())
    corr = float(np.corrcoef(s, y)[0, 1]) if s.std() > 0 else float("nan")
    yhat = p.argmax(axis=1)
    acc = float((yhat == yc).mean())
    f1s = []
    for c in (0, 1, 2):
        tp = float(((yhat == c) & (yc == c)).sum())
        fp = float(((yhat == c) & (yc != c)).sum())
        fn = float(((yhat != c) & (yc == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return dict(mae=mae, pearson=corr, acc=acc, macro_f1=float(np.mean(f1s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--tag", required=True, help="输出文件名标记")
    ap.add_argument("--out_dir", default=None, help="默认写到 ckpt_dir")
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--types", default="text,audio,vision,text+audio,text+vision,"
                                      "audio+vision,text+audio+vision")
    ap.add_argument("--positions", default="head,middle,tail,random")
    ap.add_argument("--rhos", default="0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    types = [s for s in a.types.split(",") if s]
    positions = [s for s in a.positions.split(",") if s]
    rhos = [float(s) for s in a.rhos.split(",") if s]

    cfg = Config()
    print("[LOAD] %s" % a.ckpt_dir)
    paths = find_ckpts(a.ckpt_dir)
    if not paths:
        print("[ERR] 目录下没有 student_*.pt")
        return 1
    models = load_ensemble(paths, device=a.device)

    tr, va, te = load_all(cfg)
    sp = va if a.split == "valid" else te
    print("[SPLIT] %s  n=%d" % (a.split, len(sp["ids"])))

    # ---- 干净基线 ----
    base = metrics(predict_split(models, sp, batch_size=a.batch_size, device=a.device))
    print("[CLEAN] MAE=%.4f ACC=%.4f MacroF1=%.4f r=%.4f"
          % (base["mae"], base["acc"], base["macro_f1"], base["pearson"]))

    rows = []
    for mtype in types:
        for pos in positions:
            for rho in rhos:
                pa = predict_split(models, sp, batch_size=a.batch_size, device=a.device,
                                   scenario=(mtype, pos, rho), unaware=False)
                pu = predict_split(models, sp, batch_size=a.batch_size, device=a.device,
                                   scenario=(mtype, pos, rho), unaware=True)
                ra, ru = metrics(pa), metrics(pu)
                rows.append(dict(
                    missing_type=mtype, position=pos, rho=rho,
                    aware_mae=ra["mae"], aware_acc=ra["acc"], aware_f1=ra["macro_f1"],
                    aware_corr=ra["pearson"],
                    unaware_mae=ru["mae"], unaware_acc=ru["acc"], unaware_f1=ru["macro_f1"],
                    unaware_corr=ru["pearson"],
                    d_mae=ra["mae"] - base["mae"], d_acc=ra["acc"] - base["acc"],
                    d_f1=ra["macro_f1"] - base["macro_f1"],
                    mae_gain=ru["mae"] - ra["mae"], acc_gain=ru["acc"] - ra["acc"]))
            print("  %-18s %-7s  %s" % (
                mtype, pos,
                " ".join("ρ%.1f ACC=%.3f" % (rho, r["aware_acc"])
                         for rho, r in zip(rhos, rows[-len(rhos):]))))

    import pandas as pd
    out_dir = a.out_dir or a.ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    csv = os.path.join(out_dir, "sweep_valid_%s.csv" % a.tag)
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    # ---- 摘要：按 (类型, ρ) 对位置取平均 ----
    g = df.groupby(["missing_type", "rho"]).agg(
        aware_acc=("aware_acc", "mean"), aware_mae=("aware_mae", "mean"),
        aware_f1=("aware_f1", "mean"), unaware_acc=("unaware_acc", "mean"),
        unaware_mae=("unaware_mae", "mean")).reset_index()
    g["d_acc_vs_clean"] = g["aware_acc"] - base["acc"]
    g["d_mae_vs_clean"] = g["aware_mae"] - base["mae"]
    gcsv = os.path.join(out_dir, "sweep_summary_%s.csv" % a.tag)
    g.to_csv(gcsv, index=False, encoding="utf-8-sig")

    lines = ["# 缺失退化摘要｜%s（%s 集，n=%d）" % (a.tag, a.split, len(sp["ids"])), "",
             "干净基线：MAE=%.4f，ACC=%.4f，Macro-F1=%.4f，r=%.4f"
             % (base["mae"], base["acc"], base["macro_f1"], base["pearson"]), "",
             "| 缺失类型 | ρ | ACC(aware) | ΔACC | MAE(aware) | ΔMAE | ACC(unaware 对照) |",
             "|---|---|---|---|---|---|---|"]
    for _, r in g.sort_values(["missing_type", "rho"]).iterrows():
        lines.append("| %s | %.1f | %.4f | %+.4f | %.4f | %+.4f | %.4f |"
                     % (r["missing_type"], r["rho"], r["aware_acc"], r["d_acc_vs_clean"],
                        r["aware_mae"], r["d_mae_vs_clean"], r["unaware_acc"]))
    md = os.path.join(out_dir, "sweep_summary_%s.md" % a.tag)
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_dir, "sweep_clean_%s.json" % a.tag), "w",
              encoding="utf-8") as f:
        json.dump(dict(tag=a.tag, split=a.split, clean=base, ckpts=paths),
                  f, ensure_ascii=False, indent=2)

    print("\n[CLEAN ] MAE=%.4f ACC=%.4f MacroF1=%.4f" % (base["mae"], base["acc"],
                                                         base["macro_f1"]))
    print("[SAVE] %s\n[SAVE] %s\n[SAVE] %s" % (csv, gcsv, md))
    # ---- 关键结论直接打印，便于快速判读 ----
    print("\n[KEY] 整模态缺失（ρ=1.0）的 ACC 变化：")
    for mtype in types:
        sel = g[(g["missing_type"] == mtype) & (g["rho"] == 1.0)]
        if len(sel):
            r = sel.iloc[0]
            print("   %-18s ACC %.4f → %.4f  (Δ%+.4f)" % (
                mtype, base["acc"], r["aware_acc"], r["d_acc_vs_clean"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
