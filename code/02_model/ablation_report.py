# -*- coding: utf-8 -*-
r"""
消融实验汇总（论文第 4 节）：把 runs/ablate_* 的干净指标与缺失场景评测汇总成
两张可直接粘贴进论文的表：

  1) 干净性能表：完整模型 vs A0/A1/A2/A3/A6/A7/A9，多种子 mean±std + Δ（相对完整模型）
  2) 缺失场景表：干净基线 + 4 个代表场景（text/text+audio × middle/random × ρ=0.3）
     的 MAE 与"最大退化"，用于回答"是否急剧下降"

用法：
  python 02_model\ablation_report.py --runs_dir ..\runs --out_dir ..\paper_tables
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

EXPS = [
    ("q2v3", "完整模型（MRF-Net）"),
    ("ablate_a0_plain", "A0 朴素融合（干净训练、无指示）"),
    ("ablate_a1_plain_ind", "A1 朴素融合 + 掩码指示 + 缺失注入"),
    ("ablate_a2_nomask", "A2 去掉掩码指示"),
    ("ablate_a3_norec", "A3 去掉跨模态重建损失"),
    ("ablate_a6_exp1", "A6 单专家（去掉动态专家）"),
    ("ablate_a7_nokd", "A7 去掉教师蒸馏"),
    ("ablate_a9_nocurric", "A9 去掉缺失课程学习"),
]
SCEN = [("text", "middle", 0.3), ("text", "random", 0.3),
        ("text+audio", "middle", 0.3), ("text+audio", "random", 0.3)]


def md_table(df, fmt="%.4f"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda x: ("—" if pd.isna(x) else fmt % x))
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values]
    return "\n".join([head, sep] + rows)


def collect_clean(runs_dir):
    rows = []
    for d, label in EXPS:
        vals = []
        seeds = []
        for p in sorted(glob.glob(os.path.join(runs_dir, d, "metrics_s*.json"))):
            if "_legacy" in p:
                continue
            try:
                m = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(m, dict) or "valid_clean" not in m:
                continue
            v, t = m["valid_clean"], m["test_clean"]
            vals.append((v.get("mae"), v.get("acc_head"), v.get("f1_head"),
                         t.get("mae"), t.get("acc_head"), t.get("f1_head")))
            seeds.append(os.path.basename(p))
        if not vals:
            continue
        a = np.array(vals, dtype=float)
        rows.append(dict(
            实验=label, 种子数=len(vals),
            valid_MAE=a[:, 0].mean(), valid_MAE_std=(a[:, 0].std(ddof=1) if len(vals) > 1 else np.nan),
            valid_ACC=a[:, 1].mean(), valid_ACC_std=(a[:, 1].std(ddof=1) if len(vals) > 1 else np.nan),
            valid_MacroF1=a[:, 2].mean(), test_MAE=a[:, 3].mean(),
            test_MAE_std=(a[:, 3].std(ddof=1) if len(vals) > 1 else np.nan),
            test_ACC=a[:, 4].mean(), test_ACC_std=(a[:, 4].std(ddof=1) if len(vals) > 1 else np.nan),
            test_MacroF1=a[:, 5].mean()))
    return pd.DataFrame(rows)


def collect_missing(runs_dir):
    rows = []
    for d, label in EXPS:
        # 优先用 3 种子集成扫描，其次单种子
        path = None
        for cand in ("sweep_valid_ens.csv", "sweep_valid_s42.csv"):
            p = os.path.join(runs_dir, d, cand)
            if os.path.isfile(p):
                path = p
                break
        if path is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "aware_mae" not in df.columns:
            continue
        base = df.loc[df["missing_type"] == "none", "aware_mae"]
        base = float(base.mean()) if len(base) else float("nan")
        r = dict(实验=label, 来源=os.path.basename(path), 干净基线=base)
        deltas = []
        for mt, pos, rho in SCEN:
            s = df[(df["missing_type"] == mt) & (df["position"] == pos)
                   & (np.isclose(df["rho"], rho))]["aware_mae"]
            val = float(s.mean()) if len(s) else float("nan")
            r["%s/%s/ρ=%.1f" % (mt, pos, rho)] = val
            if np.isfinite(val) and np.isfinite(base):
                deltas.append(val - base)
        r["最大退化ΔMAE"] = float(max(deltas)) if deltas else float("nan")
        r["相对恶化%"] = (100.0 * r["最大退化ΔMAE"] / base) if np.isfinite(base) and base else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=os.path.join("..", "runs"))
    ap.add_argument("--out_dir", default=os.path.join("..", "paper_tables"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    clean = collect_clean(a.runs_dir)
    miss = collect_missing(a.runs_dir)

    md = ["# 消融实验汇总（论文第 4 节用）\n",
          "> 极性口径：分类头 argmax（主口径）；强度口径：MAE。完整模型 = q2v3 配置。\n",
          "> 说明：A0 为“常规固定权重融合模型”（朴素拼接+MLP、干净数据训练、无缺失指示）；",
          "A1 = A0 + 掩码指示 + 缺失注入课程。\n",
          "\n## 1. 干净性能（多种子 mean±std）\n"]
    md.append(md_table(clean))
    if not clean.empty:
        base = clean.iloc[0]
        d = clean.copy()
        for c, bc in (("valid_MAE", "valid_MAE"), ("valid_ACC", "valid_ACC"),
                      ("valid_MacroF1", "valid_MacroF1"), ("test_MAE", "test_MAE"),
                      ("test_ACC", "test_ACC"), ("test_MacroF1", "test_MacroF1")):
            d["Δ" + c] = d[c] - base[bc]
        keep = ["实验", "种子数"] + [c for c in d.columns if c.startswith("Δ")]
        md.append("\n> 相对完整模型的差值（ΔMAE>0 或 ΔACC/ΔF1<0 表示该消融**变差**，"
                  "即被去掉的模块**有正贡献**）：\n")
        md.append(md_table(d[keep]))
    clean.to_csv(os.path.join(a.out_dir, "ablation_clean.csv"), index=False, encoding="utf-8-sig")

    md.append("\n## 2. 缺失场景下的性能（valid，ρ=0.3）\n")
    md.append("> 用于回答“常规固定权重融合模型是否急剧下降”：比较各模型的干净基线与缺失场景 MAE。\n")
    md.append(md_table(miss))
    miss.to_csv(os.path.join(a.out_dir, "ablation_missing.csv"), index=False, encoding="utf-8-sig")

    p = os.path.join(a.out_dir, "ablation.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("[SAVE] %s" % p)
    print("[SAVE] %s / %s" % (os.path.join(a.out_dir, "ablation_clean.csv"),
                              os.path.join(a.out_dir, "ablation_missing.csv")))
    print("\n".join(md))


if __name__ == "__main__":
    main()
