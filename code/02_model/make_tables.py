# -*- coding: utf-8 -*-
r"""
论文表格汇总导出（把散落的 metrics/sweep/消融结果整理成可直接粘贴的 Markdown + CSV）。

用法：
  python 02_model/make_tables.py                       # 扫描 runs\ 与 figs\
  python 02_model/make_tables.py --runs_dir ..\runs --out_dir ..\paper_tables
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def md_table(df, float_fmt="%.4f"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda x: ("—" if pd.isna(x) else float_fmt % x))
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values]
    return "\n".join([head, sep] + rows)


def collect_metrics(runs_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics_*.json"))):
        tag = os.path.basename(os.path.dirname(p))
        m = re.search(r"metrics_s(\d+)\.json", os.path.basename(p))
        seed = int(m.group(1)) if m else -1
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        r = dict(exp=tag, seed=seed, theta=d.get("theta"))
        for split in ("valid_clean", "test_clean"):
            s = d.get(split) or {}
            pre = "valid" if split.startswith("valid") else "test"
            for k in ("mae", "pearson", "ccc", "acc", "f1", "acc_head", "f1_head"):
                if k in s:
                    r["%s_%s" % (pre, k)] = s[k]
        rows.append(r)
    return pd.DataFrame(rows)


def agg_by_exp(df):
    if df.empty:
        return df
    num = [c for c in df.columns if c.startswith(("valid_", "test_"))]
    g = df.groupby("exp")[num].agg(["mean", "std"])
    g.columns = ["%s_%s" % (a, b) for a, b in g.columns]
    return g.reset_index()


def collect_sweep(runs_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "sweep_valid_*.csv"))):
        tag = os.path.basename(os.path.dirname(p))
        df = pd.read_csv(p)
        if df.empty:
            continue
        out.setdefault(tag, {})["valid"] = df
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "sweep_test_*.csv"))):
        tag = os.path.basename(os.path.dirname(p))
        out.setdefault(tag, {})["test"] = pd.read_csv(p)
    return out


def sweep_pivot(df, value="aware_mae"):
    try:
        return df.pivot_table(index="missing_type", columns="rho", values=value, aggfunc="mean")
    except Exception:
        return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "paper_tables"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    md = ["# 论文表格汇总（自动生成）\n"]
    dfs = collect_metrics(a.runs_dir)

    md.append("## 1. 主结果（每实验 × 每种子）\n")
    if dfs.empty:
        md.append("> 暂无 metrics_*.json\n")
    else:
        md.append(md_table(dfs.sort_values(["exp", "seed"])))
        md.append("\n## 2. 主结果（多随机种子 mean ± std）\n")
        agg = agg_by_exp(dfs)
        md.append(md_table(agg))
        dfs.to_csv(os.path.join(a.out_dir, "main_results_perssed.csv"),
                   index=False, encoding="utf-8-sig")
        agg.to_csv(os.path.join(a.out_dir, "main_results_summary.csv"),
                   index=False, encoding="utf-8-sig")

    sw = collect_sweep(a.runs_dir)
    md.append("\n## 3. 缺失因素影响（缺失率 × 缺失类型，单元格=MAE）\n")
    for tag, parts in sw.items():
        for split, df in parts.items():
            md.append("\n**%s / %s 集**\n" % (tag, split))
            for val, name in (("aware_mae", "本文模型 MAE"), ("unaware_mae", "未做缺失处理 MAE"),
                              ("aware_f1", "本文模型 MacroF1")):
                if val in df.columns:
                    pv = sweep_pivot(df, val)
                    md.append("\n*%s*\n" % name)
                    md.append(md_table(pv.reset_index()))
                    pv.to_csv(os.path.join(a.out_dir, "sweep_%s_%s_%s.csv" % (tag, split, val)),
                              encoding="utf-8-sig")
            if {"aware_mae", "unaware_mae"}.issubset(df.columns):
                g = df.groupby("missing_type")[["aware_mae", "unaware_mae"]].mean().reset_index()
                g["MAE 提升"] = g["unaware_mae"] - g["aware_mae"]
                g["相对提升%"] = 100 * g["MAE 提升"] / g["unaware_mae"]
                md.append("\n*掩码感知带来的收益（按缺失类型平均）*\n")
                md.append(md_table(g))

    md.append("\n## 4. 消融实验\n")
    abl = [r for r in dfs["exp"].unique()] if not dfs.empty else []
    abl_names = [x for x in abl if x.startswith(("ablate", "cmp"))]
    if abl_names:
        md.append(md_table(agg_by_exp(dfs[dfs["exp"].isin(abl_names)])))
    else:
        md.append("> 暂无消融实验目录（`runs/ablate_*`）。\n")

    rep = os.path.join(a.runs_dir, "q2", "att3_missing_report.json")
    md.append("\n## 5. 附件3 缺失分布（数据说明）\n")
    if os.path.isfile(rep):
        d = json.load(open(rep, encoding="utf-8"))
        rows = []
        for split, info in d.get("splits", {}).items():
            for m in ("text", "audio", "vision"):
                if m in info:
                    r = dict(数据集=split, 模态=m)
                    r.update({k: info[m].get(k) for k in
                              ("n_with_missing", "frac_with_missing", "mean_missing_ratio",
                               "p50", "p90", "max")})
                    it = info.get("interval") or {}
                    r.update({k: it.get(k) for k in ("len_mean", "head", "middle", "tail")})
                    rows.append(r)
        if rows:
            t = pd.DataFrame(rows)
            md.append(md_table(t))
            t.to_csv(os.path.join(a.out_dir, "att3_missing_distribution.csv"),
                     index=False, encoding="utf-8-sig")
    else:
        md.append("> 暂无 att3_missing_report.json\n")

    p = os.path.join(a.out_dir, "tables.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("[SAVE] %s" % p)
    print("[SAVE] CSV 目录 %s" % a.out_dir)
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
