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

# 合成数据/调试/早期试跑目录：论文表格必须剔除（赛题要求结果统计只能用题目数据）
DEFAULT_EXCLUDE = ("smoke", "learn", "timing", "cmp_kd", "cmp_nokd",
                   "v2check", "v3check", "baseline", "data_dummy")


def md_table(df, float_fmt="%.4f"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda x: ("—" if pd.isna(x) else float_fmt % x))
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values]
    return "\n".join([head, sep] + rows)


def collect_metrics(runs_dir, exclude=DEFAULT_EXCLUDE):
    rows = []
    skipped = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics_*.json"))):
        tag = os.path.basename(os.path.dirname(p))
        if tag.startswith("_"):        # 以 _ 开头的目录一律视为临时/冒烟实验
            skipped.append(tag)
            continue
        if exclude and tag in exclude:
            skipped.append(tag)
            continue
        if "_legacy" in os.path.basename(p):
            continue          # 旧口径备份（report_p0.py 重算时留下），不参与汇总
        m = re.search(r"metrics_s(\d+)\.json", os.path.basename(p))
        seed = int(m.group(1)) if m else -1
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        r = dict(exp=tag, seed=seed, theta=d.get("theta"),
                 primary=(d.get("primary") or "theta"))
        for split in ("valid_clean", "test_clean"):
            s = d.get(split) or {}
            pre = "valid" if split.startswith("valid") else "test"
            for k in ("mae", "pearson", "ccc"):
                if k in s:
                    r["%s_%s" % (pre, k)] = s[k]
            # θ 阈值口径（旧 schema 把 θ 口径写在 acc/f1 里；新 schema 另存 acc_theta/f1_theta）
            r["%s_acc_theta" % pre] = s.get("acc_theta", s.get("acc"))
            r["%s_f1_theta" % pre] = s.get("f1_theta", s.get("f1"))
            # 分类头口径（主口径）
            r["%s_acc_head" % pre] = s.get("acc_head")
            r["%s_f1_head" % pre] = s.get("f1_head")
        rows.append(r)
    return pd.DataFrame(rows), skipped


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
    ap.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                    help="剔除的实验目录名（合成数据/调试/早期试跑），默认已内置")
    ap.add_argument("--keep_debug", action="store_true",
                    help="保留调试实验（仅自查用，论文表格不要开）")
    ap.add_argument("--primary_run", default="q2v3",
                    help="主模型所在实验目录（其 P0 报告进入第 6 节），默认 q2v3")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    exclude = tuple() if a.keep_debug else tuple(a.exclude)
    md = ["# 论文表格汇总（自动生成）\n"]
    md.append("> **口径说明**：赛题规定模型结构、超参数与决策阈值均在**验证集**上选择，"
              "且结果统计只能使用赛题数据；故本表以 **valid 列为主报口径**，test 列仅作泛化核验。\n")
    md.append("> 已剔除合成数据/调试/早期试跑目录：%s\n"
              % (", ".join(sorted(set(exclude))) if exclude else "（未剔除）"))

    dfs, skipped = collect_metrics(a.runs_dir, exclude=exclude)
    if skipped:
        md.append("> 本次实际剔除：%s\n" % ", ".join(sorted(set(skipped))))

    # 消融实验单独汇总（第 4 节），不混入主结果表
    if dfs.empty:
        dfs_main, dfs_abl = dfs, dfs
    else:
        is_abl = dfs["exp"].astype(str).str.startswith(("ablate", "cmp"))
        dfs_main, dfs_abl = dfs[~is_abl], dfs[is_abl]

    md.append("\n## 1. 主结果（每实验 × 每种子；极性**主口径 = 分类头**，θ 口径为对照）\n")
    if dfs_main.empty:
        md.append("> 暂无 metrics_*.json\n")
    else:
        cols = ["exp", "seed", "theta",
                "valid_mae", "valid_pearson", "valid_ccc",
                "valid_acc_head", "valid_f1_head", "valid_acc_theta", "valid_f1_theta",
                "test_mae", "test_pearson", "test_ccc",
                "test_acc_head", "test_f1_head", "test_acc_theta", "test_f1_theta"]
        cols = [c for c in cols if c in dfs_main.columns]
        md.append(md_table(dfs_main[cols].sort_values(["exp", "seed"])))
        md.append("\n## 2. 主结果（多随机种子 mean ± std）\n")
        agg = agg_by_exp(dfs_main)
        md.append(md_table(agg))
        dfs_main.to_csv(os.path.join(a.out_dir, "main_results_per_seed.csv"),
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
    if not dfs_abl.empty:
        md.append(md_table(agg_by_exp(dfs_abl)))
        dfs_abl.sort_values(["exp", "seed"]).to_csv(
            os.path.join(a.out_dir, "ablation_results.csv"), index=False, encoding="utf-8-sig")
        # 与完整模型的差值（按主口径分类头、以及强度 MAE）
        try:
            full = agg_by_exp(dfs_main[dfs_main["exp"] == a.primary_run])
            if not full.empty:
                f = full.iloc[0]
                rows = []
                for _, r in agg_by_exp(dfs_abl).iterrows():
                    rows.append(dict(
                        消融=r["exp"],
                        d_valid_MAE=float(r.get("valid_mae_mean", np.nan)) - float(f.get("valid_mae_mean", np.nan)),
                        d_test_MAE=float(r.get("test_mae_mean", np.nan)) - float(f.get("test_mae_mean", np.nan)),
                        d_valid_ACC=float(r.get("valid_acc_head_mean", np.nan)) - float(f.get("valid_acc_head_mean", np.nan)),
                        d_test_ACC=float(r.get("test_acc_head_mean", np.nan)) - float(f.get("test_acc_head_mean", np.nan)),
                        d_valid_MacroF1=float(r.get("valid_f1_head_mean", np.nan)) - float(f.get("valid_f1_head_mean", np.nan)),
                        d_test_MacroF1=float(r.get("test_f1_head_mean", np.nan)) - float(f.get("test_f1_head_mean", np.nan)),
                    ))
                md.append("\n> 相对完整模型（q2v3 多种子均值）的差值：ΔMAE>0 表示变差，ΔACC/ΔMacroF1<0 表示变差。\n")
                md.append(md_table(pd.DataFrame(rows)))
        except Exception as e:
            md.append("> （差值表生成失败：%s）\n" % e)
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

    md.append("\n## 6. P0 双口径评估（valid 主报 / test 附加核验，主模型 `%s`）\n"
              % a.primary_run)
    p0 = os.path.join(a.runs_dir, a.primary_run, "report_p0.json")
    if os.path.isfile(p0):
        d = json.load(open(p0, encoding="utf-8"))
        base = d.get("base", {})
        primary = (d.get("meta") or {}).get("primary", "head")
        md.append("> 极性主口径：**%s**\n" % ("分类头 argmax" if primary == "head" else "θ 回归阈值"))
        rows = []
        for k, label in (("valid", "valid（主报）"), ("test", "test（附加核验）")):
            m = base.get(k) or {}
            r = dict(划分=label)
            if primary == "head":
                r.update({"ACC": m.get("acc_head"), "MacroF1": m.get("f1_head")})
                r.update({"ACC_θ对照": m.get("acc"), "MacroF1_θ对照": m.get("f1")})
            else:
                r.update({"ACC": m.get("acc"), "MacroF1": m.get("f1")})
                r.update({"ACC_分类头对照": m.get("acc_head"),
                          "MacroF1_分类头对照": m.get("f1_head")})
            r.update({"MAE": m.get("mae"), "Pearson": m.get("pearson"), "CCC": m.get("ccc")})
            rows.append(r)
        md.append(md_table(pd.DataFrame(rows)))
        cb = d.get("combined") or {}
        if cb:
            md.append("\n> 融合+校准（系数在 valid 上拟合）后：valid MAE %s → %s；"
                      "test MAE %s → %s；θ=%.2f\n"
                      % ("%.4f" % (base.get("valid", {}).get("mae", float("nan"))),
                         "%.4f" % (cb.get("valid", {}).get("mae", float("nan"))),
                         "%.4f" % (base.get("test", {}).get("mae", float("nan"))),
                         "%.4f" % (cb.get("test", {}).get("mae", float("nan"))),
                         cb.get("theta", float("nan"))))
        md.append("\n> 完整报告（阈值敏感性、McNemar、bootstrap CI、标签强度分布校正）见 "
                  "`runs/q2/report_p0.md`。\n")
    else:
        md.append("> 未找到 `runs/q2/report_p0.json`（先运行 `report_p0.py`）。\n")

    p = os.path.join(a.out_dir, "tables.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("[SAVE] %s" % p)
    print("[SAVE] CSV 目录 %s" % a.out_dir)
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
