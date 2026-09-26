# -*- coding: utf-8 -*-
r"""把「缺失因素影响规律」与「错误归因」汇总成论文表格（问题2 第(1)~(4)项用）。

输入
----
  runs/<tag>/sweep_valid_<name>.csv    缺失场景扫描（sweep_degradation.py 产出）
  runs/<tag>/sweep_clean_<name>.json   无缺失基线
  figs/sweep_<alias>/anova_aware_*.csv 已修正的 ANOVA（效应量）
  runs/ens_top2/error_groups_*.csv     错误归因分层

输出（paper_tables/）
----
  missing_law_by_type_rho.csv          ΔMAE/ΔACC/ΔF1：缺失类型 × 缺失率（位置取均值）
  missing_law_by_type_position.csv     ΔMAE/ΔACC/ΔF1：缺失类型 × 缺失位置（率取均值）
  missing_law_anova_summary.csv        4 个模型的效应量（eta2 / partial_eta2 / F / p）
  error_attribution_groups.csv         valid+test 分层归因合并
  missing_law_summary.md               结论摘要（可直接进论文表注/小节）

用法
----
  python missing_law_tables.py --root ..\..
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

# 模型清单：tag / 扫描 CSV / ANOVA 目录 / 论文用名 / 是否主模型
MODELS = [
    ("ablate_a0_plain", "runs/ablate_a0_plain/sweep_valid_a0_plain.csv",
     "figs/sweep_a0", "A0 朴素融合（无缺失处理）", False),
    ("ablate_a1_plain_ind", "runs/ablate_a1_plain_ind/sweep_valid_a1_plain_ind.csv",
     "figs/sweep_a1", "A1 朴素融合 + 掩码指示 + 缺失注入", False),
    ("ours_s42", "runs/ours_s42/sweep_valid_ours_s42.csv",
     "figs/sweep_ours_s42", "MRF-Net（单种子 42）", False),
    ("ens_top2", "runs/ens_top2/sweep_valid_ours_B.csv",
     "figs/sweep_B", "MRF-Net（B：42+43 集成，提交模型）", True),
]
CLEAN_JSON = {
    "ablate_a0_plain": "runs/ablate_a0_plain/sweep_clean_a0_plain.json",
    "ablate_a1_plain_ind": "runs/ablate_a1_plain_ind/sweep_clean_a1_plain_ind.json",
    "ours_s42": "runs/ours_s42/sweep_clean_ours_s42.json",
    "ens_top2": "runs/ens_top2/sweep_clean_ours_B.json",
}


def _md(df, floatfmt=".4f"):
    """极简 markdown 表格渲染（避免为复现环境引入 tabulate 依赖）。"""
    df = df.reset_index() if df.index.name else df
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(("%" + floatfmt) % v if np.isfinite(v) else "")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def load_model(root, csv_rel, clean_rel):
    df = pd.read_csv(os.path.join(root, csv_rel))
    with open(os.path.join(root, clean_rel), encoding="utf-8") as f:
        clean = json.load(f)["clean"]
    base = dict(mae=clean["mae"], acc=clean["acc"], f1=clean["macro_f1"])
    # 退化量（相对无缺失基线）：正=变差
    df["dMAE"] = df["aware_mae"] - base["mae"]
    df["dACC"] = df["aware_acc"] - base["acc"]
    df["dF1"] = df["aware_f1"] - base["f1"]
    # 掩码感知收益（unaware - aware，正=aware 更好）
    df["gainMAE"] = df["unaware_mae"] - df["aware_mae"]
    return df, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("..", ".."))
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    out = a.out_dir or os.path.join(root, "paper_tables")
    os.makedirs(out, exist_ok=True)

    rows_all, anova_rows, clean_rows = [], [], []
    data = {}
    for tag, csv_rel, anova_dir, nice, _is_main in MODELS:
        df, base = load_model(root, csv_rel, CLEAN_JSON[tag])
        data[tag] = df
        clean_rows.append(dict(model=tag, name=nice, clean_mae=base["mae"],
                               clean_acc=base["acc"], clean_f1=base["f1"]))
        d = df.copy()
        d.insert(0, "model", tag)
        d.insert(1, "name", nice)
        rows_all.append(d)
        # ANOVA 效应量
        metric_files = (("dMAE", "anova_aware_mae.csv"), ("dF1", "anova_aware_f1.csv"))
        for metric, fname in metric_files:
            p = os.path.join(root, anova_dir, fname)
            if not os.path.isfile(p):
                print("[WARN] 缺少 %s" % p)
                continue
            t = pd.read_csv(p, index_col=0)
            for term, r in t.iterrows():
                if term == "Residual":
                    continue
                anova_rows.append(dict(model=tag, name=nice, metric=metric, term=term,
                                       df=r.get("df"), F=r.get("F"), p=r.get("PR(>F)"),
                                       eta2=r.get("eta2"), partial_eta2=r.get("partial_eta2")))

    big = pd.concat(rows_all, ignore_index=True)

    # ---- 表1：类型 × 缺失率（位置取均值）----
    agg = ["dMAE", "dACC", "dF1", "gainMAE", "aware_mae", "unaware_mae"]
    by_tr = (big.groupby(["model", "name", "missing_type", "rho"])[agg]
                .mean().reset_index())
    by_tr.to_csv(os.path.join(out, "missing_law_by_type_rho.csv"),
                 index=False, encoding="utf-8-sig")

    # ---- 表2：类型 × 缺失位置（率取均值）----
    by_tp = (big.groupby(["model", "name", "missing_type", "position"])[agg]
                .mean().reset_index())
    by_tp.to_csv(os.path.join(out, "missing_law_by_type_position.csv"),
                 index=False, encoding="utf-8-sig")

    # ---- 表2b：位置 × 模型（排除 ρ=1）----
    # 整模态缺失时“位置”无意义（整条全零），若混入会稀释位置效应。
    sub = big[big["rho"] < 1.0]
    by_pos = (sub.groupby(["model", "name", "position"])[agg]
                 .mean().reset_index())
    by_pos.to_csv(os.path.join(out, "missing_law_by_position_excl_full.csv"),
                  index=False, encoding="utf-8-sig")

    # ---- 表3：ANOVA 汇总 ----
    pd.DataFrame(anova_rows).to_csv(os.path.join(out, "missing_law_anova_summary.csv"),
                                    index=False, encoding="utf-8-sig")
    pd.DataFrame(clean_rows).to_csv(os.path.join(out, "missing_law_clean_baseline.csv"),
                                    index=False, encoding="utf-8-sig")

    # ---- 表4：错误归因分层（valid+test 合并）----
    eg = []
    for s in ("valid", "test"):
        p = os.path.join(root, "runs", "ens_top2", "error_groups_%s.csv" % s)
        if os.path.isfile(p):
            eg.append(pd.read_csv(p))
    if eg:
        pd.concat(eg, ignore_index=True).to_csv(
            os.path.join(out, "error_attribution_groups.csv"),
            index=False, encoding="utf-8-sig")

    # ---- 摘要 ----
    main_df = data["ens_top2"]
    L = []
    L.append("# 缺失因素影响规律与错误归因（自动汇总）\n")
    L.append("> 生成自 `code/02_model/missing_law_tables.py`；口径：valid 集，"
             "缺失注入为场景模拟（注入缺失率），与附件3 的**实测缺失率**语义不同。\n")

    L.append("\n## 1. 无缺失基线（valid，n=728）\n")
    L.append(_md(pd.DataFrame(clean_rows)[["name", "clean_mae", "clean_acc", "clean_f1"]]))

    L.append("\n\n## 2. 缺失类型 × 缺失率 → ΔMAE（位置取均值，提交模型 B）\n")
    piv = main_df.pivot_table(index="missing_type", columns="rho", values="dMAE", aggfunc="mean")
    L.append(_md(piv))

    L.append("\n\n## 3. 缺失类型 × 缺失位置 → ΔMAE（率取均值，提交模型 B）\n")
    piv2 = main_df.pivot_table(index="missing_type", columns="position", values="dMAE", aggfunc="mean")
    L.append(_md(piv2))

    L.append("\n\n## 3b. 缺失位置 → ΔMAE（**排除 ρ=1**，此时位置无意义）\n")
    pv = sub.pivot_table(index="position", columns="name", values="dMAE", aggfunc="mean")
    L.append("**各模型在四个缺失位置上的平均 ΔMAE\n\n")
    L.append(_md(pv))
    pv2 = sub[sub["model"] == "ens_top2"].pivot_table(
        index="missing_type", columns="position", values="dMAE", aggfunc="mean")
    L.append("\n提交模型 B：缺失类型 × 位置 → ΔMAE\n\n")
    L.append(_md(pv2))

    L.append("\n\n## 4. 缺失类型 × 缺失率 → ΔACC（位置取均值，提交模型 B）\n")
    piv3 = main_df.pivot_table(index="missing_type", columns="rho", values="dACC", aggfunc="mean")
    L.append(_md(piv3))

    L.append("\n\n## 5. 三因素 ANOVA 效应量（响应 = 相对基线的退化量，ρ∈(0,0.5]）\n")
    an = pd.DataFrame(anova_rows)
    an_mae = an[an["metric"] == "dMAE"].pivot_table(
        index="term", columns="model", values="eta2", aggfunc="mean")
    L.append("**eta2（因素平方和 / 总平方和），响应 = ΔMAE**\n\n")
    L.append(_md(an_mae))

    L.append("\n\n## 6. 错误归因（提交模型 B）\n")
    e1 = os.path.join(root, "runs", "ens_top2", "error_groups_valid.csv")
    if os.path.isfile(e1):
        g = pd.read_csv(e1)
        cols = [c for c in ["dim", "group", "n", "mae", "acc", "f1", "acc_head",
                            "f1_head", "bias", "miss_text", "miss_audio", "miss_vision"]
                if c in g.columns]
        L.append(_md(g[cols].sort_values("mae", ascending=False)))

    p = os.path.join(out, "missing_law_summary.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("[SAVE] %s" % p)
    for fn in ("missing_law_by_type_rho.csv", "missing_law_by_type_position.csv",
               "missing_law_by_position_excl_full.csv",
               "missing_law_anova_summary.csv", "missing_law_clean_baseline.csv",
               "error_attribution_groups.csv"):
        q = os.path.join(out, fn)
        print("[SAVE] %s  %s" % (q, "OK" if os.path.isfile(q) else "MISSING"))


if __name__ == "__main__":
    main()
