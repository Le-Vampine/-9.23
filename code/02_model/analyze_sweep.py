# -*- coding: utf-8 -*-
"""
缺失因素规律分析（第 4 节）：
  1) 三因素方差分析（Type-III ANOVA + 偏 eta^2 效应量）
  2) 性能退化曲线拟合  dMAE(rho) = a*rho + b*rho^2 + c*rho^3
  3) 生成论文用图：退化曲线 / 热力图 / 交互效应图

用法：
  python analyze_sweep.py --csv E:\\数学建模\\runs\\q2\\sweep_valid_s42.csv --out_dir E:\\数学建模\\figs
"""
import argparse
import os

import numpy as np
import pandas as pd


def anova(df, metric):
    """三因素（缺失类型/位置/缺失率）方差分析。"""
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
    except Exception:
        print("[WARN] 未安装 statsmodels，跳过量分析。pip install statsmodels")
        return None
    sub = df.rename(columns={metric: "y"})
    model = smf.ols("y ~ C(missing_type) * C(position) * C(rho)", data=sub).fit()
    tab = anova_lm(model, typ=3)
    ss_res = tab.loc["Residual", "sum_sq"]
    tab["partial_eta2"] = tab["sum_sq"] / (tab["sum_sq"] + ss_res)
    print("\n===== ANOVA (%s) =====" % metric)
    print(tab.to_string(float_format=lambda x: "%.4f" % x))
    return tab


def fit_curve(df, metric, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(out_dir, exist_ok=True)

    # ---- 退化曲线（按缺失类型） ----
    fig, ax = plt.subplots(figsize=(6, 4))
    for mt, g in df.groupby("missing_type"):
        g = g.groupby("rho")[metric].mean().reset_index()
        coef = np.polyfit(g["rho"], g[metric], 3)
        xs = np.linspace(g["rho"].min(), g["rho"].max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "-", lw=1.2)
        ax.plot(g["rho"], g[metric], "o", ms=4, label="%s (y=%.3fx^3+%.3fx^2+%.3fx+%.3f)"
                % (mt, coef[0], coef[1], coef[2], coef[3]))
    ax.set_xlabel("缺失率 rho"); ax.set_ylabel(metric)
    ax.set_title("性能退化曲线"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "degrade_%s.png" % metric), dpi=200)
    plt.close(fig)

    # ---- 热力图（类型 x 位置） ----
    piv = df.pivot_table(index="missing_type", columns="position", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, "%.3f" % piv.values[i, j], ha="center", va="center", fontsize=7)
    ax.set_title("%s 热力图（类型 x 位置）" % metric)
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "heatmap_%s.png" % metric), dpi=200)
    plt.close(fig)
    print("[FIG] %s/degrade_%s.png, heatmap_%s.png" % (out_dir, metric, metric))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default=r"E:\数学建模\figs")
    ap.add_argument("--metrics", nargs="*", default=["aware_mae", "aware_f1"])
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    print("[DATA] %d 行, 列=%s" % (len(df), list(df.columns)))
    os.makedirs(a.out_dir, exist_ok=True)
    for metric in a.metrics:
        if metric not in df.columns:
            print("[SKIP] 缺少列 %s" % metric)
            continue
        fit_curve(df, metric, a.out_dir)
        anova(df, metric)

    # ---- 有无处理的对照（A0 基线：模型看不到掩码） ----
    if "unaware_mae" in df.columns:
        cmp = df.groupby(["missing_type", "rho"])[["aware_mae", "unaware_mae"]].mean().reset_index()
        cmp["gain"] = cmp["unaware_mae"] - cmp["aware_mae"]
        p = os.path.join(a.out_dir, "aware_vs_unaware.csv")
        cmp.to_csv(p, index=False, encoding="utf-8-sig")
        print("[SAVE] %s（gain>0 表示掩码感知带来的 MAE 收益）" % p)


if __name__ == "__main__":
    main()
