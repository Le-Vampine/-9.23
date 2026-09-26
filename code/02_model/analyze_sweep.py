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


def anova(df, metric, out_csv=None, baseline=None):
    """三因素（缺失类型/位置/缺失率）方差分析。

    默认对**相对无缺失基线的退化量 ΔMAE** 做分析：直接用 MAE 水平值时，
    截距会占掉 ~99% 的平方和，各因素的效应量会被稀释到不可读。

    若样本量不足以估计三阶交互（每格观测数 < 2），自动降阶为
    二阶交互模型或主效应模型，并在结果中标注实际使用的模型。
    """
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
    except Exception:
        print("[WARN] 未安装 statsmodels，跳过量分析。pip install statsmodels")
        return None

    KEY = ["missing_type", "position", "rho"]

    # ---- 伪重复检测 -------------------------------------------------------
    # 若同一格内各 rep 的指标**完全一致**，说明“重复”没有引入任何随机性
    # （历史缺陷：sweep_eval 的 scenario_seed 未传给 evaluate，所有 rep 都用
    #  seed=1234 → 掩码相同 → 指标逐位重复）。
    # 此时把 rep 当重复会把残差方差压到 ~0，使 F 值虚高（曾达 1e25）、p 恒为 0。
    # 处置：折叠为格均值（每格 1 个观测），让**最高阶交互**充当误差项。
    pseudo = False
    sub = df.copy()
    if "rep" in df.columns and df["rep"].nunique() > 1:
        nun = df.groupby(KEY)[metric].nunique()
        if int(nun.max()) == 1:
            pseudo = True
            print("\n[WARN] 伪重复：每格 %d 个 rep 的 %s 完全相同"
                  "（scenario_seed 未生效？）" % (df["rep"].nunique(), metric))
            print("       已折叠为格均值（每格 1 个观测），以最高阶交互作误差项。")
            num_cols = [c for c in df.columns
                        if c not in KEY + ["rep"] and pd.api.types.is_numeric_dtype(df[c])]
            sub = df.groupby(KEY, as_index=False)[num_cols].mean()

    base = float(baseline) if baseline is not None else float(sub[metric].astype(float).mean())
    sub["y"] = sub[metric].astype(float) - base
    label = ("Δ%s（基线=%.4f）" % (metric, base)) if baseline is not None else metric
    n_cells = sub.groupby(["missing_type", "position", "rho"]).size()
    per_cell = int(n_cells.min())
    candidates = ["y ~ C(missing_type) * C(position) * C(rho)",
                  "y ~ (C(missing_type) + C(position) + C(rho)) ** 2",
                  "y ~ C(missing_type) + C(position) + C(rho)"]
    if per_cell >= 2:
        order = candidates
    elif per_cell == 1:
        order = candidates[1:]
    else:
        order = candidates[2:]

    tab, used = None, None
    for formula in order:
        try:
            model = smf.ols(formula, data=sub).fit()
            t = anova_lm(model, typ=3)
            ss_res = float(t.loc["Residual", "sum_sq"])
            denom = t["sum_sq"] + ss_res
            # 残差平方和≈0 时 partial_eta2 会被算成 1.0（数学上无意义）→ 置 NaN
            t["partial_eta2"] = np.where(denom > 0, t["sum_sq"] / denom, np.nan)
            tab, used = t, formula
            break
        except Exception as e:
            print("[WARN] 模型 %s 拟合失败：%s" % (formula, e))
    if tab is None:
        print("[WARN] ANOVA 全部失败，跳过 %s" % metric)
        return None

    print("\n===== ANOVA (%s) ===== 每格观测数=%d，采用模型: %s" % (label, per_cell, used))
    ss_res_final = float(tab.loc["Residual", "sum_sq"]) if "Residual" in tab.index else float("nan")
    if not np.isfinite(ss_res_final) or ss_res_final <= 1e-12:
        print("  [WARN] 残差平方和≈0 → F 值与 p 值不可用（请检查是否存在伪重复或设计饱和）；"
              "效应量以 eta2 与 ΔMAE 实际差值为准。")
    total_ss = float(tab["sum_sq"].sum())
    tab["eta2"] = tab["sum_sq"] / total_ss if total_ss > 0 else np.nan
    show = tab.copy()
    for c in ("sum_sq", "F", "PR(>F)", "partial_eta2", "eta2"):
        if c in show.columns:
            show[c] = show[c].map(lambda x: ("%.4g" % x) if np.isfinite(x) else "NaN")
    print(show.to_string())
    try:
        top = tab.drop(index="Residual", errors="ignore").sort_values("eta2", ascending=False)
        print("  效应量排序（eta2 = 因素平方和/总平方和）：")
        for name, row in top.iterrows():
            print("    %-38s eta2=%.3f  partial_eta2=%.3f  p=%.3g"
                  % (name, row["eta2"], row["partial_eta2"], row["PR(>F)"]))
    except Exception:
        pass
    if pseudo:
        note = ("  [NOTE] 检测到伪重复，已折叠为格均值；误差项 = 最高阶交互"
                "（%s 的残差项），F/p 在无真实重复时仅作参考。" % used)
    else:
        note = ("  [NOTE] 响应变量为相对无缺失基线的 ΔMAE/ΔF1（避免截距吞掉效应量）。\n"
                "         head/middle/tail 的掩码是确定的，仅 position=random 的重复"
                "产生真实随机性；\n"
                "         故以 效应量 eta2 与 实际 ΔMAE 差值 为主要证据，p 值仅作参考。")
    print(note)
    if out_csv:
        tab.to_csv(out_csv, encoding="utf-8-sig")
        print("[SAVE] %s" % out_csv)
    return tab


def fit_curve(df, metric, out_dir, deg=3):
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
        g = g[np.isfinite(g[metric])]
        if len(g) < 3:                      # 点太少无法拟合（如无缺失基线行）
            continue
        d = int(min(deg, len(g) - 1))
        coef = np.polyfit(g["rho"], g[metric], d)
        xs = np.linspace(g["rho"].min(), g["rho"].max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "-", lw=1.2)
        ax.plot(g["rho"], g[metric], "o", ms=4)
        txt = " + ".join("%.3fx^%d" % (c, d - i) for i, c in enumerate(coef))
        ax.plot([], [], " ", label="%s: %s" % (mt, txt if d == 3 else "deg=%d" % d))
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
    ap.add_argument("--out_dir", default=os.path.join("..", "figs"))
    ap.add_argument("--metrics", nargs="*", default=["aware_mae", "aware_f1"])
    ap.add_argument("--max_rho", type=float, default=0.5,
                    help="参与 ANOVA/曲线的最大缺失率（默认 0.5，排除 rho=1 的退化极端场景）")
    ap.add_argument("--keep_full_missing", action="store_true",
                    help="保留 rho=1（整模态缺失）行")
    a = ap.parse_args()

    df_all = pd.read_csv(a.csv)
    os.makedirs(a.out_dir, exist_ok=True)
    df = df_all[df_all["rho"] > 0].copy()             # 去掉"无缺失"基线行
    if not a.keep_full_missing:
        df = df[df["rho"] <= a.max_rho + 1e-9]
    print("[DATA] 原始 %d 行 → 参与分析 %d 行（rho∈(0, %.2f]，已排除 none 基线行）"
          % (len(df_all), len(df), a.max_rho if not a.keep_full_missing else 1.0))

    # 无缺失基线（用于把响应中心化为 ΔMAE / ΔF1，避免截距吞掉效应量）
    z = df_all[np.isclose(df_all["rho"], 0.0)]
    baselines = {c: float(z[c].mean()) for c in df_all.columns
                 if c in a.metrics and not z.empty}
    df.to_csv(os.path.join(a.out_dir, "sweep_analysis_input.csv"),
              index=False, encoding="utf-8-sig")
    for metric in a.metrics:
        if metric not in df.columns:
            print("[SKIP] 缺少列 %s" % metric)
            continue
        fit_curve(df, metric, a.out_dir)
        anova(df, metric, out_csv=os.path.join(a.out_dir, "anova_%s.csv" % metric),
              baseline=baselines.get(metric))

    # ---- 有无处理的对照（A0 基线：模型看不到掩码） ----
    if "unaware_mae" in df.columns:
        cmp = df.groupby(["missing_type", "rho"])[["aware_mae", "unaware_mae"]].mean().reset_index()
        cmp["gain"] = cmp["unaware_mae"] - cmp["aware_mae"]
        p = os.path.join(a.out_dir, "aware_vs_unaware.csv")
        cmp.to_csv(p, index=False, encoding="utf-8-sig")
        print("[SAVE] %s（gain>0 表示掩码感知带来的 MAE 收益）" % p)


if __name__ == "__main__":
    main()
