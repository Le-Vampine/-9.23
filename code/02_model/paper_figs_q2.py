# -*- coding: utf-8 -*-
r"""问题2 论文配图生成（补齐缺失的 5 张数据图）。

风格：nature 配色（低饱和典雅）+ 中文宋体 / 西文 Times，去 top/right 边框，数值直接标注。
产物全部写入 05_paper/figures/，命名遵循 `Q2_{内容}_{图表类型}.png`。

用法：python paper_figs_q2.py --root ..\..
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

matplotlib.rcParams["font.family"] = ["Times New Roman", "SimSun",
                                      "Liberation Serif", "Noto Serif CJK SC"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["figure.dpi"] = 200

# nature 配色（全论文统一）
NATURE = ["#3B5F8A", "#A23B3B", "#4A7C59", "#B8B8B8", "#F0E6D3"]
BLUE, RED, GREEN, GREY, CREAM = NATURE
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[BLUE, RED, GREEN, GREY])


def fig_missing_profile(root, out):
    """附件3/4 实测缺失分布（含缺失比例 + 平均缺失率）。"""
    p = os.path.join(root, "paper_tables", "missing_profile.csv")
    if not os.path.isfile(p):
        return None
    d = pd.read_csv(p)
    d["label"] = d["dataset"].str.upper() + "-" + d["modality"].str.capitalize()
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    bars = ax.bar(x, d["frac_with_missing"] * 100, width=0.62,
                  color=[BLUE if v > 0 else GREY for v in d["frac_with_missing"]],
                  edgecolor="black", linewidth=0.5)
    for b, (_, r) in zip(bars, d.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                "%.1f%%" % (r["frac_with_missing"] * 100), ha="center", fontsize=9)
        if r["n_full_missing"] > 0:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
                    "整模态缺失 %d 条" % r["n_full_missing"], ha="center",
                    fontsize=8, color=RED)
    ax2 = ax.twinx()
    ax2.plot(x, d["rho_mean"], "D", ms=5, color=RED, label="平均缺失率 ρ")
    ax2.set_ylabel("平均实测缺失率 ρ", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(RED)
    ax2.set_ylim(0, max(0.15, float(d["rho_mean"].max()) * 2.2))
    ax.set_xticks(x)
    ax.set_xticklabels(d["label"], fontsize=9)
    ax.set_ylabel("含缺失样本比例（%）")
    ax.set_ylim(0, 82)
    ax.grid(axis="y", ls=":", lw=0.5, color="0.8")
    ax.set_axisbelow(True)
    fp = os.path.join(out, "Q2_附件3附件4缺失分布_双轴条形图.png")
    fig.savefig(fp)
    plt.close(fig)
    return fp


def fig_error_groups(root, out):
    """错误归因：5 个分层维度的 MAE（含 n 与 ACC 标注）。"""
    p = os.path.join(root, "runs", "ens_top2", "error_groups_valid.csv")
    if not os.path.isfile(p):
        return None
    d = pd.read_csv(p)
    dims = list(dict.fromkeys(d["dim"]))
    dim_cn = {"strength_bucket": "真值强度｜y｜分层", "text_len_bucket": "文本长度分位",
              "vision_len_bucket": "视觉长度分位", "polarity": "真值极性",
              "any_missing": "是否含缺失"}
    ncol = 3
    nrow = int(np.ceil(len(dims) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.5, 2.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, dim in zip(axes, dims):
        s = d[d["dim"] == dim]
        y = np.arange(len(s))[::-1]
        cols = [RED if v == s["mae"].max() else BLUE for v in s["mae"]]
        ax.barh(y, s["mae"], height=0.6, color=cols, edgecolor="black", linewidth=0.4)
        for yy, (_, r) in zip(y, s.iterrows()):
            ax.text(r["mae"] + 0.02, yy, "%.3f  (n=%d, ACC=%.3f)"
                    % (r["mae"], r["n"], r["acc_head"]), va="center", fontsize=7.5)
        ax.set_yticks(y)
        ax.set_yticklabels([str(g) for g in s["group"]], fontsize=8)
        ax.set_title(dim_cn.get(dim, dim), fontsize=9.5)
        ax.set_xlabel("MAE", fontsize=8.5)
        ax.set_xlim(0, float(s["mae"].max()) * 1.85)
        ax.grid(axis="x", ls=":", lw=0.5, color="0.85")
        ax.set_axisbelow(True)
    for ax in axes[len(dims):]:
        ax.axis("off")
    fig.tight_layout()
    fp = os.path.join(out, "Q2_错误归因五维分层_条形图.png")
    fig.savefig(fp)
    plt.close(fig)
    return fp


def fig_tune(root, out):
    """超参搜索：screen 与 full 两阶段的 valid F1 / MAE。"""
    fs = os.path.join(root, "runs", "tune", "grid_screen.csv")
    ff = os.path.join(root, "runs", "tune", "grid_full.csv")
    if not os.path.isfile(fs):
        return None
    s = pd.read_csv(fs)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
    for ax, col, title, better in ((axes[0], "valid_f1_head",
                                    "验证集 Macro-F1（分类头，越大越好）", "max"),
                                   (axes[1], "valid_mae",
                                    "验证集 MAE（越小越好）", "min")):
        x = np.arange(len(s))
        best = s[col].max() if better == "max" else s[col].min()
        bestf = s[col].min() if better == "max" else s[col].max()
        cols = [GREEN if v == best else (RED if v == bestf else BLUE) for v in s[col]]
        ax.bar(x, s[col], width=0.62, color=cols, edgecolor="black", linewidth=0.4)
        for xi, v in zip(x, s[col]):
            ax.text(xi, v + (s[col].max() - s[col].min()) * 0.03, "%.4f" % v,
                    ha="center", fontsize=7.5, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(s["tag"], rotation=35, ha="right", fontsize=8)
        ax.set_title("%s（screen 阶段，1024 子集 6/8 轮）" % title, fontsize=9)
        span = s[col].max() - s[col].min()
        ax.set_ylim(s[col].min() - span * 0.25, s[col].max() + span * 0.45)
        ax.grid(axis="y", ls=":", lw=0.5, color="0.85")
        ax.set_axisbelow(True)
    if os.path.isfile(ff):
        f = pd.read_csv(ff)
        txt = "  ".join("%s: F1=%.4f MAE=%.4f" % (r["tag"], r["valid_f1_head"],
                                                  r["valid_mae"]) for _, r in f.iterrows())
        fig.text(0.5, -0.06, "full 阶段（全量 12/14 轮）：" + txt,
                 ha="center", fontsize=8, color=RED)
    fig.tight_layout()
    fp = os.path.join(out, "Q2_超参数搜索结果_柱状图.png")
    fig.savefig(fp)
    plt.close(fig)
    return fp


def fig_degrade_models(root, out):
    """四个模型在「文本模态缺失」下的 ΔMAE 随缺失率变化（退化的主导因素）。"""
    sets = [("A0 朴素融合", "runs/ablate_a0_plain/sweep_valid_a0_plain.csv",
             "runs/ablate_a0_plain/sweep_clean_a0_plain.json", RED),
            ("A1 +掩码指示", "runs/ablate_a1_plain_ind/sweep_valid_a1_plain_ind.csv",
             "runs/ablate_a1_plain_ind/sweep_clean_a1_plain_ind.json", GREEN),
            ("MRF-Net 单种子", "runs/ours_s42/sweep_valid_ours_s42.csv",
             "runs/ours_s42/sweep_clean_ours_s42.json", GREY),
            ("MRF-Net 集成（提交）", "runs/ens_top2/sweep_valid_ours_B.csv",
             "runs/ens_top2/sweep_clean_ours_B.json", BLUE)]
    import json
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for name, csv, cj, col in sets:
        p = os.path.join(root, csv)
        if not os.path.isfile(p):
            continue
        d = pd.read_csv(p)
        base = json.load(open(os.path.join(root, cj), encoding="utf-8"))["clean"]
        d = d[d["missing_type"].isin(["text", "text+audio", "text+audio+vision"])]
        for ax, rho_col, split in ((axes[0], "aware_mae", "MAE"),
                                   (axes[1], "aware_acc", "ACC（分类头）")):
            g = d.groupby("rho")[rho_col].mean()
            y = g - (base["mae"] if split == "MAE" else base["acc"])
            ax.plot(y.index, y.values, "-o", ms=4, lw=1.4, color=col, label=name)
    for ax, split in ((axes[0], "MAE"), (axes[1], "ACC（分类头）")):
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.set_xlabel("注入缺失率 ρ")
        ax.set_ylabel("Δ" + split + ("（正＝退化）" if split == "MAE" else "（负＝退化）"))
        ax.set_title("文本类缺失下的性能变化（按缺失类型平均）", fontsize=9.5)
        ax.grid(ls=":", lw=0.5, color="0.85")
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fp = os.path.join(out, "Q2_四模型缺失退化对比_折线图.png")
    fig.savefig(fp)
    plt.close(fig)
    return fp


def fig_ablation(root, out):
    """消融：相对完整模型的 ΔMAE 与 ΔMacroF1（去该模块后变差 → 该模块有贡献）。"""
    p = os.path.join(root, "paper_tables", "ablation_clean.csv")
    if not os.path.isfile(p):
        return None
    d = pd.read_csv(p)
    full = d[d["实验"].str.contains("完整模型")].iloc[0]
    sub = d[~d["实验"].str.contains("完整模型")].copy()
    sub["d_mae"] = sub["valid_MAE"] - full["valid_MAE"]
    sub["d_f1"] = sub["valid_MacroF1"] - full["valid_MacroF1"]
    y = np.arange(len(sub))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
    for ax, col, title, pos_is_bad in (
            (axes[0], "d_mae", "ΔMAE（正＝比完整模型差）", True),
            (axes[1], "d_f1", "ΔMacro-F1（负＝比完整模型差）", False)):
        v = sub[col].values
        cols = [RED if (x > 0 if pos_is_bad else x < 0) else BLUE for x in v]
        ax.barh(y, v, height=0.6, color=cols, edgecolor="black", linewidth=0.4)
        for yy, x in zip(y, v):
            ax.text(x + (0.0008 if x >= 0 else -0.0008), yy, "%+.4f" % x,
                    va="center", ha="left" if x >= 0 else "right", fontsize=8)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["实验"], fontsize=8)
        ax.set_title(title, fontsize=9.5)
        ax.margins(x=0.22)
        ax.grid(axis="x", ls=":", lw=0.5, color="0.85")
        ax.set_axisbelow(True)
    fig.tight_layout()
    fp = os.path.join(out, "Q2_消融实验对比_发散条形图.png")
    fig.savefig(fp)
    plt.close(fig)
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("..", ".."))
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    out = a.out_dir or os.path.join(root, "05_paper", "figures")
    os.makedirs(out, exist_ok=True)
    for fn in (fig_missing_profile, fig_error_groups, fig_tune,
               fig_degrade_models, fig_ablation):
        try:
            p = fn(root, out)
            print("[%s] %s" % ("OK" if p else "SKIP", p or fn.__name__))
        except Exception as e:
            print("[FAIL] %s: %s" % (fn.__name__, e))


if __name__ == "__main__":
    main()
