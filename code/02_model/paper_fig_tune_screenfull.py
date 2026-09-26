# -*- coding: utf-8 -*-
"""超参搜索 screen 与 full 两阶段的对比图（哑铃图）。

只读 runs/tune/grid_screen.csv 与 grid_full.csv，不训练。

图的读法
--------
  每个候选在左端点（灰）与右端点（蓝）各有一个点，两点用带箭头的连线相接，
  点旁的 #n 为该阶段内的排名。左列看 MAE（越小越好），右列看分类头宏观 F1。
  连线向下表示该候选在完整训练后指标改善，排名标签的错位表示排序发生变化。
"""
import csv
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

NATURE = ["#3B5F8A", "#A23B3B", "#4A7C59", "#B8B8B8", "#F0E6D3"]
GRAY = "#9A9A9A"
ORDER = ["c06_kd05", "c00_baseline", "c04_dropout50", "c01_hidden128"]
SHORT = {
    "c06_kd05": "蒸馏权重 0.5",
    "c00_baseline": "基线配方",
    "c04_dropout50": "丢弃率 0.5",
    "c01_hidden128": "隐维度 128",
}

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "axes.unicode_minus": False,
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def read(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return {r["tag"]: r for r in csv.DictReader(f)}


def ranks(vals, lower_better):
    idx = sorted(range(len(vals)), key=lambda i: vals[i] if lower_better else -vals[i])
    out = [0] * len(vals)
    for r, i in enumerate(idx, 1):
        out[i] = r
    return out


def main():
    s = read(os.path.join(ROOT, "runs", "tune", "grid_screen.csv"))
    f_ = read(os.path.join(ROOT, "runs", "tune", "grid_full.csv"))
    tags = [t for t in ORDER if t in s and t in f_]
    if not tags:
        raise SystemExit("[ERR] 缺少 grid_screen.csv / grid_full.csv")

    s_mae = [float(s[t]["valid_mae"]) for t in tags]
    f_mae = [float(f_[t]["valid_mae"]) for t in tags]
    s_f1 = [float(s[t]["valid_f1_head"]) for t in tags]
    f_f1 = [float(f_[t]["valid_f1_head"]) for t in tags]
    rs_m, rf_m = ranks(s_mae, True), ranks(f_mae, True)
    rs_f, rf_f = ranks(s_f1, False), ranks(f_f1, False)

    y = np.arange(len(tags))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=True)

    def panel(ax, sv, fv, r_s, r_f, title, xlabel):
        lo, hi = min(sv + fv), max(sv + fv)
        pad = (hi - lo) * 0.22
        for yi, a, b, ra, rb in zip(y, sv, fv, r_s, r_f):
            ax.annotate("", xy=(b, yi), xytext=(a, yi),
                        arrowprops=dict(arrowstyle="-|>", color="#7AA0C4",
                                        lw=1.4, shrinkA=3, shrinkB=3))
        h1 = ax.scatter(sv, y, s=58, color=GRAY, zorder=3, label="快速筛选阶段",
                        edgecolor="white", linewidth=0.8)
        h2 = ax.scatter(fv, y, s=58, color=NATURE[0], zorder=3, label="完整训练阶段",
                        edgecolor="white", linewidth=0.8)
        for yi, a, ra in zip(y, sv, r_s):
            ax.text(a, yi + 0.22, "#%d" % ra, ha="center", va="bottom",
                    fontsize=8.5, color="#666666")
        for yi, b, rb in zip(y, fv, r_f):
            ax.text(b, yi + 0.22, "#%d" % rb, ha="center", va="bottom",
                    fontsize=8.5, color=NATURE[0])
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[t] for t in tags])
        ax.grid(axis="x", linestyle=":", linewidth=0.5, color="#DDDDDD")
        ax.set_axisbelow(True)
        return h1, h2

    h1, h2 = panel(axes[0], s_mae, f_mae, rs_m, rf_m, "(a) 平均绝对误差（越小越好）", "平均绝对误差")
    panel(axes[1], s_f1, f_f1, rs_f, rf_f, "(b) 分类头宏观 F1（越大越好）", "宏观 F1")
    fig.tight_layout()
    fig.legend(handles=[h1, h2], labels=["快速筛选阶段", "完整训练阶段"],
               loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    p = os.path.join(ROOT, "figs", "Q2_超参搜索screen与full对比_哑铃图.png")
    fig.savefig(p)
    plt.close(fig)
    print("[OK] %s" % os.path.basename(p))

    print("\n%-16s | %-24s | %-24s" % ("候选", "screen MAE/F1 (排名)", "full MAE/F1 (排名)"))
    for i, t in enumerate(tags):
        print("%-16s | %.4f / %.4f  (#%d/#%d) | %.4f / %.4f  (#%d/#%d)" % (
            t, s_mae[i], s_f1[i], rs_m[i], rs_f[i], f_mae[i], f_f1[i], rf_m[i], rf_f[i]))


if __name__ == "__main__":
    main()
