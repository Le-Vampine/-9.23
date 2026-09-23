# -*- coding: utf-8 -*-
r"""
论文图表生成（问题2 第(4)项要求的"可视化分析"）。

产出（保存到 --out_dir，默认 figs/）：
  1. confusion_matrix_test.png      混淆矩阵（行归一化）
  2. scatter_pred_true.png          预测-真值散点（含回归线与 MAE/Pearson/CCC）
  3. error_hist.png                 误差分布（总体 + 按真值极性分层）
  4. missing_mask_showcase.png      缺失/填充掩码可视化（若干样本的三模态时间条）
  5. train_curves_sX.png            训练曲线（损失分项 + valid MAE）
  6. sweep_degrade.png              缺失退化曲线（需 sweep CSV，存在才画）

用法：
  python viz_results.py --ckpt_dir ..\runs\q2 --out_dir ..\figs
  python viz_results.py --ckpt_dir ..\runs\smoke --out_dir ..\figs --tag smoke
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config                                  # noqa: E402
from data_utils import load_all, load_any                  # noqa: E402
import runtime                                             # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import pandas as pd                                        # noqa: E402
from sklearn.metrics import confusion_matrix               # noqa: E402
from losses import compute_metrics                         # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
LABELS = ["Negative", "Neutral", "Positive"]


def esc(s):
    """样本 id 形如 'xxx$_$yyy'，matplotlib 会把 $ 当数学模式，需转义。"""
    return str(s).replace("$", r"\$")


def fig_confusion(y_cls, pred_cls, out):
    cm = confusion_matrix(y_cls, pred_cls, labels=[0, 1, 2])
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, m, t in zip(axes, [cm, cmn], ["计数", "行归一化"]):
        im = ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_xlabel("预测"); ax.set_ylabel("真值"); ax.set_title(t)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, ("%d" % m[i, j]) if t == "计数" else ("%.2f" % m[i, j]),
                        ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > m.max() * 0.6 else "black")
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


def fig_scatter(y, p, res, out):
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(y, p, s=6, alpha=0.35)
    ax.plot([-3, 3], [-3, 3], "k--", lw=1)
    z = np.polyfit(y, p, 1)
    xs = np.linspace(-3, 3, 50)
    ax.plot(xs, np.polyval(z, xs), "r-", lw=1.2, label="拟合: y=%.3fx+%.3f" % (z[0], z[1]))
    ax.set_xlabel("真值强度"); ax.set_ylabel("预测强度")
    ax.set_title("MAE=%.3f  r=%.3f  CCC=%.3f" % (res["mae"], res["pearson"], res["ccc"]), fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


def fig_error_hist(y, p, y_cls, out):
    err = np.abs(y - p)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].hist(err, bins=40, color="#4C72B0")
    axes[0].set_title("绝对误差分布 (mean=%.3f)" % err.mean()); axes[0].set_xlabel("|y-ŷ|")
    data = [err[y_cls == c] for c in (0, 1, 2)]
    axes[1].boxplot(data, tick_labels=LABELS, showfliers=False)
    axes[1].set_title("按真值极性的误差箱线图"); axes[1].set_ylabel("|y-ŷ|")
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


def fig_missing_showcase(split, out, k=4, seed=0):
    """把三模态的有效/缺失/填充状态画成色条图（绿=有效 红=缺失 灰=尾部填充）。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(split["N"], size=min(k, split["N"]), replace=False)
    mods = ("text", "audio", "vision")
    fig, axes = plt.subplots(len(idx), 1, figsize=(7, 0.95 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        ax = axes[r][0]
        L = max(split["pad"][m].shape[1] for m in mods)
        img = np.ones((3, L, 3))
        for j, m in enumerate(mods):
            Lm = split["pad"][m].shape[1]
            valid = ~split["pad"][m][i]
            miss = split["miss"][m][i]
            row = np.ones((Lm, 3))
            row[valid] = [0.45, 0.75, 0.45]
            row[valid & miss] = [0.85, 0.35, 0.35]
            row[~valid] = [0.85, 0.85, 0.85]
            img[j, :Lm] = row
        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(list(mods), fontsize=7)
        ax.set_xticks([])
        ax.set_title(esc(split["ids"][i]), fontsize=7)
    fig.suptitle("绿=有效  红=缺失  灰=尾部填充", fontsize=9)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.03, hspace=0.4)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def fig_train_curves(hist_csv, out):
    df = pd.read_csv(hist_csv)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    st = df[df["phase"] == "student"]
    for c in [c for c in ["reg", "cls", "rec", "rec0", "sim", "diff", "bal", "kd"] if c in df.columns]:
        if c in st.columns and st[c].fillna(0).abs().sum() > 0:
            axes[0].plot(st["epoch"], st[c], label=c, lw=1)
    axes[0].set_title("学生训练损失分项"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=7)
    for ph, c in (("teacher", "#4C72B0"), ("student", "#C44E52")):
        g = df[df["phase"] == ph]
        if len(g):
            axes[1].plot(g["epoch"], g["valid_mae"], "-o", ms=3, label=ph, color=c)
    axes[1].set_title("valid MAE 曲线"); axes[1].set_xlabel("epoch"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


def fig_sweep_degrade(csv_path, out):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    first = df["missing_type"].iloc[0]
    g0 = df[df["missing_type"] == first].groupby("rho")[["aware_mae", "unaware_mae"]].mean().reset_index()
    ax.plot(g0["rho"], g0["unaware_mae"], "k--", lw=1.2, label="未做缺失处理（A0 基线）")
    for mt, g in df.groupby("missing_type"):
        g = g.groupby("rho")[["aware_mae"]].mean().reset_index()
        ax.plot(g["rho"], g["aware_mae"], "-o", ms=3, label="%s (本文模型)" % mt)
    ax.set_xlabel("缺失率 ρ"); ax.set_ylabel("MAE"); ax.grid(alpha=0.3)
    ax.set_title("缺失率-性能退化曲线"); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", default=r"E:\数学建模\figs")
    ap.add_argument("--version", default="aligned")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    sfx = ("_" + a.tag) if a.tag else ""

    cfg = Config(version=a.version)
    if a.data_dir:
        cfg.data_dir = a.data_dir
        cfg.__post_init__()
    tr, va, te = load_all(cfg)
    cks = runtime.find_ckpts(a.ckpt_dir)
    if not cks:
        raise SystemExit("未找到 checkpoint: %s" % a.ckpt_dir)
    models = runtime.load_ensemble(cks, a.device)
    theta = float(np.mean([ck["theta"] for _, ck in models]))

    print("[INFO] 在 test 上推理（无缺失时的基础性能）...")
    pt = runtime.predict_split(models, te, a.batch_size, a.device)
    res = compute_metrics(pt["y_reg"], pt["score"], pt["y_cls"], theta,
                          y_pred_cls=pt["prob"].argmax(-1))
    print("[TEST] %s" % res)
    pc = np.where(pt["score"] < -theta, 0, np.where(pt["score"] > theta, 2, 1))

    fig_confusion(pt["y_cls"], pc, os.path.join(a.out_dir, "confusion_matrix%s.png" % sfx))
    fig_scatter(pt["y_reg"], pt["score"], res, os.path.join(a.out_dir, "scatter_pred_true%s.png" % sfx))
    fig_error_hist(pt["y_reg"], pt["score"], pt["y_cls"], os.path.join(a.out_dir, "error_hist%s.png" % sfx))
    fig_missing_showcase(te, os.path.join(a.out_dir, "missing_mask%s.png" % sfx))
    print("[FIG] confusion / scatter / error_hist / missing_mask -> %s" % a.out_dir)

    for h in sorted(glob.glob(os.path.join(a.ckpt_dir, "history_*.csv"))):
        tag = os.path.basename(h).replace("history_", "").replace(".csv", "")
        fig_train_curves(h, os.path.join(a.out_dir, "train_curves_%s.png" % tag))
        print("[FIG] train_curves_%s.png" % tag)

    for s in sorted(glob.glob(os.path.join(a.ckpt_dir, "sweep_valid_*.csv"))):
        tag = os.path.basename(s).replace("sweep_valid_", "").replace(".csv", "")
        fig_sweep_degrade(s, os.path.join(a.out_dir, "sweep_degrade_%s.png" % tag))
        print("[FIG] sweep_degrade_%s.png" % tag)

    with open(os.path.join(a.out_dir, "test_metrics%s.json" % sfx), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
