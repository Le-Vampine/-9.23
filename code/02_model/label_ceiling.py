# -*- coding: utf-8 -*-
r"""
标签上限与基线测算（只用已有的预测/标签 CSV，不加载原始 pkl，不训练模型）。

回答三个问题：
  1) 数据本身的**理论性能上限**大致在哪（标签自洽性、边界样本比例、常数基线）；
  2) 与文献可比性：把三分类结果桥接成 **二分类 Acc2/F1**（CMU-MOSEI 文献常用口径）；
  3) 现有模型的差距（还差多少到"可达上限"）。

用法：
  python 02_model\label_ceiling.py --csv ..\runs\q2v3\error_groups_test.csv
  python 02_model\label_ceiling.py --csv ..\runs\q2v3\error_groups_valid.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from losses import polar_from_score                             # noqa: E402
from sklearn.metrics import accuracy_score, f1_score            # noqa: E402


def f(x, nd=4):
    return "—" if x is None or not np.isfinite(x) else ("%.*f" % (nd, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="error_groups_*.csv（含 y_reg/y_cls/pred/pred_cls）")
    ap.add_argument("--theta", type=float, default=None, help="θ 口径阈值（默认从 pred 推断为 0.45）")
    ap.add_argument("--out_json", default=None)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    df = pd.read_csv(a.csv)
    y = df["y_reg"].to_numpy(dtype=float)          # 连续强度（多标注者平均）
    c = df["y_cls"].to_numpy(dtype=int)            # 三分类极性标签
    # 兼容两种格式：error_top20_*.csv（pred/pred_cls）与 report_p0 --dump_preds（score_theta/pred_head）
    if "pred" in df.columns:
        p = df["pred"].to_numpy(dtype=float)
        pc_head = df["pred_cls"].to_numpy(dtype=int) if "pred_cls" in df.columns else None
        pc_theta = None
    else:
        p = df["score_theta"].to_numpy(dtype=float)
        pc_head = df["pred_head"].to_numpy(dtype=int) if "pred_head" in df.columns else None
        pc_theta = df["pred_theta"].to_numpy(dtype=int) if "pred_theta" in df.columns else None
    n = len(y)
    print("[DATA] %s  n=%d" % (os.path.basename(a.csv), n))
    print("[DATA] 标签强度 mean=%.3f std=%.3f  三分类分布=%s"
          % (y.mean(), y.std(), np.bincount(c, minlength=3).tolist()))

    out = dict(n=int(n), source=os.path.basename(a.csv))

    # ---------- 1) 标签自洽性：强度符号 vs 三分类标签 ----------
    sign = np.where(y > 0, 2, np.where(y < 0, 0, 1))
    agree = float((sign == c).mean())
    conflict = 1.0 - agree
    print("\n[1] 标签自洽性（sign(强度) 与三分类标签一致率）= %.4f；跨界率 = %.4f" % (agree, conflict))
    for k, name in ((0, "Negative"), (1, "Neutral"), (2, "Positive")):
        m = c == k
        if m.sum():
            print("    真值 %-9s n=%3d  强度均/标: %+.3f / %.3f  区间[%.2f, %.2f]"
                  % (name, m.sum(), y[m].mean(), y[m].std(), y[m].min(), y[m].max()))
    out["label_self_consistency"] = agree
    out["label_conflict_rate"] = conflict

    # ---------- 2) 边界样本（人类最易分歧区） ----------
    print("\n[2] 边界样本比例（|强度| 很小 → 极性本身模糊）")
    out["boundary"] = {}
    for e in (0.1, 0.25, 0.5):
        fr = float((np.abs(y) <= e).mean())
        out["boundary"]["abs_le_%.2f" % e] = fr
        print("    |y| <= %.2f: %.3f" % (e, fr))
    # 只考虑非中性真值时的"近边界"比例
    nz = c != 1
    if nz.sum():
        fr = float((np.abs(y[nz]) <= 0.5).mean())
        out["boundary"]["abs_le_0.5_non_neutral"] = fr
        print("    |y| <= 0.50（真值非中性）: %.3f" % fr)

    # ---------- 3) 常数/平凡基线 ----------
    print("\n[3] 平凡基线")
    mae0 = float(np.abs(y).mean())
    mu0 = float(np.abs(y - y.mean()).mean())
    maj = int(np.bincount(c, minlength=3).argmax())
    acc_maj = float((c == maj).mean())
    # 逐类常数最优（oracle constant，按真值类别给该类均值）
    yhat_cls_mean = np.array([y[c == k].mean() if (c == k).sum() else 0.0 for k in (0, 1, 2)])[c]
    mae_cls = float(np.abs(y - yhat_cls_mean).mean())
    print("    预测常数 0 的 MAE            = %.4f" % mae0)
    print("    预测全局均值 %.3f 的 MAE     = %.4f" % (y.mean(), mu0))
    print("    按真值类别给类均值(oracle)MAE = %.4f  （任何模型的 MAE 下限参考）" % mae_cls)
    print("    多数类 ACC = %.4f（多数类=%d）; 随机 Macro-F1 ≈ 0.333" % (acc_maj, maj))
    out["baselines"] = dict(mae_const0=mae0, mae_global_mean=mu0, mae_class_mean_oracle=mae_cls,
                            acc_majority=acc_maj, macro_f1_random=1.0 / 3.0)

    # ---------- 4) 模型现状 + 距"符号上限"的距离 ----------
    theta = a.theta if a.theta is not None else 0.45
    pc_th = pc_theta if pc_theta is not None else polar_from_score(p, theta)
    print("\n[4] 模型现状（θ=%.2f）" % theta)
    print("    强度: MAE=%.4f  (常数0基线 %.4f → 改善 %.4f)"
          % (np.abs(y - p).mean(), mae0, mae0 - np.abs(y - p).mean()))
    if pc_head is not None:
        print("    极性(分类头): ACC=%.4f  MacroF1=%.4f"
              % (accuracy_score(c, pc_head), f1_score(c, pc_head, average="macro", zero_division=0)))
    print("    极性(θ 口径): ACC=%.4f  MacroF1=%.4f"
          % (accuracy_score(c, pc_th), f1_score(c, pc_th, average="macro", zero_division=0)))
    print("    **符号上限**（若强度预测完美、仅按符号判极性）= %.4f → 距上限 %.4f"
          % (agree, agree - accuracy_score(c, pc_head if pc_head is not None else pc_th)))
    out["model"] = dict(
        mae=float(np.abs(y - p).mean()),
        acc_head=float(accuracy_score(c, pc_head)) if pc_head is not None else None,
        f1_head=float(f1_score(c, pc_head, average="macro", zero_division=0)) if pc_head is not None else None,
        acc_theta=float(accuracy_score(c, pc_th)),
        f1_theta=float(f1_score(c, pc_th, average="macro", zero_division=0)),
        theta=theta)
    out["sign_oracle_acc"] = agree

    # ---------- 5) 二分类桥接（与 CMU-MOSEI 文献口径对齐） ----------
    print("\n[5] 二分类桥接（MOSEI 文献常用 Acc2/F1；正 = 强度>0）")
    yb = (y > 0).astype(int)
    variants = {}
    for tag, pred3 in (("分类头", pc_head), ("θ口径", pc_th)):
        if pred3 is None:
            continue
        for fold, name in ((0, "中性并入负"), (2, "中性并入正")):
            pb = np.where(pred3 == 2, 1, np.where(pred3 == 1, 1 if fold == 2 else 0, 0))
            variants["%s_%s" % (tag, name)] = dict(
                acc2=float(accuracy_score(yb, pb)),
                f1=float(f1_score(yb, pb, average="binary", zero_division=0)))
    # 排除中性真值
    keep = c != 1
    if keep.sum():
        for tag, pred3 in (("分类头", pc_head), ("θ口径", pc_th)):
            if pred3 is None:
                continue
            variants["%s_排除中性" % tag] = dict(
                acc2=float(accuracy_score(yb[keep], (pred3[keep] == 2).astype(int))),
                f1=float(f1_score(yb[keep], (pred3[keep] == 2).astype(int),
                                  average="binary", zero_division=0)))
    for k, v in variants.items():
        print("    %-18s Acc2=%.4f  F1=%.4f" % (k, v["acc2"], v["f1"]))
    out["binary_bridge"] = variants

    if a.out_json:
        with open(a.out_json, "w", encoding="utf-8") as fp:
            json.dump(out, fp, ensure_ascii=False, indent=2)
        print("\n[SAVE] %s" % a.out_json)


if __name__ == "__main__":
    main()
