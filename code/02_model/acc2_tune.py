"""Acc-2 专项：口径校准 + 阈值调优（文献口径复现）。

文献 Acc-2 两种设置：
  Acc-2(neg/pos)  ：只在 |y|>0 的样本上判 负/正
  Acc-2(neg/nonneg)：判 y<0 与 y>=0
文献常见做法是**在验证集上调阈值**再测；我们此前固定用 θ=0，可能白丢分。

用法：python acc2_tune.py --valid <preds_valid.csv> --test <preds_test.csv> --tag B
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(errors="replace")


def acc2(y, s, th, mode):
    if mode == "np":                      # neg / pos，剔除中性
        m = np.abs(y) > 1e-6
        return float(((y[m] > 0) == (s[m] > th)).mean()), int(m.sum())
    return float(((y < 0) == (s < th)).mean()), len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    dv = pd.read_csv(args.valid)
    dt = pd.read_csv(args.test)
    # Acc-2 是回归口径：用**连续标签**的符号（与文献一致）
    yv, sv = dv["y_reg"].to_numpy(), dv["score_theta"].to_numpy()
    yt, st = dt["y_reg"].to_numpy(), dt["score_theta"].to_numpy()

    print("=" * 78)
    print("[%s] Acc-2 口径校准" % args.tag)
    print("=" * 78)

    for mode, name in (("np", "Acc-2(neg/pos，剔除中性)"), ("nn", "Acc-2(neg/non-neg)")):
        # 平凡基线
        if mode == "np":
            m = np.abs(yv) > 1e-6
            base_v = max((yv[m] > 0).mean(), (yv[m] < 0).mean())
            m2 = np.abs(yt) > 1e-6
            base_t = max((yt[m2] > 0).mean(), (yt[m2] < 0).mean())
        else:
            base_v = max((yv < 0).mean(), (yv >= 0).mean())
            base_t = max((yt < 0).mean(), (yt >= 0).mean())
        # valid 上选阈值
        grid = np.arange(-0.6, 0.61, 0.02)
        vals = [acc2(yv, sv, g, mode)[0] for g in grid]
        j = int(np.argmax(vals))
        th_star = float(grid[j])
        a0_v = acc2(yv, sv, 0.0, mode)[0]
        a0_t = acc2(yt, st, 0.0, mode)[0]
        ast_t = acc2(yt, st, th_star, mode)[0]
        print("\n%s" % name)
        print("  平凡多数类基线: valid %.4f / test %.4f" % (base_v, base_t))
        print("  θ=0（我们之前的口径）    : valid %.4f / test %.4f" % (a0_v, a0_t))
        print("  valid 上最优 θ=%.2f       : valid %.4f / test %.4f  (Δtest %+.4f)"
              % (th_star, vals[j], ast_t, ast_t - a0_t))

    # 头部概率口径：用分类头概率构造二分类
    if {"prob_neg", "prob_neu", "prob_pos"} <= set(dv.columns):
        print("\n[对照] 分类头概率构造 Acc-2(neg/non-neg)：argmax(p_neg, p_neu+p_pos)")
        pv = dv[["prob_neg", "prob_neu", "prob_pos"]].to_numpy()
        pt = dt[["prob_neg", "prob_neu", "prob_pos"]].to_numpy()
        predv = (pv[:, 0] > pv[:, 1] + pv[:, 2]).astype(int)
        predt = (pt[:, 0] > pt[:, 1] + pt[:, 2]).astype(int)
        print("  valid %.4f / test %.4f"
              % (float(((yv < 0).astype(int) == predv).mean()),
                 float(((yt < 0).astype(int) == predt).mean())))


if __name__ == "__main__":
    main()
