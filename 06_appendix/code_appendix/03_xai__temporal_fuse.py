# -*- coding: utf-8 -*-
r"""S4-a：三源时间重要性的归一化与融合（gamma 在验证集上按与遮挡源的一致性标定）。

三源：
  w_att   模型内生注意力（基线，已知"注意力!=重要性"）
  w_ig    积分梯度（公理化）
  w_occ   遮挡（模型无关，作为验证基准）

融合：先对每个样本、每个模态在**有效位内**做 min-max 归一化，再线性加权
      w = (gamma1.w~_att + gamma2.w~_ig + gamma3.w~_occ) / (gamma1+gamma2+gamma3)，最后再归一化一次。
gamma 的标定：在单纯形上枚举（**约束每个 gamma >= 0.1**，避免退化为"只留遮挡源"这种平凡解），
      以"融合结果与遮挡源的 Spearman 一致性（跨样本、跨模态、有效位内聚合）"为准则；
      同时报告等权 (1,1,1) 对照。

用法：
  python temporal_fuse.py --name valid --out_dir ..\..\runs\q3
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)
from xai_common import MODALITIES, save_npz, save_json, ROOT   # noqa: E402

SOURCES = ("att", "ig", "occ")


def load_sources(out_dir, name):
    ig = np.load(os.path.join(out_dir, "q3_temporal_ig_%s.npz" % name), allow_pickle=True)
    oc = np.load(os.path.join(out_dir, "q3_temporal_occ_%s.npz" % name), allow_pickle=True)
    w = dict(att=ig["w_att"], ig=ig["w_ig"], occ=oc["w_occ"])
    return w, ig["valid"], np.asarray(ig["ids"]).astype(str)


def norm_valid_mat(W, valid):
    """对 (N,3,L) 逐样本逐模态在有效位内 min-max 归一化（无效位保持 0）。"""
    A = np.zeros_like(W, dtype=np.float64)
    N, K, L = W.shape
    for i in range(N):
        for k in range(K):
            v = valid[i, k]
            if not v.any():
                continue
            seg = W[i, k, v].astype(np.float64)
            lo, hi = seg.min(), seg.max()
            if hi - lo < 1e-12:
                continue
            A[i, k, v] = (seg - lo) / (hi - lo)
    return A


def spearman(a, b):
    from scipy.stats import spearmanr
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(spearmanr(a, b).statistic)


def agreement(W, ref, valid):
    """跨样本/跨模态/有效位聚合的 Spearman 一致性。"""
    a, b = [], []
    for i in range(W.shape[0]):
        for k in range(W.shape[1]):
            v = valid[i, k]
            if v.sum() < 3:
                continue
            a.append(W[i, k, v])
            b.append(ref[i, k, v])
    if not a:
        return np.nan
    return spearman(np.concatenate(a), np.concatenate(b))


def simplex_grid(step=0.1, min_g=0.1):
    """枚举 gamma1+gamma2+gamma3=1，每个 >= min_g。"""
    out = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            g = np.array([i, j, k]) * step
            if g.min() >= min_g - 1e-9 and abs(g.sum() - 1.0) < 1e-9:
                out.append(tuple(np.round(g, 3)))
    return sorted(set(out))


def fuse(Ws, gamma):
    g = np.asarray(gamma, dtype=np.float64)
    g = g / g.sum()
    W = g[0] * Ws["att"] + g[1] * Ws["ig"] + g[2] * Ws["occ"]
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="valid")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "runs", "q3"))
    ap.add_argument("--step", type=float, default=0.1)
    a = ap.parse_args()

    W_raw, valid, ids = load_sources(a.out_dir, a.name)
    N, K, L = W_raw["occ"].shape
    print("[DATA] %s  N=%d 三源形状=(%d,%d,%d)" % (a.name, N, N, K, L))
    Ws = {k: norm_valid_mat(W_raw[k], valid) for k in SOURCES}
    for k in SOURCES:
        print("  [SRC] %-4s 归一化后均值 文/语/视 = %.4f / %.4f / %.4f"
              % (k, *[Ws[k][:, i][valid[:, i]].mean() for i in range(3)]))

    print("  [gamma] 单纯形枚举（每源 >= %.1f），以与遮挡源的一致性为准则：" % 0.1)
    rows = []
    for g in simplex_grid(a.step):
        W = fuse(Ws, g)
        ag = agreement(norm_valid_mat(W, valid), Ws["occ"], valid)
        rows.append((g, ag))
    rows_sorted = sorted(rows, key=lambda t: -(t[1] if np.isfinite(t[1]) else -9))
    for g, ag in rows_sorted[:5]:
        print("      gamma=(%.1f, %.1f, %.1f)  一致性 rho=%.4f" % (*g, ag))
    g_best = rows_sorted[0][0]
    g_eq = (1 / 3, 1 / 3, 1 / 3)
    W_best = fuse(Ws, g_best)
    W_eq = fuse(Ws, g_eq)
    print("  [gamma] 选定 gamma_att/gamma_ig/gamma_occ = (%.2f, %.2f, %.2f)（等权对照 rho=%.4f）"
          % (*g_best, agreement(norm_valid_mat(W_eq, valid), Ws["occ"], valid)))

    Wb = norm_valid_mat(W_best, valid)
    We = norm_valid_mat(W_eq, valid)
    path = save_npz(os.path.join(a.out_dir, "q3_temporal_fused_%s.npz" % a.name),
                    ids=ids, valid=valid, w_fused=Wb, w_fused_equal=We,
                    w_att=Ws["att"], w_ig=Ws["ig"], w_occ=Ws["occ"],
                    gamma=np.array(g_best))
    rep = dict(name=a.name, n=N, gamma=list(g_best), gamma_equal=list(g_eq),
               grid=[dict(gamma=list(g), spearman_occ=(None if not np.isfinite(ag) else round(ag, 4)))
                     for g, ag in rows_sorted],
               source_mean={k: [round(float(Ws[k][:, i][valid[:, i]].mean()), 4)
                                for i in range(3)] for k in SOURCES},
               agreement_with_occ={k: round(agreement(Ws[k], Ws["occ"], valid), 4)
                                   for k in SOURCES},
               data=os.path.relpath(path, ROOT))
    save_json(os.path.join(a.out_dir, "q3_gamma_%s.json" % a.name), rep)
    print("[SAVE] %s" % os.path.relpath(path, ROOT))
    print("FUSE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
