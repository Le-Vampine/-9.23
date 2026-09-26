# -*- coding: utf-8 -*-
r"""S1：模态级精确 Shapley 值（3 模态 → 8 个子集全枚举）。

数学定义（论文 §5.3 用）：
    v(S) 为只用模态子集 S 时模型的目标标量；模态 m 的 Shapley 值
        phi_m = sum_{S ⊆ M\{m}} |S|!(n-|S|-1)!/n! · [v(S∪{m}) - v(S)],  n=3
    精确性检验：sum_m phi_m = v(M) - v(∅)。

本脚本一次算完 4 个口径组合（保证可比、避免重复前向）：
    target  ∈ {mu(强度,主), p_pol = p_pos - p_neg(极性,对照)}
    semantic∈ {miss(置缺失指示,主), pad(整模态不可见,对照)}
并输出两套归一化（ReLU 归一 / softmax|φ|）供后续按验证集"模态忠实性"择优。

两类效用：
  * 样本级：v(S) 取该样本自身的预测 → 逐样本 φ（写入附件4 解释 CSV）；
  * 数据集级：v(S) 取验证集上的 -MAE 与 Macro-F1 → 全局模态重要性（论文结论）。

用法：
  python shapley.py --pkl <pkl> --name att4 --out_dir <runs/q3>
  python shapley.py --pkl <附件2 aligned_50.pkl> --split valid --name valid --with_global
"""
import argparse
import itertools
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)
from xai_common import (MODALITIES, load_backbone, load_split, make_batch,     # noqa: E402
                        predict_batch, mask_modalities, norm_valid, softmax1d,
                        save_npz, save_json, q3_meta, ROOT)

SUBSETS = [tuple(m for i, m in enumerate(MODALITIES) if (s >> i) & 1) for s in range(8)]
MASKS = list(range(8))
NAMES = {0: "none", 1: "text", 2: "audio", 4: "vision",
         3: "text+audio", 5: "text+vision", 6: "audio+vision", 7: "all"}
WEIGHTS = {}


def mods_of(mask):
    """位掩码 → 模态名列表（text=1, audio=2, vision=4）。"""
    return [MODALITIES[i] for i in range(3) if (mask >> i) & 1]


def shapley_weights(n=3):
    """精确 Shapley 的组合权重 w(S) = |S|!(n-|S|-1)!/n!（按位掩码索引）。"""
    if n not in WEIGHTS:
        w = {}
        for i in range(3):
            others = [j for j in range(3) if j != i]
            tab = {}
            for r in range(len(others) + 1):
                for S in itertools.combinations(others, r):
                    tab[sum(1 << j for j in S)] = (math.factorial(len(S)) *
                                                   math.factorial(n - len(S) - 1) /
                                                   math.factorial(n))
            w[i] = tab
        WEIGHTS[n] = w
    return WEIGHTS[n]


def phi_from_values(v):
    """由 8 个子集的 v 值（按 mask 索引）算精确 Shapley 向量 (3,)。"""
    W = shapley_weights(3)
    phi = np.zeros(3)
    for i in range(3):
        for T, wt in W[i].items():
            phi[i] += wt * (v[T | (1 << i)] - v[T])
    return phi


def normalizers(phi):
    """两套归一化（对 (N,3) 或 (3,) 的 φ 逐样本处理）。"""
    a = np.asarray(phi, dtype=np.float64)
    single = a.ndim == 1
    a = a[None] if single else a
    relu = np.maximum(a, 0.0)
    s = relu.sum(1, keepdims=True)
    pi_relu = np.where(s > 1e-12, relu / np.maximum(s, 1e-12), np.nan)
    pi_soft = np.exp(np.abs(a) / 1.0)
    pi_soft = pi_soft / pi_soft.sum(1, keepdims=True)
    return (pi_relu[0], pi_soft[0]) if single else (pi_relu, pi_soft)


def run_batch(models, b, target, semantic):
    """对一个 batch 算 8 个子集的 v 值 → V (B,8)，并返回全模态输出。"""
    B = b["x"]["text"].size(0)
    V = np.zeros((B, 8))
    full = None
    for mask in MASKS:
        bb = b if mask == 7 else mask_modalities(
            b, [m for m in MODALITIES if m not in mods_of(mask)], mode=semantic)
        out = predict_batch(models, bb, target=target)
        V[:, mask] = out["target"].detach().cpu().numpy()
        if mask == 7:
            full = dict(score=out["score"].detach().cpu().numpy(),
                        prob=out["prob"].detach().cpu().numpy())
    return V, full


def global_values(models, split, target, semantic, batch=64, device="cpu"):
    """数据集级效用：v(S) = −MAE（回归）与 Macro-F1（分类头概率）。"""
    from sklearn.metrics import f1_score
    N = split["N"]
    y = np.asarray(split["labels_reg"])
    yc = np.asarray(split["labels_cls"])
    v_mae, v_f1 = np.zeros(8), np.zeros(8)
    for mask in MASKS:
        preds, probs = [], []
        for i0 in range(0, N, batch):
            idxs = list(range(i0, min(i0 + batch, N)))
            b = make_batch(split, idxs, device)
            if mask != 7:
                b = mask_modalities(
                    b, [m for m in MODALITIES if m not in mods_of(mask)], mode=semantic)
            out = predict_batch(models, b, target=target)
            preds.append(out["score"].detach().cpu().numpy())
            probs.append(out["prob"].detach().cpu().numpy())
        sc, pb = np.concatenate(preds), np.concatenate(probs)
        v_mae[mask] = -float(np.abs(sc - y).mean())
        try:
            v_f1[mask] = float(f1_score(yc, pb.argmax(1), average="macro"))
        except Exception:
            v_f1[mask] = np.nan
        print("      [GLOBAL] S=%-12s  MAE=%.4f  MacroF1=%.4f"
              % (NAMES[mask], -v_mae[mask], v_f1[mask]))
    return v_mae, v_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=None, help="默认附件2 aligned（用 --split valid）")
    ap.add_argument("--split", default=None, help="pkl 内的 split 名（附件2 需指定 valid）")
    ap.add_argument("--name", default="att4", help="输出文件名后缀")
    ap.add_argument("--ckpt_dir", default=os.path.join(ROOT, "runs", "ens_top2"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "runs", "q3"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只算前 N 条")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--with_global", action="store_true", help="额外算数据集级 Shapley")
    ap.add_argument("--global_target", default="mu", choices=["mu", "p_pol"])
    a = ap.parse_args()

    pkl = a.pkl or os.path.join(ROOT, "附件2", "aligned_50.pkl")
    models, stats, theta = load_backbone(a.ckpt_dir, a.device)
    split, key = load_split(pkl, stats, a.split)
    if a.limit:
        from data_utils import slice_split
        split = slice_split(split, a.limit)
    N = split["N"]
    print("[DATA] %s/%s  N=%d" % (os.path.basename(pkl), key, N))

    combos = [("mu", "miss"), ("mu", "pad"), ("p_pol", "miss"), ("p_pol", "pad")]
    store, v_store = {}, {}
    score_all, prob_all = None, None
    for target, semantic in combos:
        tag = "%s_%s" % (target, semantic)
        V = np.zeros((N, 8))
        phis = np.zeros((N, 3))
        sc, pb = [], []
        for i0 in range(0, N, a.batch_size):
            idxs = list(range(i0, min(i0 + a.batch_size, N)))
            b = make_batch(split, idxs, a.device)
            v, full = run_batch(models, b, target, semantic)
            for r in range(len(idxs)):
                phis[i0 + r] = phi_from_values(v[r])
            V[i0:i0 + len(idxs)] = v
            sc.append(full["score"])
            pb.append(full["prob"])
        store[tag], v_store[tag] = phis, V
        resid = float(np.abs(phis.sum(1) - (V[:, 7] - V[:, 0])).max())
        print("  [%-10s] φ 均值 文/语/视 = %+0.4f / %+0.4f / %+0.4f ; 精确性残差 max=%.2e"
              % (tag, *phis.mean(0), resid))
        assert resid < 1e-6, "Shapley 精确性检验失败"
        if tag == "mu_miss":
            score_all, prob_all = np.concatenate(sc), np.concatenate(pb)

    # 主口径：mu + miss；同时给出两套归一化
    phi_main = store["mu_miss"]
    pi_relu, pi_soft = normalizers(phi_main)
    n_relu = int(np.isnan(pi_relu).any(1).sum())
    print("  [NORM] ReLU 归一在全负样本上退化 %d/%d 条；softmax 口径无退化" % (n_relu, N))
    miss_rate = np.zeros((N, 3))
    for i in range(N):
        for k, m in enumerate(MODALITIES):
            span = float((~split["pad"][m][i]).sum())
            if span <= 1.0:                      # 整条全零：整模态缺失（口径 D1）
                miss_rate[i, k] = 1.0
            else:
                miss_rate[i, k] = float((split["miss"][m][i] & ~split["pad"][m][i]).sum()) / span

    out = dict(ids=np.asarray(split["ids"]).astype(str),
               score=score_all, prob=prob_all, phi_mu_miss=phi_main,
               pi_relu=np.nan_to_num(pi_relu, nan=0.0), pi_soft=pi_soft,
               miss_rate=miss_rate, subset_names=np.array([NAMES[m] for m in MASKS]))
    for tag in store:
        out["phi_" + tag] = store[tag]
        out["v_" + tag] = v_store[tag]
    path = save_npz(os.path.join(a.out_dir, "q3_shapley_%s.npz" % a.name), **out)

    # CSV（逐样本）
    import pandas as pd
    df = pd.DataFrame({
        "id": out["ids"],
        "score": np.round(score_all, 4),
        "prob_neg": np.round(prob_all[:, 0], 4), "prob_neu": np.round(prob_all[:, 1], 4),
        "prob_pos": np.round(prob_all[:, 2], 4),
        "main_modality": [MODALITIES[int(np.argmax(r))] for r in pi_soft],
        "main_modality_relu": [MODALITIES[int(np.argmax(r))] if not np.isnan(r).any()
                               else "" for r in pi_relu],
        "pi_text": np.round(pi_soft[:, 0], 4), "pi_audio": np.round(pi_soft[:, 1], 4),
        "pi_vision": np.round(pi_soft[:, 2], 4),
        "phi_text": np.round(phi_main[:, 0], 4), "phi_audio": np.round(phi_main[:, 1], 4),
        "phi_vision": np.round(phi_main[:, 2], 4),
        "phi_text_padsem": np.round(store["mu_pad"][:, 0], 4),
        "phi_audio_padsem": np.round(store["mu_pad"][:, 1], 4),
        "phi_vision_padsem": np.round(store["mu_pad"][:, 2], 4),
        "miss_text": np.round(miss_rate[:, 0], 4), "miss_audio": np.round(miss_rate[:, 1], 4),
        "miss_vision": np.round(miss_rate[:, 2], 4),
    })
    csv = os.path.join(a.out_dir, "q3_shapley_%s.csv" % a.name)
    os.makedirs(a.out_dir, exist_ok=True)
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    report = dict(meta=q3_meta(a.ckpt_dir, models, theta,
                               dict(pkl=os.path.relpath(pkl, ROOT), split=key, n=N)),
                  data=os.path.relpath(path, ROOT),
                  phi_mean={k: np.round(store[k].mean(0), 4).tolist() for k in store},
                  pi_soft_mean=np.round(pi_soft.mean(0), 4).tolist(),
                  pi_relu_mean=np.round(np.nan_to_num(pi_relu, nan=0.0).mean(0), 4).tolist(),
                  pi_relu_degenerate=int(n_relu),
                  v_mean_mu_miss={NAMES[m]: round(float(v_store["mu_miss"][:, m].mean()), 4)
                                  for m in MASKS})
    if a.with_global:
        print("  [GLOBAL] 数据集级效用（%s / miss 语义）:" % a.global_target)
        v_mae, v_f1 = global_values(models, split, a.global_target, "miss", a.batch_size,
                                    a.device)
        report["global"] = dict(target=a.global_target,
                                v_negmae={NAMES[m]: round(float(v_mae[m]), 4) for m in MASKS},
                                v_macrof1={NAMES[m]: round(float(v_f1[m]), 4) for m in MASKS},
                                phi_negmae=np.round(phi_from_values(v_mae), 4).tolist(),
                                phi_macrof1=np.round(
                                    phi_from_values(np.nan_to_num(v_f1)), 4).tolist())
        print("      φ(-MAE) = %s" % report["global"]["phi_negmae"])
        print("      φ(MacroF1) = %s" % report["global"]["phi_macrof1"])
    save_json(os.path.join(a.out_dir, "q3_shapley_%s_report.json" % a.name), report)
    print("[SAVE] %s\n[SAVE] %s" % (os.path.relpath(path, ROOT), os.path.relpath(csv, ROOT)))
    print("SHAPLEY_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
