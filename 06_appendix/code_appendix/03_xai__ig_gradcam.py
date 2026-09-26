# -*- coding: utf-8 -*-
r"""S2：片段级时间重要性的梯度类来源（注意力 / 积分梯度 / 注意力 x 梯度）。

三个来源（论文 section5.3 的 (a)(b) 两族）：
  (a) 注意力源           w^att_k = alpha_k                        （模型内生，仅作基线）
  (a') 注意力 x 梯度源      w^agrad_k = alpha_k . ||d t/d x_k||_2       （Grad-CAM 式加权）
  (b) 积分梯度            w^ig_k = ||(x_k - x'_k) (*) (1/L)Sigma_l grad_{x_k} t(x' + l/L(x-x'))||_2

基线 x'：`zero`（归一化空间中即训练集均值，主口径）/ `mean`（本 batch 有效位均值，对照）。

用法：
  python ig_gradcam.py --pkl <pkl> --name att4 --steps 20 --target mu --out_dir <runs/q3>
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)
from xai_common import (MODALITIES, load_backbone, load_split, make_batch,      # noqa: E402
                        predict_batch, save_npz, save_json, q3_meta, ROOT)


def ig_batch(models, b, steps=20, baseline="zero", target="mu", need_agrad=True):
    """返回 (w_ig (B,3,L), w_att (B,3,L), w_agrad (B,3,L))。"""
    x0 = {m: b["x"][m].detach().clone() for m in MODALITIES}
    if baseline == "zero":
        base = {m: torch.zeros_like(x0[m]) for m in MODALITIES}
    else:                                     # mean：该样本有效位均值
        base = {}
        for m in MODALITIES:
            valid = (~(b["miss"][m] | b["pad"][m])).float().unsqueeze(-1)
            mu = (x0[m] * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp(min=1)
            base[m] = mu.expand_as(x0[m]).contiguous()
    gsum = {m: torch.zeros_like(x0[m]) for m in MODALITIES}
    for l in range(1, steps + 1):
        a = l / float(steps)
        xs = {m: (base[m] + a * (x0[m] - base[m])).requires_grad_(True) for m in MODALITIES}
        out = predict_batch(models, dict(x=xs, miss=b["miss"], pad=b["pad"]), target=target)
        g = torch.autograd.grad(out["target"].sum(), [xs[m] for m in MODALITIES],
                                retain_graph=False)
        for i, m in enumerate(MODALITIES):
            gsum[m] = gsum[m] + g[i].detach()
    w_ig = torch.stack([((x0[m] - base[m]) * (gsum[m] / steps)).norm(dim=-1)
                        for m in MODALITIES], dim=1)          # (B,3,L)

    # 注意力源 + 注意力 x 梯度源（一次前向 + 一次反向）
    xg = {m: x0[m].clone().requires_grad_(True) for m in MODALITIES}
    out = predict_batch(models, dict(x=xg, miss=b["miss"], pad=b["pad"]),
                        target=target, aux_keys=("alpha",))
    grads = torch.autograd.grad(out["target"].sum(), [xg[m] for m in MODALITIES],
                                retain_graph=False)
    gnorm = torch.stack([grads[i].detach().norm(dim=-1) for i in range(3)], dim=1)   # (B,3,L)
    B, _, L = gnorm.shape
    alpha = out.get("alpha")
    if alpha is not None and alpha.size(0) == B and alpha.size(1) == 3 * L:
        w_att = alpha.detach().reshape(B, 3, L)
    else:                                     # 兜底：无法取注意力时退化为梯度幅值
        w_att = gnorm.clone()
    w_agrad = w_att * gnorm
    return (w_ig.detach().cpu().numpy(), w_att.cpu().numpy(), w_agrad.detach().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--name", default="att4")
    ap.add_argument("--ckpt_dir", default=os.path.join(ROOT, "runs", "ens_top2"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "runs", "q3"))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--baseline", default="zero", choices=["zero", "mean"])
    ap.add_argument("--target", default="mu", choices=["mu", "p_pol"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    pkl = a.pkl or os.path.join(ROOT, "附件2", "aligned_50.pkl")
    models, stats, theta = load_backbone(a.ckpt_dir, a.device)
    split, key = load_split(pkl, stats, a.split)
    if a.limit:
        from data_utils import slice_split
        split = slice_split(split, a.limit)
    N = split["N"]
    L = split["L"]["text"]
    print("[DATA] %s/%s N=%d L=%d | IG steps=%d baseline=%s target=%s"
          % (os.path.basename(pkl), key, N, L, a.steps, a.baseline, a.target))

    W_IG = np.zeros((N, 3, L), np.float64)
    W_AT = np.zeros((N, 3, L), np.float64)
    W_AG = np.zeros((N, 3, L), np.float64)
    for i0 in range(0, N, a.batch_size):
        idxs = list(range(i0, min(i0 + a.batch_size, N)))
        b = make_batch(split, idxs, a.device)
        ig, att, ag = ig_batch(models, b, a.steps, a.baseline, a.target)
        W_IG[i0:i0 + len(idxs)] = ig
        W_AT[i0:i0 + len(idxs)] = att
        W_AG[i0:i0 + len(idxs)] = ag
        if (i0 // a.batch_size) % 20 == 0:
            print("    %d/%d" % (min(i0 + a.batch_size, N), N))

    # 有效位掩码（重要性只在有效位内有意义）
    valid = np.zeros((N, 3, L), bool)
    for i in range(N):
        for k, m in enumerate(MODALITIES):
            valid[i, k, :int((~split["pad"][m][i]).sum())] = True
    for W in (W_IG, W_AT, W_AG):
        W[~valid] = 0.0
    path = save_npz(os.path.join(a.out_dir, "q3_temporal_ig_%s.npz" % a.name),
                    ids=np.asarray(split["ids"]).astype(str),
                    w_ig=W_IG, w_att=W_AT, w_agrad=W_AG, valid=valid,
                    meta=np.array([a.steps, a.baseline, a.target], dtype=object))
    rep = dict(meta=q3_meta(a.ckpt_dir, models, theta,
                            dict(pkl=os.path.relpath(pkl, ROOT), split=key, n=N,
                                 steps=a.steps, baseline=a.baseline, target=a.target)),
               w_ig_mean=[round(float(W_IG[:, k].sum(1).mean()), 4) for k in range(3)],
               w_att_mean=[round(float(W_AT[:, k].sum(1).mean()), 4) for k in range(3)],
               sparsity_ig_top5=[round(float(np.sort(W_IG[i, k, valid[i, k]])[::-1][:5].sum() /
                                              max(W_IG[i, k, valid[i, k]].sum(), 1e-9)), 3)
                                 for i in range(min(N, 20)) for k in range(1)],
               data=os.path.relpath(path, ROOT))
    save_json(os.path.join(a.out_dir, "q3_temporal_ig_%s_report.json" % a.name), rep)
    print("[SAVE] %s" % os.path.relpath(path, ROOT))
    print("IG_GRADCAM_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
