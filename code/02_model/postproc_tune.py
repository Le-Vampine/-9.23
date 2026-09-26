# -*- coding: utf-8 -*-
r"""
集成/后处理策略调优（**零训练成本**，直接用现有 ckpt 对比）。

对比以下策略，全部以"分类头口径"为主口径（与提交 CSV 一致）：
  V0 概率平均（当前实现）：分类用 mean(softmax)，回归用 mean(μ)
  V1 **logit 平均**：分类用 softmax(mean(logits))（分布更尖锐，修"过度中性"）
  V2 V1 + **类别先验 logit 校正**：logits − τ·log π（π 取训练集类频），τ 在 valid 上选
  V3 **ckpt 子集**：按 valid 表现挑 top-k 个 ckpt 再平均
  V4 **TTA**：对每个样本做 K 次随机时间裁剪视图，平均预测（仅推理期，不拟合任何参数）

用法：
  python 02_model\postproc_tune.py --ckpt_dir ..\runs\q2v3 --out_dir ..\runs\q2v3
  python 02_model\postproc_tune.py --ckpt_dir ..\runs\q2v3 --tta_views 4 --subset 2
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config                                        # noqa: E402
from data_utils import load_all, MoseiDataset, collate            # noqa: E402
from losses import compute_metrics, polar_from_score              # noqa: E402
from missing_sim import crop_batch                                # noqa: E402
import runtime                                                    # noqa: E402
from train import prep, stack_masks, unstack_masks                # noqa: E402

MODALITIES = ("text", "audio", "vision")


# --------------------------------------------------------------------------
@torch.no_grad()
def per_model_outputs(ckpt_paths, split, batch_size, device):
    """逐个 ckpt 单独推理，返回 (mu[N,M], logits[N,M,3]) 便于事后做不同聚合。"""
    mus, logs = [], []
    for p in ckpt_paths:
        models = runtime.load_ensemble([p], device=device, verbose=False)
        ds = MoseiDataset(split, augment=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
        ms, ls = [], []
        for batch in loader:
            b = prep(batch, device)
            M, Pd = stack_masks(b)
            unstack_masks(b, M)
            mu, _, logits = models[0][0](b["x"], b["miss"], b["pad"])
            ms.append(mu.cpu().numpy())
            ls.append(logits.cpu().numpy())
        mus.append(np.concatenate(ms))
        logs.append(np.concatenate(ls))
    return np.stack(mus, axis=1), np.stack(logs, axis=1)      # (N,M), (N,M,3)


@torch.no_grad()
def tta_outputs(ckpt_paths, split, batch_size, device, views=4, min_keep=0.85, seed0=0):
    """TTA：K 个随机时间裁剪视图 × 全部 ckpt 平均，返回 (mu[N], prob[N,3])。"""
    models = []
    for p in ckpt_paths:
        models += [m for m, _ in runtime.load_ensemble([p], device=device, verbose=False)]
    ds = MoseiDataset(split, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    acc_mu, acc_p = None, None
    for v in range(views):
        rng = np.random.default_rng(seed0 + 1000 * v)
        mus, ps = [], []
        for batch in loader:
            b = prep(batch, device)
            if v > 0:                                   # v=0 为原始视图
                b = crop_batch(b, rng, min_keep=min_keep)
            M, Pd = stack_masks(b)
            unstack_masks(b, M)
            s, c = None, None
            for m in models:
                mu, _, logits = m(b["x"], b["miss"], b["pad"])
                s = mu if s is None else s + mu
                c = torch.softmax(logits, -1) if c is None else c + torch.softmax(logits, -1)
            mus.append((s / len(models)).cpu().numpy())
            ps.append((c / len(models)).cpu().numpy())
        mu_v = np.concatenate(mus)
        p_v = np.concatenate(ps)
        acc_mu = mu_v if acc_mu is None else acc_mu + mu_v
        acc_p = p_v if acc_p is None else acc_p + p_v
    return acc_mu / views, acc_p / views


def metrics_from(mu, cls_pred, y, c, theta):
    r = compute_metrics(y, mu, c, theta=theta, y_pred_cls=cls_pred)
    r["acc_theta"] = r["acc"]
    r["f1_theta"] = r["f1"]
    return r


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--subset", type=int, default=2, help="ckpt 子集大小（top-k by valid）")
    ap.add_argument("--tta_views", type=int, default=4)
    ap.add_argument("--tta_min_keep", type=float, default=0.85)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--version", default="aligned")
    a = ap.parse_args()
    out_dir = a.out_dir or a.ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    cfg = Config(version=a.version, out_dir=out_dir)
    cfg.__post_init__()
    tr, va, te = load_all(cfg)
    # 训练集类先验（供 logit 校正）
    cnt = np.bincount(np.asarray(tr["labels_cls"]).astype(int), minlength=3).astype(float)
    prior = cnt / cnt.sum()
    print("[PRIOR] 训练集类频 = %s" % np.round(prior, 4).tolist())

    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, "student_*.pt")))
    if not ckpts:
        raise SystemExit("未找到 ckpt")
    print("[CKPT] %d 个: %s" % (len(ckpts), [os.path.basename(p) for p in ckpts]))
    thetas = []
    for p in ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        thetas.append(float(ck["theta"]))
    theta0 = float(np.mean(thetas))

    K = min(a.subset, len(ckpts))
    mu_v, lg_v = per_model_outputs(ckpts, va, a.batch_size, a.device)
    mu_t, lg_t = per_model_outputs(ckpts, te, a.batch_size, a.device)
    yv, cv = np.asarray(va["labels_reg"], float), np.asarray(va["labels_cls"]).astype(int)
    yt, ct = np.asarray(te["labels_reg"], float), np.asarray(te["labels_cls"]).astype(int)

    # 每个 ckpt 在 valid 上的单独 MAE，用于子集选择
    per_mae = [float(np.abs(yv - mu_v[:, i]).mean()) for i in range(len(ckpts))]
    order = np.argsort(per_mae)
    sub = list(order[:K])
    print("[SUBSET] 各 ckpt valid MAE = %s → 取前 %d 个: %s"
          % (np.round(per_mae, 4).tolist(), K, [os.path.basename(ckpts[i]) for i in sub]))

    rows = []

    def add(name, mu_v_, p_v_, mu_t_, p_t_, cls_v, cls_t, note=""):
        mv = metrics_from(mu_v_, cls_v, yv, cv, theta0)
        mt = metrics_from(mu_t_, cls_t, yt, ct, theta0)
        rows.append(dict(策略=name, note=note,
                         valid_MAE=mv["mae"], valid_ACC=mv["acc_head"], valid_MacroF1=mv["f1_head"],
                         test_MAE=mt["mae"], test_ACC=mt["acc_head"], test_MacroF1=mt["f1_head"],
                         valid_ACC_theta=mv["acc_theta"], test_ACC_theta=mt["acc_theta"]))

    # V0 概率平均（当前实现）
    pv0 = np.exp(lg_v - lg_v.max(-1, keepdims=True)); pv0 /= pv0.sum(-1, keepdims=True)
    pt0 = np.exp(lg_t - lg_t.max(-1, keepdims=True)); pt0 /= pt0.sum(-1, keepdims=True)
    add("V0 概率平均（现状）", mu_v.mean(1), pv0.mean(1), mu_t.mean(1), pt0.mean(1),
        pv0.mean(1).argmax(1), pt0.mean(1).argmax(1))

    # V1 logit 平均
    lg_vm, lg_tm = lg_v.mean(1), lg_t.mean(1)
    pv1 = np.exp(lg_vm - lg_vm.max(-1, keepdims=True)); pv1 /= pv1.sum(-1, keepdims=True)
    pt1 = np.exp(lg_tm - lg_tm.max(-1, keepdims=True)); pt1 /= pt1.sum(-1, keepdims=True)
    add("V1 logit 平均", mu_v.mean(1), pv1, mu_t.mean(1), pt1, pv1.argmax(1), pt1.argmax(1))

    # V2 + 先验校正（τ 在 valid 上选，按 Macro-F1）
    log_prior = np.log(prior)
    from sklearn.metrics import f1_score
    best = None
    for tau in np.arange(0.0, 1.51, 0.1):
        adj_v = lg_vm - tau * log_prior
        pred_v = adj_v.argmax(1)
        f1 = f1_score(cv, pred_v, average="macro", zero_division=0)
        acc = float((pred_v == cv).mean())
        if best is None or f1 > best[1]:
            best = (float(tau), float(f1), acc)
    tau_star = best[0]
    pred_v2 = (lg_vm - tau_star * log_prior).argmax(1)
    pred_t2 = (lg_tm - tau_star * log_prior).argmax(1)
    add("V2 logit 平均 + 先验校正", mu_v.mean(1), None, mu_t.mean(1), None, pred_v2, pred_t2,
        note="τ*=%.1f（valid 选）" % tau_star)
    print("[PRIOR] τ* = %.1f（valid Macro-F1=%.4f）" % (tau_star, best[1]))

    # V3 logit 平均 + ckpt 子集
    lg_vs, lg_ts = lg_v[:, sub, :].mean(1), lg_t[:, sub, :].mean(1)
    add("V3 logit 平均 + top-%d ckpt" % K, mu_v[:, sub].mean(1), None, mu_t[:, sub].mean(1), None,
        lg_vs.argmax(1), lg_ts.argmax(1), note="按 valid MAE 选")

    # V4 TTA（在全部 ckpt 上）
    if a.tta_views > 0:
        try:
            mu_v4, p_v4 = tta_outputs(ckpts, va, a.batch_size, a.device, a.tta_views, a.tta_min_keep)
            mu_t4, p_t4 = tta_outputs(ckpts, te, a.batch_size, a.device, a.tta_views, a.tta_min_keep)
            add("V4 TTA（%d 视图）" % a.tta_views, mu_v4, p_v4, mu_t4, p_t4,
                (np.log(p_v4 + 1e-12) - tau_star * log_prior).argmax(1),
                (np.log(p_t4 + 1e-12) - tau_star * log_prior).argmax(1),
                note="裁剪 min_keep=%.2f" % a.tta_min_keep)
        except Exception as e:
            print("[WARN] TTA 失败：%s" % e)

    df = pd.DataFrame(rows)
    print("\n" + df.round(4).to_string(index=False))
    p = os.path.join(out_dir, "postproc_tune.csv")
    df.to_csv(p, index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "postproc_tune.json"), "w", encoding="utf-8") as f:
        json.dump(dict(prior=prior.tolist(), tau_star=tau_star, per_ckpt_valid_mae=per_mae,
                       subset=[os.path.basename(ckpts[i]) for i in sub],
                       rows=rows), f, ensure_ascii=False, indent=2)
    print("[SAVE] %s" % p)


if __name__ == "__main__":
    main()
