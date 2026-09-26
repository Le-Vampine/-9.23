# -*- coding: utf-8 -*-
r"""S4-b：解释质量指标（题目"可量化、可复核"的落地）。

指标定义（论文 §5.3）：
  忠实性 Comprehensiveness  c = t(X) − t(X 去掉 top-r 片段)      越大越好
  充分性 Sufficiency        s = t(X 只保留 top-r 片段) − t(X)    越接近 0 越好
  删除/插入 AUC            按重要性从高到低逐段移除（或逐段恢复）的 t 曲线积分
  稀疏性 Sparsity           top-5 位置的权重占该模态总权重的比例
  稳定性 Stability          输入加小扰动后重算解释的 L1 变化       越小越好

操作定义（与遮挡源保持同一族算子，保证可比）：
  * "去掉片段" = 用相邻段替换该位置（不是置零，避免同时改变"内容"与"是否可用"）
  * "只保留片段" = 其余有效位置置 miss=1（模型训练语义下的"不可用"）

用法：
  python metrics_xai.py --pkl ..\..\data_att\att4_aligned.pkl --name att4 --auc --stability
  python metrics_xai.py --pkl ..\..\附件2\aligned_50.pkl --split valid --name valid --n_max 200
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)
from xai_common import (MODALITIES, load_backbone, load_split, make_batch,     # noqa: E402
                        predict_batch, save_json, q3_meta, ROOT)
from ig_gradcam import ig_batch                                                # noqa: E402


# --------------------------------------------------------------------------
# 算子
# --------------------------------------------------------------------------
def replace_positions(b, m, pos, width=1):
    """把模态 m 的若干位置同时替换为相邻片段（一次调用内互不影响）。"""
    out = dict(x={}, miss={}, pad={})
    for k in ("y_reg", "y_cls", "idx"):
        if k in b:
            out[k] = b[k]
    L = b["x"][m].size(1)
    for mm in MODALITIES:
        out["x"][mm] = b["x"][mm].clone()
        out["miss"][mm] = b["miss"][mm].clone()
        out["pad"][mm] = b["pad"][mm].clone()
    X0 = b["x"][m]
    for j in range(X0.size(0)):
        for p in pos:
            s = max(0, int(p))
            e = min(L, s + int(width))
            if e <= s:
                continue
            off = e - s
            if e + off <= L:
                src = slice(e, e + off)
            elif s - off >= 0:
                src = slice(s - off, s)
            else:
                src = slice(max(0, e - 1), min(L, e - 1 + off)) or slice(0, off)
            out["x"][m][j, s:e] = X0[j, src]
            out["miss"][m][j, s:e] = False
    bad = (out["miss"][m] | out["pad"][m]).float().unsqueeze(-1)
    out["x"][m] = out["x"][m] * (1.0 - bad)
    return out


def keep_only(b, m, pos):
    """只保留模态 m 的 pos 位置，其余有效位设 miss=1。"""
    out = dict(x={}, miss={}, pad={})
    for k in ("y_reg", "y_cls", "idx"):
        if k in b:
            out[k] = b[k]
    L = b["x"][m].size(1)
    keep = torch.zeros((b["x"][m].size(0), L), dtype=torch.bool, device=b["x"][m].device)
    for p in pos:
        if 0 <= int(p) < L:
            keep[:, int(p)] = True
    for mm in MODALITIES:
        out["x"][mm] = b["x"][mm].clone()
        out["miss"][mm] = b["miss"][mm].clone()
        out["pad"][mm] = b["pad"][mm].clone()
    out["miss"][m] = (~keep) & (~out["pad"][m])
    bad = (out["miss"][m] | out["pad"][m]).float().unsqueeze(-1)
    out["x"][m] = out["x"][m] * (1.0 - bad)
    return out


# --------------------------------------------------------------------------
def top_positions(w_row, n_valid, r):
    n = int(max(1, min(len(w_row), n_valid)))
    idx = np.argsort(-np.asarray(w_row)[:n])[:r]
    return [int(i) for i in idx]


def metrics_one(models, b, W, valid_row, target="mu", r=5, auc_steps=5, do_auc=True):
    """对单条样本（batch=1）算解释质量指标。返回 dict(source 无关的数值）。"""
    t_full = float(predict_batch(models, b, target=target)["target"].item())
    res = dict(t_full=round(t_full, 4))
    compr, suff, spars, del_auc, ins_auc = [], [], [], [], []
    for k, m in enumerate(MODALITIES):
        nv = int(valid_row[k].sum())
        w = np.asarray(W[0, k], dtype=np.float64)
        wv = w[:nv]
        if wv.sum() <= 1e-12:
            compr.append(0.0)
            suff.append(0.0)
            spars.append(np.nan)
            del_auc.append(np.nan)
            ins_auc.append(np.nan)
            continue
        pos = top_positions(w, nv, r)
        t_del = float(predict_batch(models, replace_positions(b, m, pos), target=target,
                                   )["target"].item())
        t_keep = float(predict_batch(models, keep_only(b, m, pos), target=target,
                                    )["target"].item())
        compr.append(t_full - t_del)
        suff.append(t_keep - t_full)
        top5 = np.sort(wv)[::-1][:r]
        spars.append(float(top5.sum() / max(wv.sum(), 1e-9)))
        if do_auc:
            order = list(np.argsort(-wv))
            ks = np.linspace(0, nv, auc_steps + 1).astype(int)
            curve_d, curve_i = [], []
            for kk in ks:
                sel = order[:int(kk)]
                curve_d.append(float(predict_batch(models, replace_positions(b, m, sel),
                                                   target=target)["target"].item()))
                curve_i.append(float(predict_batch(models, keep_only(b, m, sel),
                                                   target=target)["target"].item()))
            f = ks / max(nv, 1)
            del_auc.append(float(np.trapezoid(curve_d, f)))
            ins_auc.append(float(np.trapezoid(curve_i, f)))
        else:
            del_auc.append(np.nan)
            ins_auc.append(np.nan)
    res.update(compr_text=round(compr[0], 4), compr_audio=round(compr[1], 4),
               compr_vision=round(compr[2], 4), compr_mean=round(float(np.mean(compr)), 4),
               suff_text=round(suff[0], 4), suff_audio=round(suff[1], 4),
               suff_vision=round(suff[2], 4), suff_absmean=round(float(np.abs(suff).mean()), 4),
               sparsity_mean=round(float(np.nanmean(spars)), 4),
               del_auc_mean=(None if np.isnan(del_auc).all() else round(float(np.nanmean(del_auc)), 4)),
               ins_auc_mean=(None if np.isnan(ins_auc).all() else round(float(np.nanmean(ins_auc)), 4)))
    return res


def stability_one(models, b, W_ref, steps=20, sigma=0.05, seed=0, target="mu"):
    """输入加高斯扰动后重算 IG，比较归一化重要性的 L1 变化。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    bp = dict(x={}, miss=b["miss"], pad=b["pad"])
    for m in MODALITIES:
        noise = torch.randn(b["x"][m].shape, generator=g).to(b["x"][m].device) * sigma
        valid = (~(b["miss"][m] | b["pad"][m])).float().unsqueeze(-1)
        bp["x"][m] = b["x"][m] + noise * valid
    ig, _, _ = ig_batch(models, bp, steps=steps, baseline="zero", target=target,
                        need_agrad=False)
    d = []
    for k in range(3):
        a = np.asarray(W_ref[0, k], np.float64)
        c = np.asarray(ig[0, k], np.float64)
        n = len(a)
        a2, c2 = a.copy(), c.copy()
        for v in (a2, c2):
            s = v[:n].sum()
            if s > 1e-12:
                v[:n] = v[:n] / s
        d.append(float(np.abs(a2[:n] - c2[:n]).sum()))
    return round(float(np.mean(d)), 4), [round(x, 4) for x in d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--name", default="att4")
    ap.add_argument("--ckpt_dir", default=os.path.join(ROOT, "runs", "ens_top2"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "runs", "q3"))
    ap.add_argument("--target", default="mu", choices=["mu", "p_pol"])
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--auc_steps", type=int, default=5)
    ap.add_argument("--auc", action="store_true")
    ap.add_argument("--stability", action="store_true")
    ap.add_argument("--n_max", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    pkl = a.pkl or os.path.join(ROOT, "附件2", "aligned_50.pkl")
    models, stats, theta = load_backbone(a.ckpt_dir, a.device, verbose=False)
    split, key = load_split(pkl, stats, a.split)
    if a.n_max and split["N"] > a.n_max:
        from data_utils import slice_split
        split = slice_split(split, a.n_max)
    N = split["N"]

    fused = np.load(os.path.join(a.out_dir, "q3_temporal_fused_%s.npz" % a.name),
                    allow_pickle=True)
    srcs = {k: fused["w_" + k] for k in ("att", "ig", "occ", "fused", "fused_equal")}
    valid = fused["valid"]
    ids = np.asarray(fused["ids"]).astype(str)
    print("[DATA] %s N=%d 解释源=%s" % (a.name, N, list(srcs)))

    rows = []
    stab = {}
    for i in range(N):
        b = make_batch(split, [i], a.device)
        for sname, W in srcs.items():
            r = metrics_one(models, b, W[i:i + 1], valid[i], a.target, a.r, a.auc_steps, a.auc)
            r.update(id=ids[i], source=sname)
            rows.append(r)
        if a.stability:
            d, per = stability_one(models, b, srcs["fused"][i:i + 1], target=a.target)
            mut = stability_one(models, b, srcs["ig"][i:i + 1], target=a.target)
            stab[ids[i]] = dict(stability_fused=d, stability_ig=mut[0],
                                stability_fused_per_mod=per)
        if (i + 1) % 25 == 0 or i + 1 == N:
            print("    %d/%d" % (i + 1, N))

    import pandas as pd
    df = pd.DataFrame(rows)[["id", "source", "t_full", "compr_mean", "compr_text",
                             "compr_audio", "compr_vision", "suff_absmean", "suff_text",
                             "suff_audio", "suff_vision", "sparsity_mean",
                             "del_auc_mean", "ins_auc_mean"]]
    csv = os.path.join(a.out_dir, "q3_faithfulness_%s.csv" % a.name)
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    summ = {}
    for sname in srcs:
        sub = df[df["source"] == sname]
        summ[sname] = dict(
            compr_mean=round(float(sub["compr_mean"].mean()), 4),
            suff_absmean=round(float(sub["suff_absmean"].mean()), 4),
            sparsity=round(float(sub["sparsity_mean"].mean()), 4),
            del_auc=(None if sub["del_auc_mean"].isna().all()
                     else round(float(sub["del_auc_mean"].mean()), 4)),
            ins_auc=(None if sub["ins_auc_mean"].isna().all()
                     else round(float(sub["ins_auc_mean"].mean()), 4)),
            # "忠实性−性能"加权目标：越大越好（compr 大、sufficiency 偏差小）
            score=round(float(sub["compr_mean"].mean() - sub["suff_absmean"].mean()), 4))
    print("  [SUMMARY] %s" % summ)
    rep = dict(meta=q3_meta(a.ckpt_dir, models, theta,
                            dict(pkl=os.path.relpath(pkl, ROOT), split=key, n=N,
                                 r=a.r, target=a.target, auc=bool(a.auc))),
               summary=summ, csv=os.path.relpath(csv, ROOT))
    if a.stability:
        rep["stability"] = stab
        sv = [v["stability_fused"] for v in stab.values()]
        print("  [STAB] 融合解释的稳定性（L1，越小越好）均值 %.4f" % float(np.mean(sv)))
    save_json(os.path.join(a.out_dir, "q3_faithfulness_%s_report.json" % a.name), rep)
    print("[SAVE] %s" % os.path.relpath(csv, ROOT))
    print("METRICS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
