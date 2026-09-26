# -*- coding: utf-8 -*-
r"""问题3 共享工具：骨干加载、split 构建、掩码/遮挡算子、目标标量、缓存读写。

设计要点（与 02_model 的推理链路严格一致，保证"解释的是同一个模型"）：
  * 复用 `runtime.load_ensemble` 与 `train.prep/stack_masks/unstack_masks`，
    不重写前向路径，避免"解释的模型 ≠ 预测的模型"；
  * 屏蔽模态统一用 `miss=1`（模型在训练时见过的"整模态缺失"语义），
    并同步把特征置零（与 `unstack_masks` 的做法一致）；
  * 目标标量支持 `mu`（强度，主口径）与 `p_pos/p_neg/pol/p_max`（极性，对照口径）。
"""
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
for p in (CODE, os.path.join(CODE, "02_model")):
    if p not in sys.path:
        sys.path.append(p)

from data_utils import load_any, MoseiDataset, collate   # noqa: E402
from runtime import load_ensemble, predict_split         # noqa: E402
from train import prep, stack_masks, unstack_masks       # noqa: E402

MODALITIES = ("text", "audio", "vision")
LABELS = ("Negative", "Neutral", "Positive")
TARGETS = ("mu", "p_pos", "p_neg", "p_pol", "p_max")


# --------------------------------------------------------------------------
# 骨干与数据
# --------------------------------------------------------------------------
def load_backbone(ckpt_dir=None, device="cpu", verbose=True):
    ckpt_dir = ckpt_dir or os.path.join(ROOT, "runs", "ens_top2")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "student_*.pt")))
    if not ckpts:
        raise SystemExit("未找到权重：%s/student_*.pt" % ckpt_dir)
    models = load_ensemble(ckpts, device=device, verbose=verbose)
    for m, _ in models:                      # 只解释输入，不需要参数梯度
        for p in m.parameters():
            p.requires_grad_(False)
    theta = float(np.mean([ck["theta"] for _, ck in models]))
    return models, models[0][1]["stats"], theta


def load_split(pkl, stats, key=None):
    """读任意 pkl（附件2/3/4）并用 ckpt 内统计量归一化。"""
    splits = load_any(pkl, norm_stats=stats)
    if key is None:
        key = "test" if "test" in splits else list(splits.keys())[0]
    return splits[key], key


def make_batch(split, idxs, device="cpu", mask=True):
    """按 DataLoader 同路径构造 batch（含 pad 有效位兜底）。

    mask=True 时额外执行 `apply_masks`（把 miss|pad 位置的特征置零），
    与 `infer_att3.py` 的推理链路**完全一致**——这是"解释的模型＝预测的模型"的前提：
    若不置零，填充位的 BERT 垃圾值会经编码器内的局部卷积渗入最后一个有效位，
    造成解释与提交预测的口径偏差（实测最大分数差 0.049、1/20 标签不同）。
    """
    ds = MoseiDataset(split, augment=False)
    b = prep(collate([ds[int(i)] for i in idxs]), device)
    return apply_masks(b) if mask else b


def apply_masks(b):
    """与推理链路一致：把 miss|pad 位置的特征置零。"""
    M, _ = stack_masks(b)
    unstack_masks(b, M)
    return b


def valid_len(b, m, j):
    """第 j 个样本、模态 m 的有效长度（非填充位）。"""
    return int((~b["pad"][m][j]).sum().item())


# --------------------------------------------------------------------------
# 掩码/遮挡算子
# --------------------------------------------------------------------------
def mask_modalities(b, mods, mode="miss"):
    """屏蔽若干模态。mode='miss' 置缺失指示；mode='pad' 整模态不可见。"""
    out = dict(x={}, miss={}, pad={})
    for k in ("y_reg", "y_cls", "idx"):
        if k in b:
            out[k] = b[k]
    for m in MODALITIES:
        out["x"][m] = b["x"][m].clone()
        out["miss"][m] = b["miss"][m].clone()
        out["pad"][m] = b["pad"][m].clone()
    for m in mods:
        if mode == "miss":
            out["miss"][m] = torch.ones_like(b["miss"][m])
        else:
            p = torch.ones_like(b["pad"][m])
            p[:, 0] = False                  # 安全位：避免注意力全遮 → NaN
            out["pad"][m] = p
    for m in MODALITIES:
        bad = (out["miss"][m] | out["pad"][m]).float().unsqueeze(-1)
        out["x"][m] = out["x"][m] * (1.0 - bad)
    return out


def occlude_window(b, m, start, width):
    """把模态 m 的 [start, start+width) 段用相邻段替换（保持序列长度与形态）。

    首段向右取邻段、末段向左取邻段；缺失/填充位保持不可用。
    """
    out = dict(x={}, miss={}, pad={})
    for k in ("y_reg", "y_cls", "idx"):
        if k in b:
            out[k] = b[k]
    L = b["x"][m].size(1)
    for mm in MODALITIES:
        out["x"][mm] = b["x"][mm].clone()
        out["miss"][mm] = b["miss"][mm].clone()
        out["pad"][mm] = b["pad"][mm].clone()
    s = max(0, int(start))
    e = min(L, s + int(width))
    if e <= s:
        return out
    off = e - s
    if e + off <= L:
        src = slice(e, e + off)
        dst = slice(s, e)
    elif s - off >= 0:
        src = slice(s - off, s)
        dst = slice(s, e)
    else:                                    # 序列过短：退回均值替换
        src, dst = None, slice(s, e)
    for j in range(out["x"][m].size(0)):
        if src is None:
            avail = ~(b["miss"][m][j] | b["pad"][m][j])
            fill = b["x"][m][j][avail].mean(dim=0) if avail.any() else torch.zeros_like(
                b["x"][m][j][0])
            out["x"][m][j, dst] = fill
        else:
            out["x"][m][j, dst] = b["x"][m][j][src]
        out["miss"][m][j, dst] = False       # 被替换段视为"有信息"
    bad = (out["miss"][m] | out["pad"][m]).float().unsqueeze(-1)
    out["x"][m] = out["x"][m] * (1.0 - bad)
    return out


# --------------------------------------------------------------------------
# 前向与目标标量
# --------------------------------------------------------------------------
def predict_batch(models, b, target="mu", aux_keys=()):
    """集成前向。返回 score/prob/target（均为可微张量）与可选 aux。"""
    mus, ps = [], []
    aux_sum = {k: None for k in aux_keys}
    for m, _ in models:
        if aux_keys:
            mu, _, lg, ax = m(b["x"], b["miss"], b["pad"], return_aux=True)
            for k in aux_keys:
                v = ax.get(k)
                if torch.is_tensor(v) and v.size(0) == mu.size(0):
                    v = v.detach()
                    aux_sum[k] = v if aux_sum[k] is None else aux_sum[k] + v
        else:
            mu, _, lg = m(b["x"], b["miss"], b["pad"])
        mus.append(mu)
        ps.append(torch.softmax(lg, dim=-1))
    score = torch.stack(mus).mean(0)
    prob = torch.stack(ps).mean(0)
    if target == "mu":
        t = score
    elif target == "p_pos":
        t = prob[:, 2]
    elif target == "p_neg":
        t = prob[:, 0]
    elif target == "p_pol":
        t = prob[:, 2] - prob[:, 0]
    elif target == "p_max":
        t = prob.max(dim=-1).values
    else:
        raise ValueError("未知 target=%s（可选 %s）" % (target, TARGETS))
    out = dict(score=score, prob=prob, target=t)
    for k in aux_keys:
        if aux_sum[k] is not None:
            out[k] = aux_sum[k] / len(models)
    return out


# --------------------------------------------------------------------------
# 重要性后处理与缓存
# --------------------------------------------------------------------------
def norm_valid(w, n_valid):
    """把重要性向量限制在有效位内并归一到 [0,1]（全零时返回全零）。"""
    w = np.asarray(w, dtype=np.float64).copy()
    n = int(max(1, min(len(w), n_valid)))
    seg = w[:n]
    w[n:] = 0.0
    lo, hi = float(seg.min()), float(seg.max())
    if hi - lo < 1e-12:
        return np.zeros_like(w)
    w[:n] = (seg - lo) / (hi - lo)
    return w


def softmax1d(x, tau=1.0):
    x = np.asarray(x, dtype=np.float64) / max(float(tau), 1e-9)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_jsonable)
    return path


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def q3_meta(ckpt_dir, models, theta, extra=None):
    """记录本次解释使用的骨干与阈值，供复现核对。"""
    files = sorted(os.path.basename(p) for p in
                   glob.glob(os.path.join(ckpt_dir, "student_*.pt")))
    return dict(model_files=files, n_ckpt=len(models), ckpt_dir=str(ckpt_dir),
                theta=round(float(theta), 4), extra=extra or {})
