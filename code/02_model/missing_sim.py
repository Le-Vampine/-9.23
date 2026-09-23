# -*- coding: utf-8 -*-
"""
缺失模拟引擎：按"缺失类型 / 缺失位置 / 缺失率"三因素在训练时注入随机连续区间缺失。
严格对齐题目对"模态局部缺失"的定义：一个或多个模态中存在特征值全部为零的随机连续序列区间。
"""
import numpy as np
import torch

MODALITIES = ("text", "audio", "vision")


def _pick_modalities(mode, rng):
    """返回本次注入缺失的模态下标集合。"""
    if mode == "single":
        return [int(rng.integers(0, 3))]
    if mode == "double":
        return list(rng.choice([0, 1, 2], size=2, replace=False))
    # mixed: 1~2 个模态（保证仍有可见模态，符合'局部缺失而非整模态不存在'）
    n = 1 if rng.random() < 0.6 else 2
    return list(rng.choice([0, 1, 2], size=n, replace=False))


def _pick_span(valid_len, n_miss, pos_mode, rng):
    n_miss = int(min(max(1, n_miss), valid_len))
    if pos_mode == "head":
        s = 0
    elif pos_mode == "tail":
        s = valid_len - n_miss
    elif pos_mode == "middle":
        s = max(0, (valid_len - n_miss) // 2)
    else:  # random
        s = int(rng.integers(0, valid_len - n_miss + 1))
    return s, s + n_miss


def _split_and_place(valid, total, k, pos_mode, rng):
    """把 total 个缺失步拆成 k 段并放置，返回 [(a,b), ...]（绝对索引）。
    实测附件3 的缺失是**短碎片**（区间长约 2 步），因此支持多段注入。"""
    L = len(valid)
    total = int(min(max(1, total), L))
    k = int(max(1, min(k, total)))
    parts = []
    if k == 1:
        parts = [total]
    else:
        remaining = total
        for j in range(k - 1):
            hi = max(2, remaining - (k - 1 - j))
            v = int(rng.integers(1, hi)) if hi > 1 else 1
            parts.append(v)
            remaining -= v
        parts.append(max(1, remaining))
    out = []
    if pos_mode in ("head", "middle", "tail"):
        s = {"head": 0, "tail": max(0, L - total)}.get(pos_mode, max(0, (L - total) // 2))
        for p in parts:
            out.append((s, min(L, s + p)))
            s += p
    else:
        for p in parts:
            s = int(rng.integers(0, max(1, L - p + 1)))
            out.append((s, min(L, s + p)))
    res = []
    for a, b in out:
        aa, bb = int(valid[a]), int(valid[min(b, L) - 1]) + 1
        if bb > aa:
            res.append((aa, bb))
    return res


def inject_missing(miss, pad, rho_min, rho_max, mode="mixed",
                   pos_modes=("head", "middle", "tail", "random"), rng=None,
                   n_intervals=(1, 3)):
    """
    miss: (B,L,3) float/bool，已有缺失（来自数据本身）
    pad : (B,L,3) bool，True=超出有效长度
    n_intervals: 每模态拆成几段短碎片（附件3 实测为短碎片，故默认 1~3 段）
    返回 new_miss (B,L,3) bool，以及本次注入记录 records
    """
    if rng is None:
        rng = np.random.default_rng()
    B, L, M = miss.shape
    new_miss = miss.clone().bool()
    records = []
    for i in range(B):
        if rng.random() < 0.15:          # 保留 15% 全模态样本，防止模型遗忘完整输入
            records.append([])
            continue
        mods = _pick_modalities(mode, rng)
        rec = []
        for m in mods:
            valid = (~pad[i, :, m]).nonzero(as_tuple=False).flatten()
            if valid.numel() < 4:
                continue
            valid_len = int(valid.numel())
            rho = float(rng.uniform(rho_min, rho_max))
            n_miss = int(round(rho * valid_len))
            if n_miss <= 0:
                continue
            pos_mode = pos_modes[int(rng.integers(0, len(pos_modes)))]
            k = int(rng.integers(n_intervals[0], n_intervals[1] + 1)) if n_miss >= 2 else 1
            for a, b in _split_and_place(valid, n_miss, k, pos_mode, rng):
                new_miss[i, a:b, m] = True
                rec.append((MODALITIES[m], int(a), int(b),
                            round(float((b - a) / valid_len), 4)))
        records.append(rec)
    return new_miss, records


def batch_missing_ratios(miss, pad):
    """每样本每模态的有效缺失率 (B,3)。"""
    valid = (~pad).float()
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (miss.float() * valid).sum(dim=1) / denom


def crop_batch(b, rng, min_keep=0.7):
    """
    时间裁剪增强：把样本裁成随机连续窗口（补足首尾缺失场景）。

    ⚠️ 关键：附件2 的有效数据是**前缀结构**（[0, L_i) 有效，之后为填充）。
    若直接在 [0, L) 上裁剪，某些样本裁剪后窗口与原前缀不相交 → 有效位为空
    → 注意力全被屏蔽 → 输出 NaN。因此这里把窗口限制在**各自的原有效区间**内，
    并显式保证窗口内至少保留 1 个有效时间步。
    """
    mods = list(MODALITIES)
    B = b["x"][mods[0]].size(0)
    f0 = rng.uniform(0.0, max(1e-6, 1.0 - min_keep), size=B)
    f1 = np.minimum(1.0, f0 + min_keep + rng.uniform(0.0, max(1e-6, 1.0 - min_keep), size=B))
    for m in mods:
        vlen = (~b["pad"][m]).sum(dim=1).cpu().numpy()        # 原有效长度
        L = b["x"][m].size(1)
        for i in range(B):
            li = int(min(max(vlen[i], 1), L))
            if li <= 1:
                continue
            a = int(f0[i] * li)
            e = int(round(f1[i] * li))
            a = max(0, min(a, li - 1))
            e = max(a + 1, min(e, li))
            b["x"][m][i, :a] = 0.0
            b["x"][m][i, e:] = 0.0
            b["miss"][m][i, :a] = False
            b["miss"][m][i, e:] = False
            b["pad"][m][i, :a] = True
            b["pad"][m][i, e:] = True
            b["pad"][m][i, a] = False                          # 窗口内至少 1 个有效位
    return b


# --------------------------------------------------------------------------
# 离线整组扫描（用于第 4 节三因素规律分析实验，不参与训练）
# --------------------------------------------------------------------------
def make_scenario_masks(miss_base, pad, mtype, pos, rho, seed=0):
    """
    固定某一 (缺失类型, 位置, 缺失率) 场景，返回整批统一的掩码。
    mtype: 'text'|'audio'|'vision'|'text+audio'|'text+vision'|'audio+vision'
    pos  : 'head'|'middle'|'tail'|'random'
    """
    rng = np.random.default_rng(seed)
    B, L, M = miss_base.shape
    name2idx = {m: i for i, m in enumerate(MODALITIES)}
    mods = [name2idx[x] for x in mtype.split("+")]
    out = miss_base.clone().bool()
    for i in range(B):
        for m in mods:
            valid = (~pad[i, :, m]).nonzero(as_tuple=False).flatten()
            if valid.numel() < 2:
                continue
            valid_len = int(valid.numel())
            n_miss = int(round(rho * valid_len))
            if n_miss <= 0:
                continue
            s, e = _pick_span(valid_len, n_miss, pos, rng)
            a, b = int(valid[s]), int(valid[e - 1]) + 1
            out[i, a:b, m] = True
    return out
