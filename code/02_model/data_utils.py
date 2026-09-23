# -*- coding: utf-8 -*-
"""
附件2 / 附件3 / 附件4 pkl 读取、缺失检测、归一化、Dataset。

接口严格遵循题目说明：data[split][字段][j]，即按字段集中存储。
"""
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("text", "audio", "vision")


# --------------------------------------------------------------------------
# 基础读入
# --------------------------------------------------------------------------
def load_pickle(path):
    """读 pkl，兼容 py2 遗留 pickle（MOSEI 系列常见）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到特征文件: %s" % path)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")


def _stack_pad(seqs, L, d):
    """把长度不一的序列列表补齐/截断为 (N,L,d)。"""
    out = np.zeros((len(seqs), L, d), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = np.asarray(s, dtype=np.float32)
        if s.ndim == 1:
            s = s.reshape(-1, 1)
        n = min(len(s), L)
        out[i, :n] = s[:n]
    return out


def _get(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


# --------------------------------------------------------------------------
# 缺失检测（题目：某模态某连续区间特征值全部为零）
# --------------------------------------------------------------------------
def infer_length_from_zeros(X, eps=1e-8):
    """实测发现：附件2 aligned 版本没有 *_lengths 字段，尾部填充位全为零。
    因此用"最后一个非零时间步"推断有效长度（全零行退化为 1）。"""
    nz = (np.abs(X).sum(axis=-1) > eps)                  # (N,L)
    L = nz.shape[1]
    idx = np.where(nz.any(axis=1), L - np.argmax(nz[:, ::-1], axis=1), 1)
    return np.maximum(idx, 1).astype(np.int64)


def detect_missing_mask(X, pad, min_len=2):
    """
    X: (N,L,d) 原始特征（未被归一化，零值有物理含义）
    pad: (N,L) bool，True 表示超出有效长度的填充位
    返回 miss: (N,L) bool，True = 有效区间内的真缺失（不含 pad 位）
    """
    N, L, _ = X.shape
    zero_ts = (np.abs(X).sum(axis=-1) < 1e-8)          # 该时间步全零
    miss = np.zeros((N, L), dtype=bool)
    for i in range(N):
        valid_len = int((~pad[i]).sum())
        row = zero_ts[i, :valid_len]
        j = 0
        while j < len(row):
            if row[j]:
                k = j
                while k < len(row) and row[k]:
                    k += 1
                if k - j >= min_len:                   # 连续 >= min_len 步才判为缺失
                    miss[i, j:k] = True
                j = k
            else:
                j += 1
    return miss


def missing_regions(miss_row, min_len=2):
    """把 (L,) 缺失掩码整理成 [[start,end), ...] 连续区间，用于统计与提交。"""
    regions, j = [], 0
    L = len(miss_row)
    while j < L:
        if miss_row[j]:
            k = j
            while k < L and miss_row[k]:
                k += 1
            regions.append([int(j), int(k)])
            j = k
        else:
            j += 1
    return [r for r in regions if r[1] - r[0] >= min_len]


# --------------------------------------------------------------------------
# 组装一个 split
# --------------------------------------------------------------------------
def build_split(raw, limit=0, min_zero_len=2):
    """raw = data['train'] / ['valid'] / ['test']"""
    text = np.asarray(_get(raw, ["text"]), dtype=np.float32)
    audio = np.asarray(_get(raw, ["audio"]), dtype=np.float32)
    vision = np.asarray(_get(raw, ["vision"]), dtype=np.float32)
    N = text.shape[0]

    if limit and limit > 0:
        N = min(N, limit)
        text, audio, vision = text[:N], audio[:N], vision[:N]

    L = {"text": text.shape[1], "audio": audio.shape[1], "vision": vision.shape[1]}
    D = {"text": text.shape[2], "audio": audio.shape[2], "vision": vision.shape[2]}

    # ---- 有效长度 ----
    pad = {}
    for m, X in (("text", text), ("audio", audio), ("vision", vision)):
        ln = _get(raw, ["%s_lengths" % m])
        if ln is not None:
            ln = np.asarray(ln, dtype=np.int64)[:N]
            ln = np.clip(ln, 1, X.shape[1])
            src = "field"
        else:
            ln = infer_length_from_zeros(X)              # 尾部全零 = 填充
            src = "inferred"
        p = np.ones((N, X.shape[1]), dtype=bool)
        for i in range(N):
            p[i, :ln[i]] = False
        p[:, 0] = False          # 安全位：保证每条样本至少有 1 个有效位置（避免注意力全遮 NaN）
        pad[m] = p
        pad["src_" + m] = src

    # ---- 文本长度：优先用 text_bert 的 attention mask ----
    tb = _get(raw, ["text_bert"])
    if tb is not None:
        tb = np.asarray(tb)[:N]
        if tb.ndim == 3 and tb.shape[1] >= 2:
            am = tb[:, 1, :].astype(np.int64)          # (N,L) 注意力掩码
            p = np.ones_like(pad["text"])
            for i in range(N):
                nz = np.nonzero(am[i])[0]
                ln = int(nz[-1] + 1) if len(nz) else 1
                p[i, :min(ln, L["text"])] = False
            p[:, 0] = False
            pad["text"] = p

    X = {"text": text, "audio": audio, "vision": vision}
    miss = {m: detect_missing_mask(X[m], pad[m], min_zero_len) for m in MODALITIES}

    # ---- 标签 ----
    reg = _get(raw, ["regression_labels"])
    cls = _get(raw, ["classification_labels"])
    ann = _get(raw, ["annotations"])
    ids = _get(raw, ["id", "ids"])
    raw_text = _get(raw, ["raw_text"])
    has_labels = reg is not None                 # 附件3/附件4 为无标签测试集
    if not has_labels:
        reg = np.zeros(N, dtype=np.float32)
    reg = np.asarray(reg, dtype=np.float32).reshape(-1)[:N]
    if cls is None:
        cls = np.where(reg < 0, 0, np.where(reg > 0, 2, 1)).astype(np.int64)
    else:
        cls = np.asarray(cls).reshape(-1)[:N].astype(np.int64)
    if ids is None:
        ids = np.array(["sample_%d" % i for i in range(N)], dtype=object)
    else:
        ids = np.asarray(ids).reshape(-1)[:N]

    return dict(X=X, pad=pad, miss=miss, labels_reg=reg, labels_cls=cls,
                ids=ids, ann=ann, raw_text=raw_text, L=L, D=D, N=N,
                has_labels=has_labels)


# --------------------------------------------------------------------------
# 归一化统计（只用训练集，且只用"有效且非缺失"位置）
# --------------------------------------------------------------------------
def compute_norm_stats(split):
    stats = {}
    for m in MODALITIES:
        X = split["X"][m]
        good = (~split["pad"][m]) & (~split["miss"][m])
        v = X[good]
        if v.shape[0] < 2:
            mean = np.zeros(X.shape[-1], np.float32)
            std = np.ones(X.shape[-1], np.float32)
        else:
            mean = v.mean(axis=0).astype(np.float32)
            std = v.std(axis=0).astype(np.float32)
            std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        stats[m] = {"mean": mean, "std": std}
    return stats


def apply_norm(split, stats):
    for m in MODALITIES:
        split["X"][m] = (split["X"][m] - stats[m]["mean"]) / stats[m]["std"]
    return split


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class MoseiDataset(Dataset):
    def __init__(self, split, augment=False):
        self.s = split
        self.augment = augment
        self.n = split["N"]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        out = {}
        for m in MODALITIES:
            x = self.s["X"][m][i].copy()
            if self.augment and x.shape[0] > 8:
                # 时间抖动：随机丢掉 1 个时间步并前向填充（保留有效长度）
                if np.random.rand() < 0.3:
                    k = np.random.randint(0, x.shape[0] - 1)
                    x[k] = x[k - 1] if k > 0 else x[k + 1]
            out[m] = torch.from_numpy(x)
            out["miss_" + m] = torch.from_numpy(self.s["miss"][m][i].astype(np.float32))
            out["pad_" + m] = torch.from_numpy(self.s["pad"][m][i])
        out["y_reg"] = torch.tensor(float(self.s["labels_reg"][i]))
        out["y_cls"] = torch.tensor(int(self.s["labels_cls"][i]), dtype=torch.long)
        out["idx"] = torch.tensor(i, dtype=torch.long)
        return out


def collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def load_all(cfg):
    """返回 (train, valid, test) 三个已归一化的 split 字典。"""
    data = load_pickle(cfg.pkl_path)
    splits = {}
    for k in ("train", "valid", "test"):
        if k not in data:
            raise KeyError("特征文件缺少顶层键 '%s'，实际键: %s" % (k, list(data.keys())))
        splits[k] = build_split(data[k], limit=cfg.limit, min_zero_len=cfg.min_zero_len)
    stats = compute_norm_stats(splits["train"])
    for k in splits:
        apply_norm(splits[k], stats)
    for k in splits:
        s = splits[k]
        s["stats"] = stats
    return splits["train"], splits["valid"], splits["test"]


def load_any(pkl_path, limit=0, min_zero_len=2, norm_stats=None):
    """
    通用读取：附件2（含 train/valid/test）与附件3/4（单 split、无标签）都能读。
    返回 dict: split_name -> split
    """
    data = load_pickle(pkl_path)
    names = [k for k in ("train", "valid", "test")
             if isinstance(data, dict) and k in data]
    out = {}
    if names:
        for k in names:
            out[k] = build_split(data[k], limit=limit, min_zero_len=min_zero_len)
    else:
        out["test"] = build_split(data, limit=limit, min_zero_len=min_zero_len)
    if norm_stats is not None:
        for k in out:
            apply_norm(out[k], norm_stats)
            out[k]["stats"] = norm_stats
    return out


def slice_split(sp, n):
    """把 split 截断到前 n 条样本。"""
    if n <= 0 or n >= sp["N"]:
        return sp
    for m in MODALITIES:
        sp["X"][m] = sp["X"][m][:n]
        sp["pad"][m] = sp["pad"][m][:n]
        sp["miss"][m] = sp["miss"][m][:n]
    sp["labels_reg"] = sp["labels_reg"][:n]
    sp["labels_cls"] = sp["labels_cls"][:n]
    sp["ids"] = sp["ids"][:n]
    if sp.get("raw_text") is not None:
        sp["raw_text"] = np.asarray(sp["raw_text"])[:n]
    sp["N"] = n
    return sp


def concat_splits(splits):
    """把多个 split（附件3/4 的分文件）沿样本维拼接成一个 split。"""
    if len(splits) == 1:
        return splits[0]
    out = dict(X={}, pad={}, miss={})
    for m in MODALITIES:
        out["X"][m] = np.concatenate([s["X"][m] for s in splits], axis=0)
        out["pad"][m] = np.concatenate([s["pad"][m] for s in splits], axis=0)
        out["miss"][m] = np.concatenate([s["miss"][m] for s in splits], axis=0)
        out["pad"]["src_" + m] = splits[0]["pad"].get("src_" + m, "?")
    out["labels_reg"] = np.concatenate([s["labels_reg"] for s in splits])
    out["labels_cls"] = np.concatenate([s["labels_cls"] for s in splits])
    out["ids"] = np.concatenate([s["ids"] for s in splits])
    out["has_labels"] = all(bool(s.get("has_labels", True)) for s in splits)
    rts = [s.get("raw_text") for s in splits]
    out["raw_text"] = (np.concatenate([np.asarray(x) for x in rts])
                       if all(x is not None for x in rts) else None)
    out["ann"] = None
    out["N"] = sum(int(s["N"]) for s in splits)
    out["L"] = splits[0]["L"]
    out["D"] = splits[0]["D"]
    out["stats"] = splits[0].get("stats")
    return out


def _raw_of(d):
    """定位一个 pkl 内的字段字典（兼容顶层直接是字段 / 顶层是 train|valid|test|data）。"""
    raw = d
    if isinstance(d, dict):
        for k in ("test", "data", "train", "valid"):
            if k in d and isinstance(d[k], dict):
                raw = d[k]
                break
    return raw


def load_many(pkl_paths, limit=0, min_zero_len=2, norm_stats=None, verbose=True):
    """读取多个 pkl（附件3/4 的分文件组织方式）并拼接为单个 split。"""
    splits = []
    for p in pkl_paths:
        raw = _raw_of(load_pickle(p))
        sp = build_split(raw, limit=0, min_zero_len=min_zero_len)
        splits.append(sp)
        if verbose:
            print("  + %-34s N=%d" % (os.path.basename(p), sp["N"]))
    sp = concat_splits(splits)
    if norm_stats is not None:
        apply_norm(sp, norm_stats)
        sp["stats"] = norm_stats
    sp = slice_split(sp, limit)
    return sp


def describe(split, name="split"):
    """数据核查报告（写进论文'数据说明'小节）。
    填充率与真缺失率分开报告：附件2 的"零"主要是尾部填充，附件3 才是真缺失。"""
    lines = ["[%s] N=%d" % (name, split["N"])]
    for m in MODALITIES:
        X = split["X"][m]
        ln = (~split["pad"][m]).sum(axis=1)
        pad_ratio = float(split["pad"][m].mean())
        valid = ~split["pad"][m]
        miss_in_valid = float((split["miss"][m] & valid).mean())
        lines.append("  %-6s shape=%s  长度来源=%-8s 有效长度 min/mean/max=%d/%.1f/%d"
                     % (m, str(X.shape), split["pad"].get("src_" + m, "?"),
                        ln.min(), ln.mean(), ln.max()))
        lines.append("         填充率=%.3f 有效区间内缺失率=%.4f" % (pad_ratio, miss_in_valid))
    reg = split["labels_reg"]
    cls = split["labels_cls"]
    if split.get("has_labels", True):
        lines.append("  强度 label: min/mean/max=%.3f/%.3f/%.3f" % (reg.min(), reg.mean(), reg.max()))
        lines.append("  极性标签取值集合=%s" % sorted(set(cls.tolist())))
        lines.append("  极性分布 Neg/Neu/Pos = %d/%d/%d"
                     % ((cls == 0).sum(), (cls == 1).sum(), (cls == 2).sum()))
    else:
        lines.append("  （无标签：专项测试集）")
    return "\n".join(lines)
