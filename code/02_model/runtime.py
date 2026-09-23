# -*- coding: utf-8 -*-
"""
共享推理运行时：加载多随机种子模型集成 + 对任意 split 推理（供 viz/error_analysis 复用）。
"""
import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import MoseiDataset, collate          # noqa: E402
from missing_sim import make_scenario_masks, batch_missing_ratios   # noqa: E402
from model import MRFNet                               # noqa: E402
from train import prep, stack_masks, unstack_masks     # noqa: E402

MODALITIES = ("text", "audio", "vision")


def find_ckpts(ckpt_dir, pattern="student_*.pt"):
    return sorted(glob.glob(os.path.join(ckpt_dir, pattern)))


def load_ensemble(ckpt_paths, device="cpu", verbose=True):
    """返回 [(model, ckpt_dict), ...]；自动兼容 float16 存储的权重。"""
    models = []
    for p in ckpt_paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        c = ck["cfg"]
        ver = ck.get("version", "aligned")
        downs = ({"text": 1, "audio": 1, "vision": 1} if ver == "aligned"
                 else {"text": 1, "audio": 10, "vision": 10})
        m = MRFNet(dims=ck["dims"], hidden=c["hidden"], nhead=c["nhead"],
                   enc_layers=c["enc_layers"], n_experts=c["n_experts"],
                   dropout=c["dropout"], downs=downs,
                   target_len=c["target_len"]).to(device)
        m.load_state_dict({k: (v.float() if v.is_floating_point() else v)
                           for k, v in ck["state"].items()})
        m.eval()
        models.append((m, ck))
        if verbose:
            print("[LOAD] %-28s version=%s theta=%.2f seed=%s"
                  % (os.path.basename(p), ver, ck["theta"], ck.get("seed", "-")))
    return models


@torch.no_grad()
def predict_split(models, split, batch_size=64, device="cpu",
                  scenario=None, unaware=False, seed=1234):
    """
    对单个 split 做集成推理。
    scenario = (missing_type, position, rho) 时覆盖缺失掩码，用于缺失场景评测。
    unaware=True 时把掩码置零喂给模型（模拟"未做缺失处理的常规模型"，作为 A0 基线）。
    """
    ds = MoseiDataset(split, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    S, P_ = [], []
    RH = []
    for batch in loader:
        b = prep(batch, device)
        M, Pd = stack_masks(b)
        if scenario is not None:
            M = make_scenario_masks(M, Pd, scenario[0], scenario[1], scenario[2], seed=seed)
        unstack_masks(b, M)
        feed = ({m: torch.zeros_like(b["miss"][m]) for m in MODALITIES} if unaware else b["miss"])
        s, c = None, None
        for m, _ in models:
            mu, _, logits = m(b["x"], feed, b["pad"])
            s = mu if s is None else s + mu
            c = torch.softmax(logits, -1) if c is None else c + torch.softmax(logits, -1)
        S.append((s / len(models)).cpu().numpy())
        P_.append((c / len(models)).cpu().numpy())
        RH.append(batch_missing_ratios(M, Pd).cpu().numpy())
    return dict(score=np.concatenate(S), prob=np.concatenate(P_),
                rho=np.concatenate(RH), ids=np.asarray(split["ids"]),
                y_reg=np.asarray(split["labels_reg"]), y_cls=np.asarray(split["labels_cls"]),
                has_labels=bool(split.get("has_labels", True)),
                length=np.stack([(~split["pad"][m]).sum(axis=1) for m in MODALITIES], axis=1))


def polarize(score, theta):
    from losses import polar_from_score
    return polar_from_score(score, theta)
