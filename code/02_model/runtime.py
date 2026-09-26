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
from model import MRFNet, PlainFusion                    # noqa: E402
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
        m_kind = ck.get("model_kind", "mrf")
        if m_kind == "plain":
            m = PlainFusion(dims=ck["dims"], hidden=c["hidden"], dropout=c["dropout"],
                            target_len=c["target_len"],
                            use_indicator=not c.get("ablate_mask_indicator", False)).to(device)
        else:
            m = MRFNet(dims=ck["dims"], hidden=c["hidden"], nhead=c["nhead"],
                       enc_layers=c["enc_layers"], n_experts=c["n_experts"],
                       dropout=c["dropout"], downs=downs,
                       target_len=c["target_len"],
                       mag_levels=int(ck.get("mag_levels", 0) or 0),
                       ms_pool=bool(ck.get("ms_pool", False)),
                       ms_windows=tuple(ck.get("ms_windows", (5, 25, 50))),
                       use_pol=bool(ck.get("use_pol", False))).to(device)
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
                  scenario=None, unaware=False, seed=1234, aux_keys=None):
    """
    对单个 split 做集成推理。
    scenario = (missing_type, position, rho) 时覆盖缺失掩码，用于缺失场景评测。
    unaware=True 时把掩码置零喂给模型（模拟"未做缺失处理的常规模型"，作为 A0 基线）。

    额外输出（后处理分析用，不改变默认行为）：
      member_mu / member_logvar / score_std / logvar  —— 不确定性拆分
      aux_keys=(...)：按名取出模型 aux 字典中的张量（如 "logits_mag"）并列名返回，
                     取不到的键会被跳过（不报错）。
    """
    aux_keys = tuple(aux_keys or ())
    ds = MoseiDataset(split, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    S, P_ = [], []
    RH = []
    MU, LV = [], []          # 各成员的回归分数 / logvar（不确定性分析用）
    AX = {k: [] for k in aux_keys}
    for batch in loader:
        b = prep(batch, device)
        M, Pd = stack_masks(b)
        if scenario is not None:
            M = make_scenario_masks(M, Pd, scenario[0], scenario[1], scenario[2], seed=seed)
        unstack_masks(b, M)
        feed = ({m: torch.zeros_like(b["miss"][m]) for m in MODALITIES} if unaware else b["miss"])
        s, c = None, None
        mus, lvs = [], []
        axb = {k: [] for k in aux_keys}                   # 本 batch 各成员的 aux 张量
        for m, _ in models:
            if aux_keys:
                mu, logvar, logits, ax = m(b["x"], feed, b["pad"], return_aux=True)
                for k in aux_keys:
                    v = ax.get(k)
                    if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == mu.size(0):
                        axb[k].append(v)
            else:
                mu, logvar, logits = m(b["x"], feed, b["pad"])
            mus.append(mu)
            lvs.append(logvar)
            s = mu if s is None else s + mu
            c = torch.softmax(logits, -1) if c is None else c + torch.softmax(logits, -1)
        S.append((s / len(models)).cpu().numpy())
        P_.append((c / len(models)).cpu().numpy())
        RH.append(batch_missing_ratios(M, Pd).cpu().numpy())
        MU.append(torch.stack(mus, 0).cpu().numpy())      # (K, B)
        LV.append(torch.stack(lvs, 0).cpu().numpy())      # (K, B)
        for k in aux_keys:                                # 成员均值，语义与 score 一致
            if axb[k]:
                AX[k].append(torch.stack(axb[k], 0).mean(0).detach().cpu().numpy())
    # 不确定性拆分：σ_model（logvar 头，偶然） / σ_ens（成员间分歧，认知）
    member_mu = np.concatenate(MU, axis=1).T              # (N, K)
    member_lv = np.concatenate(LV, axis=1).T              # (N, K)
    score_std = (member_mu.std(axis=1) if member_mu.shape[1] > 1
                 else np.zeros(len(member_mu)))
    out = dict(score=np.concatenate(S), prob=np.concatenate(P_),
               rho=np.concatenate(RH), ids=np.asarray(split["ids"]),
               y_reg=np.asarray(split["labels_reg"]), y_cls=np.asarray(split["labels_cls"]),
               has_labels=bool(split.get("has_labels", True)),
               length=np.stack([(~split["pad"][m]).sum(axis=1) for m in MODALITIES], axis=1),
               member_mu=member_mu, member_logvar=member_lv,
               score_std=score_std, logvar=member_lv.mean(axis=1))
    for k in aux_keys:
        if AX[k]:
            out[k] = np.concatenate(AX[k], axis=0)
    return out


def polarize(score, theta):
    from losses import polar_from_score
    return polar_from_score(score, theta)
