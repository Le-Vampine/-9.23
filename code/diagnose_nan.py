# -*- coding: utf-8 -*-
r"""定位训练 NaN：逐阶段检查数据/裁剪/前向输出是否出现 NaN。"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "02_model"))
from config import Config                                     # noqa: E402
from data_utils import load_all, MoseiDataset, collate         # noqa: E402
from missing_sim import crop_batch, inject_missing             # noqa: E402
from train import prep, stack_masks, unstack_masks, build_model  # noqa: E402

MODS = ("text", "audio", "vision")


def nan_of(t):
    return bool(torch.isnan(t).any().item())


def main():
    cfg = Config(version="aligned", limit=384, hidden=128, batch_size=64, device="cpu")
    tr, va, te = load_all(cfg)
    print("[数据] NaN:%s  Inf:%s" % (
        {m: bool(np.isnan(tr["X"][m]).any()) for m in MODS},
        {m: bool(np.isinf(tr["X"][m]).any()) for m in MODS}))
    print("[数据] 归一化后 padding 区是否非零: %s"
          % {"text": float((tr["X"]["text"][tr["pad"]["text"]].__abs__().sum(-1) > 1e-6).mean())})

    dl = DataLoader(MoseiDataset(tr), batch_size=64, shuffle=False, collate_fn=collate)
    batch = next(iter(dl))
    b = prep(batch, "cpu")
    x_full = {m: b["x"][m].clone() for m in MODS}
    print("\n[原始 batch] x NaN=%s pad 均值=%s"
          % ({m: nan_of(b["x"][m]) for m in MODS},
             {m: round(float(b["pad"][m].float().mean()), 3) for m in MODS}))

    dims = {m: tr["D"][m] for m in MODS}
    model = build_model(cfg, dims)

    # ---------- (1) 不做裁剪 ----------
    b1 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in b.items()}
    for m in MODS:
        b1["x"][m] = b["x"][m].clone()
        b1["pad"][m] = b["pad"][m].clone()
        b1["miss"][m] = b["miss"][m].clone()
    M, P = stack_masks(b1)
    unstack_masks(b1, M)
    mu, lv, lg, aux = model(b1["x"], b1["miss"], b1["pad"], return_aux=True)
    print("\n[(1) 无裁剪] z NaN=%s mu NaN=%s rec NaN=%s logits NaN=%s"
          % (nan_of(aux["z"]), nan_of(mu), {m: nan_of(aux["rec"][m]) for m in MODS}, nan_of(lg)))

    # ---------- (2) 做裁剪（当前训练配置） ----------
    b2 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in b.items()}
    for m in MODS:
        b2["x"][m] = b["x"][m].clone()
        b2["pad"][m] = b["pad"][m].clone()
        b2["miss"][m] = b["miss"][m].clone()
    crop_batch(b2, np.random.default_rng(0), min_keep=0.7)
    M2, P2 = stack_masks(b2)
    unstack_masks(b2, M2)
    mu2, lv2, lg2, aux2 = model(b2["x"], b2["miss"], b2["pad"], return_aux=True)
    print("[(2) 裁剪后] pad 均值=%s  z NaN=%s mu NaN=%s rec NaN=%s"
          % ({m: round(float(b2["pad"][m].float().mean()), 3) for m in MODS},
             nan_of(aux2["z"]), nan_of(mu2), {m: nan_of(aux2["rec"][m]) for m in MODS}))

    # ---------- (3) 教师式输入（未裁剪特征 + 裁剪 pad） ----------
    mu3, lv3, lg3, aux3 = model(x_full, {m: torch.zeros_like(b2["miss"][m]) for m in MODS},
                                b2["pad"], return_aux=True)
    print("[(3) 教师式输入] z NaN=%s mu NaN=%s rec NaN=%s"
          % (nan_of(aux3["z"]), nan_of(mu3), {m: nan_of(aux3["rec"][m]) for m in MODS}))

    # ---------- (4) 多段缺失注入 ----------
    b4 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in b.items()}
    for m in MODS:
        b4["x"][m] = b["x"][m].clone()
        b4["pad"][m] = b["pad"][m].clone()
        b4["miss"][m] = b["miss"][m].clone()
    M4, P4 = stack_masks(b4)
    M5, recs = inject_missing(M4, P4, 0.1, 0.25, "mixed",
                              ("middle", "middle", "head", "tail", "random"),
                              rng=np.random.default_rng(0), n_intervals=(1, 3))
    unstack_masks(b4, M5)
    mu4, lv4, lg4, aux4 = model(b4["x"], b4["miss"], b4["pad"], return_aux=True)
    print("[(4) 多段缺失] 注入记录示例=%s" % (recs[0][:3],))
    print("              z NaN=%s mu NaN=%s rec NaN=%s"
          % (nan_of(aux4["z"]), nan_of(mu4), {m: nan_of(aux4["rec"][m]) for m in MODS}))

    # ---------- (5) 反向是否产生 NaN 梯度 ----------
    from losses import total_loss
    b5 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in b.items()}
    for m in MODS:
        b5["x"][m] = b["x"][m].clone()
        b5["pad"][m] = b["pad"][m].clone()
        b5["miss"][m] = b["miss"][m].clone()
    M6 = M2
    unstack_masks(b5, M6)
    mu5, lv5, lg5, aux5 = model(b5["x"], b5["miss"], b5["pad"], return_aux=True)
    out = dict(aux5); out.update(mu=mu5, logvar=lv5, logits=lg5)
    cfg.cls_weight = [1.17, 1.49, 0.68]
    loss, det = total_loss(out, b5, cfg)
    print("\n[(5) 损失] loss=%s 明细=%s" % (float(loss.detach()), det))
    loss.backward()
    gmax, gnan = 0.0, 0
    for n, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                gnan += 1
                print("   [NaN 梯度] %s" % n)
            gmax = max(gmax, float(p.grad.abs().max()))
    print("    梯度最大绝对值=%.4g  NaN 参数数=%d" % (gmax, gnan))


if __name__ == "__main__":
    main()
