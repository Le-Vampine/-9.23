# -*- coding: utf-8 -*-
r"""D4/D9：可解释性证据头（冻结骨干 + 并联解释头 + 解释性正则训练）。

设计（论文 §5.3"可解释性模型的网络结构、目标函数、训练方案与关键参数"）：
  骨干**完全冻结**并保持预测通路不变；证据头挂在其**编码器输出**（逐模态 50×d 表示）上，
  只用于产生解释：s^m_k = softmax_k(w_m^T tanh(W_m h^m_k + b_m))，
  模态作用度 π = softmax_m(u^T tanh(V h̄^m))，h̄^m = Σ_k s^m_k h^m_k。

目标函数（全部可微；用"软删除/软保留"把忠实性写成可训练的损失）：
  L = α_f [ -(t(X) − t(X; miss=s))² + (t(X) − t(X; miss=1−s))² ]
      + α_1 Σ_m ‖s^m‖_1 + α_2 Σ_m TV(s^m) + α_3 Σ_m (π_m − π_m^Shapley)²

  第一项：按 s 软删除后的预测变化越大越好（**忠实性**）；
  第二项：只保留 s 的软权重时的预测应接近全模态（**充分性**）；
  第三项/第四项：稀疏性与时间平滑；第五项（可选）：与精确 Shapley 的模态作用度对齐。

训练方案：只训证据头（骨干冻结、eval 模式），AdamW，lr 1e-3，30 轮，按验证集
          "忠实性 − 充分性偏差"加权目标早停；关键参数 α 在验证集上网格择优（默认给定值）。

用法：
  python train_evidence_head.py --epochs 12 --out ..\..\runs\q3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)
from xai_common import (MODALITIES, load_backbone, load_split, make_batch,     # noqa: E402
                        predict_batch, save_json, q3_meta, ROOT)


class EvidenceHead(nn.Module):
    """逐模态时间证据头 + 模态门控（并联，不接入预测通路）。"""

    def __init__(self, hidden, n_mod=3):
        super().__init__()
        self.hidden = hidden
        self.W = nn.ModuleDict({m: nn.Linear(hidden, hidden) for m in MODALITIES})
        self.w = nn.ParameterDict({m: nn.Parameter(torch.zeros(hidden)) for m in MODALITIES})
        self.V = nn.Linear(hidden, hidden)
        self.u = nn.Linear(hidden, 1)

    def forward(self, h, pad):
        """h: 模态->(B,L,H)；pad: 模态->(B,L)；返回 s（B,3,L）、pi（B,3）。"""
        S, pi_logit = [], []
        for m in MODALITIES:
            z = torch.tanh(self.W[m](h[m]))                  # (B,L,H)
            sc = torch.einsum("blh,h->bl", z, self.w[m])     # (B,L)
            sc = sc.masked_fill(pad[m], -1e9)
            s = torch.softmax(sc, dim=1)
            S.append(s)
            hbar = (s.unsqueeze(-1) * h[m]).sum(dim=1)       # (B,H)
            pi_logit.append(self.u(torch.tanh(self.V(hbar))).squeeze(-1))
        S = torch.stack(S, dim=1)                             # (B,3,L)
        pi = torch.softmax(torch.stack(pi_logit, dim=1), dim=1)
        return S, pi


def capture(mods, models):
    """抓取骨干各模态编码器输出 h[m] 的 hook（兼容 [(model, ckpt), ...] 与 [model, ...]）。"""
    store, handles = {}, []
    m0 = models[0][0] if isinstance(models[0], tuple) else models[0]
    for m in mods:
        def mk(name):
            def hook(module, inp, out):
                store[name] = out
            return hook
        handles.append(m0.enc[m].register_forward_hook(mk(m)))
    return store, handles


def forward_with_miss(models, b, miss_override):
    """用给定的 miss（可为软值）跑集成前向，返回 (score, prob)。x 同步按软掩码缩权。"""
    bb = dict(x={}, miss={}, pad={})
    for m in MODALITIES:
        bb["pad"][m] = b["pad"][m]
        bb["miss"][m] = miss_override[m]
        bad = (miss_override[m] + b["pad"][m].float()).clamp(0, 1).unsqueeze(-1)
        bb["x"][m] = b["x"][m] * (1.0 - bad)
    out = predict_batch(models, bb, target="mu")
    return out["score"], out["prob"]


def tv(s):
    return (s[:, :, 1:] - s[:, :, :-1]).abs().mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha_f", type=float, default=1.0)
    ap.add_argument("--alpha1", type=float, default=1e-3)
    ap.add_argument("--alpha2", type=float, default=1e-3)
    ap.add_argument("--alpha3", type=float, default=0.0,
                    help="与 Shapley 对齐项权重（>0 需先算训练集 Shapley）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "q3"))
    ap.add_argument("--ckpt_dir", default=os.path.join(ROOT, "runs", "ens_top2"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    models, stats, theta = load_backbone(a.ckpt_dir, a.device, verbose=False)
    for m, _ in models:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    tr, _ = load_split(os.path.join(ROOT, "附件2", "aligned_50.pkl"), stats, "train")
    va, _ = load_split(os.path.join(ROOT, "附件2", "aligned_50.pkl"), stats, "valid")
    if a.limit:
        from data_utils import slice_split
        tr = slice_split(tr, a.limit)
    hidden = models[0][1]["cfg"]["hidden"]
    print("[DATA] train N=%d  valid N=%d  hidden=%d" % (tr["N"], va["N"], hidden))

    head = EvidenceHead(hidden).to(a.device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    hist = []
    best = dict(score=-1e9, epoch=-1, state=None)

    for ep in range(1, a.epochs + 1):
        head.train()
        tot, nb = 0.0, 0
        perm = np.random.permutation(tr["N"])
        for i0 in range(0, tr["N"], a.batch_size):
            idxs = [int(i) for i in perm[i0:i0 + a.batch_size]]
            b = make_batch(tr, idxs, a.device)
            store, handles = capture(MODALITIES, models)
            with torch.no_grad():
                t_full, _ = forward_with_miss(models, b,
                                             {m: b["miss"][m].float() for m in MODALITIES})
            for hd in handles:
                hd.remove()
            S, pi = head(store, b["pad"])
            miss_soft = {m: (S[:, k]).clamp(0, 1) for k, m in enumerate(MODALITIES)}
            keep_soft = {m: (1.0 - S[:, k]).clamp(0, 1) for k, m in enumerate(MODALITIES)}
            t_del, _ = forward_with_miss(models, b, miss_soft)
            t_keep, _ = forward_with_miss(models, b, keep_soft)
            l_f = -((t_full - t_del) ** 2).mean() + ((t_full - t_keep) ** 2).mean()
            l1 = sum(S[:, k].mean() for k in range(3)) / 3.0
            l_tv = sum(tv(S[:, k:k + 1]) for k in range(3)) / 3.0
            loss = a.alpha_f * l_f + a.alpha1 * l1 + a.alpha2 * l_tv
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        # ---- 验证集评估：忠实性 − 充分性偏差（越大越好） ----
        head.eval()
        compr, suff = [], []
        with torch.no_grad():
            for i0 in range(0, min(va["N"], 256), 64):
                idxs = list(range(i0, min(i0 + 64, va["N"])))
                b = make_batch(va, idxs, a.device)
                store, handles = capture(MODALITIES, models)
                t_full, _ = forward_with_miss(models, b,
                                              {m: b["miss"][m].float() for m in MODALITIES})
                for hd in handles:
                    hd.remove()
                S, pi = head(store, b["pad"])
                t_del, _ = forward_with_miss(models, b, {m: S[:, k] for k, m in enumerate(MODALITIES)})
                t_keep, _ = forward_with_miss(models, b,
                                              {m: (1 - S[:, k]) for k, m in enumerate(MODALITIES)})
                compr.append((t_full - t_del).abs().mean().item())
                suff.append((t_full - t_keep).abs().mean().item())
        c, s = float(np.mean(compr)), float(np.mean(suff))
        sc = c - s
        hist.append(dict(epoch=ep, loss=round(tot / max(nb, 1), 5), valid_compr=round(c, 5),
                         valid_suff=round(s, 5), score=round(sc, 5)))
        print("  [ep%02d] loss=%.5f  valid 忠实性=%.5f 充分性偏差=%.5f 目标=%.5f"
              % (ep, tot / max(nb, 1), c, s, sc))
        if sc > best["score"]:
            best = dict(score=sc, epoch=ep,
                        state={k: v.detach().clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best["state"])
    torch.save(dict(state=head.state_dict(), cfg=dict(hidden=hidden, alpha_f=a.alpha_f,
                                                      alpha1=a.alpha1, alpha2=a.alpha2,
                                                      alpha3=a.alpha3, epochs=a.epochs,
                                                      lr=a.lr),
                    meta=q3_meta(a.ckpt_dir, models, theta, dict(train_n=tr["N"])),
                    history=hist, best_epoch=best["epoch"]),
               os.path.join(a.out, "q3_evidence_head.pt"))
    save_json(os.path.join(a.out, "q3_evidence_head_report.json"),
              dict(best_epoch=best["epoch"], best_score=best["score"], history=hist,
                   args=dict(alpha_f=a.alpha_f, alpha1=a.alpha1, alpha2=a.alpha2,
                             alpha3=a.alpha3, epochs=a.epochs, lr=a.lr, batch=a.batch_size)))
    print("[SAVE] runs/q3/q3_evidence_head.pt（最佳轮 %d，目标 %.5f）" % (best["epoch"], best["score"]))
    print("EVIDENCE_HEAD_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
