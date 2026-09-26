"""缺失模式探针：不同缺失配置下的**指标**与**模型内部权重**。

回答两个问题：
  (1) 不同缺失模式下模型的表现差多少？（MAE / ACC3 / Acc-2 / Acc-7 / Macro-F1）
  (2) 处理不同缺失时模型的“权重”是不是不一样？
      - 专家门控 g ∈ R^E（输入含 7 维可用性特征）
      - 代理令牌门控 beta_m（该模态被代理令牌替换的比例，缺失越多应越大）
      - 帧级注意力 alpha 的集中度（熵）

Acc-2 / Acc-7 按 MOSEI 文献口径计算，以便与参考文献直接对比：
  Acc-2 : 二分类（负/正 与 负/非负 两种设置）
  Acc-7 : round(ŷ) == round(y)，7 个整数类 {-3..3}（等距分箱，不是按权重）

用法：
  python missing_probe.py --ckpt_dir ../../runs/exp_d4 --out ../../runs/exp_d4/probe.md
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(errors="replace")

from data_utils import MoseiDataset, collate                    # noqa: E402
from missing_sim import make_scenario_masks                     # noqa: E402
from runtime import find_ckpts, load_ensemble                   # noqa: E402
from train import prep, stack_masks, unstack_masks              # noqa: E402

MODALITIES = ("text", "audio", "vision")

SCEN = [
    ("full",             None),
    # —— 附件3 真实场景：**文本永不缺失**，只有 audio+vision 局部缺失 ——
    ("AV局部 rho=0.18(实测均值)", ("audio+vision", "random", 0.18)),
    ("AV局部 rho=0.33(实测p90)",  ("audio+vision", "random", 0.33)),
    ("AV局部 rho=0.48(实测max)",  ("audio+vision", "random", 0.48)),
    # —— 整模态缺失（附件3 不存在，仅作鲁棒性边界分析）——
    ("缺文本",            ("text", "middle", 1.0)),
    ("缺语音",            ("audio", "middle", 1.0)),
    ("缺视觉",            ("vision", "middle", 1.0)),
    ("缺文本+语音",        ("text+audio", "middle", 1.0)),
    ("缺文本+视觉",        ("text+vision", "middle", 1.0)),
    ("缺语音+视觉",        ("audio+vision", "middle", 1.0)),
    # —— 对照：含文本的历史口径 ——
    ("TA局部 rho=0.30(旧口径)", ("text+audio", "random", 0.30)),
]


def acc_of_cls(y, p):
    return float((y == p).mean())


def macro_f1(y, p, k=3):
    out = []
    for c in range(k):
        tp = float(((y == c) & (p == c)).sum())
        fp = float(((y != c) & (p == c)).sum())
        fn = float(((y == c) & (p != c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(out))


def acc2(y_reg, score):
    """MOSEI 文献口径的 Acc-2：负/正（剔除中性）与 负/非负。"""
    m = np.abs(y_reg) > 1e-6
    a = float(((y_reg[m] > 0) == (score[m] > 0)).mean()) if m.sum() else float("nan")
    b = float(((y_reg < 0) == (score < 0)).mean())
    return a, b


def acc7(y_reg, score):
    return float((np.round(y_reg) == np.round(score)).mean())


@torch.no_grad()
def probe(models, split, scenario, device="cpu"):
    ds = MoseiDataset(split, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
    S, P, YR, YC, G, B, A, PL = [], [], [], [], [], [], [], []
    for batch in loader:
        b = prep(batch, device)
        M, Pd = stack_masks(b)
        if scenario is not None:
            M = make_scenario_masks(M, Pd, scenario[0], scenario[1], scenario[2], seed=1234)
        unstack_masks(b, M)
        s = c = g_acc = None
        bt = {m: [] for m in MODALITIES}
        al = []
        pl = []
        for mdl, _ in models:
            mu, _, logits, aux = mdl(b["x"], b["miss"], b["pad"], return_aux=True)
            s = mu if s is None else s + mu
            c = torch.softmax(logits, -1) if c is None else c + torch.softmax(logits, -1)
            if "pol" in aux:
                pl.append(torch.sigmoid(aux["pol"]))
            g_acc = aux["gate"] if g_acc is None else g_acc + aux["gate"]
            for m in MODALITIES:
                bt[m].append(aux["beta"][m])
            al.append(aux["alpha"].squeeze(-1))
        n = len(models)
        S.append((s / n).cpu().numpy())
        P.append((c / n).cpu().numpy())
        G.append((g_acc / n).cpu().numpy())
        B.append({m: torch.stack(bt[m], dim=0).mean(dim=0).cpu().numpy()
                  for m in MODALITIES})
        A.append(torch.stack(al, dim=0).mean(dim=0).cpu().numpy())
        if pl:
            PL.append(torch.stack(pl, dim=0).mean(dim=0).cpu().numpy())
        YR.append(b["y_reg"].cpu().numpy())
        YC.append(b["y_cls"].cpu().numpy())
    y_reg = np.concatenate(YR)
    y_cls = np.concatenate(YC)
    score = np.concatenate(S)
    prob = np.concatenate(P)
    head = prob.argmax(-1)
    gate = np.concatenate(G)
    beta = {m: np.concatenate([bb[m] for bb in B]) for m in MODALITIES}
    alpha = np.concatenate(A)
    a2_pos, a2_non = acc2(y_reg, score)
    pol = np.concatenate(PL) if PL else None
    return dict(
        mae=float(np.abs(score - y_reg).mean()),
        acc3_head=acc_of_cls(y_cls, head),
        acc3_theta=acc_of_cls(y_cls, np.where(score < 0, 0, np.where(score > 0, 2, 1))),
        macro_f1=macro_f1(y_cls, head),
        acc2_negpos=a2_pos, acc2_neg_nonneg=a2_non,
        acc2_headprob=(float(((y_reg < 0) == (prob[:, 0] > prob[:, 1] + prob[:, 2])).mean())
                       if prob.shape[1] == 3 else float("nan")),
        acc2_polhead=(float(((y_reg < 0) == (pol < 0.5)).mean()) if pol is not None
                      else float("nan")),
        acc7=acc7(y_reg, score),
        gate_mean=gate.mean(axis=0).round(4).tolist(),
        gate_std=gate.std(axis=0).round(4).tolist(),
        beta_mean={m: round(float(beta[m].mean()), 4) for m in MODALITIES},
        alpha_entropy=float(-(alpha * np.log(alpha + 1e-12)).sum(axis=1).mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from data_utils import load_all
    from config import Config
    cfg = Config()
    cfg.version = "aligned"
    data = load_all(cfg)
    tr, va, te = data
    split = va if args.split == "valid" else te
    models = load_ensemble(find_ckpts(args.ckpt_dir), verbose=False)
    print("[LOAD] %d ckpt(s) from %s" % (len(models), args.ckpt_dir))

    rows = []
    for name, scen in SCEN:
        r = probe(models, split, scen)
        r["scenario"] = name
        rows.append(r)
        print("%-24s MAE=%.4f ACC3=%.4f Acc2=%.4f/%.4f(pol %.4f) Acc7=%.4f F1=%.4f | g=%s beta=%s" %
              (name, r["mae"], r["acc3_head"], r["acc2_negpos"], r["acc2_neg_nonneg"],
               r["acc2_polhead"], r["acc7"], r["macro_f1"],
               r["gate_mean"], r["beta_mean"]))

    md = ["# 缺失模式探针（%s, %s）" % (os.path.basename(args.ckpt_dir.rstrip("\\/")), args.split), "",
          "| 场景 | MAE | ACC(三类/头) | Acc-2(负/正) | Acc-2(负/非负) | Acc-7 | Macro-F1 |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |" %
                  (r["scenario"], r["mae"], r["acc3_head"], r["acc2_negpos"],
                   r["acc2_neg_nonneg"], r["acc7"], r["macro_f1"]))
    md += ["", "## 内部权重（不同缺失模式是否不同）", "",
           "| 场景 | 专家门控 g（均值） | 专家门控 g（标准差） | beta 文本 | beta 语音 | beta 视觉 | 帧注意力熵 |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %s | %s | %.3f | %.3f | %.3f | %.2f |" %
                  (r["scenario"], r["gate_mean"], r["gate_std"],
                   r["beta_mean"]["text"], r["beta_mean"]["audio"],
                   r["beta_mean"]["vision"], r["alpha_entropy"]))
    txt = "\n".join(md)
    print("\n" + txt)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)
        with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print("\n[SAVE] %s" % args.out)


if __name__ == "__main__":
    main()
