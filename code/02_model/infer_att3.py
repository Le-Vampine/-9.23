# -*- coding: utf-8 -*-
"""
附件3（模态局部缺失）推理 + 结果 CSV 生成。

特性：
  * 自动检测每个样本每个模态的"连续全零区间"作为缺失（题目定义）
  * 多随机种子模型集成（5 个 ckpt 取均值），输出置信度
  * 训练时的归一化统计从 ckpt 内读取，保证口径一致

用法：
  python infer_att3.py --pkl E:\\数学建模\\data\\att3.pkl ^
      --ckpt_dir E:\\数学建模\\runs\\q2 --out_csv E:\\数学建模\\submission\\pred_att3.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import load_any, MoseiDataset, collate, missing_regions  # noqa: E402
from model import MRFNet                                                  # noqa: E402
from losses import polar_from_score                                       # noqa: E402
from train import prep, stack_masks, unstack_masks                        # noqa: E402

MODALITIES = ("text", "audio", "vision")
LABELS = ("Negative", "Neutral", "Positive")


def load_models(ckpt_paths, device):
    models = []
    for p in ckpt_paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        cfgd = ck["cfg"]
        m = MRFNet(dims=ck["dims"], hidden=cfgd["hidden"], nhead=cfgd["nhead"],
                   enc_layers=cfgd["enc_layers"], n_experts=cfgd["n_experts"],
                   dropout=cfgd["dropout"],
                   downs={"text": 1, "audio": 1, "vision": 1} if ck["version"] == "aligned"
                   else {"text": 1, "audio": 10, "vision": 10},
                   target_len=cfgd["target_len"]).to(device)
        m.load_state_dict({k: (v.float() if v.is_floating_point() else v)
                           for k, v in ck["state"].items()})
        m.eval()
        models.append((m, ck))
        print("[LOAD] %s (version=%s theta=%.2f)" % (os.path.basename(p), ck["version"], ck["theta"]))
    return models


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="附件3 特征文件")
    ap.add_argument("--ckpt_dir", default=None, help="从该目录自动收集 student_*.pt")
    ap.add_argument("--ckpt", nargs="*", default=None, help="显式指定 ckpt 列表")
    ap.add_argument("--out_csv", default=r"E:\数学建模\submission\pred_att3.csv")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    device = a.device
    ckpts = a.ckpt or sorted(glob.glob(os.path.join(a.ckpt_dir or ".", "student_*.pt")))
    if not ckpts:
        raise SystemExit("未找到任何 ckpt，请检查 --ckpt_dir / --ckpt")
    models = load_models(ckpts, device)
    stats = models[0][1]["stats"]
    theta = float(np.mean([ck["theta"] for _, ck in models]))

    # split: 附件3 通常只有单 split
    splits = load_any(a.pkl, norm_stats=stats)
    key = "test" if "test" in splits else list(splits.keys())[0]
    sp = splits[key]
    print("[DATA] %s: N=%d, dims=%s" % (key, sp["N"], sp["D"]))

    ds = MoseiDataset(sp, augment=False)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate)

    scores, probs = [], []
    rho_all, nregion, regions_all = [], [], []
    for batch in loader:
        b = prep(batch, device)
        M, P = stack_masks(b)
        unstack_masks(b, M)
        s_sum, p_sum = 0.0, 0.0
        for m, ck in models:
            mu, _, logits = m(b["x"], b["miss"], b["pad"])
            s_sum = s_sum + mu
            p_sum = p_sum + torch.softmax(logits, dim=-1)
        s_sum = s_sum / len(models)
        p_sum = p_sum / len(models)
        scores.append(s_sum.cpu().numpy())
        probs.append(p_sum.cpu().numpy())
        rho_all.append(np.stack([M[..., i].float().mean(1).cpu().numpy() for i in range(3)], -1))
        for j in range(M.size(0)):
            row = []
            for i in range(3):
                row += missing_regions(M[j, :, i].cpu().numpy().astype(bool))
            regions_all.append(row)
            nregion.append(len(row))

    score = np.concatenate(scores)
    prob = np.concatenate(probs)
    rho = np.concatenate(rho_all)
    pred_cls = polar_from_score(score, theta)

    import pandas as pd
    out = pd.DataFrame({
        "id": sp["ids"],
        "pred_label": [LABELS[c] for c in pred_cls],
        "pred_score": np.round(score, 4),
        "prob_neg": np.round(prob[:, 0], 4),
        "prob_neu": np.round(prob[:, 1], 4),
        "prob_pos": np.round(prob[:, 2], 4),
        "confidence": np.round(prob.max(axis=1), 4),
        "missing_text_ratio": np.round(rho[:, 0], 4),
        "missing_audio_ratio": np.round(rho[:, 1], 4),
        "missing_vision_ratio": np.round(rho[:, 2], 4),
        "n_missing_intervals": nregion,
    })
    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    out.to_csv(a.out_csv, index=False, encoding="utf-8-sig")
    print("[SAVE] %s  (rows=%d, theta=%.2f)" % (a.out_csv, len(out), theta))

    full = out.copy()
    full["missing_regions"] = [json.dumps(r, ensure_ascii=False) for r in regions_all]
    full_path = a.out_csv.replace(".csv", "_full.csv")
    full.to_csv(full_path, index=False, encoding="utf-8-sig")

    dist = out["pred_label"].value_counts().to_dict()
    print("[STAT] 极性分布: %s" % dist)
    print("[STAT] 平均缺失率 text/audio/vision = %.3f/%.3f/%.3f"
          % (rho[:, 0].mean(), rho[:, 1].mean(), rho[:, 2].mean()))
    print("[STAT] 强度 mean=%.3f std=%.3f min=%.3f max=%.3f"
          % (score.mean(), score.std(), score.min(), score.max()))


if __name__ == "__main__":
    main()
