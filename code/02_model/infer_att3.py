# -*- coding: utf-8 -*-
"""
附件3（模态局部缺失）推理 + 结果 CSV 生成。

特性：
  * 自动检测每个样本每个模态的"连续全零区间"作为缺失（题目定义）
  * 多随机种子模型集成（多个 ckpt 取均值），输出置信度
  * 训练时的归一化统计从 ckpt 内读取，保证口径一致

缺失率口径（**规范定义 D1**，与 `paper_tables/missing_profile.{csv,md}` 完全一致）：
  * 缺失判定：连续 ≥2 帧全零（题目原文「部分**连续**时间段不可用」）
  * 统计区间：`[0, 末个非零帧]`；首部零帧计入缺失，末个非零帧之后视为尾部填充、不计入
  * `missing_*_ratio` = 缺失帧数 / 统计区间长度
  * 整条全零的样本记为**整模态缺失**，ratio = 1.0，`missing_regions` 记 `[[0, L]]`
  * 故本文件的 `missing_*_ratio` 与 `n_missing_intervals` 可与 `missing_profile.csv`
    直接交叉核对（旧版曾用固定总长 50 作分母，数值偏小约一半，已修）

CSV 口径说明（两套极性口径并列写出）：
  * `pred_label`（**主口径**）：分类头 argmax —— 直接由附件2 `classification_labels` 监督，
    与赛题"分类任务用 Accuracy/F1 评价"的训练目标一致，且不受强度分数尺度（收缩/校准）影响；
  * `pred_label_theta`（对照）：由连续强度分数按阈值判定
    （score < −θ → Negative；|score| ≤ θ → Neutral；score > θ → Positive），θ 在验证集上选出；
  * `confidence`：分类头最大概率（与主口径同源）；
  * `confidence_theta`：分类头在 θ 口径判定类别上的概率（仅作对照）。

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
from model import MRFNet, PlainFusion                                     # noqa: E402
from losses import polar_from_score                                       # noqa: E402
from train import prep, stack_masks, unstack_masks                        # noqa: E402

MODALITIES = ("text", "audio", "vision")
LABELS = ("Negative", "Neutral", "Positive")


def load_models(ckpt_paths, device):
    models = []
    for p in ckpt_paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        cfgd = ck["cfg"]
        if ck.get("model_kind", "mrf") == "plain":
            m = PlainFusion(dims=ck["dims"], hidden=cfgd["hidden"], dropout=cfgd["dropout"],
                            target_len=cfgd["target_len"],
                            use_indicator=not cfgd.get("ablate_mask_indicator", False)).to(device)
        else:
            m = MRFNet(dims=ck["dims"], hidden=cfgd["hidden"], nhead=cfgd["nhead"],
                       enc_layers=cfgd["enc_layers"], n_experts=cfgd["n_experts"],
                       dropout=cfgd["dropout"],
                       downs={"text": 1, "audio": 1, "vision": 1} if ck["version"] == "aligned"
                       else {"text": 1, "audio": 10, "vision": 10},
                       target_len=cfgd["target_len"],
                       mag_levels=int(ck.get("mag_levels", 0) or 0),
                       ms_pool=bool(ck.get("ms_pool", False)),
                       ms_windows=tuple(ck.get("ms_windows", (5, 25, 50))),
                       use_pol=bool(ck.get("use_pol", False))).to(device)
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
    # 后处理参数（与 report_p0.py 在 valid 上选出的结果保持一致；默认不改动原始输出）
    ap.add_argument("--fusion_alpha", type=float, default=0.0,
                    help="双头融合系数 α：score' = score + α·(P(pos) − P(neg))；0 表示不融合")
    ap.add_argument("--calib_a", type=float, default=None, help="仿射校准斜率 a（valid 上拟合）")
    ap.add_argument("--calib_b", type=float, default=None, help="仿射校准截距 b（valid 上拟合）")
    ap.add_argument("--theta", type=float, default=None, help="覆盖 ckpt 内的决策阈值 θ")
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
        # ---- 缺失率：规范口径 D1（分母 = 有效区间长度 = [0, 末个非零帧]）----
        # P 为 pad 掩码，~P 恰好就是 D1 的统计区间（build_split 由"末个非零帧"推断长度）。
        # 旧实现用 M[..., i].mean(1)（固定除以总长 50），会把缺失率人为稀释约 2 倍，
        # 与 paper_tables/missing_profile.csv 对不上号。
        span = (~P).sum(1).clamp(min=1).float()            # (B,3) 统计区间长度
        mcnt = M.float().sum(1)                            # (B,3) 缺失帧数
        rho_b = mcnt / span
        # 兜底：整条全零的样本，有效区间退化为安全位 1 帧，掩码判不出缺失
        # → 记为整模态缺失，rho = 1.0（附件4 视觉有 1 条这样的样本）
        deg = (span <= 1.0)
        rho_b = torch.where(deg, torch.ones_like(rho_b), rho_b)
        rho_all.append(rho_b.cpu().numpy())
        for j in range(M.size(0)):
            row = []
            for i in range(3):
                if bool(deg[j, i]):                        # 整模态缺失：整段不可用
                    row.append([0, int(M.size(1))])
                    continue
                row += missing_regions(M[j, :, i].cpu().numpy().astype(bool))
            regions_all.append(row)
            nregion.append(len(row))

    score_raw = np.concatenate(scores)
    prob = np.concatenate(probs)
    rho = np.concatenate(rho_all)

    # 后处理：融合 → 仿射校准（参数均由 report_p0.py 在验证集上选出，默认不启用）
    score = score_raw.copy()
    if a.fusion_alpha:
        score = score + float(a.fusion_alpha) * (prob[:, 2] - prob[:, 0])
    if a.calib_a is not None or a.calib_b is not None:
        ca = 1.0 if a.calib_a is None else float(a.calib_a)
        cb = 0.0 if a.calib_b is None else float(a.calib_b)
        score = ca * score + cb
    if a.theta is not None:
        theta = float(a.theta)
    print("[POST] fusion_alpha=%.3f calib_a=%s calib_b=%s theta=%.2f"
          % (a.fusion_alpha, a.calib_a, a.calib_b, theta))

    pred_theta = polar_from_score(score, theta)        # θ 阈值口径（对照）
    pred_head = prob.argmax(axis=1)                    # 分类头口径（**主口径**）
    n = len(score)

    import pandas as pd
    out = pd.DataFrame({
        "id": sp["ids"],
        "pred_label": [LABELS[c] for c in pred_head],
        "pred_label_theta": [LABELS[c] for c in pred_theta],
        "pred_score": np.round(score, 4),
        "pred_score_raw": np.round(score_raw, 4),
        "prob_neg": np.round(prob[:, 0], 4),
        "prob_neu": np.round(prob[:, 1], 4),
        "prob_pos": np.round(prob[:, 2], 4),
        "confidence": np.round(prob.max(axis=1), 4),
        "confidence_theta": np.round(prob[np.arange(n), pred_theta], 4),
        "theta": round(float(theta), 3),
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
    print("[STAT] 极性分布(主口径=分类头): %s" % dist)
    print("[STAT] 极性分布(θ 口径对照): %s" % out["pred_label_theta"].value_counts().to_dict())
    print("[STAT] 两口径一致率: %.3f" % float((pred_head == pred_theta).mean()))
    print("[STAT] 平均缺失率 text/audio/vision = %.3f/%.3f/%.3f"
          % (rho[:, 0].mean(), rho[:, 1].mean(), rho[:, 2].mean()))
    print("[STAT] 强度 mean=%.3f std=%.3f min=%.3f max=%.3f"
          % (score.mean(), score.std(), score.min(), score.max()))


if __name__ == "__main__":
    main()
