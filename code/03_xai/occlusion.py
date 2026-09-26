# -*- coding: utf-8 -*-
r"""S3：模型无关的遮挡重要性（作为解释的验证基准）。

定义：把模态 m 的窗口 [start, start+w) 用**相邻段**替换（保持序列长度与形态，
      避免"删掉一段导致长度变短"引入的伪影），测量目标标量的变化
          d_k = t(X) - t(X ∖ window_k),
      位置 k 的重要性取覆盖它的所有窗口的 |d| 均值。

为什么用相邻段替换而不是置零：置零在归一化空间里等价于"该段取训练均值"，
      会同时改变"语义内容"与"是否存在信息"两件事；相邻段替换只改变内容，
      更接近"如果这段时间说的是别的内容"的反事实，且对输出的扰动量级可解释。

用法：
  python occlusion.py --pkl <pkl> --name att4 --width 3 --stride 1 --out_dir <runs/q3>
  python occlusion.py --pkl ..\\附件2\\aligned_50.pkl --split valid --name valid --stride 2
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)
from xai_common import (MODALITIES, load_backbone, load_split, make_batch,    # noqa: E402
                        predict_batch, occlude_window, save_npz, save_json,
                        q3_meta, ROOT)


def occl_batch(models, b, width=3, stride=1, target="mu"):
    """返回 (w_occ (B,3,L), d_win 列表, 窗口起点列表)。"""
    B = b["x"]["text"].size(0)
    L = b["x"]["text"].size(1)
    base = predict_batch(models, b, target=target)["target"].detach()
    W = np.zeros((B, 3, L), np.float64)
    CNT = np.zeros((B, 3, L), np.float64)
    d_all = {}
    starts_all = {}
    for k, m in enumerate(MODALITIES):
        nvalid = int((~(b["pad"][m][0])).sum().item()) if B else L
        starts = list(range(0, max(1, nvalid - width + 1), stride))
        if not starts:
            starts = [0]
        starts_all[m] = starts
        d = np.zeros((B, len(starts)), np.float64)
        for j, st in enumerate(starts):
            bo = occlude_window(b, m, st, width)
            t = predict_batch(models, bo, target=target)["target"].detach()
            dd = (base - t).cpu().numpy()
            d[:, j] = dd
            for r in range(B):
                e = min(L, st + width)
                W[r, k, st:e] += np.abs(dd[r])
                CNT[r, k, st:e] += 1.0
        d_all[m] = d
    W = W / np.maximum(CNT, 1.0)
    return W, d_all, starts_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--name", default="att4")
    ap.add_argument("--ckpt_dir", default=os.path.join(ROOT, "runs", "ens_top2"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "runs", "q3"))
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--target", default="mu", choices=["mu", "p_pol"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    pkl = a.pkl or os.path.join(ROOT, "附件2", "aligned_50.pkl")
    models, stats, theta = load_backbone(a.ckpt_dir, a.device)
    split, key = load_split(pkl, stats, a.split)
    if a.limit:
        from data_utils import slice_split
        split = slice_split(split, a.limit)
    N, L = split["N"], split["L"]["text"]
    print("[DATA] %s/%s N=%d L=%d | 窗口 w=%d 步长 %d target=%s"
          % (os.path.basename(pkl), key, N, L, a.width, a.stride, a.target))

    W = np.zeros((N, 3, L), np.float64)
    for i0 in range(0, N, a.batch_size):
        idxs = list(range(i0, min(i0 + a.batch_size, N)))
        b = make_batch(split, idxs, a.device)
        w, _, _ = occl_batch(models, b, a.width, a.stride, a.target)
        W[i0:i0 + len(idxs)] = w
        if (i0 // a.batch_size) % 20 == 0:
            print("    %d/%d" % (min(i0 + a.batch_size, N), N))

    valid = np.zeros((N, 3, L), bool)
    for i in range(N):
        for k, m in enumerate(MODALITIES):
            valid[i, k, :int((~split["pad"][m][i]).sum())] = True
    W[~valid] = 0.0
    path = save_npz(os.path.join(a.out_dir, "q3_temporal_occ_%s.npz" % a.name),
                    ids=np.asarray(split["ids"]).astype(str), w_occ=W, valid=valid,
                    meta=np.array([a.width, a.stride, a.target], dtype=object))
    rep = dict(meta=q3_meta(a.ckpt_dir, models, theta,
                            dict(pkl=os.path.relpath(pkl, ROOT), split=key, n=N,
                                 width=a.width, stride=a.stride, target=a.target)),
               w_occ_sum_mean=[round(float(W[:, k].sum(1).mean()), 4) for k in range(3)],
               data=os.path.relpath(path, ROOT))
    save_json(os.path.join(a.out_dir, "q3_temporal_occ_%s_report.json" % a.name), rep)
    print("[SAVE] %s" % os.path.relpath(path, ROOT))
    print("OCCLUSION_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
