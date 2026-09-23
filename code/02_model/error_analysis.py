# -*- coding: utf-8 -*-
r"""
验证集/测试集错误归因（问题2 第(4)项要求的"错误归因结论"）。

分层维度：
  A 按真值强度 |y| 分桶（中性附近 vs 强情感）
  B 按文本有效长度分位（极短转写是否更依赖语音/视觉）
  C 按视觉有效长度/填充率分桶（画面质量影响）
  D 按各模态填充率分桶（序列长度效应）
  E 按真值极性
并输出 top-K 高误差样本清单（含 raw_text 摘要与缺失情况）供人工复核。

用法：
  python error_analysis.py --ckpt_dir ..\runs\q2 --out_dir ..\runs\q2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config                                  # noqa: E402
from data_utils import load_all                            # noqa: E402
import runtime                                             # noqa: E402
from losses import compute_metrics                         # noqa: E402


def bucket_report(df, group_col, out_rows, split, theta=0.0):
    g = df.groupby(group_col, dropna=False)
    for key, sub in g:
        if len(sub) < 10:
            continue
        res = compute_metrics(sub["y_reg"], sub["pred"], sub["y_cls"], theta=theta)
        out_rows.append(dict(split=split, dim=group_col, group=str(key),
                             n=len(sub), mae=res["mae"], pearson=res["pearson"],
                             acc=res["acc"], f1=res["f1"],
                             bias=float((sub["pred"] - sub["y_reg"]).mean()),
                             miss_text=float(sub["rho_text"].mean()),
                             miss_audio=float(sub["rho_audio"].mean()),
                             miss_vision=float(sub["rho_vision"].mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--version", default="aligned")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=20)
    a = ap.parse_args()
    out_dir = a.out_dir or a.ckpt_dir
    os.makedirs(out_dir, exist_ok=True)

    cfg = Config(version=a.version)
    if a.data_dir:
        cfg.data_dir = a.data_dir
        cfg.__post_init__()
    tr, va, te = load_all(cfg)
    models = runtime.load_ensemble(runtime.find_ckpts(a.ckpt_dir), a.device)
    theta = float(np.mean([ck["theta"] for _, ck in models]))

    rows = []
    for name, sp in (("valid", va), ("test", te)):
        pr = runtime.predict_split(models, sp, a.batch_size, a.device)
        pred_cls = np.where(pr["score"] < -theta, 0, np.where(pr["score"] > theta, 2, 1))
        df = pd.DataFrame({
            "id": pr["ids"], "y_reg": pr["y_reg"], "y_cls": pr["y_cls"],
            "pred": pr["score"], "pred_cls": pred_cls,
            "err": np.abs(pr["y_reg"] - pr["score"]),
            "rho_text": pr["rho"][:, 0], "rho_audio": pr["rho"][:, 1], "rho_vision": pr["rho"][:, 2],
            "len_text": pr["length"][:, 0], "len_audio": pr["length"][:, 1],
            "len_vision": pr["length"][:, 2],
        })
        df["abs_y"] = np.abs(df["y_reg"])
        df["strength_bucket"] = pd.cut(df["abs_y"], [-0.001, 0.5, 1.5, 3.01],
                                       labels=["|y|<=0.5", "0.5<|y|<=1.5", "|y|>1.5"])
        df["text_len_bucket"] = pd.qcut(df["len_text"].rank(method="first"), 3,
                                        labels=["短", "中", "长"])
        df["vision_len_bucket"] = pd.qcut(df["len_vision"].rank(method="first"), 3,
                                          labels=["短", "中", "长"])
        df["polarity"] = df["y_cls"].map({0: "Negative", 1: "Neutral", 2: "Positive"})
        df["any_missing"] = (df[["rho_text", "rho_audio", "rho_vision"]].sum(axis=1) > 1e-6).map(
            {True: "含缺失", False: "无缺失"})

        # ---- 分层统计 ----
        base = compute_metrics(pr["y_reg"], pr["score"], pr["y_cls"], theta,
                              y_pred_cls=pr["prob"].argmax(-1))
        print("\n[%s] 总体: MAE=%.4f ACC=%.4f MacroF1=%.4f r=%.3f CCC=%.3f"
              % (name, base["mae"], base["acc"], base["f1"], base["pearson"], base["ccc"]))
        before = len(rows)
        for col in ["strength_bucket", "text_len_bucket", "vision_len_bucket",
                    "polarity", "any_missing"]:
            bucket_report(df, col, rows, name, theta=theta)
        tab = pd.DataFrame(rows[before:])
        print(tab.to_string(index=False, float_format=lambda x: "%.4f" % x))
        tab.to_csv(os.path.join(out_dir, "error_groups_%s.csv" % name),
                   index=False, encoding="utf-8-sig")

        # ---- top-K 高误差样本 ----
        df = df.sort_values("err", ascending=False).head(a.topk).copy()
        if sp.get("raw_text") is not None:
            rt = [str(x)[:120] for x in np.asarray(sp["raw_text"])]
            id2txt = {str(sp["ids"][i]): rt[i] for i in range(min(sp["N"], len(rt)))}
            df["raw_text"] = [id2txt.get(str(i), "") for i in df["id"]]
        df.to_csv(os.path.join(out_dir, "error_top%d_%s.csv" % (a.topk, name)),
                  index=False, encoding="utf-8-sig")
        print("[SAVE] %s / error_top%d_%s.csv" % (out_dir, a.topk, name))

        # ---- 结论性提示 ----
        sub = pd.DataFrame(rows[before:])
        if len(sub):
            w = sub.groupby("dim").apply(lambda g: (g["n"] * g["mae"]).sum() / g["n"].sum())
            worst = sub.sort_values("mae", ascending=False).head(3)
            print("[归因要点] 各维度上 MAE 最大的分组:")
            for _, r in worst.iterrows():
                print("   %s=%s  n=%d  MAE=%.4f  bias=%.4f" %
                      (r["dim"], r["group"], r["n"], r["mae"], r["bias"]))


if __name__ == "__main__":
    main()
