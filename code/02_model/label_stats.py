# -*- coding: utf-8 -*-
r"""
标签层面的性能上限与基线测算（**只读标签，不训练、不推理**）。

目的：回答"在附件2 给定标注下，性能的理论上限大致在哪"。
注意：附件2 只提供"多标注者平均后的强度"与"三分类极性"，**不含个体标注**，
因此无法严格计算贝叶斯误差；本脚本给出三个可计算的**代理上界/参照**：

  U1 符号上限（极性=强度符号时的分类准确率上限） = 1 − 跨界率
  U2 类别 oracle 的回归 MAE（完美知道三分类标签、按类条件中位数预测）
  U3 平凡基线（常数 0 / 全局均值 / 类均值 / 多数类）

用法：
  python 02_model\label_stats.py --pkl ..\附件2\aligned_50.pkl --out_json ..\runs\label_stats.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_labels(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    out = {}
    for k in ("train", "valid", "test"):
        if k not in d:
            continue
        y = np.asarray(d[k]["regression_labels"], dtype=np.float64)
        c = np.asarray(d[k]["classification_labels"]).astype(np.int64)
        out[k] = (y, c)
    return out


def stats(y, c, name, out):
    n = len(y)
    sign = np.where(y > 0, 2, np.where(y < 0, 0, 1))
    agree = float((sign == c).mean())
    cnt = np.bincount(c, minlength=3)
    maj = int(cnt.argmax())
    res = dict(n=int(n), label_self_consistency=agree, conflict_rate=float(1 - agree),
               class_counts=cnt.tolist(), majority_acc=float((c == maj).mean()),
               y_mean=float(y.mean()), y_std=float(y.std()),
               mae_const0=float(np.abs(y).mean()),
               mae_global_mean=float(np.abs(y - y.mean()).mean()))
    # 类条件统计
    cls = {}
    for k, nm in ((0, "Negative"), (1, "Neutral"), (2, "Positive")):
        m = c == k
        if not m.sum():
            continue
        cls[nm] = dict(n=int(m.sum()), y_mean=float(y[m].mean()), y_std=float(y[m].std()),
                       y_median=float(np.median(y[m])),
                       y_min=float(y[m].min()), y_max=float(y[m].max()),
                       frac=y[m].size / n)
    res["class_conditional"] = cls
    # U2 类别 oracle：按类条件中位数预测强度（"分类信息用尽"的 MAE 参照）
    med = np.array([np.median(y[c == k]) if (c == k).sum() else 0.0 for k in (0, 1, 2)])
    res["mae_class_median_oracle"] = float(np.abs(y - med[c]).mean())
    # 边界样本比例（极性本身模糊）
    res["boundary"] = {("abs_le_%.2f" % e): float((np.abs(y) <= e).mean())
                       for e in (0.1, 0.25, 0.5)}
    nzm = c != 1
    res["boundary"]["abs_le_0.5_non_neutral"] = float((np.abs(y[nzm]) <= 0.5).mean()) if nzm.sum() else None
    print("\n===== %s (n=%d) =====" % (name, n))
    print("  类别分布: neg=%d neu=%d pos=%d（多数类 ACC=%.4f）"
          % (cnt[0], cnt[1], cnt[2], res["majority_acc"]))
    print("  强度: mean=%.3f std=%.3f  |y|: mean=%.3f" % (res["y_mean"], res["y_std"], res["mae_const0"]))
    print("  [U1] sign(强度) 与三分类标签一致率 = **%.4f** → 若极性严格等于强度符号，分类 ACC 上限 ≈ %.4f"
          % (agree, agree))
    print("  [U2] 类别 oracle 回归 MAE（按类中位数）= %.4f" % res["mae_class_median_oracle"])
    print("  [U3] 基线: MAE(常数0)=%.4f  MAE(全局均值)=%.4f  多数类ACC=%.4f  随机MacroF1≈0.333"
          % (res["mae_const0"], res["mae_global_mean"], res["majority_acc"]))
    print("  边界: |y|<=0.1 %.3f  <=0.25 %.3f  <=0.5 %.3f  （非中性中 |y|<=0.5 %.3f）"
          % (res["boundary"]["abs_le_0.10"], res["boundary"]["abs_le_0.25"],
             res["boundary"]["abs_le_0.50"], res["boundary"]["abs_le_0.5_non_neutral"]))
    for nm, v in cls.items():
        print("    %-9s n=%4d (%.3f)  强度 %+.3f ± %.3f  [%+.2f, %+.2f]"
              % (nm, v["n"], v["frac"], v["y_mean"], v["y_std"], v["y_min"], v["y_max"]))
    out[name] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=os.path.join("..", "附件2", "aligned_50.pkl"))
    ap.add_argument("--out_json", default=os.path.join("..", "runs", "label_stats.json"))
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    print("[DATA] 读取标签：%s" % os.path.abspath(a.pkl))
    labs = load_labels(a.pkl)
    out = {}
    for k, (y, c) in labs.items():
        stats(y, c, k, out)
    # 全体
    y = np.concatenate([labs[k][0] for k in labs])
    c = np.concatenate([labs[k][1] for k in labs])
    stats(y, c, "all", out)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[SAVE] %s" % os.path.abspath(a.out_json))


if __name__ == "__main__":
    main()
