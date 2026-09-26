# -*- coding: utf-8 -*-
"""把"本文 vs 公开基准"对照表落成可引用文件（问题二指标的外部定位）。

数据来源与口径：
  * 公开基线：MMSA 统一复现表（github.com/thuiar/MMSA, results/result-stat.md，
    该表为社区标准的同代码库复现结果，MOSEI 划分 16326/1871/4659）；
  * 本文：paper_tables/main_results_summary.csv 与 runs/ens_top2 的 per-class 报告；
  * 训练规模：赛题附件二 4850 条（3395/728/727），公开划分为 16326/1871/4659。

注意：两侧不可直接比较（训练样本相差 4.8 倍、特征来源不同、测试划分不同），
本表只用于"量级定位"，任何引用都必须在正文写明差异。

输出：paper_tables/lit_vs_ours.csv
用法：python lit_benchmark.py
"""
from __future__ import annotations

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "paper_tables", "lit_vs_ours.csv")

# MMSA 复现表（MOSEI）：三分类 Acc_3 / F1_3；回归 MAE / Corr；Acc-2 取 Has0 口径
PUBLIC = [
    # 方法, Acc-2, Acc-3, F1_3, MAE, Corr
    ("EF-LSTM",            77.84, 66.09, 63.68, 0.6005, 0.6825),
    ("LF-DNN",             80.60, 67.35, 64.65, 0.5802, 0.7087),
    ("TFN",                78.50, 66.63, 63.93, 0.5726, 0.7141),
    ("LMF",                80.54, 66.59, 64.86, 0.5757, 0.7169),
    ("MFN",                78.94, 66.59, 64.31, 0.5733, 0.7182),
    ("Graph-MFN",          81.28, 66.39, 64.00, 0.5745, 0.7133),
    ("MulT",               81.15, 67.04, 65.01, 0.5593, 0.7331),
    ("MISA",               80.67, 67.63, 65.39, 0.5575, 0.7515),
    ("Self-MM",            83.76, None,  None,  0.5309, 0.7649),
    ("TETFN",              84.12, None,  None,  0.5373, 0.7696),
    ("CENet",              83.52, None,  None,  0.5259, 0.7775),
]

# 本文（附件二划分：训练 3395 / 验证 728 / 测试 727）
OURS = [
    ("本文 MRF-Net 集成（验证集）",  None, 62.23, 61.58, 0.5866, 0.6543),
    ("本文 MRF-Net 集成（测试划分）", 85.14, 65.34, 63.38, 0.6216, 0.6807),
]


def main():
    rows = []
    for name, a2, a3, f1, mae, corr in PUBLIC:
        rows.append({
            "来源": "公开基线（MMSA 统一复现，MOSEI）", "方法": name,
            "训练样本数": 16326, "模态": "文本+语音+视觉", "缺失设定": "全模态",
            "Acc-2": "" if a2 is None else "%.2f" % a2,
            "Acc-3": "" if a3 is None else "%.2f" % a3,
            "F1-3": "" if f1 is None else "%.2f" % f1,
            "MAE": "%.4f" % mae, "Corr": "%.4f" % corr,
            "口径备注": "Acc-2 为 Has0 口径（负 vs 非负）；Acc-3/F1-3 为该表分类设定下的值",
        })
    for name, a2, a3, f1, mae, corr in OURS:
        rows.append({
            "来源": "本文（赛题附件二）", "方法": name,
            "训练样本数": 3395, "模态": "文本+语音+视觉",
            "缺失设定": "全模态（内部含缺失注入训练）",
            "Acc-2": "" if a2 is None else "%.2f" % a2,
            "Acc-3": "" if a3 is None else "%.2f" % a3,
            "F1-3": "" if f1 is None else "%.2f" % f1,
            "MAE": "%.4f" % mae, "Corr": "%.4f" % corr,
            "口径备注": "Acc-2 由分类头概率构造（p_neg vs p_neu+p_pos）；Acc-3 为分类头主口径",
        })
    # 下界参照
    rows.append({
        "来源": "本文（赛题附件二）", "方法": "多数类平凡基线", "训练样本数": 0,
        "模态": "—", "缺失设定": "—", "Acc-2": "71.53", "Acc-3": "—", "F1-3": "—",
        "MAE": "—", "Corr": "—", "口径备注": "不做任何建模的下界参照（负 vs 非负）",
    })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[SAVE]", os.path.relpath(OUT, ROOT), "| 行数", len(rows))
    print("[提示] 本文训练样本 3395 vs 公开 16326（1/4.8）；两侧不可直接比较，引用须写明")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
