# -*- coding: utf-8 -*-
"""指标口径自检：确认 ACC/F1 同时给出 θ 阈值口径与分类头口径。"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_model"))
from losses import compute_metrics, tune_threshold, format_metrics, polar_from_score  # noqa: E402

rng = np.random.default_rng(0)
n = 300
y = rng.uniform(-3, 3, n)                        # 真值强度
p = np.clip(y * 0.6 + rng.normal(0, 0.5, n), -3, 3)   # 模拟"向均值收缩"的预测
cls = np.where(y < 0, 0, np.where(y > 0, 2, 1)).astype(np.int64)

th, f1 = tune_threshold(p, cls)
print("tuned theta=%.2f  MacroF1(theta)=%.4f" % (th, f1))
head = (p > 0.2).astype(np.int64) * 2            # 模拟一个较差的分类头
res = compute_metrics(y, p, cls, theta=th, y_pred_cls=head)
print(format_metrics(res, tag="[自检]"))
print("\nkeys =", sorted(res.keys()))
print("\n判定：主指标 acc/f1 来自 θ 阈值口径（与提交 CSV 一致）；"
      "acc_head/f1_head 为分类头口径。")
