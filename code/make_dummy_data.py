# -*- coding: utf-8 -*-
"""
生成与附件2/附件3 同构的**合成数据**，用于在没有真实附件前跑通全流程（冒烟测试）。
注意：仅用于代码自检，绝不用于论文结果。

用法：
  python make_dummy_data.py --out_dir E:\\数学建模\\data --version aligned --n_train 256 --n_valid 128 --n_test 128
"""
import argparse
import os
import pickle

import numpy as np

MODALITIES = ("text", "audio", "vision")


def synth(n, rng, unaligned=False, missing_ratio=0.0):
    """构造一个 split：特征与标签存在可学习的相关性。"""
    Lt = 50
    La = 500 if unaligned else 50
    Lv = 500 if unaligned else 50
    text = rng.normal(0, 1, (n, Lt, 768)).astype(np.float32)
    audio = rng.normal(0, 1, (n, La, 74)).astype(np.float32)
    vision = rng.normal(0, 1, (n, Lv, 35)).astype(np.float32)

    y = rng.uniform(-3, 3, n).astype(np.float32)
    # 注入信号：文本第 0 维、语音第 0~2 维与情感强度正相关（仅前若干时间步）
    text[:, :, 0] += y[:, None] * 1.2
    audio[:, :, :3] += (y[:, None, None] * 0.6)
    vision[:, :, 0] += y[:, None] * 0.3

    if unaligned:
        alen = rng.integers(120, 501, n).astype(np.int64)
        vlen = rng.integers(120, 501, n).astype(np.int64)
        for i in range(n):
            text[i, :] += 0.0
    else:
        alen = np.full(n, 50, dtype=np.int64)
        vlen = np.full(n, 50, dtype=np.int64)

    if missing_ratio > 0:
        for i in range(n):
            for X, L in ((text, Lt), (audio, La), (vision, Lv)):
                if rng.random() < 0.8:
                    m = int(rng.integers(0, 3))
                    ln = int(alen[i] if (X is audio) else (vlen[i] if X is vision else Lt))
                    ln = min(ln, L)
                    w = max(2, int(missing_ratio * ln))
                    s = int(rng.integers(0, max(1, ln - w)))
                    X[i, s:s + w, :] = 0.0

    cls = np.where(y < 0, 0, np.where(y > 0, 2, 1)).astype(np.int64)
    ids = np.array(["vid%03d$_$clip%03d" % (i // 5, i) for i in range(n)], dtype=object)
    ann = np.array(["Negative", "Neutral", "Positive"], dtype=object)[cls]
    raw = np.array(["dummy transcript %d" % i for i in range(n)], dtype=object)

    split = dict(text=text, audio=audio, vision=vision,
                 audio_lengths=alen, vision_lengths=vlen,
                 regression_labels=y, classification_labels=cls,
                 annotations=ann, id=ids, raw_text=raw)
    # text_bert: (N,3,L) 词元编号/注意力掩码/分段编号
    tb = np.zeros((n, 3, Lt), dtype=np.int64)
    tb[:, 1, :] = rng.integers(1, 20, (n, Lt)) + 1
    split["text_bert"] = tb
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=r"E:\数学建模\data_dummy")
    ap.add_argument("--version", default="aligned", choices=["aligned", "unaligned"])
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_valid", type=int, default=128)
    ap.add_argument("--n_test", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    un = a.version == "unaligned"

    data = dict(train=synth(a.n_train, rng, un, 0.0),
                valid=synth(a.n_valid, rng, un, 0.0),
                test=synth(a.n_test, rng, un, 0.10))
    p = os.path.join(a.out_dir, "%s_50.pkl" % a.version)
    with open(p, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print("[OK] %s" % p)

    # 附件3 风格：单 split、无标签、含连续区间缺失
    att3 = synth(a.n_test, rng, un, 0.35)
    for k in ("regression_labels", "classification_labels", "annotations"):
        att3.pop(k, None)
    p3 = os.path.join(a.out_dir, "att3_like.pkl")
    with open(p3, "wb") as f:
        pickle.dump(att3, f, protocol=4)
    print("[OK] %s" % p3)


if __name__ == "__main__":
    main()
