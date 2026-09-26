# -*- coding: utf-8 -*-
r"""非流利标记与文本帧的对齐关系探查（为 6.2 转写清洗方案定路）。

要回答三个问题
--------------
1. `raw_text` 的空格分词数 是否等于文本有效帧数（即是否存在 1:1 的 词→帧 映射）？
2. 有多少样本含非流利标记（`(umm) (uhh) (stutter) {laughs}` 等）？密度多大？
3. 若 1:1 成立，则"清洗"可等价为**把标记所在帧置零并标为缺失**（保留对齐、复用问题2 的缺失框架）。

用法
----
  python inspect_disfluency.py --version aligned
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config                                  # noqa: E402
from data_utils import load_all                            # noqa: E402

# 非流利/副语言标记：括号型 (umm)/(uhh)/(stutter)/{laughs} 等
MARK_RE = re.compile(r"\((?:[^()]{1,20})\)|\{[^{}]{1,20}\}")


def marker_positions(text):
    """返回 raw_text 中属于非流利标记的**词索引**集合。"""
    s = " " + str(text).strip() + " "
    spans, idx = [], 0
    tokens = str(text).split()
    pos = []
    cur = 0
    for w in tokens:
        # 该词里是否含标记
        if MARK_RE.search(w):
            pos.append(cur)
        cur += 1
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="aligned")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--examples", type=int, default=5)
    a = ap.parse_args()

    cfg = Config(version=a.version, out_dir=os.path.dirname(os.path.abspath(__file__)))
    if a.data_dir:
        cfg.data_dir = a.data_dir
        cfg.__post_init__()
    tr, va, te = load_all(cfg)

    for name, sp in (("train", tr), ("valid", va), ("test", te)):
        if sp.get("raw_text") is None:
            print("[%s] 无 raw_text，跳过" % name)
            continue
        rt = np.asarray(sp["raw_text"])
        n_valid = (~sp["pad"]["text"]).sum(axis=1)                 # 有效文本帧数
        n_words = np.array([len(str(x).split()) for x in rt])
        n_mark_tok = np.array([len(marker_positions(x)) for x in rt])

        print("\n" + "=" * 78)
        print("[%s] N=%d   文本帧 L=%d" % (name, len(rt), sp["L"]["text"]))
        print("  词数 == 有效帧数 的比例：%.4f  （差值中位数 %.1f）"
              % (float((n_words == n_valid).mean()),
                 float(np.median(n_words - n_valid))))
        print("  词数分布：min=%d p50=%.0f p90=%.0f max=%d"
              % (n_words.min(), np.percentile(n_words, 50),
                 np.percentile(n_words, 90), n_words.max()))
        print("  有效帧数分布：min=%d p50=%.0f max=%d"
              % (n_valid.min(), np.percentile(n_valid, 50), n_valid.max()))
        has = n_mark_tok > 0
        print("  含非流利标记的样本：%d/%d = **%.1f%%**"
              % (int(has.sum()), len(rt), 100.0 * has.mean()))
        if has.any():
            m = n_mark_tok[has]
            print("    标记词数（仅含标记样本）：mean=%.2f p50=%.0f p90=%.0f max=%d"
                  % (m.mean(), np.percentile(m, 50), np.percentile(m, 90), m.max()))
            print("    占该样本有效帧比例：mean=%.1f%%（等价于平均文本局部缺失率）"
                  % (100.0 * (m / np.maximum(n_valid[has], 1)).mean()))
        # 标记类型频次（前 10）
        from collections import Counter
        c = Counter()
        for x in rt:
            for mm in MARK_RE.findall(str(x)):
                c[mm.lower()] += 1
        print("    高频标记 top10：%s"
              % ", ".join("%s×%d" % (k, v) for k, v in c.most_common(10)))

        # 示例映射
        shown = 0
        for i in range(len(rt)):
            if not has[i]:
                continue
            toks = str(rt[i]).split()
            pos = set(marker_positions(rt[i]))
            demo = " ".join(("[%s]" if j in pos else "%s") % t
                            for j, t in enumerate(toks[:18]))
            print("    ex%d  id=%s  有效帧=%d 词数=%d 标记词=%s"
                  % (shown, sp["ids"][i], n_valid[i], len(toks), sorted(pos)[:6]))
            print("        %s" % demo)
            shown += 1
            if shown >= a.examples:
                break


if __name__ == "__main__":
    main()
