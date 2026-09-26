# -*- coding: utf-8 -*-
"""按 label 删除 main.tex 中的图表环境块（用于正文页数压缩）。

用法：
    python pack_paper.py --list                 # 仅列出将删除的块
    python pack_paper.py --apply                # 执行删除（自动备份）

设计要点：
  * 以 \label{...} 行为锚点，向前找最近的 \begin{figure|table}，向后找配对的 \end{...}；
  * 环境内不允许嵌套同名环境（本文满足）；
  * 删除后报告残留的 \ref{label} 位置，提醒作者改写正文引用；
  * 备份文件名带时间戳，便于回滚。
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.abspath(os.path.join(HERE, "..", "..", "05_paper", "main.tex"))

# 待删除的人工清单（按 label）
DROP_LABELS = [
    # ---- 第四轮：数字已入正文的图表 ----
    "fig:q2errgroup",   # 错误归因五维分层
    "fig:q2att3comp",    # 附件三预测构成对比
    "fig:q3faith",       # 解释忠实性对比（数字已入正文）
    "tab:anova",         # 三因素方差分析表（数字已入正文）
]

ENV_BEGIN = re.compile(r"^\s*\\begin\{(figure|table)\}")
ENV_END = re.compile(r"^\s*\\end\{(figure|table)\}")


def find_block(lines, label):
    """返回 (start, end) 闭区间行号（0-based），找不到返回 None。"""
    anchor = None
    for i, ln in enumerate(lines):
        if ("\\label{%s}" % label) in ln:
            anchor = i
            break
    if anchor is None:
        return None
    start = None
    for j in range(anchor, -1, -1):
        m = ENV_BEGIN.match(lines[j])
        if m:
            start = j
            kind = m.group(1)
            break
    if start is None:
        return None
    end = None
    for k in range(anchor, len(lines)):
        m = ENV_END.match(lines[k])
        if m and m.group(1) == kind:
            end = k
            break
    if end is None:
        return None
    return start, end, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    with open(MAIN, encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")

    blocks = []
    for lab in DROP_LABELS:
        got = find_block(lines, lab)
        if got is None:
            print("  [跳过] 未定位: %s" % lab)
            continue
        s, e, kind = got
        blocks.append((s, e, kind, lab))

    # 自后向前删除，避免行号漂移
    blocks.sort(reverse=True)
    total_lines = 0
    for s, e, kind, lab in blocks:
        # 连同前后各一行空行一起删除
        s2, e2 = s, e
        while s2 - 1 >= 0 and lines[s2 - 1].strip() == "":
            s2 -= 1
        while e2 + 1 < len(lines) and lines[e2 + 1].strip() == "":
            e2 += 1
        n = e2 - s2 + 1
        total_lines += n
        print("  [删] %-16s %s 环境 %3d 行（源 %d-%d）" % (lab, kind, n, s + 1, e + 1))
        del lines[s2:e2 + 1]

    new = "\n".join(lines)

    # 检查残留引用
    left = []
    for lab in DROP_LABELS:
        if lab in new:
            for i, ln in enumerate(new.split("\n")):
                if lab in ln:
                    left.append((lab, i + 1, ln.strip()[:70]))
    print("\n[残留引用] %d 处（需人工改写正文句子）：" % len(left))
    for lab, lineno, snippet in left:
        print("  %-16s L%-5d %s" % (lab, lineno, snippet))

    print("\n[统计] 删除 %d 个块、%d 行；剩余行数 %d -> %d"
          % (len(blocks), total_lines, len(text.split("\n")), len(lines)))

    if args.apply:
        bak = MAIN + ".bak_pack%d" % int(time.time())
        shutil.copy2(MAIN, bak)
        with open(MAIN, "w", encoding="utf-8") as f:
            f.write(new)
        print("[APPLY] 已写入 %s\n[备份] %s" % (MAIN, bak))
    else:
        print("（未写入，加 --apply 执行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
