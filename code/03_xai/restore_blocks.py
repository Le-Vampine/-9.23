# -*- coding: utf-8 -*-
"""把此前为控页数而下架的图表块从备份中回填到 main.tex（本赛题无页数上限）。

数据来源：`05_paper/main.tex.bak_pack1790322984`（第一轮删除前的完整版本，含全部 21 个块）。
做法：对每个 label，在备份中取出 `\\begin{figure|table}...\\end{...}` 整块，
插到当前 main.tex 中"锚点行"之后，并补一句正文引用（满足"图表就近"）。

用法：
  python restore_blocks.py            # 干跑：报告锚点命中、图件是否存在、将插入的块
  python restore_blocks.py --apply    # 写真（自动备份）
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PAPER = os.path.join(ROOT, "05_paper")
MAIN = os.path.join(PAPER, "main.tex")
BAK = os.path.join(PAPER, "main.tex.bak_pack1790322984")
FIGS = os.path.join(PAPER, "figures")

ENV_BEGIN = re.compile(r"^\s*\\begin\{(figure|table)\}")
ENV_END = re.compile(r"^\s*\\end\{(figure|table)\}")

# (label, 锚点子串, 需要额外插入的引用句, 插入前还是后)
ITEMS = [
    # ---- 问题二·求解结果 ----
    ("fig:q2scatter",  "模型的强度预测行为可由预测值与真值的对应关系直接观察",
     "强度预测与真值的对应关系见图~\\ref{fig:q2scatter}。"),
    ("fig:q2cm",       "强度误差只能说明回归头的表现",
     "三分类混淆矩阵见图~\\ref{fig:q2cm}。"),
    ("fig:q2theta",    "极性判定不依赖单一阈值",
     "阈值敏感性曲线见图~\\ref{fig:q2theta}。"),
    ("fig:q2mask",     "极性判定不依赖单一阈值",
     "缺失掩码与填充掩码的时序可视化见图~\\ref{fig:q2mask}。"),
    # ---- 问题二·缺失规律 ----
    ("tab:anova",      "其中 $C(\\cdot)$ 表示按类别变量处理",
     "各效应项的平方和占比即效应量 $\\eta^2$，完整数值列于表~\\ref{tab:anova}。"),
    ("fig:q2heat",     "交叉缺失类型与缺失位置两个维度后，可得退化的二维分布",
     "缺失类型与缺失位置的退化热力图见图~\\ref{fig:q2heat}。"),
    # ---- 问题二·消融与调参 ----
    ("fig:q2tune",     "结构选型分两阶段进行",
     "两阶段超参搜索的对比见图~\\ref{fig:q2tune}。"),
    # ---- 问题二·错误归因 ----
    ("fig:q2errhist",  "强度绝对误差的分布存在明显长尾",
     "强度绝对误差的分布形态见图~\\ref{fig:q2errhist}。"),
    ("fig:q2errgroup", "为定位误差来源，本文沿强度幅值",
     "五维分层的完整结果见图~\\ref{fig:q2errgroup}。"),
    # ---- 问题二·附件三/四推理 ----
    ("fig:q2att3comp", "预测极性构成与缺失状态的关系为",
     "附件三的预测构成见图~\\ref{fig:q2att3comp}。"),
    ("fig:q2att3dist", "预测极性构成与缺失状态的关系为",
     "预测强度与置信度的分布见图~\\ref{fig:q2att3dist}。"),
    ("fig:q2att34",    "预测结果之外，还需交代两个专项测试集本身的输入条件",
     "两个专项测试集的实测缺失分布见图~\\ref{fig:q2att34}。"),
    # ---- 问题三 ----
    ("fig:q3cards2",   "取附件四中预测为正向的样本 17",
     "负向样本的解释卡见图~\\ref{fig:q3cards2}。"),
    ("tab:q3pi",       "验证集 728 条样本的模态作用度统计见图~\\ref{fig:q3pilab}",
     "按真值极性分层的数值列于表~\\ref{tab:q3pi}。"),
    ("fig:q3miss",     "进一步地，附件三与附件四的样本带有实测缺失位置",
     "作用度与实测缺失率的关系见图~\\ref{fig:q3miss}。"),
    ("tab:q3faith",    "四种解释来源的定量对比如下",
     "四种来源的指标汇总见表~\\ref{tab:q3faith}。"),
    ("fig:q3faith",    "四种解释来源的定量对比如下",
     "解释质量对比见图~\\ref{fig:q3faith}。"),
    ("fig:q3att4",     "附件四 20 条样本的预测与解释汇总见表~\\ref{tab:q3att4}",
     "汇总统计见图~\\ref{fig:q3att4}。"),
    ("tab:q3err",      "在此基础上，本文进一步从解释维度做错误归因",
     "解释维度的错误归因结果见表~\\ref{tab:q3err}。"),
    ("fig:q3err",      "在此基础上，本文进一步从解释维度做错误归因",
     "解释熵与置信度的对比见图~\\ref{fig:q3err}。"),
    # ---- 问题二·参数表（2 号表）----
    ("tab:q2hparam2",  "表中所列取值与代码配置文件逐项对应，可作为复现依据",
     "损失权重、缺失课程与决策参数的完整取值见表~\\ref{tab:q2hparam2}。"),
]


def extract_block(lines, label):
    anchor = None
    for i, l in enumerate(lines):
        if ("\\label{%s}" % label) in l:
            anchor = i
            break
    if anchor is None:
        return None
    start = kind = None
    for j in range(anchor, -1, -1):
        m = ENV_BEGIN.match(lines[j])
        if m:
            start, kind = j, m.group(1)
            break
    if start is None:
        return None
    for k in range(anchor, len(lines)):
        m = ENV_END.match(lines[k])
        if m and m.group(1) == kind:
            return "\n".join(lines[start:k + 1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    bak_lines = open(BAK, encoding="utf-8").read().split("\n")
    cur = open(MAIN, encoding="utf-8").read()
    cur_lines = cur.split("\n")

    plan = []          # (anchor_line_idx, cite, block, label)
    missing_files = set()
    for label, anchor_sub, cite in ITEMS:
        block = extract_block(bak_lines, label)
        if block is None:
            print("  [跳过] 备份中未找到块: %s" % label)
            continue
        hits = [i for i, l in enumerate(cur_lines) if anchor_sub in l and not l.strip().startswith("%")]
        if not hits:
            print("  [锚点缺失] %-14s ← '%s'" % (label, anchor_sub[:24]))
            continue
        if len(hits) > 1:
            print("  [锚点多个] %-14s ← %d 处，取最后一行" % (label, len(hits)))
        ax = hits[-1]
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", block):
            f = os.path.join(PAPER, m.group(1))
            if not os.path.exists(f):
                missing_files.add(m.group(1))
        plan.append((ax, cite, block, label))

    # 同一锚点行的多项按声明顺序依次插入
    plan.sort(key=lambda t: t[0])
    print("\n将插入 %d 个块；缺失图件 %d 个" % (len(plan), len(missing_files)))
    for f in sorted(missing_files):
        print("   [缺图]", f)
    for i, (ax, cite, block, label) in enumerate(plan[:60]):
        print("   + %-14s ← 行%-5d %s" % (label, ax + 1, cite[:30]))

    if not args.apply:
        print("\n（干跑结束，加 --apply 执行）")
        return 0

    # 由后向前插入，避免行号漂移
    for ax, cite, block, label in sorted(plan, key=lambda t: -t[0]):
        ins = ["", cite, "", block]
        cur_lines[ax + 1:ax + 1] = ins
    out = "\n".join(cur_lines)
    bak2 = MAIN + ".bak_restore%d" % int(time.time())
    shutil.copy2(MAIN, bak2)
    open(MAIN, "w", encoding="utf-8").write(out)
    print("\n[APPLY] 已插入 %d 块；备份 %s" % (len(plan), bak2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
