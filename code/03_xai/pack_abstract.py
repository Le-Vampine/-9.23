# -*- coding: utf-8 -*-
"""将 main.tex 的摘要整段替换为压缩版（要求标题与摘要同页）。

用法：python pack_abstract.py [--apply]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.abspath(os.path.join(HERE, "..", "..", "05_paper", "main.tex"))

NEW = r"""多模态情感识别需同时给出情感极性与情感强度，而真实采集场景中常出现部分连续时段的模态信息缺失，常规固定权重融合模型在此类场景下性能急剧退化。本文以 CMU-MOSEI 派生数据为唯一数据来源，依次完成多模态特征提取与时序对齐、模态缺失条件下的鲁棒情感预测、可解释性情感预测三项建模任务。

针对问题一，本文将文本、语音、视觉三模态建模为等长时序特征矩阵，以帧级有效性掩码显式分离填充位与真实缺失位：填充位由尾部连续零值推断，仅作补齐不计入缺失；真实缺失位由连续不少于两帧的全零判定。并在此基础上给出跨模态时序对齐的数学定义、统一特征维度与归一化统计，形成后两问共用的输入接口。

针对问题二，本文建立掩码感知的多模态鲁棒融合模型 MRF-Net：以共享-特有分解刻画模态间的共性与个性，以动态专家门控按样本自适应分配模态权重，以缺失指示参与注意力计算使模型“知道”哪一段不可用，并以掩码加权池化只在不缺失的时间步上聚合信息；训练采用教师--学生蒸馏与缺失课程学习。最终模型在验证集取得 **ACC=0.6223**、**Macro-F1=0.6158**、**MAE=0.5866**、**Pearson=0.6543**，在附件二测试划分取得 **ACC=0.6534**、**Macro-F1=0.6338**；按分类头概率构造的二分类口径准确率达 **0.8514**。缺失扫描覆盖 6 种缺失类型、4 种缺失位置与 5 档缺失率共 120 个场景，三因素方差分析表明退化由“模态类型×缺失率”交互主导，本文模型把该交互的效应量由 **0.9517 降至 0.8415**，整体退化降低 **1.5 至 3.3 倍**。

针对问题三，本文在冻结问题二骨干的前提下构建三层可解释通路：模态级用可精确枚举的三模态 Shapley 值量化作用程度，片段级用积分梯度与模型无关的遮挡定位关键片段，证据级把结论回映到原文词句、语音时段与视频关键帧。验证集上文本、语音、视觉的全局 Shapley 值（以负平均绝对误差为效用）为 **0.1842**、**0.0004**、**0.0103**；仅用文本即可把平均绝对误差由全屏蔽的 0.7816 降至 0.5936，接近全模态的 0.5866。模态作用度与真实删模态退化的秩相关达 **1.000**，与问题二的缺失注入实验排序完全一致；再用附件三的实测缺失位置做免标注机制检验，作用度随缺失率上升而下降（秩相关 **−0.401**）。模型内生的注意力权重与遮挡基准的秩相关仅 **−0.033**，故本文不以注意力权重作为解释结论。

本文的特点是缺失建模与决策口径的规范性：全部超参数、阈值与校准系数均在验证集上确定，缺失率采用统一定义并与提交文件逐位对齐；同时如实报告两条经验证无效的改进路径。模型的局限在于中性类识别精确率偏低与强度分数存在系统性收缩，可由分布型回归与语用级建模进一步改进。"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    with open(MAIN, encoding="utf-8") as f:
        text = f.read()

    m = re.search(r"(\\begin\{abstract\}\n)(.*?)(\\keywords\{)", text, re.S)
    if not m:
        print("[ERR] 未定位 abstract 环境")
        return 1
    old_body = m.group(2)
    n_old = len(old_body)
    new_body = NEW + "\n"
    n_new = len(new_body)
    print("[摘要] 字数 %d -> %d（压缩 %.0f%%）"
          % (n_old, n_new, 100 * (1 - n_new / n_old)))
    print("[预览] 新摘要前 80 字：", new_body[:80].replace("\n", " "))

    if args.apply:
        text = text[:m.start(2)] + new_body + text[m.end(2):]
        bak = MAIN + ".bak_abs%d" % int(time.time())
        shutil.copy2(MAIN, bak)
        with open(MAIN, "w", encoding="utf-8") as f:
            f.write(text)
        print("[APPLY] 已写入；备份 %s" % bak)
    else:
        print("（未写入，加 --apply 执行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
