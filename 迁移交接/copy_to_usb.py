# -*- coding: utf-8 -*-
r"""
一键复制工作区到目标目录（移动硬盘 / 网络盘 / 新机器）。

用法：
  python copy_to_usb.py                    # 默认目标见 TARGET
  python copy_to_usb.py "F:\\数学建模迁移"     # 指定目标
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)                      # 工作区根目录
TARGET = sys.argv[1] if len(sys.argv) > 1 else r"E:\migrate_mathmodel"

# (相对路径, 是否必需, 说明)
ITEMS = [
    ("复杂场景下多模态情感识别的数学建模与算法设计.docx", True, "赛题原文"),
    ("赛事文件.txt", True, "赛题纯文本"),
    ("解决方案", True, "方案文档"),
    ("code", True, "全部代码"),
    ("附件2", True, "标准化特征（aligned + unaligned 全量，共 3.7GB）"),
    ("附件3-模态缺失特征样本", True, "附件3"),
    ("附件4-可解释专项视频样本与特征文件", True, "附件4 + videos"),
    ("data_att", True, "附件3/4 文本特征（必带）"),
    ("runs", True, "模型权重与实验记录"),
    ("submission", True, "预测结果 CSV"),
    ("figs", True, "论文图"),
    ("paper_tables", True, "论文表格"),
    ("迁移交接", True, "本交接包"),
    (r"E:\hf_cache", False, "BERT 权重缓存（可选，也可在新机器重下）"),
]


def copy(src, dst, name):
    if not os.path.exists(src):
        print("  [跳过] %s 不存在" % name)
        return
    os.makedirs(dst, exist_ok=True)
    if os.path.isdir(src):
        cmd = ["robocopy", src, dst, "/E", "/XD", "__pycache__", ".git",
               "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/R:1", "/W:1"]
    else:
        cmd = ["robocopy", os.path.dirname(src), dst, os.path.basename(src),
               "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/R:1", "/W:1"]
    r = subprocess.run(cmd)
    ok = r.returncode < 8                      # robocopy: 0-7 都算成功
    print("  [%s] %s" % ("OK" if ok else "失败", name))


def main():
    print("源目录  : %s" % SRC)
    print("目标目录: %s" % TARGET)
    if os.path.abspath(TARGET).startswith(os.path.abspath(SRC)):
        print("!! 目标不能放在源目录内部，请换一个路径（移动硬盘/网络盘）")
        return
    for rel, required, desc in ITEMS:
        src = rel if os.path.isabs(rel) else os.path.join(SRC, rel)
        tag = "必需" if required else "可选"
        print("\n[%s] %s -- %s" % (tag, rel, desc))
        copy(src, os.path.join(TARGET, os.path.basename(rel)), rel)
    print("\n全部完成。把 %s 整个目录拷到新机器即可。" % TARGET)
    print("新机器上先读 迁移交接\\README_交接.md，第 3 节有环境搭建步骤。")


if __name__ == "__main__":
    main()
