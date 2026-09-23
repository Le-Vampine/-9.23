# -*- coding: utf-8 -*-
r"""
迁移体积核算 + 生成迁移清单与交接包。

用法：python make_migration_package.py [--target D:\数学建模] [--make_zip]
"""
import argparse
import csv
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def dsize(path):
    """目录/文件大小（字节）与文件数。"""
    if not os.path.exists(path):
        return 0, 0
    if os.path.isfile(path):
        return os.path.getsize(path), 1
    tot, n = 0, 0
    for base, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(base, f))
                n += 1
            except OSError:
                pass
    return tot, n


def mb(x):
    return x / 1048576.0


def collect():
    items = []
    base = os.path.basename(ROOT)
    items.append(("【必需】赛题与方案", [
        (os.path.join(ROOT, "复杂场景下多模态情感识别的数学建模与算法设计.docx"), "赛题原文"),
        (os.path.join(ROOT, "赛事文件.txt"), "赛题纯文本"),
        (os.path.join(ROOT, "解决方案"), "6 份方案文档"),
    ]))
    items.append(("【必需】代码", [
        (os.path.join(ROOT, "code"), "全部脚本（含 02_model/）"),
    ]))
    items.append(("【必需】数据", [
        (os.path.join(ROOT, "附件2"), "附件2（含 aligned/unaligned/label.xlsx）"),
        (os.path.join(ROOT, "附件3-模态缺失特征样本"), "附件3（对齐/未对齐）"),
        (os.path.join(ROOT, "附件4-可解释专项视频样本与特征文件"), "附件4（含 videos/）"),
    ]))
    items.append(("【必需】中间产物", [
        (os.path.join(ROOT, "data_att"), "附件3/4 的文本特征（已用 BERT 补齐）"),
        (os.path.join(ROOT, "runs"), "checkpoint / metrics / sweep / 日志"),
        (os.path.join(ROOT, "submission"), "预测结果 CSV"),
        (os.path.join(ROOT, "figs"), "论文图"),
        (os.path.join(ROOT, "paper_tables"), "论文表格"),
    ]))
    items.append(("【可选】模型缓存", [
        (r"E:\hf_cache", "bert-base-uncased 权重（也可在新机器重新下载）"),
    ]))
    items.append(("【可删】临时", [
        (os.path.join(ROOT, "data_dummy"), "合成数据（可由 make_dummy_data.py 重建）"),
        (os.path.join(ROOT, ".vscode"), "VS Code 任务配置（含中文路径 label）"),
    ]))

    rows = []
    print("=" * 84)
    for group, entries in items:
        print("\n%s" % group)
        for path, desc in entries:
            sz, n = dsize(path)
            exists = os.path.exists(path)
            print("  %-58s %9.1f MB  %5d 文件  %s"
                  % (os.path.relpath(path, ROOT) if path.startswith(ROOT) else path,
                     mb(sz), n, "" if exists else "(不存在)"))
            rows.append(dict(group=group, path=path, desc=desc, exists=exists,
                             size_mb=round(mb(sz), 2), files=n))
    total_required = sum(r["size_mb"] for r in rows if r["exists"] and "必需" in r["group"])
    total_optional = sum(r["size_mb"] for r in rows if r["exists"] and "可选" in r["group"])
    print("\n" + "=" * 84)
    print("必需合计 = %.1f MB (%.2f GB)" % (total_required, total_required / 1024))
    print("可选（BERT 缓存）合计 = %.1f MB" % total_optional)
    print("迁移决定：附件2 全量迁移（含 unaligned 2897 MB）")
    print("整包约 %.2f GB（含 BERT 缓存）" % ((total_required + total_optional) / 1024))
    return rows


def write_manifest(rows, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["group", "path", "desc", "exists", "size_mb", "files"])
        w.writeheader()
        w.writerows(rows)
    print("[SAVE] %s" % out_csv)


def write_copy_script(rows, target, out_py, out_cmd):
    """
    生成复制脚本。
    ⚠️ 必须用 Python 而不是 .cmd：cmd.exe 按 OEM 代码页读文件，中文路径会乱掉。
       Python 通过 CreateProcessW 传参，中文路径安全。
    """
    tgt = target or r"E:\migrate_mathmodel"
    py = '''# -*- coding: utf-8 -*-
r"""
一键复制工作区到目标目录（移动硬盘 / 网络盘 / 新机器）。

用法：
  python copy_to_usb.py                    # 默认目标见 TARGET
  python copy_to_usb.py "F:\\\\数学建模迁移"     # 指定目标
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)                      # 工作区根目录
TARGET = sys.argv[1] if len(sys.argv) > 1 else r"TARGET_PLACEHOLDER"

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
    (r"E:\\hf_cache", False, "BERT 权重缓存（可选，也可在新机器重下）"),
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
        print("\\n[%s] %s -- %s" % (tag, rel, desc))
        copy(src, os.path.join(TARGET, os.path.basename(rel)), rel)
    print("\\n全部完成。把 %s 整个目录拷到新机器即可。" % TARGET)
    print("新机器上先读 迁移交接\\\\README_交接.md，第 3 节有环境搭建步骤。")


if __name__ == "__main__":
    main()
'''.replace("TARGET_PLACEHOLDER", tgt)

    with open(out_py, "w", encoding="utf-8") as f:
        f.write(py)
    print("[SAVE] %s" % out_py)

    cmd = "\n".join([
        "@echo off",
        "REM Wrapper: run the Python copy script (avoids cmd.exe encoding issues with CJK paths)",
        "set PYTHONUTF8=1",
        'cd /d "%~dp0"',
        "python -u copy_to_usb.py %*",
        "pause",
    ])
    with open(out_cmd, "w", encoding="ascii") as f:
        f.write(cmd)
    print("[SAVE] %s" % out_cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=r"E:\migrate_mathmodel")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join(ROOT, "迁移交接")
    os.makedirs(out_dir, exist_ok=True)

    rows = collect()
    write_manifest(rows, os.path.join(out_dir, "file_manifest.csv"))
    write_copy_script(rows, a.target,
                      os.path.join(out_dir, "copy_to_usb.py"),
                      os.path.join(out_dir, "copy_to_usb.cmd"))
    print("\n[交接包内容] %s" % out_dir)
    for f in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, f)
        print("   %-26s %9.1f KB" % (f, os.path.getsize(p) / 1024))
    print("\n[提示] 复制脚本用 Python 实现（cmd.exe 读中文路径会乱码）。"
          "运行 `python copy_to_usb.py <目标目录>` 或用 .cmd 包装。")


if __name__ == "__main__":
    main()
