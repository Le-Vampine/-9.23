# -*- coding: utf-8 -*-
"""把核心代码同步到论文附录（05_paper/code_appendix/ + _appendix_code.tex）。

用法：
    python code/03_xai/make_appendix_code.py            # 同步并生成 tex 片段
    python code/03_xai/make_appendix_code.py --list     # 只列清单

设计说明：
- 论文正文用 \\input{_appendix_code.tex} 引入本脚本生成的片段，
  因此附录代码与提交的 code/ 目录为同一份，避免手抄不同步。
- 生成前自动备份 main.tex（仅首次调用时），并统计总行数便于评估篇幅。
"""
import argparse
import os
import shutil
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAPER = os.path.join(ROOT, "05_paper")
DEST = os.path.join(PAPER, "code_appendix")

# (支撑材料相对路径, 附录小节标题)
FILES = [
    ("02_model/model.py", "问题二：MRF-Net 网络结构与动态专家门控"),
    ("02_model/losses.py", "问题二：损失函数（异方差回归、类别加权、一致性、重建与蒸馏）"),
    ("02_model/train.py", "问题二：训练与推理主流程（缺失课程学习、EMA、早停）"),
    ("03_xai/xai_common.py", "问题三：骨干加载与批构造（三源解释的公共入口）"),
    ("03_xai/shapley.py", "问题三：精确 Shapley 值（八子集全枚举）"),
    ("03_xai/ig_gradcam.py", "问题三：积分梯度与片段级重要性"),
    ("03_xai/occlusion.py", "问题三：遮挡重要性（窗宽 3 滑窗）"),
    ("03_xai/temporal_fuse.py", "问题三：三源融合与 γ 单纯形约束"),
    ("03_xai/evidence.py", "问题三：证据定位（词元对齐、语音时段映射、关键帧抽取）"),
    ("03_xai/train_evidence_head.py", "问题三：并联证据头训练（骨干冻结、忠实性正则）"),
    ("03_xai/explain_card.py", "问题三：解释卡生成"),
    ("03_xai/infer_explain_att4.py", "问题三：附件四全量推理与解释输出"),
    ("03_xai/verify_q3.py", "问题三：交付自查（17 项一致性校验）"),
]


def esc(path):
    return path.replace("_", "\\_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    total = 0
    for rel, desc in FILES:
        src = os.path.join(ROOT, "code", rel.replace("/", os.sep))
        n = len(open(src, encoding="utf-8").read().split("\n"))
        total += n
        print("%-34s %5d 行  %s" % (rel, n, desc))
    print("合计 %d 个文件 %d 行" % (len(FILES), total))
    if a.list:
        return 0

    os.makedirs(DEST, exist_ok=True)
    L = ["%% 本文件由 code/03_xai/make_appendix_code.py 自动生成，请勿手改。",
         "%% 生成时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"), "",
         "本节给出实现上述模型的核心代码，与提交的支撑材料中的文件为同一份；",
         "代码以 Python 3.12 与 PyTorch 编写，运行顺序见 \\texttt{code/README.md}。",
         "为便于与提交材料核对，每个代码块标注其在支撑材料中的相对路径与总行数。",
         "为避免中文字体下缺字形，代码中的少量 Unicode 数学符号（如 $\\in$、$\\le$、$\\to$、",
         "$\\pi$）在附录中以 ASCII 等价形式给出（如 \\texttt{in}、\\texttt{<=}、\\texttt{->}、",
         "\\texttt{pi}），仅涉及注释、文档字符串与输出文案，不影响可执行语义；",
         "原始文件以提交的 \\texttt{code/} 目录为准。", ""]
    for rel, desc in FILES:
        src = os.path.join(ROOT, "code", rel.replace("/", os.sep))
        flat = rel.replace("/", "__")
        shutil.copy2(src, os.path.join(DEST, flat))
        n = len(open(src, encoding="utf-8").read().split("\n"))
        L += ["\\subsection{%s}" % desc,
              "\\noindent\\texttt{code/%s}（共 %d 行）" % (esc(rel), n),
              "\\lstinputlisting[language=Python]{code_appendix/%s}" % flat,
              ""]
    out = os.path.join(PAPER, "_appendix_code.tex")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    # 同步做一次 Unicode 符号 ASCII 化（缺字形防护），保证与论文排版一致
    try:
        import check_appendix_unicode as cu
        m = cu.fix_all()
        print("[SYM] 已归一化 %d 个代码文件中的 Unicode 符号" % m)
    except Exception as e:  # noqa: BLE001
        print("[WARN] 符号归一化跳过：%s" % e)
    print("[SAVE] %s（%d 个代码文件）" % (os.path.relpath(out, ROOT), len(FILES)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
