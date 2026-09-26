# -*- coding: utf-8 -*-
r"""问题3 交接包：把交付物整理成 handover_problem3/（结构对齐问题2 的 handover_problem2/）。

产出：
  handover_problem3/
    ├── 00_问题3输出清单.md     自动生成（核心数字 + 阅读顺序 + 论文位置）
    ├── manifest.csv            机器可读清单（分组/路径/说明/字节）
    ├── 01_实验数据表/           paper_tables/Q3_*.csv + runs/q3 的报告 JSON
    ├── 02_图表/                 figs/Q3_*.png
    ├── 03_流程与架构/           tikz/Q3_*.tex 与编译产物
    ├── 04_核心代码/             code/03_xai/*.py + code/assets 说明
    ├── 05_方案文档/             解决方案/03、11、12
    ├── 06_提交文件/             pred_explain_att4.csv + explain_cards/ + evidence/
    └── 07_论文/                 main.tex + main.pdf（成稿）

用法：
  python make_handover.py                    # 复制 + 生成清单
  python make_handover.py --dry_run          # 只打印将要复制的内容与体积
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
DEST = os.path.join(ROOT, "handover_problem3")


def sz(p):
    return os.path.getsize(p) if os.path.isfile(p) else 0


def collect():
    """返回 [(分组, 源路径, 目标相对路径, 说明)]"""
    items = []
    # 01 实验数据表
    for p in sorted(glob.glob(os.path.join(ROOT, "paper_tables", "Q3_*.csv"))):
        items.append(("01_实验数据表", p, os.path.join("01_实验数据表", os.path.basename(p)),
                      "论文问题三用表"))
    for pat, desc in ((("q3_shapley_*_report.json"), "精确 Shapley（含数据集级效用）"),
                      (("q3_temporal_ig_*_report.json"), "积分梯度配置与稀疏性"),
                      (("q3_temporal_occ_*_report.json"), "遮挡重要性配置"),
                      (("q3_gamma_*.json"), "三源融合权重标定"),
                      (("q3_faithfulness_*_report.json"), "解释质量指标"),
                      (("q3_valid_analysis.json"), "验证集分析汇总"),
                      (("q3_evidence_att4.json"), "证据定位记录"),
                      (("q3_cards_att4.json"), "解释卡数据"),
                      (("q3_fig_captions.json"), "论文图说明文字"),
                      (("q3_explain_meta.json"), "提交文件元信息"),
                      (("d1_recalib_report.json"), "D1 文本长度口径修正报告"),
                      (("q3_evidence_head_report.json"), "证据头训练记录"),
                      (("verify_q3.json"), "交付自查报告")):
        for p in sorted(glob.glob(os.path.join(ROOT, "runs", "q3", pat))):
            items.append(("01_实验数据表", p,
                          os.path.join("01_实验数据表", os.path.basename(p)), desc))
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "q3", "q3_shapley_*.csv"))):
        items.append(("01_实验数据表", p, os.path.join("01_实验数据表", os.path.basename(p)),
                      "逐样本模态作用度"))
    for name in ("lit_vs_ours.csv",):
        p = os.path.join(ROOT, "paper_tables", name)
        if os.path.isfile(p):
            items.append(("01_实验数据表", p, os.path.join("01_实验数据表", name),
                          "文献基准与本文结果对照"))
    # 02 图表
    for p in sorted(glob.glob(os.path.join(ROOT, "figs", "Q3_*.png"))):
        items.append(("02_图表", p, os.path.join("02_图表", os.path.basename(p)), "论文用图"))
    # 03 流程与架构
    for p in sorted(glob.glob(os.path.join(ROOT, "05_paper", "tikz", "Q3_*"))):
        items.append(("03_流程与架构", p, os.path.join("03_流程与架构", os.path.basename(p)),
                      "TikZ 源与预览"))
    # 04 核心代码
    for p in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        if os.path.basename(p).startswith("_"):
            continue
        items.append(("04_核心代码", p, os.path.join("04_核心代码", os.path.basename(p)),
                      "问题3 代码"))
    for p in glob.glob(os.path.join(CODE, "assets", "*")):
        items.append(("04_核心代码", p, os.path.join("04_核心代码", "assets", os.path.basename(p)),
                      "词表与资源清单"))
    for p in glob.glob(os.path.join(ROOT, "runs", "q3", "q3_evidence_head.pt")):
        items.append(("04_核心代码", p, os.path.join("04_核心代码", os.path.basename(p)),
                      "证据头权重（若已训练）"))
    # 05 方案文档
    for name in ("03_问题3_可解释性情感预测.md", "11_问题3实施方案与执行计划.md",
                 "12_问题3加分建议与预计产出清单.md",
                 "13_问题3实施记录与问题2附件三口径修订.md"):
        p = os.path.join(ROOT, "解决方案", name)
        if os.path.isfile(p):
            items.append(("05_方案文档", p, os.path.join("05_方案文档", name), "方案/计划"))
    # 06 提交文件
    for p in (os.path.join(ROOT, "submission", "pred_explain_att4.csv"),
              os.path.join(ROOT, "submission", "pred_att4.csv")):
        if os.path.isfile(p):
            items.append(("06_提交文件", p, os.path.join("06_提交文件", os.path.basename(p)),
                          "附件四预测与解释"))
    for p in sorted(glob.glob(os.path.join(ROOT, "submission", "explain_cards", "*.json"))):
        items.append(("06_提交文件", p,
                      os.path.join("06_提交文件", "explain_cards", os.path.basename(p)),
                      "解释卡 JSON"))
    for p in sorted(glob.glob(os.path.join(ROOT, "submission", "evidence", "*"))):
        items.append(("06_提交文件", p, os.path.join("06_提交文件", "evidence",
                                                     os.path.basename(p)), "关键帧"))
    # 07 论文
    for name in ("main.tex", "main.pdf"):
        p = os.path.join(ROOT, "05_paper", name)
        if os.path.isfile(p):
            items.append(("07_论文", p, os.path.join("07_论文", name), "论文成稿"))
    return items


def brief():
    """从 JSON 报告里抽取给论文用的核心数字。"""
    def j(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return {}
    out = {}
    sv = j(os.path.join(ROOT, "runs", "q3", "q3_shapley_valid_report.json"))
    if sv:
        out["全局φ(-MAE)"] = sv.get("global", {}).get("phi_negmae")
        out["全局φ(MacroF1)"] = sv.get("global", {}).get("phi_macrof1")
        out["八子集 MAE"] = sv.get("global", {}).get("v_negmae")
        out["八子集 MacroF1"] = sv.get("global", {}).get("v_macrof1")
        out["π_soft 均值"] = sv.get("pi_soft_mean")
        out["ReLU 归一退化样本数"] = sv.get("pi_relu_degenerate")
    va = j(os.path.join(ROOT, "runs", "q3", "q3_valid_analysis.json"))
    if va:
        out["模态忠实性"] = va.get("modality_faithfulness", {}).get("delta_sample")
        out["与问题2 一致性秩相关"] = va.get("consistency_with_q2", {}).get("spearman")
        out["π-缺失率一致性"] = va.get("missing_rate_consistency")
        out["错误归因"] = va.get("error_attribution")
    fh = j(os.path.join(ROOT, "runs", "q3", "q3_faithfulness_att4_report.json"))
    if fh:
        out["解释质量（附件4）"] = fh.get("summary")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--dest", default=DEST)
    a = ap.parse_args()
    items = collect()
    print("=== 交接包 %s ===" % os.path.relpath(a.dest, ROOT))
    tot = 0
    per = {}
    for grp, src, rel, desc in items:
        tot += sz(src)
        per[grp] = per.get(grp, 0) + sz(src)
    for grp in sorted(per):
        n = len([1 for it in items if it[0] == grp])
        print("  %-14s %3d 项  %7.2f MB" % (grp, n, per[grp] / 1024 / 1024))
    print("  合计 %d 项  %.2f MB" % (len(items), tot / 1024 / 1024))
    if a.dry_run:
        for grp, src, rel, desc in items:
            print("    %-14s %s" % (grp, rel))
        return 0

    os.makedirs(a.dest, exist_ok=True)
    rows = []
    for grp, src, rel, desc in items:
        dst = os.path.join(a.dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        rows.append(dict(分组=grp, 路径=rel.replace("\\", "/"), 说明=desc, 字节=sz(src)))
    with open(os.path.join(a.dest, "manifest.csv"), "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["分组", "路径", "说明", "字节"])
        w.writeheader()
        w.writerows(rows)

    b = brief()
    L = ["# 问题3 输出清单（给写论文的同学）", "",
         "> 本文件由 `code/03_xai/make_handover.py` 自动生成；核心数字取自 `01_实验数据表/` 下的 JSON 报告，可逐项核对。",
         "> 模型：`runs/ens_top2`（MRF-Net 两种子集成，与问题2 同一骨干）；决策阈值 θ=0.325（valid 上选定）。", ""]
    L += ["## 一、核心数字", ""]
    for k, v in b.items():
        L.append("- **%s**：`%s`" % (k, json.dumps(v, ensure_ascii=False)))
    L += ["", "## 二、目录结构", ""]
    for grp in sorted(per):
        L.append("- `%s/`（%d 项，%.2f MB）" % (grp, len([1 for it in items if it[0] == grp]),
                                              per[grp] / 1024 / 1024))
    L += ["", "## 三、建议阅读顺序", "",
          "1. 本文件第一节（核心数字）→ 2. `05_方案文档/11`（实施方案）→ 3. `01_实验数据表/` 按论文小节取数",
          "→ 4. `02_图表/`（Q3_*.png，说明文字见 `01_实验数据表/q3_fig_captions.json`）→ 5. `04_核心代码/`（复现说明见 README）。", ""]
    L += ["## 四、论文问题三章节对应关系", "",
          "| 题目要求 | 论文小节 | 用哪些输出 |", "|---|---|---|",
          "| 可解释性模型的建模原理、网络结构、目标函数、训练方案与关键参数 | §5.3.1–5.3.2 | 证据头训练记录 `q3_evidence_head_report.json`、结构图 TikZ |",
          "| 典型样本解释卡 | §5.3.4 | `02_图表/Q3_解释卡_*.png`、`06_提交文件/explain_cards/*.json` |",
          "| 局部片段重要性可视化与三模态作用差异 | §5.3.5 | 热力图、箱线图、`Q3_模态作用度_按真值极性.csv` |",
          "| 附件四全量预测与解释汇总 | §5.3.6 | `06_提交文件/pred_explain_att4.csv`、`Q3_附件4预测与解释汇总_分组条形图.png` |",
          "| 验证集基础性能、可视化与错误归因 | §5.3.6 | `Q3_错误归因.csv`、`Q3_解释与错误关联_箱线图.png` |", ""]
    with open(os.path.join(a.dest, "00_问题3输出清单.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("[SAVE] %s（%d 项文件 + manifest + 清单）"
          % (os.path.relpath(a.dest, ROOT), len(rows)))
    print("HANDOVER_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
