# -*- coding: utf-8 -*-
"""附件3 专项测试集的结果汇总与展示（只读预测文件，不训练、不改预测）。

产出
----
  paper_tables/att3_pred_summary.csv     机器可读汇总
  paper_tables/att3_pred_summary.md      论文表格的文本来源
  figs/Q2_附件3预测结果构成_分组条形图.png
  figs/Q2_附件3预测强度与置信度分布_直方图.png

设计口径
--------
  * 极性同时给出两个口径：分类头 argmax（主口径）与强度阈值口径，两者一致率一并汇报，
    与 `解决方案/07_问题2指标口径与合规性说明.md` 的定义保持一致。
  * "含缺失"指三个模态中任一模态的实测缺失率大于零；该字段由 predict 阶段直接写入提交文件。
  * 本脚本只读 `submission/pred_att3.csv`，所有数字均可在提交文件上逐行核对。
"""
import csv
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 与全文一致的自然配色（visualization.md 指定）
NATURE = ["#3B5F8A", "#A23B3B", "#4A7C59", "#B8B8B8", "#F0E6D3"]
CLASS_NAME = ["负向", "中性", "正向"]
# 提交文件里的极性写成英文，统一映射到中文，避免统计落空
EN2CN = {"Negative": "负向", "Neutral": "中性", "Positive": "正向"}

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "axes.unicode_minus": False,
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def composition(labels):
    lab = ["负向", "中性", "正向"]
    out = {}
    n = len(labels)
    for c in lab:
        k = sum(1 for x in labels if x == c)
        out[c] = (k, k / n if n else 0.0)
    return out


def main():
    pred_path = os.path.join(ROOT, "submission", "pred_att3.csv")
    if not os.path.isfile(pred_path):
        raise SystemExit("[ERR] 缺少 submission/pred_att3.csv")
    rows = read_csv(pred_path)
    n = len(rows)
    print("[INFO] 附件3 提交文件 %d 条样本，%d 个字段" % (n, len(rows[0])))
    unknown = sorted({r["pred_label"] for r in rows} - set(EN2CN))
    if unknown:
        raise SystemExit("[ERR] pred_label 出现未预期取值：%s" % unknown)

    head = [EN2CN.get(r["pred_label"], r["pred_label"]) for r in rows]        # 分类头主口径
    th = [EN2CN.get(r["pred_label_theta"], r["pred_label_theta"]) for r in rows]  # 阈值口径
    score = np.array([float(r["pred_score"]) for r in rows])
    raw = np.array([float(r["pred_score_raw"]) for r in rows])
    conf = np.array([float(r["confidence"]) for r in rows])
    conf_th = np.array([float(r["confidence_theta"]) for r in rows])
    m_t = np.array([float(r["missing_text_ratio"]) for r in rows])
    m_a = np.array([float(r["missing_audio_ratio"]) for r in rows])
    m_v = np.array([float(r["missing_vision_ratio"]) for r in rows])
    n_int = np.array([int(r["n_missing_intervals"]) for r in rows])
    has_miss = (m_t + m_a + m_v) > 0

    agree = float(np.mean([a == b for a, b in zip(head, th)]))
    comp_h = composition(head)
    comp_t = composition(th)

    print("\n[极性构成 · 分类头主口径]")
    for c in CLASS_NAME:
        k, p = comp_h[c]
        print("   %s %2d 条 (%.1f%%)" % (c, k, 100 * p))
    print("[极性构成 · 阈值口径]  " + "  ".join(
        "%s %d" % (c, comp_t[c][0]) for c in CLASS_NAME))
    print("[两口径一致率] %.4f" % agree)

    print("\n[强度分数]  均值 %.4f  标准差 %.4f  中位数 %.4f  范围 [%.4f, %.4f]"
          % (score.mean(), score.std(ddof=1), np.median(score), score.min(), score.max()))
    print("[绝对强度]  均值 %.4f  中位数 %.4f  最大 %.4f"
          % (np.abs(score).mean(), np.median(np.abs(score)), np.abs(score).max()))
    print("[置信度]    均值 %.4f  标准差 %.4f  最低 %.4f"
          % (conf.mean(), conf.std(ddof=1), conf.min()))

    print("\n[实测缺失]")
    for nm, m in (("文本", m_t), ("语音", m_a), ("视觉", m_v)):
        print("   %s 含缺失 %2d/%d 条 (%.1f%%)，平均缺失率 %.4f，最大 %.4f"
              % (nm, int((m > 0).sum()), n, 100 * (m > 0).mean(), m.mean(), m.max()))
    print("   任一模态含缺失 %d/%d 条 (%.1f%%)，缺失区间数均值 %.2f（最大 %d）"
          % (int(has_miss.sum()), n, 100 * has_miss.mean(), n_int.mean(), n_int.max()))

    print("\n[含缺失 vs 无缺失]")
    sub = {}
    for tag, mask in (("含缺失", has_miss), ("无缺失", ~has_miss)):
        k = int(mask.sum())
        if k == 0:
            sub[tag] = None
            print("   %s 0 条" % tag)
            continue
        cc = composition([head[i] for i in range(n) if mask[i]])
        s = dict(n=k,
                 neg=cc["负向"][0], neu=cc["中性"][0], pos=cc["正向"][0],
                 score=float(score[mask].mean()),
                 abs_score=float(np.abs(score[mask]).mean()),
                 conf=float(conf[mask].mean()),
                 big=float((np.abs(score[mask]) > 1.0).mean()))
        sub[tag] = s
        print("   %s %2d 条  极性 负/中/正 = %d/%d/%d  平均强度 %+.4f  平均绝对强度 %.4f"
              "  平均置信度 %.4f  |强度|>1 占比 %.3f"
              % (tag, k, s["neg"], s["neu"], s["pos"], s["score"], s["abs_score"], s["conf"], s["big"]))

    # ------------------------------------------------------------ 图1 构成
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    groups = ["全样本", "含缺失", "无缺失"]
    data = []
    for g in groups:
        if g == "全样本":
            data.append([comp_h[c][0] for c in CLASS_NAME])
        else:
            s = sub.get(g)
            data.append([s["neg"], s["neu"], s["pos"]] if s else [0, 0, 0])
    x = np.arange(len(groups))
    w = 0.26
    for j, c in enumerate(CLASS_NAME):
        axes[0].bar(x + (j - 1) * w, [d[j] for d in data], w * 0.92,
                    color=NATURE[j], edgecolor="white", linewidth=0.6, label=c)
        for xi, d in zip(x, data):
            if d[j] > 0:
                axes[0].text(xi + (j - 1) * w, d[j] + 0.25, str(d[j]),
                             ha="center", va="bottom", fontsize=8.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups)
    axes[0].set_ylabel("样本数")
    axes[0].set_title("(a) 预测极性构成")
    axes[0].set_ylim(0, max(max(d) for d in data) * 1.40)
    axes[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper right")

    # 右图：平均绝对强度与平均置信度
    labels2 = groups
    absv = [float(np.abs(score).mean())] + [sub[g]["abs_score"] if sub.get(g) else 0 for g in groups[1:]]
    cfv = [float(conf.mean())] + [sub[g]["conf"] if sub.get(g) else 0 for g in groups[1:]]
    x2 = np.arange(len(labels2))
    axes[1].bar(x2 - 0.17, absv, 0.32, color=NATURE[0], edgecolor="white", label="平均绝对强度")
    axes[1].bar(x2 + 0.17, cfv, 0.32, color=NATURE[1], edgecolor="white", label="平均置信度")
    for xi, v in zip(x2 - 0.17, absv):
        axes[1].text(xi, v + 0.012, "%.3f" % v, ha="center", va="bottom", fontsize=8.5)
    for xi, v in zip(x2 + 0.17, cfv):
        axes[1].text(xi, v + 0.012, "%.3f" % v, ha="center", va="bottom", fontsize=8.5)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels2)
    axes[1].set_ylim(0, max(absv + cfv) * 1.30)
    axes[1].set_title("(b) 强度幅值与置信度")
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p1 = os.path.join(ROOT, "figs", "Q2_附件3预测结果构成_分组条形图.png")
    fig.savefig(p1)
    plt.close(fig)
    print("\n[OK] %s" % os.path.basename(p1))

    # ------------------------------------------------------------ 图2 分布
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    bins = np.linspace(-2.0, 2.0, 21)
    axes[0].hist(score[~has_miss], bins=bins, color=NATURE[3], edgecolor="white",
                 linewidth=0.6, label="无缺失")
    axes[0].hist(score[has_miss], bins=bins, color=NATURE[0], alpha=0.82,
                 edgecolor="white", linewidth=0.6, label="含缺失")
    for t in (-0.325, 0.325):
        axes[0].axvline(t, color=NATURE[1], linestyle="--", linewidth=1.0)
    axes[0].axvline(0.0, color="#666666", linewidth=0.8)
    axes[0].set_xlabel("预测情感强度")
    axes[0].set_ylabel("样本数")
    axes[0].set_title("(a) 预测强度分布")
    axes[0].legend(frameon=False, fontsize=9)

    cbins = np.linspace(0.28, 1.0, 25)
    axes[1].hist(conf, bins=cbins, color=NATURE[2], edgecolor="white", linewidth=0.6)
    axes[1].axvline(conf.mean(), color=NATURE[1], linestyle="--", linewidth=1.1)
    axes[1].annotate("均值 %.3f" % conf.mean(),
                     xy=(conf.mean(), axes[1].get_ylim()[1] * 0.82),
                     xytext=(conf.mean() + 0.03, axes[1].get_ylim()[1] * 0.86),
                     fontsize=9, color=NATURE[1])
    axes[1].set_xlabel("分类头置信度")
    axes[1].set_ylabel("样本数")
    axes[1].set_title("(b) 置信度分布")
    fig.tight_layout()
    p2 = os.path.join(ROOT, "figs", "Q2_附件3预测强度与置信度分布_直方图.png")
    fig.savefig(p2)
    plt.close(fig)
    print("[OK] %s" % os.path.basename(p2))

    # ------------------------------------------------------------ 汇总落盘
    out_dir = os.path.join(ROOT, "paper_tables")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "att3_pred_summary.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["指标", "全样本", "含缺失", "无缺失"])
        s1, s2 = sub.get("含缺失"), sub.get("无缺失")
        w.writerow(["样本数", n, s1["n"] if s1 else 0, s2["n"] if s2 else 0])
        w.writerow(["预测负向", comp_h["负向"][0], s1["neg"] if s1 else 0, s2["neg"] if s2 else 0])
        w.writerow(["预测中性", comp_h["中性"][0], s1["neu"] if s1 else 0, s2["neu"] if s2 else 0])
        w.writerow(["预测正向", comp_h["正向"][0], s1["pos"] if s1 else 0, s2["pos"] if s2 else 0])
        w.writerow(["平均强度", "%.4f" % score.mean(),
                    "%.4f" % s1["score"] if s1 else "",
                    "%.4f" % s2["score"] if s2 else ""])
        w.writerow(["平均绝对强度", "%.4f" % np.abs(score).mean(),
                    "%.4f" % s1["abs_score"] if s1 else "",
                    "%.4f" % s2["abs_score"] if s2 else ""])
        w.writerow(["平均置信度", "%.4f" % conf.mean(),
                    "%.4f" % s1["conf"] if s1 else "",
                    "%.4f" % s2["conf"] if s2 else ""])
        w.writerow(["两口径一致率", "%.4f" % agree, "", ""])
        w.writerow(["实测缺失率(文本)", "%.4f" % m_t.mean(), "", ""])
        w.writerow(["实测缺失率(语音)", "%.4f" % m_a.mean(), "", ""])
        w.writerow(["实测缺失率(视觉)", "%.4f" % m_v.mean(), "", ""])

    with open(os.path.join(out_dir, "att3_pred_summary.md"), "w", encoding="utf-8") as f:
        f.write("# 附件3 预测结果汇总\n\n")
        f.write("样本数 %d；分类头主口径与阈值口径的一致率 %.4f。\n\n" % (n, agree))
        f.write("| 指标 | 全样本 | 含缺失 | 无缺失 |\n|---|---:|---:|---:|\n")
        f.write("| 样本数 | %d | %s | %s |\n" % (
            n, s1["n"] if s1 else 0, s2["n"] if s2 else 0))
        f.write("| 预测负向 | %d | %d | %d |\n" % (
            comp_h["负向"][0], s1["neg"] if s1 else 0, s2["neg"] if s2 else 0))
        f.write("| 预测中性 | %d | %d | %d |\n" % (
            comp_h["中性"][0], s1["neu"] if s1 else 0, s2["neu"] if s2 else 0))
        f.write("| 预测正向 | %d | %d | %d |\n" % (
            comp_h["正向"][0], s1["pos"] if s1 else 0, s2["pos"] if s2 else 0))
        f.write("| 平均强度 | %.4f | %.4f | %.4f |\n" % (
            score.mean(), s1["score"] if s1 else np.nan, s2["score"] if s2 else np.nan))
        f.write("| 平均绝对强度 | %.4f | %.4f | %.4f |\n" % (
            np.abs(score).mean(), s1["abs_score"] if s1 else np.nan,
            s2["abs_score"] if s2 else np.nan))
        f.write("| 平均置信度 | %.4f | %.4f | %.4f |\n" % (
            conf.mean(), s1["conf"] if s1 else np.nan, s2["conf"] if s2 else np.nan))
        f.write("\n实测缺失：文本 %.4f，语音 %.4f，视觉 %.4f（平均缺失率）。\n"
                % (m_t.mean(), m_a.mean(), m_v.mean()))
    print("[OK] paper_tables/att3_pred_summary.csv / .md")


if __name__ == "__main__":
    main()
