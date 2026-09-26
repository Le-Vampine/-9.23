# -*- coding: utf-8 -*-
"""缺失「时长」口径的量化说明（只读附件2，不训练）。

回答一个审稿人必然会问的问题
------------------------------
题目要求分析"缺失**时长**"对性能的影响，而退化扫描用的因子是缺失**率** ρ。
两者不等价：绝对缺失时长（帧数）= ρ × 有效长度，而各样本的有效长度并不相同。

本脚本用附件2 验证划分的实测有效长度，把扫描中使用的每一档 ρ
换算成三种模态各自的平均绝对缺失帧数，从而给出 ρ ↔ 时长的明确对应表。
同时给出有效长度的分布，说明为何必须用归一化缺失率而非绝对帧数做跨样本比较。

输出
----
  paper_tables/missing_duration_map.csv
  paper_tables/missing_duration_map.md
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data_utils import load_pickle, infer_length_from_zeros   # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODS = ["text", "audio", "vision"]
MOD_CN = {"text": "文本", "audio": "语音", "vision": "视觉"}
RHOS = [0.1, 0.2, 0.3, 0.5, 1.0]


def main():
    pkl = os.path.join(ROOT, "附件2", "aligned_50.pkl")
    if not os.path.isfile(pkl):
        raise SystemExit("[ERR] 缺少 %s" % pkl)
    raw = load_pickle(pkl)
    sp = raw["valid"] if "valid" in raw else raw
    print("[INFO] 附件2 验证划分，字段：%s" % sorted(sp.keys()) if isinstance(sp, dict) else "")

    lens = {}
    for m in MODS:
        X = np.asarray(sp[m], dtype=np.float32)
        ln = infer_length_from_zeros(X).astype(np.float64)   # 该函数按批处理，返回 (N,)
        lens[m] = ln
        print("[LEN ] %-7s n=%d  mean=%.3f  median=%.1f  p10=%.1f  p90=%.1f  min=%d  max=%d"
              % (m, len(ln), ln.mean(), np.median(ln),
                 np.percentile(ln, 10), np.percentile(ln, 90), ln.min(), ln.max()))

    print("\n[有效长度分布]")
    for m in MODS:
        ln = lens[m]
        hist = [(k, int((ln == k).sum())) for k in range(1, 51)]
        hist = [h for h in hist if h[1] > 0]
        print("  %s: %s" % (MOD_CN[m], "  ".join("%d帧×%d" % h for h in hist[:14])))

    rows = []
    for rho in RHOS:
        for m in MODS:
            frames = rho * lens[m].mean()
            rows.append(dict(rho=rho, modality=m, mean_len=float(lens[m].mean()),
                             abs_frames=float(frames),
                             int_frames=int(round(frames))))
            # 逐样本取整后求均值，反映实际注入几何
            rows[-1]["abs_frames_rounded"] = float(np.mean(np.round(rho * lens[m])))

    print("\n[ρ → 平均绝对缺失帧数]")
    print("  rho   " + "".join("%-16s" % MOD_CN[m] for m in MODS))
    for rho in RHOS:
        cells = []
        for m in MODS:
            r = [x for x in rows if x["rho"] == rho and x["modality"] == m][0]
            cells.append("%.2f 帧 (%d)" % (r["abs_frames"], r["int_frames"]))
        print("  %.1f   %s" % (rho, "".join("%-16s" % c for c in cells)))

    # 跨模态可比性：同一 ρ 下三模态绝对帧数的离散程度
    print("\n[跨模态可比性] 同一 ρ 下三模态绝对帧数的极差与均值之比")
    worst = 0.0
    for rho in RHOS:
        v = [x["abs_frames"] for x in rows if x["rho"] == rho]
        ratio = (max(v) - min(v)) / np.mean(v)
        worst = max(worst, ratio)
        print("  rho=%.1f  极差/均值 = %.4f" % (rho, ratio))
    print("  最大相对极差 %.4f" % worst)
    print("  → 同一缺失率下文本丢失的帧数约为语音/视觉的两倍（L_text=50 vs 24.6/23.8），")
    print("    故须再用「绝对帧数匹配」复核结论（见 matched_duration_check.py）。")

    out = os.path.join(ROOT, "paper_tables")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "missing_duration_map.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        f.write("rho,modality,mean_effective_len,abs_missing_frames,abs_missing_frames_int,"
                "abs_frames_per_sample_mean\n")
        for r in rows:
            f.write("%.2f,%s,%.4f,%.4f,%d,%.4f\n" % (
                r["rho"], r["modality"], r["mean_len"], r["abs_frames"],
                r["int_frames"], r["abs_frames_rounded"]))

    with open(os.path.join(out, "missing_duration_map.md"), "w", encoding="utf-8") as f:
        f.write("# 缺失率与缺失时长的对应关系\n\n")
        f.write("附件二验证划分的实测有效长度（50 个序列位置中非尾部填充的位置数）：\n\n")
        f.write("| 模态 | 均值 | 中位数 | 10 分位 | 90 分位 | 最小 | 最大 |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for m in MODS:
            ln = lens[m]
            f.write("| %s | %.2f | %.1f | %.1f | %.1f | %d | %d |\n" % (
                MOD_CN[m], ln.mean(), np.median(ln), np.percentile(ln, 10),
                np.percentile(ln, 90), ln.min(), ln.max()))
        f.write("\n扫描中各档缺失率对应的平均绝对缺失帧数：\n\n")
        f.write("| 缺失率 | " + " | ".join(MOD_CN[m] for m in MODS) + " |\n|---|" + "---:|" * len(MODS) + "\n")
        for rho in RHOS:
            cells = []
            for m in MODS:
                r = [x for x in rows if x["rho"] == rho and x["modality"] == m][0]
                cells.append("%.2f（%d）" % (r["abs_frames"], r["int_frames"]))
            f.write("| %.1f | %s |\n" % (rho, " | ".join(cells)))
        f.write("\n同一缺失率下三模态绝对帧数的最大相对极差为 %.4f。文本模态全长 50 帧，"
                "语音与视觉平均有效长度约 24 帧，故同一缺失率下文本实际丢失的帧数约为另两者的两倍。"
                "为避免该差异干扰“退化由文本驱动”的结论，本文另在绝对缺失帧数匹配的条件下重做了对比，"
                "结果见 `code/02_model/matched_duration_check.py` 与论文相应小节。\n" % worst)

    print("\n[OK] paper_tables/missing_duration_map.csv / .md")


if __name__ == "__main__":
    main()
