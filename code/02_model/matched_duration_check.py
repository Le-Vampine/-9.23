# -*- coding: utf-8 -*-
"""缺失「时长」混淆的核验：在**绝对缺失帧数匹配**下重比三模态退化。

问题背景
--------
退化扫描的因子是缺失率 ρ，但三种模态的有效长度并不相同：
文本恒为 50 帧，语音与视觉约 24 帧（见 `missing_duration_map.py`）。
因此在同一个 ρ 下，文本实际丢失的帧数约为语音、视觉的两倍。
"退化由文本模态驱动"这一结论若只建立在同 ρ 比较上，会受该混淆影响。

本脚本的做法
------------
从既有扫描结果中挑出**绝对缺失帧数相互匹配**的场景对，在同一位置下重比退化量：
    ~5  帧：文本 ρ=0.1  ↔  语音 ρ=0.2  ↔  视觉 ρ=0.2
    ~25 帧：文本 ρ=0.5  ↔  语音 ρ=1.0  ↔  视觉 ρ=1.0
若在匹配时长下文本仍显著更差，则"文本驱动"的结论不依赖于时长混淆。

只读既有扫描产物，不训练。
"""
import csv
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 由 missing_duration_map.py 实测得到（附件2 验证划分）
MEAN_LEN = {"text": 50.000, "audio": 24.588, "vision": 23.819}
MOD_CN = {"text": "文本", "audio": "语音", "vision": "视觉"}
POS_CN = {"head": "头部", "middle": "中部", "tail": "尾部", "random": "随机"}

# 绝对帧数匹配组：(标签, {模态: rho})
MATCH = [
    ("约 5 帧", {"text": 0.1, "audio": 0.2, "vision": 0.2}),
    ("约 25 帧", {"text": 0.5, "audio": 1.0, "vision": 1.0}),
]

SWEEP = [
    ("本文模型", "runs/ens_top2/sweep_valid_ours_B.csv"),
    ("朴素融合基线", "figs/sweep_a0/sweep_analysis_input.csv"),
    ("朴素融合加指示", "figs/sweep_a1/sweep_analysis_input.csv"),
]


def load(rel):
    p = os.path.join(ROOT, rel.replace("/", os.sep))
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get(rows, mtype, pos, rho, key):
    for r in rows:
        if (r["missing_type"] == mtype and r["position"] == pos
                and abs(float(r["rho"]) - rho) < 1e-6):
            return float(r[key])
    return float("nan")


def main():
    print("=" * 78)
    print("绝对缺失帧数匹配表（由附件2 验证划分实测有效长度换算）")
    print("=" * 78)
    for lab, spec in MATCH:
        cells = ["%s ρ=%.1f → %.2f 帧" % (MOD_CN[m], spec[m], spec[m] * MEAN_LEN[m])
                 for m in ("text", "audio", "vision")]
        v = [spec[m] * MEAN_LEN[m] for m in ("text", "audio", "vision")]
        print("  %-9s %s   相对极差 %.2f%%"
              % (lab, "  |  ".join(cells), 100 * (max(v) - min(v)) / np.mean(v)))

    for tag, rel in SWEEP:
        rows = load(rel)
        if not rows:
            print("\n[MISS] %s" % rel)
            continue
        print("\n" + "=" * 78)
        print("%s   (%s)" % (tag, rel))
        print("=" * 78)
        print("  %-6s | %-9s | %-24s | %-24s" % ("匹配组", "位置", "绝对缺失帧增量 MAE", "绝对缺失帧增量 ACC"))
        for lab, spec in MATCH:
            for pos in ("head", "middle", "tail", "random"):
                dm = [get(rows, m, pos, spec[m], "d_mae") for m in ("text", "audio", "vision")]
                da = [get(rows, m, pos, spec[m], "d_acc") for m in ("text", "audio", "vision")]
                if any(x != x for x in dm):
                    continue
                cellm = "  ".join("%s %+.4f" % (MOD_CN[m], x)
                                  for m, x in zip(("text", "audio", "vision"), dm))
                cella = "  ".join("%s %+.4f" % (MOD_CN[m], x)
                                  for m, x in zip(("text", "audio", "vision"), da))
                print("  %-6s | %-9s | %-24s | %-24s" % (lab, POS_CN[pos], cellm, cella))

        # 位置平均
        print("\n  [位置平均]")
        for lab, spec in MATCH:
            out_m, out_a = [], []
            for m in ("text", "audio", "vision"):
                vals_m, vals_a = [], []
                for pos in ("head", "middle", "tail", "random"):
                    x = get(rows, m, pos, spec[m], "d_mae")
                    y = get(rows, m, pos, spec[m], "d_acc")
                    if x == x:
                        vals_m.append(x)
                    if y == y:
                        vals_a.append(y)
                out_m.append(np.mean(vals_m) if vals_m else np.nan)
                out_a.append(np.mean(vals_a) if vals_a else np.nan)
            print("   %-9s ΔMAE  文本 %+.5f  语音 %+.5f  视觉 %+.5f   文本/语音 = %.1f 倍"
                  % (lab, out_m[0], out_m[1], out_m[2],
                     out_m[0] / out_m[1] if out_m[1] not in (0, np.nan) and out_m[1] != 0 else float("nan")))
            print("   %-9s ΔACC  文本 %+.5f  语音 %+.5f  视觉 %+.5f"
                  % (lab, out_a[0], out_a[1], out_a[2]))


if __name__ == "__main__":
    main()
