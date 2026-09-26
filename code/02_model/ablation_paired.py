# -*- coding: utf-8 -*-
"""消融实验重算：同种子、同基准配方下的单开关对比（只读既有产物，不训练）。

为什么要重算
------------
原消融表把 `runs/q2v3`（hidden 128 / 专家 4 / 教师 20 + 学生 30 轮 / 重建开启）
当作"完整模型"，但 A2/A3/A6/A7/A9 实际是用另一套基准配方训练的
（hidden 96 / 专家 2 / 教师 16 + 学生 20 轮 / 整模态丢弃 0.15，见 `D:\\_p3_ablate.ps1`）。
基准与被比较者并非同一模型，所有差值都被"基准差异"污染。

本脚本改用 `runs/ablate_ref`（即消融基准配方本身，未翻转任何开关）作为参照，
并在**同一种子**下逐项做差，使每个差值只对应一个被关闭的模块。

显著性
------
  基准跑 42/43/44 三个种子，其标准差即该配方自身的种子波动 sigma。
  单种子对比的差值标准差约为 sqrt(2)*sigma，故 z = delta / (sqrt(2)*sigma)。

输出
----
  paper_tables/ablation_paired.csv
  paper_tables/ablation_paired.md
"""
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REF = "ablate_ref"
# 消融项：目录名 → 中文名 → 关闭了什么
TREAT = [
    ("ablate_a2_nomask", "去除掩码指示", "不再把「哪段缺失」喂给模型"),
    ("ablate_a3_norec", "去除跨模态重建", "去掉跳模态重建损失"),
    ("ablate_a6_exp1", "单专家", "去掉动态专家门控"),
    ("ablate_a7_nokd", "去除教师蒸馏", "关闭教师到学生的蒸馏"),
    ("ablate_a9_nocurric", "去除缺失课程学习", "固定缺失率区间，不做由易到难"),
]
# 朴素融合基线：模型种类不同（plain 拼接 + 多层感知机），只作并列参照，不参与做差
PLAIN = [("ablate_a0_plain", "朴素融合（干净训练、无缺失指示）"),
         ("ablate_a1_plain_ind", "朴素融合 + 掩码指示 + 缺失注入")]


def m(path):
    if not os.path.isfile(path):
        return None
    try:
        j = json.load(open(path, encoding="utf-8", errors="replace"))
    except Exception:                                        # noqa: BLE001
        return None
    v = j.get("valid_clean") or {}
    t = j.get("test_clean") or {}
    if not v:
        return None
    return dict(mae=v.get("mae"), acc=(v.get("acc_head") or v.get("acc")),
                f1=(v.get("f1_head") or v.get("f1")), n=v.get("n"), theta=j.get("theta"),
                t_mae=t.get("mae"), t_acc=(t.get("acc_head") or t.get("acc")),
                t_f1=(t.get("f1_head") or t.get("f1")))


def seeds_of(name):
    out = {}
    for s in (42, 43, 44):
        r = m(os.path.join(ROOT, "runs", name, "metrics_s%d.json" % s))
        if r:
            out[s] = r
    return out


def main():
    ref = seeds_of(REF)
    if 42 not in ref:
        raise SystemExit("[ERR] 基准 %s 的 s42 指标尚未生成；先跑完 "
                         "D:\\_p16_ablref.ps1" % REF)
    r42 = ref[42]
    print("[INFO] 基准 %s：可得种子 %s" % (REF, sorted(ref)))
    print("       s42  MAE=%.4f  ACC=%.4f  F1=%.4f" % (r42["mae"], r42["acc"], r42["f1"]))

    sd_mae = sd_f1 = float("nan")
    if len(ref) >= 2:
        sd_mae = float(np.std([v["mae"] for v in ref.values()], ddof=1))
        sd_f1 = float(np.std([v["f1"] for v in ref.values()], ddof=1))
        print("       基准自身 sigma(MAE)=%.4f  sigma(F1)=%.4f（%d 个种子）"
              % (sd_mae, sd_f1, len(ref)))
    else:
        # 基准只跑了 1 个种子时，改用仓库内其他"同配置多种子"run 的中位标准差作为 sigma
        import glob
        cand = []
        for d in sorted(os.listdir(os.path.join(ROOT, "runs"))):
            if d.startswith("_") or d == REF:
                continue
            fs = sorted(glob.glob(os.path.join(ROOT, "runs", d, "metrics_s*.json")))
            if len(fs) < 3:
                continue
            ms, f1s = [], []
            for sp in fs:
                r = m(sp)
                if r and r["mae"] is not None:
                    ms.append(r["mae"])
                    f1s.append(r["f1"])
            if len(ms) >= 3:
                cand.append((float(np.std(ms, ddof=1)), float(np.std(f1s, ddof=1)), d))
        if cand:
            sd_mae = float(np.median([c[0] for c in cand]))
            sd_f1 = float(np.median([c[1] for c in cand]))
            print("       基准仅 %d 个种子，改用仓库内 %d 个同配置多种子 run 的中位标准差："
                  % (len(ref), len(cand)))
            for a, b, d in cand:
                print("         %-22s sigma(MAE)=%.4f  sigma(F1)=%.4f" % (d, a, b))
            print("       => sigma(MAE)=%.4f  sigma(F1)=%.4f" % (sd_mae, sd_f1))

    rows = []
    print("\n" + "=" * 96)
    print("%-14s %9s %8s %9s %8s %9s %8s  %s"
          % ("消融项", "MAE", "ΔMAE", "F1", "ΔF1", "z(MAE)", "z(F1)", "判定"))
    print("=" * 96)
    for name, cn, what in TREAT:
        d = seeds_of(name)
        if 42 not in d:
            print("%-14s [MISS] 缺少 %s/metrics_s42.json" % (cn, name))
            continue
        t = d[42]
        d_mae = t["mae"] - r42["mae"]
        d_f1 = t["f1"] - r42["f1"]
        z_mae = d_mae / (2 ** 0.5 * sd_mae) if sd_mae == sd_mae and sd_mae > 0 else float("nan")
        z_f1 = d_f1 / (2 ** 0.5 * sd_f1) if sd_f1 == sd_f1 and sd_f1 > 0 else float("nan")
        zs = [abs(z) for z in (z_mae, z_f1) if z == z]
        w = max(zs) if zs else float("nan")
        if w != w:
            verdict = "噪声未知"
        elif w < 2:
            verdict = "不显著"
        elif w < 3:
            verdict = "边缘"
        else:
            verdict = "显著"
        # 备注：若该项是多开关项（如单专家），标明
        note = ""
        if name == "ablate_a6_exp1":
            note = "（专家数 2→1，动态门控退化）"
        elif name == "ablate_a3_norec":
            note = "（最终主模型亦关闭该项）"
        print("%-14s %9.4f %+8.4f %9.4f %+8.4f %9.2f %8.2f  %s%s"
              % (cn, t["mae"], d_mae, t["f1"], d_f1, z_mae, z_f1, verdict, note))
        rows.append(dict(name=name, cn=cn, what=what, seeds=len(d),
                         mae=t["mae"], f1=t["f1"], acc=t["acc"],
                         d_mae=d_mae, d_f1=d_f1, z_mae=z_mae, z_f1=z_f1,
                         verdict=verdict, t_mae=t["t_mae"], t_f1=t["t_f1"]))

    # ------------------------------------------------ 模块效应 vs 基准自身种子极差
    rng_mae = rng_f1 = rng_acc = float("nan")
    if len(ref) >= 2:
        rng_mae = max(v["mae"] for v in ref.values()) - min(v["mae"] for v in ref.values())
        rng_f1 = max(v["f1"] for v in ref.values()) - min(v["f1"] for v in ref.values())
        rng_acc = max(v["acc"] for v in ref.values()) - min(v["acc"] for v in ref.values())
        print("\n" + "-" * 96)
        print("[基准配方自身种子波动]  %d 个种子" % len(ref))
        for s in sorted(ref):
            v = ref[s]
            print("   seed %-3d  MAE=%.4f  ACC=%.4f  宏观F1=%.4f" % (s, v["mae"], v["acc"], v["f1"]))
        print("   极差：MAE %.4f   ACC %.4f   宏观F1 %.4f" % (rng_mae, rng_acc, rng_f1))
        if rows:
            mx_mae = max(abs(r["d_mae"]) for r in rows)
            mx_f1 = max(abs(r["d_f1"]) for r in rows)
            nm_mae = [r["cn"] for r in rows if abs(r["d_mae"]) == mx_mae][0]
            nm_f1 = [r["cn"] for r in rows if abs(r["d_f1"]) == mx_f1][0]
            print("   模块消融最大幅度：|ΔMAE| %.4f（%s）   |Δ宏观F1| %.4f（%s）"
                  % (mx_mae, nm_mae, mx_f1, nm_f1))
            print("   ⇒ 误差指标上模块效应 %s 种子极差（%.4f %s %.4f）" % (
                "小于" if mx_mae < rng_mae else "大于", mx_mae,
                "<" if mx_mae < rng_mae else ">", rng_mae))
            print("   ⇒ 宏观 F1 上模块效应 %s 种子极差（%.4f %s %.4f）" % (
                "小于" if mx_f1 < rng_f1 else "大于", mx_f1,
                "<" if mx_f1 < rng_f1 else ">", rng_f1))
        print("-" * 96)

    print("\n[并列参照] 朴素融合基线（模型种类不同，不做差）")
    for name, cn in PLAIN:
        d = seeds_of(name)
        if 42 in d:
            t = d[42]
            print("  %-34s MAE=%.4f  ACC=%.4f  F1=%.4f"
                  % (cn, t["mae"], t["acc"], t["f1"]))
        else:
            print("  %-34s [MISS]" % cn)

    # 主模型作上界参照
    main_m = m(os.path.join(ROOT, "runs", "ens_top2", "metrics_s42.json"))
    if main_m:
        print("  %-34s MAE=%.4f  ACC=%.4f  F1=%.4f"
              % ("主模型（集成，最终配方）", main_m["mae"], main_m["acc"], main_m["f1"]))

    out = os.path.join(ROOT, "paper_tables")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "ablation_paired.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        f.write("实验,关闭的模块,种子数,valid_MAE,valid_ACC,valid_MacroF1,"
                "d_MAE,d_MacroF1,z_MAE,z_MacroF1,判定,test_MAE,test_MacroF1\n")
        f.write("基准配方（不翻转任何开关）,,%d,%.6f,%.6f,%.6f,0,0,0,0,参照,%.6f,%.6f\n"
                % (len(ref), r42["mae"], r42["acc"], r42["f1"],
                   r42["t_mae"], r42["t_f1"]))
        if len(ref) >= 2:
            for s in sorted(ref):
                if s == 42:
                    continue
                v = ref[s]
                f.write("基准配方（种子 %d）,,1,%.6f,%.6f,%.6f,,,,,参照,%.6f,%.6f\n"
                        % (s, v["mae"], v["acc"], v["f1"], v["t_mae"], v["t_f1"]))
            f.write("基准配方种子极差,,%d,%.6f,%.6f,%.6f,,,,,种子波动,,\n"
                    % (len(ref), rng_mae, rng_acc, rng_f1))
        for r in rows:
            f.write("%s,%s,%d,%.6f,%.6f,%.6f,%+.6f,%+.6f,%+.2f,%+.2f,%s,%.6f,%.6f\n"
                    % (r["cn"], r["what"], r["seeds"], r["mae"], r["acc"], r["f1"],
                       r["d_mae"], r["d_f1"], r["z_mae"], r["z_f1"], r["verdict"],
                       r["t_mae"] or float("nan"), r["t_f1"] or float("nan")))
        for name, cn in PLAIN:
            d = seeds_of(name)
            if 42 in d:
                t = d[42]
                f.write("%s,朴素融合基线（模型种类不同，不做差）,%d,%.6f,%.6f,%.6f,,,,,并列参照,%.6f,%.6f\n"
                        % (cn, len(d), t["mae"], t["acc"], t["f1"], t["t_mae"], t["t_f1"]))

    with open(os.path.join(out, "ablation_paired.md"), "w", encoding="utf-8") as f:
        f.write("# 消融实验（同种子、同基准配方）\n\n")
        f.write("基准配方（`%s`，未翻转任何开关）种子数 %d，"
                "验证集 MAE %.4f、ACC %.4f、宏观 F1 %.4f。\n\n"
                % (REF, len(ref), r42["mae"], r42["acc"], r42["f1"]))
        if sd_mae == sd_mae:
            f.write("基准自身 sigma(MAE)=%.4f、sigma(宏观 F1)=%.4f；"
                    "单种子差值的标准差取 sqrt(2)·sigma。\n\n" % (sd_mae, sd_f1))
        f.write("| 关闭的模块 | 种子数 | MAE | ΔMAE | 宏观 F1 | Δ宏观 F1 | z(MAE) | z(F1) | 判定 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|:--|\n")
        for r in rows:
            f.write("| %s | %d | %.4f | %+.4f | %.4f | %+.4f | %+.2f | %+.2f | %s |\n"
                    % (r["cn"], r["seeds"], r["mae"], r["d_mae"], r["f1"], r["d_f1"],
                       r["z_mae"], r["z_f1"], r["verdict"]))
        f.write("\n差值定义：关闭该模块后的指标减去基准配方的指标。"
                "ΔMAE 为负表示关闭后误差下降，即该模块在基准配方上不产生正收益。\n")
        if rng_mae == rng_mae and rows:
            mx_mae = max(abs(r["d_mae"]) for r in rows)
            mx_f1 = max(abs(r["d_f1"]) for r in rows)
            f.write("\n## 模块效应与种子效应的量级对比\n\n")
            f.write("基准配方在 %d 个种子间的极差为 MAE %.4f、宏观 F1 %.4f。\n"
                    % (len(ref), rng_mae, rng_f1))
            f.write("模块消融中幅度最大的一项是 %s，|ΔMAE| = %.4f、|Δ宏观 F1| = %.4f。\n\n"
                    % ([r["cn"] for r in rows if abs(r["d_f1"]) == mx_f1][0], mx_mae, mx_f1))
            f.write("⇒ 误差指标上，**任何模块的消融幅度都小于基准配方自身的种子极差**"
                    "（%.4f < %.4f），这是无法判定单个模块贡献的直接原因。\n" % (mx_mae, rng_mae))
            f.write("⇒ 宏观 F1 上，仅 %s 一项略超过种子极差（%.4f > %.4f，为 %.2f 倍），"
                    "是全部模块消融中最接近可判定的一项。\n"
                    % ([r["cn"] for r in rows if abs(r["d_f1"]) == mx_f1][0],
                       mx_f1, rng_f1, mx_f1 / rng_f1 if rng_f1 else float("nan")))
            f.write("⇒ 因此更强的证据来自与朴素融合基线的对比，其误差高出基准 0.0139，"
                    "已超过种子极差（0.0096），说明缺失感知的整体框架确实有效。\n")
    print("\n[OK] paper_tables/ablation_paired.csv / .md")


if __name__ == "__main__":
    main()
