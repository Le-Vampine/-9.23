# -*- coding: utf-8 -*-
"""超参搜索 screen vs full 阶段对比与显著性核验（只读，不训练）。

背景
----
  stage=screen : 训练集子集(1024) + 短轮数(教师6/学生8) → 快速淘汰
  stage=full   : 全量训练集 + 与主实验一致轮数(教师12/学生14) → 复赛胜出者

本脚本回答三个问题：
  Q1 p15d 全量阶段是否确实跑完、结果如何；
  Q2 screen 阶段的排序能否预测 full 阶段的排序（"排序可迁移性"）；
  Q3 c06_kd05 相对基线 c00_baseline 的改善是否超出单种子噪声（显著性）。

显著性口径
----------
  从仓库中"同配置多种子"的历史 run 目录估计单种子标准差 sigma，
  再以 z = delta / sqrt(2) / sigma 近似两独立单种子配置之差的显著性
  （两个配置各只训了 1 个种子，方差为 2*sigma^2）。

输出
----
  paper_tables/tune_screen_vs_full.csv
  paper_tables/tune_seed_noise.csv
  paper_tables/tune_full_report.md
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TUNE = os.path.join(ROOT, "runs", "tune")
OUT = os.path.join(ROOT, "paper_tables")
TAGS = ["c00_baseline", "c06_kd05", "c04_dropout50", "c01_hidden128"]


def read_csv(path):
    """极简 CSV 读取（列内不含逗号），避免依赖 pandas 的编码差异。"""
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    head = lines[0].split(",")
    rows = []
    for ln in lines[1:]:
        vals = ln.split(",")
        if len(vals) != len(head):
            continue
        rows.append(dict(zip(head, vals)))
    return rows


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------- Q1 & Q2
screen = {r["tag"]: r for r in read_csv(os.path.join(TUNE, "grid_screen.csv"))}
full = {r["tag"]: r for r in read_csv(os.path.join(TUNE, "grid_full.csv"))}

print("=" * 78)
print("Q1  全量阶段(full)完成情况")
print("=" * 78)
if not full:
    print("[ERR] 未找到 grid_full.csv —— 全量阶段未产出")
    sys.exit(1)

# 日志完成标志
import glob
for tag in TAGS:
    lg = os.path.join(TUNE, tag + ".log")
    ok = os.path.exists(lg) and "[SAVE]" in open(lg, "r", encoding="utf-8", errors="replace").read()
    print("  %-16s log=%s" % (tag, "完成(含[SAVE])" if ok else "缺失或未完成"))

print()
print("  %-16s %8s %8s %8s %8s %8s %8s  %s" % (
    "tag", "v_MAE", "v_F1h", "v_ACCh", "t_MAE", "t_F1h", "t_ACCh", "门禁"))
for tag in sorted(full, key=lambda t: -f(full[t]["valid_f1_head"])):
    r = full[tag]
    print("  %-16s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f  %s" % (
        tag, f(r["valid_mae"]), f(r["valid_f1_head"]), f(r["valid_acc_head"]),
        f(r["test_mae"]), f(r["test_f1_head"]), f(r["test_acc_head"]), r["mae_ok"]))

print()
print("=" * 78)
print("Q2  screen → full 排序可迁移性")
print("=" * 78)
rows_cmp = []
for tag in TAGS:
    s, fl = screen.get(tag), full.get(tag)
    if not s or not fl:
        continue
    rows_cmp.append(dict(
        tag=tag, desc=fl["desc"],
        s_mae=f(s["valid_mae"]), f_mae=f(fl["valid_mae"]),
        s_f1=f(s["valid_f1_head"]), f_f1=f(fl["valid_f1_head"]),
    ))

# 排名（1 = 最好）
rk_s_mae = {r["tag"]: i + 1 for i, r in enumerate(sorted(rows_cmp, key=lambda r: r["s_mae"]))}
rk_f_mae = {r["tag"]: i + 1 for i, r in enumerate(sorted(rows_cmp, key=lambda r: r["f_mae"]))}
rk_s_f1 = {r["tag"]: i + 1 for i, r in enumerate(sorted(rows_cmp, key=lambda r: -r["s_f1"]))}
rk_f_f1 = {r["tag"]: i + 1 for i, r in enumerate(sorted(rows_cmp, key=lambda r: -r["f_f1"]))}

print("  %-16s | %-22s | %-22s" % ("tag", "screen(MAE rank / F1 rank)", "full(MAE rank / F1 rank)"))
for r in rows_cmp:
    t = r["tag"]
    print("  %-16s | %6.4f (#%d)  %6.4f (#%d)   | %6.4f (#%d)  %6.4f (#%d)" % (
        t, r["s_mae"], rk_s_mae[t], r["s_f1"], rk_s_f1[t],
        r["f_mae"], rk_f_mae[t], r["f_f1"], rk_f_f1[t]))


def spearman(a, b):
    """无 scipy 依赖的 Spearman（n 很小，直接算）。"""
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da * db > 0 else float("nan")


print()
print("  Spearman(screen MAE rank, full MAE rank) = %+.3f" % spearman(
    [rk_s_mae[r["tag"]] for r in rows_cmp], [rk_f_mae[r["tag"]] for r in rows_cmp]))
print("  Spearman(screen F1  rank, full F1  rank) = %+.3f" % spearman(
    [rk_s_f1[r["tag"]] for r in rows_cmp], [rk_f_f1[r["tag"]] for r in rows_cmp]))
print("  注: +1 完全一致, 0 无关, -1 完全反转; n=4 时 |rho|>=0.8 才算强相关")

# ---------------------------------------------------------------- Q3 种子噪声
print()
print("=" * 78)
print("Q3  单种子噪声估计（同配置、不同 seed 的历史 run）")
print("=" * 78)
multi = {}
for d in sorted(os.listdir(os.path.join(ROOT, "runs"))):
    # 排除烟雾测试 / 联调目录（前缀 _），其结果不代表真实训练方差
    if d.startswith("_"):
        continue
    dd = os.path.join(ROOT, "runs", d)
    if not os.path.isdir(dd):
        continue
    seeds = sorted(glob.glob(os.path.join(dd, "metrics_s*.json")))
    if len(seeds) < 2:
        continue
    recs = []
    for sp in seeds:
        try:
            j = json.load(open(sp, "r", encoding="utf-8", errors="replace"))
        except Exception:
            continue
        v = j.get("valid_clean")
        if not isinstance(v, dict):
            continue
        rec = {"seed": os.path.basename(sp).replace("metrics_", "").replace(".json", "")}
        for k in ("mae", "f1_head", "acc_head"):
            rec[k] = v.get(k) if isinstance(v.get(k), (int, float)) else None
        recs.append(rec)
    # 需至少 3 个**互不相同**的种子，否则 sd 不稳（如 s42 与 s42_legacy 重复）
    if len({r["seed"] for r in recs}) >= 3:
        multi[d] = recs


def _num(xs):
    """筛出有限浮点数（同时排掉 None 与 nan）。"""
    out = []
    for x in xs:
        if isinstance(x, (int, float)):
            xf = float(x)
            if xf == xf and abs(xf) != float("inf"):
                out.append(xf)
    return out


def std(xs):
    xs = _num(xs)
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def med(xs):
    xs = sorted(_num(xs))
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


noise_rows = []
for d, recs in multi.items():
    row = dict(run=d, n=len(recs))
    for k in ("mae", "f1_head", "acc_head"):
        row[k + "_sd"] = std([r[k] for r in recs])
    # 该配置至少要有一个指标可用于估计噪声
    if not _num([row["mae_sd"], row["f1_head_sd"], row["acc_head_sd"]]):
        continue
    noise_rows.append(row)
    vals = "  ".join("%s:MAE=%.4f/F1h=%s" % (
        r["seed"], r["mae"], ("%.4f" % r["f1_head"]) if isinstance(r["f1_head"], (int, float)) else "--")
        for r in recs)
    print("  %-20s n=%d  sd(MAE)=%.4f  sd(F1h)=%.4f  sd(ACCh)=%.4f" % (
        d, len(recs), row["mae_sd"], row["f1_head_sd"], row["acc_head_sd"]))
    print("      %s" % vals)

if not noise_rows:
    print("  [WARN] 未找到多种子 run，无法估计噪声")
    sd_mae = sd_f1 = float("nan")
else:
    # 取各 run 标准差的中位数（对个别方差偏大的配置更稳健）
    sd_mae = med([r["mae_sd"] for r in noise_rows])
    sd_f1 = med([r["f1_head_sd"] for r in noise_rows])
print()
print("  sigma 取各 run 标准差的中位数（稳健）:")
print("    sigma(valid MAE)     ~ %.4f   [范围 %.4f ~ %.4f]" % (
    sd_mae, min(_num([r["mae_sd"] for r in noise_rows])), max(_num([r["mae_sd"] for r in noise_rows]))))
print("    sigma(valid F1_head) ~ %.4f   [范围 %.4f ~ %.4f]" % (
    sd_f1, min(_num([r["f1_head_sd"] for r in noise_rows])), max(_num([r["f1_head_sd"] for r in noise_rows]))))

print()
print("=" * 78)
print("Q3b 各候选相对基线 c00_baseline 的改善显著性（单种子对比, z = delta/(sqrt(2)*sigma)）")
print("=" * 78)
base = full.get("c00_baseline")
if base:
    b_mae, b_f1 = f(base["valid_mae"]), f(base["valid_f1_head"])
    print("  %-16s %9s %8s %9s %8s   %s" % ("tag", "dMAE", "z(MAE)", "dF1h", "z(F1h)", "判定"))
    verdicts = {}
    for tag in sorted(full, key=lambda t: -f(full[t]["valid_f1_head"])):
        if tag == "c00_baseline":
            continue
        d_mae = f(full[tag]["valid_mae"]) - b_mae
        d_f1 = f(full[tag]["valid_f1_head"]) - b_f1
        z_mae = d_mae / (2 ** 0.5 * sd_mae) if sd_mae == sd_mae and sd_mae > 0 else float("nan")
        z_f1 = d_f1 / (2 ** 0.5 * sd_f1) if sd_f1 == sd_f1 and sd_f1 > 0 else float("nan")
        zs = _num([z_mae, z_f1])
        worst = max(abs(z) for z in zs) if zs else float("nan")
        if worst != worst:
            verdict = "噪声未知"
        elif worst < 2.0:
            verdict = "差异不显著(|z|<2)"
        elif worst < 3.0:
            verdict = "边缘显著(2<=|z|<3)"
        else:
            verdict = "显著(|z|>=3)"
        verdicts[tag] = (d_mae, z_mae, d_f1, z_f1, verdict)
        print("  %-16s %+9.4f %8.2f %+9.4f %8.2f   %s" % (tag, d_mae, z_mae, d_f1, z_f1, verdict))

# ---------------------------------------------------------------- 落盘
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(OUT, "tune_screen_vs_full.csv"), "w", encoding="utf-8-sig", newline="") as fh:
    fh.write("tag,desc,screen_mae,screen_f1_head,full_mae,full_f1_head,d_mae,d_f1_head,"
             "rank_screen_mae,rank_full_mae,rank_screen_f1,rank_full_f1\n")
    for r in sorted(rows_cmp, key=lambda r: r["f_f1"], reverse=True):
        t = r["tag"]
        fh.write("%s,%s,%.6f,%.6f,%.6f,%.6f,%+.6f,%+.6f,%d,%d,%d,%d\n" % (
            t, r["desc"], r["s_mae"], r["s_f1"], r["f_mae"], r["f_f1"],
            r["f_mae"] - r["s_mae"], r["f_f1"] - r["s_f1"],
            rk_s_mae[t], rk_f_mae[t], rk_s_f1[t], rk_f_f1[t]))

with open(os.path.join(OUT, "tune_seed_noise.csv"), "w", encoding="utf-8-sig", newline="") as fh:
    fh.write("run,n,sd_valid_mae,sd_valid_f1_head,sd_valid_acc_head\n")
    for r in noise_rows:
        fh.write("%s,%d,%.6f,%.6f,%.6f\n" % (r["run"], r["n"], r["mae_sd"], r["f1_head_sd"], r["acc_head_sd"]))

# markdown 摘要
with open(os.path.join(OUT, "tune_full_report.md"), "w", encoding="utf-8", newline="") as fh:
    fh.write("# 超参搜索：screen 阶段 vs full 阶段\n\n")
    fh.write("## 1 全量阶段结果（验证集，n=728）\n\n")
    fh.write("| 配置 | 说明 | valid MAE | valid F1(分类头) | valid ACC(分类头) | 误差门禁 |\n")
    fh.write("|---|---|---:|---:|---:|:--:|\n")
    for tag in sorted(full, key=lambda t: -f(full[t]["valid_f1_head"])):
        r = full[tag]
        fh.write("| %s | %s | %.4f | %.4f | %.4f | %s |\n" % (
            tag, r["desc"], f(r["valid_mae"]), f(r["valid_f1_head"]), f(r["valid_acc_head"]), r["mae_ok"]))
    fh.write("\n## 2 screen → full 可迁移性\n\n")
    fh.write("screen 阶段使用 1024 条训练子集与更短轮数（教师 6 / 学生 8），"
             "full 阶段使用全量训练集与主实验轮数（教师 12 / 学生 14）。\n\n")
    fh.write("| 配置 | screen MAE(排名) | full MAE(排名) | screen F1(排名) | full F1(排名) |\n")
    fh.write("|---|---:|---:|---:|---:|\n")
    for r in rows_cmp:
        t = r["tag"]
        fh.write("| %s | %.4f (#%d) | %.4f (#%d) | %.4f (#%d) | %.4f (#%d) |\n" % (
            t, r["s_mae"], rk_s_mae[t], r["f_mae"], rk_f_mae[t], r["s_f1"], rk_s_f1[t], r["f_f1"], rk_f_f1[t]))
    fh.write("\nSpearman 秩相关：MAE %+.3f；F1 %+.3f（n=4）。\n" % (
        spearman([rk_s_mae[r["tag"]] for r in rows_cmp], [rk_f_mae[r["tag"]] for r in rows_cmp]),
        spearman([rk_s_f1[r["tag"]] for r in rows_cmp], [rk_f_f1[r["tag"]] for r in rows_cmp])))
    fh.write("\n## 3 单种子噪声与显著性\n\n")
    fh.write("| run | 种子数 | sd(valid MAE) | sd(valid F1_head) | sd(valid ACC_head) |\n")
    fh.write("|---|---:|---:|---:|---:|\n")
    for r in noise_rows:
        fh.write("| %s | %d | %.4f | %.4f | %.4f |\n" % (
            r["run"], r["n"], r["mae_sd"], r["f1_head_sd"], r["acc_head_sd"]))
    fh.write("\n稳健估计（中位数）：sigma(MAE) ~ %.4f，sigma(F1_head) ~ %.4f。\n\n" % (sd_mae, sd_f1))
    fh.write("| 配置 | dMAE | z(MAE) | dF1_head | z(F1_head) | 判定 |\n")
    fh.write("|---|---:|---:|---:|---:|:--|\n")
    for tag, (d_mae, z_mae, d_f1, z_f1, verdict) in verdicts.items():
        fh.write("| %s | %+.4f | %+.2f | %+.4f | %+.2f | %s |\n" % (tag, d_mae, z_mae, d_f1, z_f1, verdict))

print()
print("[OK] 已写出 paper_tables/tune_screen_vs_full.csv")
print("[OK] 已写出 paper_tables/tune_seed_noise.csv")
print("[OK] 已写出 paper_tables/tune_full_report.md")
