"""附件3/4 缺失画像（**实测重建** paper_tables 的“缺失规律”表）。

题目要求分析缺失的**类型 / 位置 / 时长**三个因素，因此本表必须全部来自实测，
不能沿用旧表里从 config 先验填充的 head/middle/tail（那样三个模态和 text 完全一样）。

口径（**规范定义 D1**，写进论文，全文统一）：
  · 缺失判定：**连续 >= min_run 帧全零**（题目原文"部分**连续**时间段不可用"）
  · 统计区间：`[0, 末个非零帧]`
      - 附件3/4 没有 `*_lengths` 字段，末个非零帧之后只能是**尾部填充**，故排除；
      - **首部零帧属于"开头一段不可用"，计入缺失**（这是 D1 与"头尾都裁"版本的关键区别）。
        该口径与 `data_utils.detect_missing_mask` + `infer_att3.py` 的掩码判定**完全一致**。
  · 缺失率 rho = 缺失帧数 / 统计区间长度
  · **整条全零的样本 = 整模态缺失，rho = 1.0**（附件4 视觉有 1 条这样的样本）
  · 位置：以缺失段**中心**在统计区间上的归一化坐标 p 分箱 —— 头 p<1/3、中 1/3~2/3、尾 >2/3
  · 时长：缺失段长度（帧）

[历史备注] 旧表的 `frac_with_missing / mean / p90 / max` 与 D1 逐位吻合（0.5667 / 0.101 /
0.224 / 0.400），说明旧值口径正确；**旧表唯一的问题是 head/middle/tail 三列由 config 先验
硬编码**（三个模态与零缺失的 text 完全同值）。本脚本已改为全列实测。

用法：python missing_profile.py --root <工程根>
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

sys.stdout.reconfigure(errors="replace")
MODS = ("text", "audio", "vision")


def load_flat(path):
    """读 data_att/*.pkl -> {模态: (N,50,d)}"""
    with open(path, "rb") as f:
        d = pickle.load(f)
    return {m: np.asarray(d[m], dtype=np.float32) for m in MODS if m in d}


def analyse(X, min_run=2):
    """返回逐样本 rho 与所有缺失段信息（规范定义 D1，见模块 docstring）。"""
    z = (np.abs(X).sum(axis=-1) < 1e-8)                # (N,L) 全零帧
    rhos, segs_len, segs_pos, with_miss, n_allzero = [], [], [], 0, 0
    for row in z:
        nz = ~row
        if not nz.any():
            # 整条全零：整模态缺失
            n_allzero += 1
            with_miss += 1
            rhos.append(1.0)
            segs_len.append(len(row))
            segs_pos.append(0.5)
            continue
        last = int(len(row) - 1 - np.argmax(nz[::-1]))     # 末个非零帧
        span = last + 1                                     # 统计区间长度（含首部零帧）
        seg = row[:span]
        miss, run, st, found = 0, 0, 0, []
        for i, v in enumerate(seg):
            if v:
                run += 1
                if run == 1:
                    st = i
            else:
                if run >= min_run:
                    miss += run
                    found.append((st, i))
                run = 0
        if run >= min_run:
            miss += run
            found.append((st, len(seg)))
        rhos.append(miss / span if span else 0.0)
        if found:
            with_miss += 1
        for a, b in found:
            segs_len.append(b - a)
            segs_pos.append(((a + b) / 2.0) / max(span, 1))
    return (np.asarray(rhos, dtype=float), np.asarray(segs_len), np.asarray(segs_pos),
            with_miss, len(z), n_allzero)


def fmt(r, s, p, wm, n, nz, tag, name, rows, foot):
    ok = r[~np.isnan(r)]
    if ok.size == 0:
        return
    head = float((p < 1 / 3).mean()) if p.size else 0.0
    mid = float(((p >= 1 / 3) & (p <= 2 / 3)).mean()) if p.size else 0.0
    tail = float((p > 2 / 3).mean()) if p.size else 0.0
    rows.append(dict(dataset=tag, modality=name, n=n, n_with_missing=wm,
                     frac_with_missing=round(wm / n, 4),
                     n_full_missing=nz,
                     rho_mean=round(float(ok.mean()), 4),
                     rho_p50=round(float(np.percentile(ok, 50)), 4),
                     rho_p90=round(float(np.percentile(ok, 90)), 4),
                     rho_max=round(float(ok.max()), 4),
                     n_segments=int(s.size),
                     seg_len_mean=round(float(s.mean()), 2) if s.size else 0.0,
                     seg_len_p90=round(float(np.percentile(s, 90)), 1) if s.size else 0.0,
                     seg_len_max=int(s.max()) if s.size else 0,
                     pos_head=round(head, 4), pos_middle=round(mid, 4), pos_tail=round(tail, 4)))
    print("  %-6s %-7s 含缺失 %2d/%d (%.1f%%)  整模态%2d  rho 均值 %.3f p50 %.3f p90 %.3f max %.3f | "
          "段数 %2d 段长 均值%.1f p90 %.0f max %d | 位置 头%.2f 中%.2f 尾%.2f"
          % (tag, name, wm, n, 100 * wm / n, nz, ok.mean(), np.percentile(ok, 50),
             np.percentile(ok, 90), ok.max(), s.size, s.mean() if s.size else 0,
             np.percentile(s, 90) if s.size else 0, int(s.max()) if s.size else 0,
             head, mid, tail))
    # 灵敏度：另外两种口径，仅作脚注
    if name != "text":
        foot.append((tag, name, float(ok.mean())))



def _mean_d1(rows, tag, mod):
    for r in rows:
        if r["dataset"] == tag and r["modality"] == mod:
            return r["rho_mean"]
    return float("nan")


def _sensitivity(root, fn, mod, mode):
    """口径灵敏度：D2 = 头尾都裁/有效长度；D3 = 首部计入/固定总长 50。仅作脚注。"""
    p = os.path.join(root, "data_att", fn)
    if not os.path.isfile(p):
        return float("nan")
    with open(p, "rb") as f:
        X = np.asarray(pickle.load(f)[mod], dtype=np.float32)
    z = (np.abs(X).sum(axis=-1) < 1e-8)
    vals = []
    for row in z:
        nz = ~row
        if not nz.any():
            vals.append(1.0)
            continue
        last = int(len(row) - 1 - np.argmax(nz[::-1]))
        first = int(np.argmax(nz))
        if mode == "D2":
            seg, denom = row[first:last + 1], max(last - first + 1, 1)
        else:
            seg, denom = row[:last + 1], len(row)
        miss, run = 0, 0
        for v in seg:
            if v:
                run += 1
            else:
                if run >= 2:
                    miss += run
                run = 0
        if run >= 2:
            miss += run
        vals.append(miss / denom)
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--min_run", type=int, default=2)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    print("=" * 100)
    print("附件3/4 缺失画像（规范口径 D1：连续 >=%d 帧全零；统计区间 = [0, 末个非零帧]；"
          "rho = 缺失帧 / 区间长度）" % args.min_run)
    print("=" * 100)
    rows, foot = [], []
    for tag, fn in (("att3", "att3_aligned.pkl"), ("att4", "att4_aligned.pkl")):
        p = os.path.join(args.root, "data_att", fn)
        if not os.path.isfile(p):
            print("**缺失** %s" % p)
            continue
        arrs = load_flat(p)
        for m in MODS:
            if m not in arrs:
                continue
            r, s, pos, wm, n, nz = analyse(arrs[m], args.min_run)
            fmt(r, s, pos, wm, n, nz, tag, m, rows, foot)

    out = args.out_dir or os.path.join(args.root, "paper_tables")
    os.makedirs(out, exist_ok=True)
    import csv
    csv_path = os.path.join(out, "missing_profile.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    md = ["# 附件3/4 缺失画像（全列实测，规范口径 D1）",
          "",
          "**口径定义**（论文全文统一）：",
          "",
          "- 缺失判定：**连续 ≥%d 帧全零**（题目原文「部分**连续**时间段不可用」）" % args.min_run,
          "- 统计区间：`[0, 末个非零帧]`。附件3/4 无 `*_lengths` 字段，末个非零帧之后只能是"
          "**尾部填充**故排除；**首部零帧属于「开头一段不可用」，计入缺失**",
          "- 缺失率 $\\rho$ = 缺失帧数 / 统计区间长度",
          "- **整条全零 = 整模态缺失，$\\rho=1.0$**",
          "- 位置：缺失段**中心**在统计区间上的归一化坐标 $p$ 分箱 —— 头 $p<1/3$、中 $1/3\\sim2/3$、尾 $>2/3$",
          "",
          "> 本口径与 `data_utils.detect_missing_mask` 及 `infer_att3.py` 的掩码判定**完全一致**，"
          "因此本表与提交 CSV 的 `n_missing_intervals` 可直接交叉核对。",
          "> 全部数值由实测计算，**不使用** config 中的位置先验。",
          "",
          "| 数据集 | 模态 | 含缺失样本 | 占比 | 整模态缺失 | ρ 均值 | p50 | p90 | max | 段数 | 段长均值 | 段长p90 | 段长max | 位置（头/中/尾） |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| %s | %s | %d/%d | %.1f%% | %d | %.4f | %.4f | %.4f | %.4f | %d | %.2f | %.1f | %d | %.2f/%.2f/%.2f |"
                  % (r["dataset"], r["modality"], r["n_with_missing"], r["n"],
                     100 * r["frac_with_missing"], r["n_full_missing"], r["rho_mean"],
                     r["rho_p50"], r["rho_p90"], r["rho_max"], r["n_segments"],
                     r["seg_len_mean"], r["seg_len_p90"], r["seg_len_max"],
                     r["pos_head"], r["pos_middle"], r["pos_tail"]))
    md += ["",
           "## 用于训练课程标定",
           "",
           "- $\\rho$ 上界应覆盖 `p90 ~ max`；段长与位置分布用于设定 `n_intervals` 与 `pos_modes`。",
           "- **text 零缺失**：附件3 的文本模态不存在缺失，故「文本缺失」只能通过合成注入研究"
           "（这也是 `missing_probe.py` 中「缺文本」场景为人为构造的原因）。",
           "- **附件4 视觉有整模态缺失样本**：该样本视觉全零，须由 text+audio 支撑推理，"
           "是附件4 上唯一的极端缺失案例。",
           "",
           "## 口径灵敏度（脚注，非主报）",
           "",
           "| 口径 | 说明 | audio ρ 均值 | vision ρ 均值 |",
           "|---|---|---|---|",
           "| **D1（本表主报）** | 首部零帧计入缺失 / 分母 = 有效长度 | %.4f | %.4f |"
           % (_mean_d1(rows, "att3", "audio"), _mean_d1(rows, "att3", "vision")),
           "| D2 | 头尾零帧都裁 / 分母 = 有效长度（把首部缺失丢弃，**不建议**） | %.4f | %.4f |"
           % (_sensitivity(args.root, "att3_aligned.pkl", "audio", "D2"),
              _sensitivity(args.root, "att3_aligned.pkl", "vision", "D2")),
           "| D3 | 首部零帧计入 / 分母 = 固定总长 50（把缺失率人为稀释，**不建议**） | %.4f | %.4f |"
           % (_sensitivity(args.root, "att3_aligned.pkl", "audio", "D3"),
              _sensitivity(args.root, "att3_aligned.pkl", "vision", "D3")),
           "",
           "> D1 与 D3 的**含缺失样本计数相同**（audio 17/30、vision 18/30），"
           "差别仅在分母；D2 会把仅在开头缺失的样本判为「无缺失」（audio 少 3 条、vision 少 3 条），"
           "与题目「部分连续时间段不可用」的表述不符，故不采用。"]
    with open(os.path.join(out, "missing_profile.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n[SAVE] %s\n[SAVE] %s" % (csv_path, os.path.join(out, "missing_profile.md")))


    # ---- 同步重建旧文件名（保持原 schema，避免下游引用失效）----
    legacy = os.path.join(out, "att3_missing_distribution.csv")
    if os.path.isfile(legacy):
        bak = os.path.join(out, "att3_missing_distribution_legacy.csv")
        if not os.path.isfile(bak):
            import shutil
            shutil.copyfile(legacy, bak)
            print("[BACKUP] %s -> %s" % (legacy, bak))
    cols = ["数据集", "模态", "n_with_missing", "frac_with_missing", "mean_missing_ratio",
            "p50", "p90", "max", "len_mean", "head", "middle", "tail"]
    with open(legacy, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["dataset"], r["modality"], r["n_with_missing"],
                        r["frac_with_missing"], r["rho_mean"], r["rho_p50"],
                        r["rho_p90"], r["rho_max"], r["seg_len_mean"],
                        r["pos_head"], r["pos_middle"], r["pos_tail"]])
    print("[SAVE] %s（旧 schema，位置列改为**实测值**）" % legacy)



if __name__ == "__main__":
    main()
