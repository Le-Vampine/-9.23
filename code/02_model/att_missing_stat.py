"""附件3/4 的模态缺失实况统计（只做「全零帧」判据，用于判断文本缺失是否真的出现）。

动机：优化「缺文本」之前必须确认这一场景在附件3 里是否出现、出现多少。
对齐版无长度字段 → pad 与 miss 混在一起，因此只能统计「全零帧」比例与最长连续段。

用法：python att_missing_stat.py --root <工程根>
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.stdout.reconfigure(errors="replace")
MODS = ("text", "audio", "vision")


def zero_split(X):
    """把零帧拆成 头部零(前导)/尾部零(后随)/中间零(碎片缺失)，用于区分 pad 与 miss。"""
    z = (np.abs(X).sum(axis=-1) == 0)                     # (N,L)
    nz = ~z
    lead, trail, inner = [], [], []
    for row, row_nz in zip(z, nz):
        L = len(row)
        if not row_nz.any():
            lead.append(L); trail.append(0); inner.append(0)
            continue
        f = int(np.argmax(row_nz))                        # 第一个非零
        l = int(L - 1 - np.argmax(row_nz[::-1]))          # 最后一个非零
        lead.append(f)
        trail.append(L - 1 - l)
        inner.append(int(row[f:l + 1].sum()))
    tot = z.sum(axis=1)
    tot = np.where(tot == 0, 1, tot)
    return (np.array(lead) / tot, np.array(trail) / tot, np.array(inner) / tot)


def rho_dist(X):
    """逐样本真实缺失率 rho = 有效长度内的零帧数 / 有效长度（尾部填充先被剔除）。"""
    z = (np.abs(X).sum(axis=-1) == 0)
    nz = ~z
    out = []
    for row, row_nz in zip(z, nz):
        if not row_nz.any():
            out.append(np.nan)
            continue
        f = int(np.argmax(row_nz))
        l = int(len(row) - 1 - np.argmax(row_nz[::-1]))
        valid = l - f + 1
        miss = int(row[f:l + 1].sum())
        out.append(miss / valid if valid else np.nan)
    r = np.asarray(out, dtype=float)
    return r


def rho_run_dist(X, min_run=2):
    """**题目口径**的真实缺失率：只把“连续 >= min_run 帧全零”判为缺失（题目说“连续时间段不可用”）。

    与 rho_dist 的区别：后者把任意孤立单帧零也算作缺失，口径更宽松。
    """
    z = (np.abs(X).sum(axis=-1) == 0)
    nz = ~z
    out = []
    for row, row_nz in zip(z, nz):
        if not row_nz.any():
            out.append(np.nan)
            continue
        f = int(np.argmax(row_nz))
        l = int(len(row) - 1 - np.argmax(row_nz[::-1]))
        valid = l - f + 1
        if valid <= 0:
            out.append(np.nan)
            continue
        seg = row[f:l + 1]
        miss = 0
        run = 0
        for v in seg:
            if v:
                run += 1
            else:
                if run >= min_run:
                    miss += run
                run = 0
        if run >= min_run:
            miss += run
        out.append(miss / valid)
    return np.asarray(out, dtype=float)


def report_run_rho(name, arrs, n, min_run=2):
    print("\n  【题目口径 rho：仅计连续 >=%d 帧全零（=训练注入的可比口径）】" % min_run)
    print("  %-8s %-8s %-8s %-8s %-8s %-10s" %
          ("模态", "均值", "p50", "p90", "最大", "含缺失样本"))
    for m in MODS:
        r = rho_run_dist(arrs[m], min_run)
        ok = r[~np.isnan(r)]
        if ok.size == 0:
            continue
        print("  %-8s %-8.3f %-8.3f %-8.3f %-8.3f %-10s"
              % (m, ok.mean(), np.percentile(ok, 50), np.percentile(ok, 90),
                 ok.max(), "%.1f%%" % (100 * (ok > 0).mean())))


def report_rho(name, arrs, n):
    print("\n  【真实缺失率 rho 分布】有效长度内的零帧占比（逐样本）")
    print("  %-8s %-8s %-8s %-8s %-8s %-10s" %
          ("模态", "均值", "p50", "p90", "最大", "含缺失样本"))
    for m in MODS:
        r = rho_dist(arrs[m])
        ok = r[~np.isnan(r)]
        if ok.size == 0:
            continue
        print("  %-8s %-8.3f %-8.3f %-8.3f %-8.3f %-10s"
              % (m, ok.mean(), np.percentile(ok, 50), np.percentile(ok, 90),
                 ok.max(), "%.1f%%" % (100 * (ok > 0).mean())))
    print("\n  用途：训练课程的 rho 上界应覆盖到测试集 p90~max，但不必超出太多。")


def zero_stats(X):
    """返回每个样本的 (全零帧比例, 最长连续全零段)。"""
    z = (np.abs(X).sum(axis=-1) == 0)               # (N,L)
    ratios = z.mean(axis=1)
    runs = []
    for row in z:
        best = cur = 0
        for v in row:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        runs.append(best)
    return ratios, np.asarray(runs)


def report_pos(name, arrs, n):
    """pad / miss 分离：零帧中头/尾/中间各占多少。"""
    print("\n  【pad vs miss 分离】零帧构成（占该模态零帧总数的比例）")
    print("  %-8s %-14s %-14s %-14s" % ("模态", "头部零(前导)", "尾部零(后随)", "中间零(碎片=真缺失)"))
    for m in MODS:
        ld, tr, inr = zero_split(arrs[m])
        print("  %-8s %-14.3f %-14.3f %-14.3f"
              % (m, float(np.mean(ld)), float(np.mean(tr)), float(np.mean(inr))))
    print("\n  判定规则：头部/尾部零 = 有效长度外的填充；中间零 = 长度内的缺失片段。")


def report(name, arrs, n):
    print("\n" + "=" * 74)
    print("== %s  样本数=%d  序列长度=%d" % (name, n, arrs[MODS[0]].shape[1]))
    print("=" * 74)
    print("  %-8s %-34s %-8s %-8s %-8s" % ("模态", "全零帧比例(均值/中位/最大)",
                                          "含零样本", "≥50%零", "全零样本"))
    for m in MODS:
        r, run = zero_stats(arrs[m])
        print("  %-8s %6.3f / %6.3f / %6.3f            %5.1f%%   %5.1f%%   %5.1f%%"
              % (m, r.mean(), np.median(r), r.max(),
                 100 * (r > 0).mean(), 100 * (r >= 0.5).mean(), 100 * (r >= 0.999).mean()))
    # 逐样本明细（只看有零的）
    report_pos(name, arrs, n)
    report_rho(name, arrs, n)
    report_run_rho(name, arrs, n, min_run=2)
    print("\n  逐样本明细（仅列出含全零帧的样本）：")
    print("  %-6s %-10s %-10s %-10s %-26s" % ("idx", "text", "audio", "vision", "说明"))
    for i in range(n):
        row = []
        note = []
        for m in MODS:
            r, run = zero_stats(arrs[m][i:i + 1])
            row.append((r[0], int(run[0])))
        if all(v[0] == 0 for v in row):
            continue
        if row[0][0] > 0:
            note.append("文本含全零段(最长%d)" % row[0][1])
        print("  %-6d %-10s %-10s %-10s %s"
              % (i, "%.2f/%d" % row[0], "%.2f/%d" % row[1], "%.2f/%d" % row[2],
                 ";".join(note)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--which", default="att3")
    args = ap.parse_args()

    cand = {"att3": "att3_aligned.pkl", "att4": "att4_aligned.pkl",
            "train": "train_aligned.pkl"}
    path = os.path.join(args.root, "data_att", cand[args.which])
    if not os.path.isfile(path):
        print("**缺失**: %s" % path)
        return
    with open(path, "rb") as f:
        d = pickle.load(f)
    n = len(d["id"])
    arrs = {m: np.asarray(d[m], dtype=np.float32) for m in MODS}
    report("%s (%s)" % (args.which, os.path.basename(path)), arrs, n)
    if args.which == "att3":
        print("\n  raw_text 示例:", str(d.get("raw_text", [None])[0])[:60])


if __name__ == "__main__":
    main()
