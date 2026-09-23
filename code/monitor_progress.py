# -*- coding: utf-8 -*-
r"""
训练进度实时监控（每 N 秒刷新终端仪表盘，并同步写 runs/progress.md）。

用法：
  python monitor_progress.py                 # 每 15 秒刷新，Ctrl+C 退出
  python monitor_progress.py --once          # 只打印一次快照
  python monitor_progress.py --interval 5    # 每 5 秒刷新
  python monitor_progress.py --log xxx.log   # 指定日志（默认取 code/ 下最新的 *.log）
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RE_SEED = re.compile(r"^=+ seed (\d+) =+", re.M)
RE_HEAD = re.compile(r"\[SEED (\d+)\].*epochs\(t/s\)=(\d+)/(\d+)")
RE_T = re.compile(r"\[T ep(\d+)\] loss=([\d.]+|nan) mae=([\d.]+|nan) rec=([\d.]+|nan) \(([\d.]+)s\)")
RE_S = re.compile(r"\[S ep(\d+)\] loss=([\d.]+|nan) mae=([\d.]+|nan) rec=([\d.]+|nan) kd=([\d.]+|nan)"
                  r".*rho~\[([\d.]+),([\d.]+)\] (\w+) \(([\d.]+)s\)")
RE_BEST = re.compile(r"\[T\] best valid MAE = ([\d.]+)")
RE_TH = re.compile(r"\[TH\] theta=([\d.]+) \(valid MacroF1=([\d.]+)\)")
RE_VAL = re.compile(r"\[VALID clean\] (.*)")
RE_TEST = re.compile(r"\[TEST  clean\] (.*)")
RE_SAVE = re.compile(r"\[SAVE\] (\S+) \(?([\d.]+) MB\)?")


def newest_log():
    cands = [p for p in glob.glob(os.path.join(HERE, "*.log"))
             if os.path.getsize(p) > 0 and "_status" not in p and "_verif" not in p]
    if not cands:
        return None
    named = [p for p in cands if os.path.basename(p).startswith(("_train", "_ablate", "_sweep"))]
    pool = named or cands
    return max(pool, key=os.path.getmtime)


def parse(path):
    if not path or not os.path.isfile(path):
        return None
    txt = open(path, encoding="utf-8", errors="ignore").read()
    lines = txt.splitlines()
    seeds = [int(m.group(1)) for m in RE_SEED.finditer(txt)]
    # 一个日志里可能有多次训练（快速验证 + 正式训练）→ 只统计最后一次
    parts = re.split(r"(?=\[SEED \d+\])", txt)
    cur = parts[-1] if len(parts) > 1 else txt
    n_t, n_s = 20, 30
    m = RE_HEAD.search(cur)
    if m:
        n_t, n_s = int(m.group(2)), int(m.group(3))
    T = [(int(a), b, c, d, float(e)) for a, b, c, d, e in RE_T.findall(cur)]
    S = [(int(a), b, c, d, e, f, g, h, float(i)) for a, b, c, d, e, f, g, h, i in RE_S.findall(cur)]
    tail = lines[-1] if lines else ""
    return dict(log=path, text=txt, cur=cur, seeds=seeds, n_t=n_t, n_s=n_s,
                T=T, S=S, best=RE_BEST.findall(cur), th=RE_TH.findall(cur),
                val=RE_VAL.findall(cur), test=RE_TEST.findall(cur),
                saves=RE_SAVE.findall(cur), last=tail,
                finished="ALL_TRAINING_FINISHED" in txt,
                mtime=os.path.getmtime(path))


def spark(vals, w=24):
    blocks = "▁▂▃▄▅▆▇█"
    v = [x for x in vals if isinstance(x, float) and x == x]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    if hi - lo < 1e-9:
        return blocks[3] * min(len(v), w)
    out = "".join(blocks[min(7, int((x - lo) / (hi - lo) * 7.999))] for x in v[-w:])
    return out


def bar(frac, width=40):
    frac = max(0.0, min(1.0, frac))
    n = int(frac * width)
    return "[" + "#" * n + "." * (width - n) + "] %5.1f%%" % (frac * 100)


def fmt_eta(sec):
    if sec is None or sec != sec or sec < 0 or sec > 10 ** 8:
        return "—"
    h, m = int(sec // 3600), int(sec % 3600 // 60)
    s = int(sec % 60)
    return ("%d h %02d min" % (h, m)) if h else ("%d min %02d s" % (m, s))


def build(st, planned_seeds=2):
    L = []
    add = L.append
    now = datetime.now()
    add("# 训练进度（自动刷新）\n")
    if not st:
        add("> 未找到训练日志。请确认训练任务已启动。\n")
        return "\n".join(L), None

    started = len(st["seeds"])
    total_seeds = max(planned_seeds, started, 1)
    done_seeds = max(0, started - 1)
    left_seeds = max(0, total_seeds - started)
    # 当前处于教师还是学生阶段
    phase = "teacher"
    if st["S"] and (not st["T"] or st["S"][-1][0] >= 0) and st["last"].startswith(("[S", "[TH", "[VALID", "[TEST", "[SAVE")):
        phase = "student"
    if st["T"] and len(st["T"]) >= st["n_t"]:
        phase = "student"
    dt, ds = len(st["T"]), len(st["S"])

    t_times = [r[4] for r in st["T"]]
    s_times = [r[8] for r in st["S"]]
    t_mean = sum(t_times) / len(t_times) if t_times else 150.0
    s_mean = sum(s_times) / len(s_times) if s_times else t_mean * 1.25

    rem_t = max(0, st["n_t"] - dt)
    rem_s = max(0, st["n_s"] - ds)
    eta = 0.0
    if st["finished"]:
        eta = 0.0
    else:
        if phase == "teacher":
            eta = rem_t * t_mean + st["n_s"] * s_mean
        else:
            eta = rem_s * s_mean
        eta += left_seeds * (st["n_t"] * t_mean + st["n_s"] * s_mean)

    total_epochs = total_seeds * (st["n_t"] + st["n_s"])
    done_epochs = done_seeds * (st["n_t"] + st["n_s"]) + dt + ds
    frac = done_epochs / total_epochs if total_epochs else 0

    add("```")
    add("时间      %s        日志  %s" % (now.strftime("%Y-%m-%d %H:%M:%S"), os.path.basename(st["log"])))
    add("进程      %s        种子 %d/%d（已完成 %d，排队 %d）"
        % ("运行中" if not st["finished"] else "全部完成", started, total_seeds,
           done_seeds, left_seeds))
    add("")
    add("总进度    %s   (%d/%d 轮)" % (bar(frac), done_epochs, total_epochs))
    add("当前阶段  %s   教师 %d/%d 轮   学生 %d/%d 轮"
        % ("教师(全模态)" if phase == "teacher" else "学生(缺失+蒸馏)", dt, st["n_t"], ds, st["n_s"]))
    if not st["finished"]:
        add("预计剩余  %s   预计完成  %s"
            % (fmt_eta(eta), (now + timedelta(seconds=eta)).strftime("%H:%M")))
    add("")
    add("单轮耗时  教师 %.1f s   学生 %.1f s" % (t_mean, s_mean))
    add("")
    if st["T"]:
        add("教师 loss  %s   当前 %s" % (spark([float(r[1]) for r in st["T"] if r[1] != "nan"]),
                                          st["T"][-1][1]))
        add("教师 valMAE %s   当前 %s   (越小越好)" % (spark([float(r[2]) for r in st["T"] if r[2] != "nan"]),
                                                    st["T"][-1][2]))
    if st["S"]:
        add("学生 loss  %s   当前 %s" % (spark([float(r[1]) for r in st["S"] if r[1] != "nan"]),
                                          st["S"][-1][1]))
        add("学生 valMAE %s   当前 %s" % (spark([float(r[2]) for r in st["S"] if r[2] != "nan"]),
                                          st["S"][-1][2]))
    add("```\n")

    # ---- 明细表 ----
    def rows(name, data, idx_mae, idx_loss, extra=("",)):
        if not data:
            return
        add("**%s（最近 8 轮）**\n" % name)
        add("| 轮次 | 损失 | valid MAE | 重建 |" + (" 蒸馏 |" if name.startswith("学生") else "") + " 耗时 |")
        add("|---|---|---|---|" + ("---|" if name.startswith("学生") else "") + "---|")
        for r in data[-8:]:
            if name.startswith("学生"):
                add("| %d | %s | %s | %s | %s | %.1f s |" % (r[0], r[1], r[2], r[3], r[4], r[8]))
            else:
                add("| %d | %s | %s | %s | %.1f s |" % (r[0], r[1], r[2], r[3], r[4]))
        add("")

    rows("教师阶段", st["T"], 2, 1)
    rows("学生阶段", st["S"], 2, 1)

    if st["best"]:
        add("**教师最优 valid MAE**：%s\n" % " → ".join(st["best"]))
    if st["th"]:
        add("**阈值/验证指标**：%s\n" % "; ".join("theta=%s, validF1=%s" % x for x in st["th"]))
    if st["val"]:
        add("**VALID**：%s\n" % st["val"][-1])
    if st["test"]:
        add("**TEST**：%s\n" % st["test"][-1])
    if st["saves"]:
        add("**已保存模型**：")
        for p, mb in st["saves"]:
            add("- `%s`  (%.2f MB)" % (p, float(mb)))
        add("")
    add("**最近日志行**：\n```\n%s\n```\n" % "\n".join(st.get("cur", st["text"]).splitlines()[-6:]))
    return "\n".join(L), dict(frac=frac, eta=eta, phase=phase, dt=dt, ds=ds,
                              done_seeds=done_seeds, total_seeds=total_seeds,
                              finished=st["finished"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--interval", type=float, default=15)
    ap.add_argument("--seeds", type=int, default=2, help="计划训练的随机种子总数（用于估算剩余时间）")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--md", default=os.path.join(ROOT, "runs", "progress.md"))
    a = ap.parse_args()

    log = a.log or newest_log()
    explicit = a.log is not None
    if a.once:
        st = parse(log)
        md, info = build(st, a.seeds)
        print(md)
        return
    print("监控中：%s（每 %.0f 秒刷新，Ctrl+C 退出）" % (log, a.interval))
    time.sleep(1.2)
    try:
        while True:
            st = parse(log if explicit else newest_log())
            md, info = build(st, a.seeds)
            os.makedirs(os.path.dirname(a.md), exist_ok=True)
            with open(a.md, "w", encoding="utf-8") as f:
                f.write(md)
            try:
                os.system("cls")
            except Exception:
                pass
            print(md)
            if info and info.get("finished"):
                print("\n*** 训练全部完成 —— 产物在 runs/q2/，可执行推理命令 ***")
                break
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\n已停止监控（训练仍在后台继续）。")


if __name__ == "__main__":
    main()
