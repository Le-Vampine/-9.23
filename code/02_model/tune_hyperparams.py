# -*- coding: utf-8 -*-
r"""
超参数/结构搜索（题目要求：在验证集上选择模型结构、超参数与决策阈值）。

做法：在验证集上评估若干候选配置，记录 valid MAE / MacroF1，输出排序表。
采用"小轮数筛选 + 最优配置完整训练"的两段式，避免在 CPU 上耗时过长。

用法：
  python tune_hyperparams.py --stage screen            # 快速筛选（默认 8/10 轮）
  python tune_hyperparams.py --stage full --topk 3     # 对最优 3 个配置完整训练
  python tune_hyperparams.py --list                    # 只打印候选配置
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

GRID_SMALL = {          # 默认：只搜影响最大的 3 个维度（8 组）
    "hidden": [96, 128],
    "enc_layers": [1, 2],
    "lam_kd0": [0.0, 0.3],
}
GRID_FULL = {           # --full_grid：32 组完整网格
    "hidden": [96, 128],
    "enc_layers": [1, 2],
    "n_experts": [2, 4],
    "lam_rec": [0.5, 1.0],
    "lam_kd0": [0.0, 0.3],
}
FIXED_SCREEN = dict(epochs_teacher=6, epochs_student=8, seed=42)
SCREEN_LIMIT = 768      # 筛选阶段只用训练集子集（CPU 友好）


def candidates(grid):
    keys = list(grid.keys())
    out = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        out.append(dict(zip(keys, vals)))
    return out


def run_one(cfg, tag, stage, extra=None):
    out_dir = os.path.join(ROOT, "runs", "tune", tag)
    cmd = [sys.executable, os.path.join(HERE, "train.py"),
           "--out_dir", out_dir, "--device", "cpu", "--version", "aligned"]
    for k, v in cfg.items():
        cmd += ["--" + k, str(v)]
    if extra:
        cmd += extra
    print("[RUN] %s" % " ".join(cmd))
    log = os.path.join(ROOT, "runs", "tune", "%s.log" % tag)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    m = os.path.join(out_dir, "metrics_s%d.json" % cfg.get("seed", 42))
    if not os.path.isfile(m):
        return None
    d = json.load(open(m, encoding="utf-8"))
    v, t = d.get("valid_clean", {}), d.get("test_clean", {})
    return dict(tag=tag, **cfg, theta=d.get("theta"),
                valid_mae=v.get("mae"), valid_acc=v.get("acc"), valid_f1=v.get("f1"),
                test_mae=t.get("mae"), test_acc=t.get("acc"), test_f1=t.get("f1"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen", "full"], default="screen")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--full_grid", action="store_true", help="使用 32 组完整网格")
    ap.add_argument("--screen_limit", type=int, default=SCREEN_LIMIT)
    ap.add_argument("--epochs_teacher", type=int, default=20)
    ap.add_argument("--epochs_student", type=int, default=30)
    a = ap.parse_args()

    cands = candidates(GRID_FULL if a.full_grid else GRID_SMALL)
    print("候选配置 %d 个：" % len(cands))
    for i, c in enumerate(cands):
        print("  #%02d %s" % (i, c))
    if a.list:
        return

    rows = []
    for i, c in enumerate(cands):
        cfg = dict(c)
        if a.stage == "screen":
            cfg.update(FIXED_SCREEN)
            extra = ["--limit", str(a.screen_limit)]
        else:
            cfg.update(dict(epochs_teacher=a.epochs_teacher,
                            epochs_student=a.epochs_student, seed=42))
            extra = []
        tag = "cfg%02d_%s" % (i, "_".join("%s%s" % (k, v) for k, v in c.items()))
        r = run_one(cfg, tag, a.stage, extra=extra)
        if r:
            rows.append(r)
            print("[OK] %s -> valid MAE=%.4f ACC=%.4f F1=%.4f"
                  % (tag, r["valid_mae"], r["valid_acc"], r["valid_f1"]))

    df = pd.DataFrame(rows).sort_values("valid_mae")
    out = os.path.join(ROOT, "runs", "tune", "grid_%s.csv" % a.stage)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n[GRID] 已写入 %s" % out)
    print(df.head(10).to_string(index=False))

    if a.stage == "screen" and len(df) >= a.topk:
        print("\n[建议] 用 --stage full 对前 %d 个配置做完整训练" % a.topk)
        for _, r in df.head(a.topk).iterrows():
            print("   %s" % r["tag"])


if __name__ == "__main__":
    main()
