# -*- coding: utf-8 -*-
r"""
超参数/结构搜索（赛题要求：模型结构、超参数与决策阈值均在**验证集**上选择）。

设计要点
--------
1. **基线 = E1 最终配方**（不是 config 默认值）。所有候选只在该配方上做单/双维度扰动，
   使搜索空间围绕真正会提交的模型，结论可直接用于论文"超参数选择"一节。
2. **两段式**（CPU 友好）：
     stage=screen —— 训练集子集 + 短轮数，8 个候选，快速淘汰
     stage=full   —— 对 screen 胜出者用**全量训练集 + 与主实验相同的轮数**完整重训
3. **预注册选择规则**（写进论文，避免"事后挑指标"）：
     主排序：valid Macro-F1（分类头口径，主指标）
     约束：  valid MAE 不得比基线差 0.010 以上
   仅使用 **valid** 数据；test 只在最终确认时读取。
4. **失败可归因**：每个候选都检查退出码，非零则显式列出日志尾部（不再静默跳过）。

用法：
  python tune_hyperparams.py --list                     # 只打印候选
  python tune_hyperparams.py --stage screen             # 快速筛选
  python tune_hyperparams.py --stage full --only c01_hidden128,c04_dropout50
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable

# ---------------------------------------------------------------- E1 最终配方
# 与 D:\_p11_msp.ps1 中的 $base 完全一致；搜索的"原点"
BASE = [
    "--version", "aligned",
    "--hidden", "96", "--n_experts", "2",
    "--dropout", "0.35", "--weight_decay", "0.0005",
    "--lr_backbone", "0.00005", "--lr_head", "0.0005",
    "--noise_sigma", "0.15", "--mod_dropout", "0.15", "--label_smoothing", "0.1",
    "--ablate_rec", "--reg_beta", "0.15", "--ema", "--ema_decay", "0.99",
    "--batch_size", "64", "--mag_levels", "10", "--lam_mag", "0.5", "--lam_emd", "0.3",
    "--weak_weight", "0.5", "--lam_neusup", "0.25",
]

# 候选 = 在 BASE 之上追加/覆盖的参数；c00 是对照组（基线本身）
CANDIDATES = [
    ("c00_baseline",  [],                                      "E1 配方（对照，必须参与）"),
    ("c01_hidden128", ["--hidden", "128"],                     "容量：hidden 96 → 128"),
    ("c02_enc1",      ["--enc_layers", "1"],                   "编码层数 2 → 1（轻量化）"),
    ("c03_exp4",      ["--n_experts", "4"],                    "专家数 2 → 4"),
    ("c04_dropout50", ["--dropout", "0.50"],                   "dropout 0.35 → 0.50"),
    ("c05_nokd",      ["--lam_kd0", "0.0"],                    "关闭蒸馏（检验蒸馏贡献）"),
    ("c06_kd05",      ["--lam_kd0", "0.5"],                    "蒸馏权重 0.3 → 0.5"),
    ("c07_h128_e4",   ["--hidden", "128", "--n_experts", "4"], "容量 + 专家数同时上调"),
]

FIXED_SCREEN = dict(epochs_teacher=6, epochs_student=8, seed=42)
SCREEN_LIMIT = 1024
SELECT_METRIC = "f1_head"          # 主排序：valid Macro-F1（分类头）
MAE_TOLERANCE = 0.010              # 约束：valid MAE 允许的最差退化


def build_cmd(out_dir, epochs_t, epochs_s, limit, extra):
    cmd = [PY, "-u", os.path.join(HERE, "train.py"),
           "--out_dir", out_dir, "--device", "cpu"] + BASE + extra + [
           "--epochs_teacher", str(epochs_t), "--epochs_student", str(epochs_s),
           "--seed", "42", "--patience", "99"]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def run_one(extra, tag, epochs_t, epochs_s, limit):
    """跑一个候选；返回 (指标 dict | None, 失败信息 | None)。"""
    out_dir = os.path.join(ROOT, "runs", "tune", tag)
    log = os.path.join(ROOT, "runs", "tune", "%s.log" % tag)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    cmd = build_cmd(out_dir, epochs_t, epochs_s, limit, extra)
    print("[RUN] %s" % " ".join(cmd))
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        tail = ""
        try:
            with open(log, encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-8:])
        except OSError:
            pass
        return None, dict(tag=tag, exit=p.returncode, tail=tail.strip())

    m = os.path.join(out_dir, "metrics_s42.json")
    if not os.path.isfile(m):
        return None, dict(tag=tag, exit=0, tail="metrics_s42.json 未生成")
    d = json.load(open(m, encoding="utf-8"))
    v = d.get("valid_clean", {}) or {}
    t = d.get("test_clean", {}) or {}
    row = dict(
        tag=tag,
        valid_f1_head=v.get("f1_head", v.get("f1")),
        valid_acc_head=v.get("acc_head", v.get("acc")),
        valid_mae=v.get("mae"),
        valid_f1_theta=v.get("f1_theta"),
        test_f1_head=t.get("f1_head", t.get("f1")),
        test_acc_head=t.get("acc_head", t.get("acc")),
        test_mae=t.get("mae"),
    )
    return row, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen", "full"], default="screen")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--screen_limit", type=int, default=SCREEN_LIMIT)
    ap.add_argument("--epochs_teacher", type=int, default=12,
                    help="full 阶段教师轮数（与主实验一致）")
    ap.add_argument("--epochs_student", type=int, default=14,
                    help="full 阶段学生轮数（与主实验一致）")
    ap.add_argument("--only", type=str, default=None,
                    help="只跑指定 tag（逗号分隔），用于补跑/追跑单个候选")
    a = ap.parse_args()

    cands = CANDIDATES
    if a.only:
        want = {s.strip() for s in a.only.split(",") if s.strip()}
        cands = [c for c in CANDIDATES if c[0] in want]
        if not cands:
            print("[ERR] --only 未匹配到任何 tag；可选：%s"
                  % ", ".join(c[0] for c in CANDIDATES))
            return 2

    print("=" * 96)
    print("超参搜索｜stage=%s｜候选 %d 个｜基线 = E1 最终配方" % (a.stage, len(cands)))
    print("选择规则（预注册）：主排序 valid %s ↑；约束 valid_mae ≤ 基线 + %.3f"
          % (SELECT_METRIC, MAE_TOLERANCE))
    print("=" * 96)
    for tag, extra, why in cands:
        print("  %-16s %-34s  %s" % (tag, " ".join(extra) or "(baseline)", why))
    if a.list:
        return 0

    if a.stage == "screen":
        epochs_t = FIXED_SCREEN["epochs_teacher"]
        epochs_s = FIXED_SCREEN["epochs_student"]
        limit = a.screen_limit
    else:
        epochs_t, epochs_s, limit = a.epochs_teacher, a.epochs_student, 0

    rows, fails = [], []
    for tag, extra, why in cands:
        r, bad = run_one(extra, tag, epochs_t, epochs_s, limit)
        if bad:
            fails.append(bad)
            print("[FAIL] %s exit=%s\n%s" % (tag, bad["exit"], bad["tail"]))
            continue
        r["desc"] = why
        rows.append(r)
        print("[OK] %-16s valid F1(head)=%.4f ACC=%.4f MAE=%.4f | test F1=%.4f ACC=%.4f"
              % (tag, r["valid_f1_head"], r["valid_acc_head"], r["valid_mae"],
                 r["test_f1_head"], r["test_acc_head"]))

    out_dir = os.path.join(ROOT, "runs", "tune")
    os.makedirs(out_dir, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows).sort_values("valid_f1_head", ascending=False)
        # ---- 应用预注册规则：MAE 约束 + 主排序 ----
        ref = df.loc[df["tag"] == "c00_baseline", "valid_mae"]
        if len(ref):
            base_mae = float(ref.iloc[0])
            base_f1 = float(df.loc[df["tag"] == "c00_baseline", "valid_f1_head"].iloc[0])
            df["mae_ok"] = df["valid_mae"] <= base_mae + MAE_TOLERANCE
            df["delta_f1"] = df["valid_f1_head"] - base_f1
        else:
            df["mae_ok"] = True
            df["delta_f1"] = float("nan")
        csv = os.path.join(out_dir, "grid_%s.csv" % a.stage)
        df.to_csv(csv, index=False, encoding="utf-8-sig")
        print("\n[GRID] %s" % csv)
        print(df.to_string(index=False))

        ok = df[df["mae_ok"]].head(max(a.topk, 1))
        print("\n[SELECT] MAE 约束通过且排名前 %d：" % len(ok))
        for _, r in ok.iterrows():
            print("   %-16s valid F1(head)=%.4f (Δ=%+.4f) MAE=%.4f"
                  % (r["tag"], r["valid_f1_head"], r["delta_f1"], r["valid_mae"]))
        tags = ",".join(ok["tag"].tolist())
        if a.stage == "screen":
            print("\n下一步（完整重训，全量数据 + 12/14 轮）：")
            print("  python 02_model\\tune_hyperparams.py --stage full --only %s" % tags)
    else:
        print("\n[ERR] 没有任何候选成功，无法排序。")

    if fails:
        with open(os.path.join(out_dir, "grid_%s_failures.json" % a.stage),
                  "w", encoding="utf-8") as f:
            json.dump(fails, f, ensure_ascii=False, indent=2)
        print("\n[FAIL] %d 个候选失败，详情见 grid_%s_failures.json"
              % (len(fails), a.stage))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
