# -*- coding: utf-8 -*-
r"""D1=A：用修正后的文本长度口径重出附件3/附件4 预测文件，并生成口径修订报告。

流程：
  1. 备份现有 submission/pred_att3.csv、pred_att4.csv（含 _full 版）到 submission/_backup_preD1/；
  2. 用 runs/ens_top2 集成、θ=0.325 重跑推理（同一命令、同一阈值，只改了输入口径）；
  3. 对比新旧文件的分类标签、强度分数、极性分布，写出报告
     runs/q3/d1_recalib_report.{json,md}。

用法：python d1_recalib.py
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
SUB = os.path.join(ROOT, "submission")
BAK = os.path.join(SUB, "_backup_preD1")
RUNS = os.path.join(ROOT, "runs", "q3")
PY = sys.executable
THETA = "0.325"          # 与问题2 最终口径一致（valid 上选定）

JOBS = [
    ("att3", os.path.join(ROOT, "data_att", "att3_aligned.pkl"), "pred_att3.csv"),
    ("att4", os.path.join(ROOT, "data_att", "att4_aligned.pkl"), "pred_att4.csv"),
]


def backup():
    os.makedirs(BAK, exist_ok=True)
    for _, _, name in JOBS:
        for f in (name, name.replace(".csv", "_full.csv")):
            src = os.path.join(SUB, f)
            dst = os.path.join(BAK, f)
            if os.path.isfile(src) and not os.path.isfile(dst):
                shutil.copy2(src, dst)
                print("[BAK] %s -> %s" % (f, os.path.relpath(dst, ROOT)))


def run_infer(pkl, out_csv):
    cmd = [PY, "-u", os.path.join(CODE, "02_model", "infer_att3.py"),
           "--pkl", pkl, "--ckpt_dir", os.path.join(ROOT, "runs", "ens_top2"),
           "--out_csv", out_csv, "--theta", THETA, "--device", "cpu"]
    print("[RUN] %s" % " ".join(os.path.basename(c) if c.startswith(ROOT) else c for c in cmd))
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, cwd=CODE, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    tail = [l for l in (r.stdout or "").splitlines() if l.strip()][-6:]
    for l in tail:
        print("      " + l)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit("推理失败：%s" % out_csv)
    return r.stdout


def compare(tag, name):
    old = pd.read_csv(os.path.join(BAK, name))
    new = pd.read_csv(os.path.join(SUB, name))
    assert len(old) == len(new), "行数不一致"
    assert list(old["id"]) == list(new["id"]), "id 顺序不一致"
    lab_chg = int((old["pred_label"] != new["pred_label"]).sum())
    rec = dict(
        n=int(len(new)),
        label_changed=lab_chg,
        label_changed_ratio=round(lab_chg / len(new), 4),
        score_absdiff_mean=round(float(np.abs(old["pred_score"] - new["pred_score"]).mean()), 4),
        score_absdiff_max=round(float(np.abs(old["pred_score"] - new["pred_score"]).max()), 4),
        mean_score_old=round(float(old["pred_score"].mean()), 4),
        mean_score_new=round(float(new["pred_score"].mean()), 4),
        dist_old=old["pred_label"].value_counts().to_dict(),
        dist_new=new["pred_label"].value_counts().to_dict(),
        mean_conf_old=round(float(old["confidence"].mean()), 4),
        mean_conf_new=round(float(new["confidence"].mean()), 4),
    )
    print("[CMP %s] 标签改动 %d/%d (%.1f%%)  |Δscore| 均值 %.4f 最大 %.4f"
          % (tag, lab_chg, len(new), 100 * lab_chg / len(new),
             rec["score_absdiff_mean"], rec["score_absdiff_max"]))
    print("        强度均值 %.4f → %.4f ; 分布 %s → %s"
          % (rec["mean_score_old"], rec["mean_score_new"], rec["dist_old"], rec["dist_new"]))
    return rec


def main():
    os.makedirs(RUNS, exist_ok=True)
    backup()
    rep = {}
    for tag, pkl, name in JOBS:
        run_infer(pkl, os.path.join(SUB, name))
        rep[tag] = compare(tag, name)
    with open(os.path.join(RUNS, "d1_recalib_report.json"), "w", encoding="utf-8") as f:
        json.dump(dict(theta=float(THETA), model="runs/ens_top2 (2 ckpt)", report=rep),
                  f, ensure_ascii=False, indent=2)
    md = ["# D1 文本长度口径修正报告（附件3/4 重出）", "",
          "模型：`runs/ens_top2`（2 种子集成）｜决策阈值 θ=%s（valid 上选定，与问题2 一致）" % THETA,
          "修正内容：文本有效长度由「恒为 50」改为「`text_bert` 注意力掩码长度」；数值数组逐位未变。", ""]
    for tag, r in rep.items():
        md += ["## %s" % tag, "",
               "| 指标 | 修正前 | 修正后 |", "|---|---:|---:|",
               "| 极性分布（负/中/正） | %s | %s |" % (r["dist_old"], r["dist_new"]),
               "| 平均强度 | %.4f | %.4f |" % (r["mean_score_old"], r["mean_score_new"]),
               "| 平均置信度 | %.4f | %.4f |" % (r["mean_conf_old"], r["mean_conf_new"]),
               "| 分类标签改动 | — | %d/%d（%.1f%%） |" % (r["label_changed"], r["n"],
                                                          100 * r["label_changed_ratio"]),
               "| 强度分数 \\|Δ\\| 均值/最大 | — | %.4f / %.4f |" % (r["score_absdiff_mean"],
                                                                     r["score_absdiff_max"]), ""]
    with open(os.path.join(RUNS, "d1_recalib_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("[SAVE] runs/q3/d1_recalib_report.{json,md}")
    print("D1_RECALIB_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
