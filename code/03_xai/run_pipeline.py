# -*- coding: utf-8 -*-
r"""问题3 一键流水线：把 S1–S9 串起来跑，逐步打印状态并落日志。

每一步都是独立的 .py 子进程，日志写到 runs/q3/logs/{step}.log（UTF-8，可直接阅读）。
任一步失败不中断后续步骤（除非 --strict），最后打印汇总表；失败步骤可用 --only 重跑。

用法：
  python run_pipeline.py                       # 全部步骤（已存在的产物会覆盖更新）
  python run_pipeline.py --only fuse,metrics   # 只跑指定步骤
  python run_pipeline.py --steps 3,4,5         # 按序号跑（便于分阶段执行）
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
PY = sys.executable
RUNS = os.path.join(ROOT, "runs", "q3")
LOGS = os.path.join(RUNS, "logs")
DATA_ATT = os.path.join(ROOT, "data_att")
ATT2 = os.path.join(ROOT, "附件2", "aligned_50.pkl")

STEPS = [
    ("1_shapley_att3", ["shapley.py", "--pkl", os.path.join(DATA_ATT, "att3_aligned.pkl"),
                        "--name", "att3", "--out_dir", RUNS]),
    ("2_shapley_att4", ["shapley.py", "--pkl", os.path.join(DATA_ATT, "att4_aligned.pkl"),
                        "--name", "att4", "--out_dir", RUNS]),
    ("3_shapley_valid", ["shapley.py", "--pkl", ATT2, "--split", "valid", "--name", "valid",
                         "--out_dir", RUNS, "--with_global"]),
    ("4_ig_att4", ["ig_gradcam.py", "--pkl", os.path.join(DATA_ATT, "att4_aligned.pkl"),
                   "--name", "att4", "--steps", "20", "--out_dir", RUNS]),
    ("5_ig_valid", ["ig_gradcam.py", "--pkl", ATT2, "--split", "valid", "--name", "valid",
                    "--steps", "20", "--out_dir", RUNS]),
    ("6_occ_att4", ["occlusion.py", "--pkl", os.path.join(DATA_ATT, "att4_aligned.pkl"),
                    "--name", "att4", "--width", "3", "--stride", "1", "--out_dir", RUNS]),
    ("7_occ_valid", ["occlusion.py", "--pkl", ATT2, "--split", "valid", "--name", "valid",
                     "--width", "3", "--stride", "2", "--out_dir", RUNS]),
    ("8_fuse_att4", ["temporal_fuse.py", "--name", "att4", "--out_dir", RUNS]),
    ("9_fuse_valid", ["temporal_fuse.py", "--name", "valid", "--out_dir", RUNS]),
    ("10_metrics_att4", ["metrics_xai.py", "--pkl", os.path.join(DATA_ATT, "att4_aligned.pkl"),
                         "--name", "att4", "--auc", "--stability", "--out_dir", RUNS]),
    ("11_metrics_valid", ["metrics_xai.py", "--pkl", ATT2, "--split", "valid", "--name", "valid",
                          "--n_max", "200", "--out_dir", RUNS]),
    ("12_evidence_att4", ["evidence.py", "--name", "att4", "--out_dir", RUNS]),
    ("13_card_att4", ["explain_card.py", "--name", "att4", "--out_dir", RUNS]),
    ("14_infer_explain", ["infer_explain_att4.py", "--out_dir", RUNS]),
    ("15_valid_analysis", ["q3_valid_analysis.py", "--out_dir", RUNS]),
    ("16_paper_figs", ["paper_figs_q3.py", "--out_dir", RUNS]),
    ("17_verify", ["verify_q3.py", "--out_dir", RUNS]),
]


def run_one(name, args, strict=False):
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, "%s.log" % name)
    cmd = [PY, "-u"] + args
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    print("[RUN ] %-16s %s" % (name, " ".join(os.path.basename(c) for c in cmd[2:])))
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    ok = (r.returncode == 0)
    tail = ""
    try:
        lines = [l.rstrip() for l in open(log, encoding="utf-8") if l.strip()]
        tail = lines[-1] if lines else ""
    except OSError:
        pass
    print("[%s] %-16s %5.1fs  %s" % ("OK  " if ok else "FAIL", name, dt, tail[:90]))
    if not ok:
        print("        见 %s" % os.path.relpath(log, ROOT))
        if strict:
            raise SystemExit("步骤 %s 失败" % name)
    return ok, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="逗号分隔的步骤名")
    ap.add_argument("--steps", default=None, help="逗号分隔的步骤序号（1 起）")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    sel = STEPS
    if a.only:
        want = [s.strip() for s in a.only.split(",")]
        sel = [s for s in STEPS if s[0] in want]
    if a.steps:
        idx = [int(x) - 1 for x in a.steps.split(",")]
        sel = [STEPS[i] for i in idx]

    print("=== 问题3 流水线：%d 个步骤 ===" % len(sel))
    res = []
    for name, args in sel:
        ok, dt = run_one(name, args, a.strict)
        res.append((name, ok, dt))
    print("\n=== 汇总 ===")
    for name, ok, dt in res:
        print("  %-16s %s  %5.1fs" % (name, "OK" if ok else "FAIL", dt))
    bad = [n for n, ok, _ in res if not ok]
    print("PIPELINE_%s" % ("OK" if not bad else "PARTIAL_FAIL:" + ",".join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
