# -*- coding: utf-8 -*-
r"""
缺失因素三因素扫描（**独立于训练**，可对任意已训练 ckpt 直接运行）。

题目要求：分析**缺失模态类型、缺失位置、缺失时长（缺失率）**对预测性能的影响规律。
本脚本对指定划分构造 类型 × 位置 × 缺失率 的场景网格，并在两种推理方式下评测：

  * `aware_*`   ：本文模型（模型能收到"缺失掩码"指示）
  * `unaware_*` ：A0 基线（把掩码置零喂给模型，模拟"未做缺失处理的常规固定权重融合模型"）

输出 CSV 的列与 `train.py --sweep 1` 完全一致，可直接交给 `analyze_sweep.py`
做 Type-III ANOVA、三次多项式退化曲线与热力图。

用法：
  # 完整网格（valid），7 类型 × 4 位置 × 6 档 ≈ 30 分钟
  python 02_model\sweep_eval.py --ckpt_dir ..\runs\q2v3 --out_dir ..\runs\q2v3 --tag s42

  # 快速自检（只跑少量组合）
  python 02_model\sweep_eval.py --ckpt_dir ..\runs\q2 --types text audio --rhos 0.2 0.4

  # 代表性场景在 test 上核验
  python 02_model\sweep_eval.py --ckpt_dir ..\runs\q2v3 --out_dir ..\runs\q2v3 --tag s42 --split test
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config                                    # noqa: E402
from data_utils import load_all                              # noqa: E402
from losses import compute_metrics                           # noqa: E402
import runtime                                               # noqa: E402

TYPES = ["text", "audio", "vision", "text+audio", "text+vision",
         "audio+vision", "text+audio+vision"]
POSITIONS = ["head", "middle", "tail", "random"]
RHOS = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]


def evaluate(models, split, theta, batch_size, device, scenario=None, unaware=False):
    p = runtime.predict_split(models, split, batch_size=batch_size, device=device,
                              scenario=scenario, unaware=unaware)
    res = compute_metrics(p["y_reg"], p["score"], p["y_cls"], theta=theta,
                          y_pred_cls=p["prob"].argmax(-1))
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tag", default="s42")
    ap.add_argument("--split", default="valid", choices=["valid", "test", "both"])
    ap.add_argument("--types", nargs="*", default=TYPES)
    ap.add_argument("--positions", nargs="*", default=POSITIONS)
    ap.add_argument("--rhos", nargs="*", type=float, default=RHOS)
    ap.add_argument("--scenario_seed", type=int, default=1234)
    ap.add_argument("--reps", type=int, default=1,
                    help="每个场景重复次数（不同随机种子）。ANOVA 的三因素交互需要重复，建议 ≥3")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--version", default="aligned")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--baseline", type=int, default=1, help="1=额外输出无缺失基线行")
    ap.add_argument("--mask_off", action="store_true",
                    help="A2 消融模型专用：主列也关闭缺失指示（该模型训练时就看不到指示）")
    a = ap.parse_args()

    out_dir = a.out_dir or a.ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    cfg = Config(version=a.version, out_dir=out_dir)
    if a.data_dir:
        cfg.data_dir = a.data_dir
    cfg.__post_init__()
    print("[DATA] pkl = %s" % cfg.pkl_path)
    tr, va, te = load_all(cfg)

    ckpts = runtime.find_ckpts(a.ckpt_dir)
    if not ckpts:
        raise SystemExit("未找到 ckpt：%s" % a.ckpt_dir)
    models = runtime.load_ensemble(ckpts, device=a.device)
    theta = float(np.mean([ck["theta"] for _, ck in models]))
    print("[INFO] 集成 %d 个 ckpt，theta=%.2f" % (len(ckpts), theta))

    splits = [("valid", va)] if a.split == "valid" else \
             ([("test", te)] if a.split == "test" else [("valid", va), ("test", te)])

    # 场景网格：rho=1（整模态缺失）仅对单模态类型有意义，避免多模态组合退化为全空
    grid = []
    for mt in a.types:
        for pos in a.positions:
            for rho in a.rhos:
                if rho >= 1.0 and "+" in mt:
                    continue
                grid.append((mt, pos, float(rho)))

    for name, sp in splits:
        t0 = time.time()
        rows = []
        print("\n[%s] N=%d，共 %d 个场景 × %d 次重复 × 2（aware/unaware）"
              % (name, sp["N"], len(grid), a.reps))
        try:
            base = evaluate(models, sp, theta, a.batch_size, a.device)      # 无缺失基线
            for r in range(a.reps):
                rows.append(dict(missing_type="none", position="none", rho=0.0, rep=r,
                                 aware_mae=base["mae"], aware_f1=base["f1"],
                                 aware_acc=base["acc"], aware_corr=base["pearson"],
                                 unaware_mae=base["mae"], unaware_f1=base["f1"],
                                 unaware_acc=base["acc"], unaware_corr=base["pearson"]))
            print("  [base ] MAE=%.4f F1=%.4f ACC=%.4f" % (base["mae"], base["f1"], base["acc"]))
        except Exception as e:
            print("  [WARN] 基线评测失败：%s" % e)

        total = len(grid) * a.reps
        i = 0
        for rep in range(a.reps):
            # 每个 rep 使用不同的场景种子。
            # 注意：`evaluate` 的 seed 默认是 1234，之前**没有把它传下去**，
            # 导致所有 rep 生成完全相同的缺失掩码、指标逐位重复（伪重复），
            # 进而使 analyze_sweep 的 ANOVA 残差方差≈0、F 值虚高到 1e25 量级。
            sd = int(a.scenario_seed) + 1000 * rep
            for mt, pos, rho in grid:
                i += 1
                try:
                    ra = evaluate(models, sp, theta, a.batch_size, a.device,
                                  scenario=(mt, pos, rho), unaware=bool(a.mask_off),
                                  seed=sd)
                    ru = evaluate(models, sp, theta, a.batch_size, a.device,
                                  scenario=(mt, pos, rho), unaware=True, seed=sd)
                except Exception as e:
                    print("  [WARN] 场景 %s/%s/%.2f rep%d 失败：%s" % (mt, pos, rho, rep, e))
                    continue
                rows.append(dict(missing_type=mt, position=pos, rho=float(rho), rep=rep,
                                 aware_mae=ra["mae"], aware_f1=ra["f1"],
                                 aware_acc=ra["acc"], aware_corr=ra["pearson"],
                                 unaware_mae=ru["mae"], unaware_f1=ru["f1"],
                                 unaware_acc=ru["acc"], unaware_corr=ru["pearson"]))
                if i % 20 == 0 or i == total:
                    print("  [%3d/%3d] %s/%s/rho=%.2f rep%d  aware MAE=%.4f | unaware MAE=%.4f  (%.0fs)"
                          % (i, total, mt, pos, rho, rep, ra["mae"], ru["mae"], time.time() - t0))

        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, "sweep_%s_%s.csv" % (name, a.tag))
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print("[SAVE] %s（%d 行，用时 %.1f 分钟）" % (path, len(df), (time.time() - t0) / 60))

        # 简要规律摘要（给论文正文用）
        try:
            sub = df[df["rho"] > 0]
            if not sub.empty:
                g1 = sub.groupby("missing_type")[["aware_mae", "unaware_mae"]].mean()
                g1["gain"] = g1["unaware_mae"] - g1["aware_mae"]
                print("\n  按缺失类型平均 MAE（aware / unaware / 收益）：")
                print(g1.round(4).to_string())
                g2 = sub.groupby("position")[["aware_mae", "unaware_mae"]].mean()
                print("\n  按缺失位置平均 MAE：")
                print(g2.round(4).to_string())
                g3 = sub.groupby("rho")[["aware_mae", "unaware_mae"]].mean()
                print("\n  按缺失率平均 MAE：")
                print(g3.round(4).to_string())
        except Exception as e:
            print("  [WARN] 摘要失败：%s" % e)


if __name__ == "__main__":
    main()
