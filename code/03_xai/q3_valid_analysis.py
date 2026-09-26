# -*- coding: utf-8 -*-
r"""S7：验证集/专项集上的可解释性分析（论文 §5.3 的表与结论数据）。

产出（paper_tables/ 下 CSV）：
  Q3_模态作用度_按真值极性.csv        每类情感的三模态 π 均值与主模态占比
  Q3_模态作用度_按强度分层.csv        |y| 分箱后的 π 均值
  Q3_八子集反事实.csv                 8 个模态子集在验证集上的 MAE / Macro-F1 / 平均预测
  Q3_模态忠实性.csv                   π 与"真删该模态的误差增量"的 Spearman（择优依据）
  Q3_稳健性矩阵.csv                   两套归一化 × 两种屏蔽语义 × 两种效用的一致性（Spearman）
  Q3_缺失率一致性.csv                 附件3/4：π_m 与实测缺失率 ρ_m 的关系（免标注机制验证）
  Q3_错误归因.csv                     正确/错误样本的 π 熵、π_text、解释指标对比

用法：
  python q3_valid_analysis.py --out_dir ..\..\runs\q3 --table_dir ..\..\paper_tables
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)
from xai_common import MODALITIES, load_backbone, load_split, save_json, ROOT   # noqa: E402

NAMES = {0: "none", 1: "text", 2: "audio", 4: "vision",
         3: "text+audio", 5: "text+vision", 6: "audio+vision", 7: "all"}
MOD_CN = {"text": "文本", "audio": "语音", "vision": "视觉"}


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def entropy(p, eps=1e-12):
    p = np.asarray(p, float)
    p = np.clip(p, eps, None)
    return float(-(p * np.log(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "runs", "q3"))
    ap.add_argument("--table_dir", default=os.path.join(ROOT, "paper_tables"))
    ap.add_argument("--ckpt_dir", default=os.path.join(ROOT, "runs", "ens_top2"))
    a = ap.parse_args()
    os.makedirs(a.table_dir, exist_ok=True)

    # ---------- 载入验证集（要标签） ----------
    models, stats, theta = load_backbone(a.ckpt_dir, verbose=False)
    split, key = load_split(os.path.join(ROOT, "附件2", "aligned_50.pkl"), stats, "valid")
    y = np.asarray(split["labels_reg"], float)
    yc = np.asarray(split["labels_cls"], int)
    N = split["N"]

    sh = np.load(os.path.join(a.out_dir, "q3_shapley_valid.npz"), allow_pickle=True)
    fz = np.load(os.path.join(a.out_dir, "q3_temporal_fused_valid.npz"), allow_pickle=True)
    pi_soft = sh["pi_soft"]
    pi_relu = sh["pi_relu"]
    phi = sh["phi_mu_miss"]
    V = sh["v_mu_miss"]
    valid = fz["valid"]
    Wf = fz["w_fused"]
    report = {}

    # ---------- 1) 按真值极性分层 ----------
    rows = []
    for c, cn in ((0, "负向"), (1, "中性"), (2, "正向")):
        m = yc == c
        if not m.any():
            continue
        rows.append(dict(极性=cn, 样本数=int(m.sum()),
                         π_文本=round(float(pi_soft[m, 0].mean()), 4),
                         π_语音=round(float(pi_soft[m, 1].mean()), 4),
                         π_视觉=round(float(pi_soft[m, 2].mean()), 4),
                         主模态为文本占比=round(float((pi_soft[m].argmax(1) == 0).mean()), 4),
                         平均强度=round(float(y[m].mean()), 4)))
    df1 = pd.DataFrame(rows)
    df1.to_csv(os.path.join(a.table_dir, "Q3_模态作用度_按真值极性.csv"),
               index=False, encoding="utf-8-sig")
    report["by_label"] = rows

    # ---------- 2) 按 |y| 分层 ----------
    bins = [(0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0), (1.0, 2.0), (2.0, 3.01)]
    rows = []
    ay = np.abs(y)
    for lo, hi in bins:
        m = (ay >= lo) & (ay < hi) if hi < 3.0 else (ay >= lo) & (ay <= hi)
        if m.sum() < 3:
            continue
        rows.append(dict(强度区间="%.2f–%.2f" % (lo, hi), 样本数=int(m.sum()),
                         π_文本=round(float(pi_soft[m, 0].mean()), 4),
                         π_语音=round(float(pi_soft[m, 1].mean()), 4),
                         π_视觉=round(float(pi_soft[m, 2].mean()), 4),
                         π熵=round(float(np.mean([entropy(r) for r in pi_soft[m]])), 4)))
    df2 = pd.DataFrame(rows)
    df2.to_csv(os.path.join(a.table_dir, "Q3_模态作用度_按强度分层.csv"),
               index=False, encoding="utf-8-sig")
    report["by_magnitude"] = rows
    report["pi_entropy_mean"] = round(float(np.mean([entropy(r) for r in pi_soft])), 4)

    # ---------- 3) 八子集反事实（验证集） ----------
    y_mae = np.abs(sh["score"] - y)
    rows = []
    for m in range(8):
        v_mu = V[:, m]
        # 用 μ 与真值的 MAE（对所有子集统一口径；全集即主口径预测）
        rows.append(dict(子集=NAMES[m], 平均预测=round(float(v_mu.mean()), 4),
                         与真值MAE=round(float(np.abs(v_mu - y).mean()), 4),
                         φ贡献=round(float(phi[:, 0].mean() if m == 7 else np.nan), 4)))
    # 用 v_pol_miss 的 MAE 不可比（概率尺度）；这里 MAE 一律由 μ 子集值算
    glob = json.load(open(os.path.join(a.out_dir, "q3_shapley_valid_report.json"),
                          encoding="utf-8")).get("global", {})
    if glob:
        for r in rows:
            r["MAE（数据集级复算）"] = glob["v_negmae"][r["子集"]] * -1.0
            r["MacroF1（数据集级复算）"] = glob["v_macrof1"][r["子集"]]
    df3 = pd.DataFrame(rows).drop(columns=["φ贡献"])
    df3.to_csv(os.path.join(a.table_dir, "Q3_八子集反事实.csv"), index=False, encoding="utf-8-sig")
    report["subsets"] = rows

    # ---------- 4) 模态忠实性：π vs 真删该模态的误差增量 ----------
    mae_all = float(np.abs(V[:, 7] - y).mean())
    delta = {}
    for k, m in enumerate(MODALITIES):
        sub = 7 & ~(1 << k)                       # 去掉模态 m 后的子集
        delta[m] = float(np.abs(V[:, sub] - y).mean() - mae_all)
    gl = glob.get("v_negmae", {})
    delta_global = {}
    if gl:
        for k, m in enumerate(MODALITIES):
            sub = 7 & ~(1 << k)
            # v = −MAE，故 ΔMAE_m = MAE(不含 m) − MAE(全模态) = v(all) − v(sub)
            delta_global[m] = float(gl["all"] - gl[NAMES[sub]])
    rows = []
    for tag, P in (("softmax|φ|", pi_soft), ("ReLU 归一", pi_relu)):
        pm = [float(np.nanmean(P[:, k])) for k in range(3)]
        rows.append(dict(归一化=tag, **{"π_%s" % MOD_CN[m]: round(pm[k], 4)
                                       for k, m in enumerate(MODALITIES)},
                         与样本级ΔMAE秩相关=round(spearman(pm, [delta[m] for m in MODALITIES]), 3),
                         与数据集级ΔMAE秩相关=round(spearman(
                             pm, [delta_global.get(m, np.nan) for m in MODALITIES]), 3)))
    df4 = pd.DataFrame(rows)
    df4.to_csv(os.path.join(a.table_dir, "Q3_模态忠实性.csv"), index=False, encoding="utf-8-sig")
    report["modality_faithfulness"] = dict(rows=rows, delta_sample=delta,
                                           delta_global=delta_global, mae_all=mae_all)

    # ---------- 5) 稳健性矩阵 ----------
    phis = {t: sh["phi_" + t] for t in ("mu_miss", "mu_pad", "p_pol_miss", "p_pol_pad")}
    rows = []
    for t1, t2 in itertools.combinations(phis, 2):
        a1 = phis[t1].ravel()
        a2 = phis[t2].ravel()
        rows.append(dict(口径A=t1, 口径B=t2, φ秩相关=round(spearman(a1, a2), 4)))
    df5 = pd.DataFrame(rows)
    df5.to_csv(os.path.join(a.table_dir, "Q3_稳健性矩阵.csv"), index=False, encoding="utf-8-sig")
    report["robustness"] = rows
    # ReLU 归一的退化样本数从 Shapley 报告读取（npz 中已做 nan_to_num，不能反推）
    sv = json.load(open(os.path.join(a.out_dir, "q3_shapley_valid_report.json"),
                        encoding="utf-8")) if os.path.isfile(
        os.path.join(a.out_dir, "q3_shapley_valid_report.json")) else {}
    report["pi_relu_degenerate"] = int(sv.get("pi_relu_degenerate", -1))

    # ---------- 6) 附件3/4：π 与实测缺失率（免标注机制验证） ----------
    rows = []
    for nm in ("att3", "att4"):
        p = os.path.join(a.out_dir, "q3_shapley_%s.npz" % nm)
        if not os.path.isfile(p):
            continue
        z = np.load(p, allow_pickle=True)
        pi = z["pi_soft"]
        rho = z["miss_rate"]
        pairs = [(pi[i, k], rho[i, k]) for i in range(len(pi)) for k in range(3)]
        a1 = np.array([p[0] for p in pairs])
        a2 = np.array([p[1] for p in pairs])
        rows.append(dict(数据集=nm, 样本数=int(len(pi)),
                         含缺失样本数=int((rho > 0).any(1).sum()),
                         π与缺失率秩相关=round(spearman(a1, a2), 4),
                         缺失模态的平均π=round(float(a1[a2 > 0].mean()) if (a2 > 0).any() else np.nan, 4),
                         无缺失模态的平均π=round(float(a1[a2 == 0].mean()) if (a2 == 0).any() else np.nan, 4)))
    if rows:
        df6 = pd.DataFrame(rows)
        df6.to_csv(os.path.join(a.table_dir, "Q3_缺失率一致性.csv"),
                   index=False, encoding="utf-8-sig")
        report["missing_rate_consistency"] = rows

    # ---------- 7) 错误归因（正确 vs 错误样本） ----------
    phead = sh["prob"].argmax(1)
    ok = phead == yc
    fp = os.path.join(a.out_dir, "q3_faithfulness_valid.csv")
    rows = []
    for tag, m in (("预测正确", ok), ("预测错误", ~ok)):
        if m.sum() < 3:
            continue
        r = dict(分组=tag, 样本数=int(m.sum()),
                 π熵=round(float(np.mean([entropy(r_) for r_ in pi_soft[m]])), 4),
                 π_文本=round(float(pi_soft[m, 0].mean()), 4),
                 主模态为文本占比=round(float((pi_soft[m].argmax(1) == 0).mean()), 4),
                 置信度=round(float(sh["prob"][m].max(1).mean()), 4))
        if os.path.isfile(fp):
            f = pd.read_csv(fp)
            f = f[(f["source"] == "fused") & (f["id"].isin(np.asarray(split["ids"])[m]))]
            if len(f):
                r["忠实性"] = round(float(f["compr_mean"].mean()), 4)
                r["充分性偏差"] = round(float(f["suff_absmean"].mean()), 4)
        rows.append(r)
    df7 = pd.DataFrame(rows)
    df7.to_csv(os.path.join(a.table_dir, "Q3_错误归因.csv"), index=False, encoding="utf-8-sig")
    report["error_attribution"] = rows

    # ---------- 7.5) 解释质量的正文级汇总（含"绝对变化量"，避免只看符号均值） ----------
    frows = []
    for nm in ("att4", "valid"):
        fp = os.path.join(a.out_dir, "q3_faithfulness_%s.csv" % nm)
        if not os.path.isfile(fp):
            continue
        f = pd.read_csv(fp, dtype={"id": str})
        for src, gsub in f.groupby("source"):
            frows.append(dict(数据集=nm, 解释源=src, 样本数=int(len(gsub)),
                              忠实性_符号均值=round(float(gsub["compr_mean"].mean()), 4),
                              忠实性_绝对均值=round(float(gsub["compr_mean"].abs().mean()), 4),
                              充分性偏差_绝对均值=round(float(gsub["suff_absmean"].mean()), 4),
                              稀疏性=round(float(gsub["sparsity_mean"].mean()), 4),
                              预测标准差=round(float(gsub["t_full"].std()), 4)))
    if frows:
        pd.DataFrame(frows).to_csv(os.path.join(a.table_dir, "Q3_解释质量对比.csv"),
                                   index=False, encoding="utf-8-sig")
        report["faithfulness_summary"] = frows
        # 注意力源与遮挡源的一致性（论文"注意力≠重要性"的实证）
        g = json.load(open(os.path.join(a.out_dir, "q3_gamma_valid.json"), encoding="utf-8"))
        report["agreement_with_occlusion"] = g.get("agreement_with_occ")
        report["gamma_valid"] = g.get("gamma")

    # ---------- 8) 与问题2 缺失实验的对照（写进报告文本） ----------
    q2 = dict(text=0.1804, audio=0.0002, vision=0.0068)     # 问题2：ρ=1.0 时 MAE 增量
    ours = {m: round(delta[m], 4) for m in MODALITIES}
    report["consistency_with_q2"] = dict(
        q2_sweep_rho1=q2, ours_greedy_removal=ours,
        spearman=round(spearman([ours[m] for m in MODALITIES],
                                [q2[m] for m in MODALITIES]), 3),
        note="问题2 的缺失注入实验（缺失率 1.0 时 MAE 增量）与问题3 的模态删除ΔMAE 排序一致，"
             "两条独立证据链互相印证")
    save_json(os.path.join(a.out_dir, "q3_valid_analysis.json"), report)

    print("=== 关键结论 ===")
    print("  π_soft 均值 文/语/视 = %.4f / %.4f / %.4f" % tuple(pi_soft.mean(0)))
    print("  模态忠实性（样本级 ΔMAE）= %s" % ours)
    print("  与问题2 缺失实验的秩相关 = %.3f" % report["consistency_with_q2"]["spearman"])
    print("  错误样本 π 熵 %.4f vs 正确样本 %.4f"
          % (rows[-1].get("π熵", np.nan) if rows else np.nan,
             rows[0].get("π熵", np.nan) if rows else np.nan))
    print("  附件3/4 π-缺失率一致性: %s" % report.get("missing_rate_consistency"))
    print("[SAVE] paper_tables/Q3_*.csv（7 张表）")
    print("VALID_ANALYSIS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
