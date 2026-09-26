# -*- coding: utf-8 -*-
r"""L1 探针：有序回归（CORAL 头）的期望能否替代点回归 μ？

背景
----
`MRFNet` 已带 CORAL 有序头（`mag_w` + 可学习累计阈值 `mag_b`），
按 |y|×3 离散为 K=10 档（步长 1/3），已用 `coral_loss` 训练（`lam_mag=0.5`）。
本探针**不重训**，只在推理期把累计 logit 转成期望幅值：

    p_j = sigmoid(logits_mag_j) = P(k > j),  k = round(3|y|)
    E[k]   = Σ_j p_j
    E|y|   = E[k] / 3

再与若干**符号**来源组合成新的点估计，与现有 μ 对比，从而判断
"把有序分布提升为主参数化"（6.1 的 L2）是否有依据。

预注册判据（避免事后挑指标）
----------------------------
  1) 主排序 = **valid MAE**；
  2) **test MAE 必须同向改善**；
  3) 极性 ACC(分类头) 不受影响（本次不改分类头）；θ 口径重新在 valid 选 θ 后报告；
  4) 若没有任何"合法符号来源"的变体在 valid MAE 上优于 μ → 放弃 L2/L3。

用法
----
  python ordinal_probe.py --ckpt_dir ..\runs\ens_top2 --out_dir ..\runs\ens_top2 --tag ens_top2
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config                                  # noqa: E402
from data_utils import load_all                            # noqa: E402
import runtime                                             # noqa: E402
from losses import polar_from_score                        # noqa: E402


def _mae(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).mean())


def _pearson(a, b):
    from scipy.stats import pearsonr
    try:
        return float(pearsonr(a, b)[0])
    except Exception:
        return float("nan")


def _ccc(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    mt, mp, vt, vp = yt.mean(), yp.mean(), yt.var(), yp.var()
    cov = ((yt - mt) * (yp - mp)).mean()
    return float(2 * cov / (vt + vp + (mt - mp) ** 2 + 1e-12))


def _acc(y_cls, pred):
    from sklearn.metrics import accuracy_score, f1_score
    return (float(accuracy_score(y_cls, pred)),
            float(f1_score(y_cls, pred, average="macro", zero_division=0)))


def sweep_theta(score, y_cls, y_reg, grid=np.round(np.arange(0.0, 1.51, 0.02), 3)):
    """在 valid 上按 Macro-F1 选 θ（与 report_p0 同口径）。"""
    best = (None, -1.0, -1.0)
    for th in grid:
        p = polar_from_score(score, th)
        a, f = _acc(y_cls, p)
        if f > best[1]:
            best = (float(th), f, a)
    return best


def eval_variant(name, pred, y_reg, y_cls, theta=None):
    absy = np.abs(y_reg)
    yhat_cls = polar_from_score(pred, theta if theta is not None else 0.0)
    a, f = _acc(y_cls, yhat_cls)
    return dict(variant=name, mae=_mae(y_reg, pred), pearson=_pearson(y_reg, pred),
                ccc=_ccc(y_reg, pred), bias=float((pred - y_reg).mean()),
                mae_strong=_mae(y_reg[absy > 1.5], pred[absy > 1.5]),
                mae_weak=_mae(y_reg[absy <= 0.5], pred[absy <= 0.5]),
                bias_neg=float((pred[y_reg < 0] - y_reg[y_reg < 0]).mean()),
                bias_pos=float((pred[y_reg > 0] - y_reg[y_reg > 0]).mean()),
                acc_theta=a, f1_theta=f, theta=float(theta if theta is not None else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--version", default="aligned")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--split", default="both", choices=["valid", "test", "both"])
    a = ap.parse_args()
    out_dir = a.out_dir or a.ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = a.tag or os.path.basename(os.path.normpath(a.ckpt_dir))

    cfg = Config(version=a.version, out_dir=out_dir)
    if a.data_dir:
        cfg.data_dir = a.data_dir
        cfg.__post_init__()
    _tr, va, te = load_all(cfg)
    models = runtime.load_ensemble(runtime.find_ckpts(a.ckpt_dir), a.device)
    n_lvl = int(models[0][1].get("mag_levels", 0) or 0)
    print("[INFO] 集成 %d 个 ckpt，mag_levels=%d" % (len(models), n_lvl))
    if n_lvl <= 1:
        raise SystemExit("该 ckpt 没有 CORAL 有序头（mag_levels=%d），L1 探针不适用。" % n_lvl)
    step = 1.0 / 3.0

    splits = [("valid", va)] if a.split == "valid" else \
             ([("test", te)] if a.split == "test" else [("valid", va), ("test", te)])

    rep = dict(tag=tag, ckpt_dir=os.path.abspath(a.ckpt_dir), n_ckpt=len(models),
               mag_levels=n_lvl, level_step=step, splits={})
    all_rows = []

    for name, sp in splits:
        pr = runtime.predict_split(models, sp, a.batch_size, a.device,
                                  aux_keys=("logits_mag",))
        if "logits_mag" not in pr:
            raise SystemExit("模型未返回 logits_mag —— 检查 mag_levels 是否被正确加载。")
        y_reg, y_cls = pr["y_reg"], pr["y_cls"]
        mu = pr["score"]
        prob = pr["prob"]
        pred_head = prob.argmax(-1)
        lm = pr["logits_mag"]
        pp = 1.0 / (1.0 + np.exp(-lm))                 # p_j = P(k > j)
        e_k = pp.sum(axis=1)                           # E[k]
        e_abs = np.clip(e_k * step, 0.0, 3.0)          # E|y|（幅值期望）
        absy = np.abs(y_reg)

        # 幅值建模质量的直接检验（与符号无关）
        mag_diag = dict(
            spearman_Eabs_vs_absy=float(pd.Series(e_abs).corr(pd.Series(absy), method="spearman")),
            mae_Eabs_vs_absy=_mae(absy, e_abs),
            mae_absmu_vs_absy=_mae(absy, np.abs(mu)),
            mean_Eabs=float(e_abs.mean()), mean_absy=float(absy.mean()),
            mean_absmu=float(np.abs(mu).mean()))
        print("\n===== [%s] n=%d =====" % (name, len(y_reg)))
        print("  幅值建模：MAE(E|y| vs |y|)=%.4f   MAE(|μ| vs |y|)=%.4f   "
              "Spearman(E|y|,|y|)=%.4f"
              % (mag_diag["mae_Eabs_vs_absy"], mag_diag["mae_absmu_vs_absy"],
                 mag_diag["spearman_Eabs_vs_absy"]))
        print("  幅值均值：E|y|=%.4f  |μ|=%.4f  真值|y|=%.4f"
              % (mag_diag["mean_Eabs"], mag_diag["mean_absmu"], mag_diag["mean_absy"]))

        # ---- 符号来源 ----
        s_mu = np.sign(mu)
        s_vote = np.sign(pr["member_mu"].sum(axis=1))          # 集成成员符号投票
        s_cls_hard = np.where(pred_head == 2, 1.0, np.where(pred_head == 0, -1.0, 0.0))
        s_cls_soft = prob[:, 2] - prob[:, 0]
        s_true = np.sign(y_reg)                                # oracle（仅诊断用）

        variants = {
            "V0 μ（现状基线）": mu,
            "V1 E|y|×sgn(μ)": s_mu * e_abs,
            "V2 E|y|×sgn(集成投票)": s_vote * e_abs,
            "V3 E|y|×硬符号(分类头)": s_cls_hard * e_abs,
            "V4 E|y|×软符号(p₊−p₋)": s_cls_soft * e_abs,
            "V5 E|y|×oracle符号（诊断上界）": s_true * e_abs,
            "V6 0.5·(μ + V1)": 0.5 * (mu + s_mu * e_abs),
        }

        # θ 在 valid 上选（test 沿用 valid 的 θ，避免用测试集调参）
        theta_of = {}
        if name == "valid":
            for k, v in variants.items():
                theta_of[k] = sweep_theta(v, y_cls, y_reg)[0]
            rep["theta_selected"] = theta_of
        else:
            theta_of = {k: float(v) for k, v in rep.get("theta_selected", {}).items()}

        rows = []
        for k, v in variants.items():
            r = eval_variant(k, v, y_reg, y_cls, theta=theta_of.get(k))
            rows.append(r)
        tab = pd.DataFrame(rows)
        tab.insert(0, "split", name)
        all_rows.append(tab)
        print(tab[["variant", "mae", "pearson", "ccc", "bias", "mae_strong",
                   "mae_weak", "bias_neg", "bias_pos"]]
              .to_string(index=False, float_format=lambda x: "%.4f" % x))
        print(tab[["variant", "theta", "acc_theta", "f1_theta"]]
              .to_string(index=False, float_format=lambda x: "%.4f" % x))
        rep["splits"][name] = dict(n=len(y_reg), mag_diag=mag_diag,
                                   theta={k: float(v) for k, v in theta_of.items()},
                                   variants=tab.to_dict("records"))

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(os.path.join(out_dir, "ordinal_probe.csv"),
                index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "ordinal_probe.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    # ---- 预注册判据裁决 ----
    print("\n" + "=" * 78)
    print("预注册判据裁决")
    print("=" * 78)
    v0 = full[full["variant"] == "V0 μ（现状基线）"]
    verdict = []
    for k in full["variant"].unique():
        if k == "V0 μ（现状基线）":
            continue
        mv = full[(full["variant"] == k) & (full["split"] == "valid")]["mae"].iloc[0]
        mt = full[(full["variant"] == k) & (full["split"] == "test")]["mae"].iloc[0]
        bv = v0[v0["split"] == "valid"]["mae"].iloc[0]
        bt = v0[v0["split"] == "test"]["mae"].iloc[0]
        d_v, d_t = mv - bv, mt - bt
        ok = (d_v < 0) and (d_t < 0)
        oracle = k.startswith("V5")
        verdict.append((k, d_v, d_t, ok, oracle))
        print("  %-34s valid ΔMAE=%+.4f  test ΔMAE=%+.4f  %s"
              % (k, d_v, d_t,
                 ("通过" if ok else "不通过") + ("（oracle 符号，仅诊断）" if oracle else "")))
    legal = [v for v in verdict if not v[4] and v[3]]
    print("\n  合法符号方案中通过两级判据的数量：%d" % len(legal))
    if legal:
        best = min(legal, key=lambda x: x[1])
        print("  → 最佳合法方案：%s（valid ΔMAE=%+.4f, test ΔMAE=%+.4f）**建议进入 L2**"
              % (best[0], best[1], best[2]))
    else:
        print("  → 没有合法符号方案同时通过 valid 与 test 判据 → **建议放弃 L2/L3**")
    print("\n[SAVE] %s" % os.path.join(out_dir, "ordinal_probe.csv"))
    print("[SAVE] %s" % os.path.join(out_dir, "ordinal_probe.json"))


if __name__ == "__main__":
    main()
