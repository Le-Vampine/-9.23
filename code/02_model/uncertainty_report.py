# -*- coding: utf-8 -*-
r"""不确定性与拒识分析（问题2 附加亮点）。

模型自带 `head_lv`（logvar 头，NLL 训练）→ 逐样本不确定性是"现成的"，无需额外训练。

三个不确定性来源
----------------
  sigma_model = exp(logvar/2)          logvar 头学到的**偶然不确定度**（aleatoric）
  sigma_ens   = std_k(mu_k)            集成成员间分歧（**认知不确定度**，epistemic）
  sigma_total = sqrt(sigma_model^2 + sigma_ens^2)  总预测不确定度
  entropy     = 分类头概率熵            第四个代理（判别不确定性）

分析内容
--------
  1) 不确定性 vs 绝对误差 的 Pearson / Spearman 相关
  2) 按 sigma_total 五等分分桶 → 各桶 MAE / ACC(head) / F1(head)
  3) 风险-覆盖率曲线（risk-coverage）+ AURC，与随机排序 / oracle 排序对比
  4) 拒识效果表（覆盖 100/90/80/70/50% 时的 MAE 与 ACC）
  5) 与错误归因呼应：弱强度(|y|<=0.5)、含缺失、判错样本的不确定性是否更高

用法
----
  python uncertainty_report.py --ckpt_dir ..\runs\ens_top2 --out_dir ..\runs\ens_top2 --tag ens_top2
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

COVERAGES = np.round(np.arange(0.05, 1.0001, 0.05), 4)
REJECT_AT = [1.00, 0.90, 0.80, 0.70, 0.50]

# numpy 2.x 移除了 np.trapz（改名 trapezoid）
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def _mae(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).mean())


def _acc(pred, true):
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(true, pred))


def _f1(pred, true):
    from sklearn.metrics import f1_score
    return float(f1_score(true, pred, average="macro", zero_division=0))


def curve(order, y_reg, score, y_cls, pred_head, coverages=COVERAGES):
    """给定样本排序（越可信越靠前），返回各 coverage 下的 MAE/ACC/MacroF1。

    oracle 排序（按真实误差 / 按真实对错）不可实现，仅作性能上界参照。
    """
    rows, n = [], len(order)
    for q in coverages:
        k = max(int(round(q * n)), 1)
        i = order[:k]
        rows.append(dict(coverage=float(q), n=k,
                         mae=_mae(y_reg[i], score[i]),
                         acc=_acc(pred_head[i], y_cls[i]),
                         f1=_f1(pred_head[i], y_cls[i])))
    return pd.DataFrame(rows)


def correlate(u, err):
    from scipy.stats import pearsonr, spearmanr
    out = {}
    for key, v in u.items():
        try:
            out[key + "_pearson"] = float(pearsonr(v, err)[0])
        except Exception:
            out[key + "_pearson"] = float("nan")
        try:
            out[key + "_spearman"] = float(spearmanr(v, err)[0])
        except Exception:
            out[key + "_spearman"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--version", default="aligned")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--figs_dir", default=None)
    ap.add_argument("--split", default="both", choices=["valid", "test", "both"])
    a = ap.parse_args()
    out_dir = a.out_dir or a.ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = a.tag or os.path.basename(os.path.normpath(a.ckpt_dir))

    cfg = Config(version=a.version, out_dir=out_dir)
    if a.data_dir:
        cfg.data_dir = a.data_dir
        cfg.__post_init__()
    tr, va, te = load_all(cfg)
    models = runtime.load_ensemble(runtime.find_ckpts(a.ckpt_dir), a.device)
    print("[INFO] 集成 %d 个 ckpt" % len(models))

    splits = [("valid", va)] if a.split == "valid" else \
             ([("test", te)] if a.split == "test" else [("valid", va), ("test", te)])

    rep = dict(tag=tag, ckpt_dir=os.path.abspath(a.ckpt_dir), n_ckpt=len(models),
               splits={})
    bins_all, rc_all, sub_all, rej_all = [], [], [], []

    for name, sp in splits:
        pr = runtime.predict_split(models, sp, a.batch_size, a.device)
        y_reg, y_cls, score = pr["y_reg"], pr["y_cls"], pr["score"]
        prob, pred_head = pr["prob"], pr["prob"].argmax(-1)
        err = np.abs(y_reg - score)
        correct = (pred_head == y_cls)          # 极性是否判对（供 oracle_acc 排序）
        sigma_model = np.exp(pr["logvar"] / 2.0)
        sigma_ens = pr["score_std"]
        sigma_total = np.sqrt(sigma_model ** 2 + sigma_ens ** 2)
        entropy = -(prob * np.log(prob + 1e-12)).sum(-1)
        U = dict(sigma_model=sigma_model, sigma_ens=sigma_ens,
                 sigma_total=sigma_total, entropy=entropy)

        base = dict(n=len(y_reg), mae=_mae(y_reg, score),
                    acc_head=_acc(pred_head, y_cls), f1_head=_f1(pred_head, y_cls),
                    mae_sigma_mean=float(sigma_model.mean()),
                    ens_sigma_mean=float(sigma_ens.mean()),
                    total_sigma_mean=float(sigma_total.mean()),
                    entropy_mean=float(entropy.mean()))
        rep["splits"][name] = dict(base=base, corr=correlate(U, err))

        print("\n===== [%s] n=%d  MAE=%.4f  ACC(head)=%.4f  MacroF1(head)=%.4f" %
              (name, base["n"], base["mae"], base["acc_head"], base["f1_head"]))
        print("  不确定性均值：σ_model=%.4f  σ_ens=%.4f  σ_total=%.4f  H=%.4f"
              % (base["mae_sigma_mean"], base["ens_sigma_mean"],
                 base["total_sigma_mean"], base["entropy_mean"]))
        print("  与 |误差| 的相关：")
        for k in ("sigma_model", "sigma_ens", "sigma_total", "entropy"):
            print("    %-12s Pearson=%+.4f  Spearman=%+.4f"
                  % (k, rep["splits"][name]["corr"][k + "_pearson"],
                     rep["splits"][name]["corr"][k + "_spearman"]))

        # ---- 2) 分位分桶 ----
        g = pd.qcut(pd.Series(sigma_total).rank(method="first"), 5,
                    labels=["Q1(最确定)", "Q2", "Q3", "Q4", "Q5(最不确定)"])
        rows = []
        for k, idx in pd.Series(range(len(g))).groupby(g.values, observed=True):
            i = np.asarray(idx)
            rows.append(dict(split=name, bin=str(k), n=len(i),
                             sigma_total=float(sigma_total[i].mean()),
                             entropy=float(entropy[i].mean()),
                             mae=_mae(y_reg[i], score[i]),
                             acc_head=_acc(pred_head[i], y_cls[i]),
                             f1_head=_f1(pred_head[i], y_cls[i]),
                             mean_abs_y=float(np.abs(y_reg[i]).mean()),
                             miss_rate=float((pr["rho"][i].sum(axis=1) > 1e-6).mean())))
        db = pd.DataFrame(rows)
        bins_all.append(db)
        print(db.to_string(index=False, float_format=lambda x: "%.4f" % x))

        # ---- 3) 风险-覆盖率：两种排序键 + oracle + 随机 ----
        orders = {
            "sigma_total(升序)": np.argsort(sigma_total),
            "entropy(升序)": np.argsort(entropy),
            "oracle_mae": np.argsort(err),
            "oracle_acc": np.argsort(~correct),
        }
        aurc = {}
        for key, od in orders.items():
            c = curve(od, y_reg, score, y_cls, pred_head)
            c.insert(0, "order_by", key)
            c.insert(0, "split", name)
            rc_all.append(c)
            def _at(c, q, col):
                return float(c.loc[np.isclose(c["coverage"], q), col].iloc[0])
            aurc[key] = dict(aurc_mae=float(_TRAPZ(c["mae"], c["coverage"])),
                             aurc_err=float(_TRAPZ(1.0 - c["acc"], c["coverage"])),
                             mae_at_90=_at(c, 0.90, "mae"), acc_at_90=_at(c, 0.90, "acc"),
                             mae_at_70=_at(c, 0.70, "mae"), acc_at_70=_at(c, 0.70, "acc"))
        aurc["random"] = dict(aurc_mae=base["mae"], aurc_err=1.0 - base["acc_head"],
                              mae_at_90=base["mae"], acc_at_90=base["acc_head"],
                              mae_at_70=base["mae"], acc_at_70=base["acc_head"])
        rep["splits"][name]["aurc"] = aurc
        print("\n  风险-覆盖率（AURC 越低越好；random = 全量水平）")
        print("    %-18s %10s %10s %10s %10s" % ("order_by", "AURC_MAE", "AURC_ERR",
                                                 "MAE@90%", "ACC@90%"))
        for key in list(orders.keys()) + ["random"]:
            d = aurc[key]
            print("    %-18s %10.4f %10.4f %10.4f %10.4f"
                  % (key, d["aurc_mae"], d["aurc_err"], d["mae_at_90"], d["acc_at_90"]))

        # ---- 4) 拒识效果表（两种排序键）----
        print("\n  拒识效果（剔除最不可信的部分）：")
        for key in ("sigma_total(升序)", "entropy(升序)"):
            od = orders[key]
            for q in REJECT_AT:
                k = max(int(round(q * len(od))), 1)
                i = od[:k]
                rej_all.append(dict(split=name, order_by=key, coverage=q,
                                    keep_n=k, rejected=int(len(od) - k),
                                    mae=_mae(y_reg[i], score[i]),
                                    acc_head=_acc(pred_head[i], y_cls[i]),
                                    f1_head=_f1(pred_head[i], y_cls[i]),
                                    d_mae=_mae(y_reg[i], score[i]) - base["mae"],
                                    d_acc=_acc(pred_head[i], y_cls[i]) - base["acc_head"]))
        dr = pd.DataFrame([r for r in rej_all if r["split"] == name])
        print(dr[["order_by", "coverage", "mae", "d_mae", "acc_head", "d_acc"]]
              .to_string(index=False, float_format=lambda x: "%.4f" % x))

        # ---- 5) 子群对照 ----
        absy = np.abs(y_reg)
        groups = {
            "强度 |y|<=0.5": absy <= 0.5,
            "强度 0.5<|y|<=1.5": (absy > 0.5) & (absy <= 1.5),
            "强度 |y|>1.5": absy > 1.5,
            "含缺失": pr["rho"].sum(axis=1) > 1e-6,
            "无缺失": pr["rho"].sum(axis=1) <= 1e-6,
            "极性判对": pred_head == y_cls,
            "极性判错": pred_head != y_cls,
        }
        for gname, m in groups.items():
            if m.sum() < 5:
                continue
            sub_all.append(dict(split=name, group=gname, n=int(m.sum()),
                                sigma_model=float(sigma_model[m].mean()),
                                sigma_ens=float(sigma_ens[m].mean()),
                                sigma_total=float(sigma_total[m].mean()),
                                entropy=float(entropy[m].mean()),
                                mae=_mae(y_reg[m], score[m]),
                                acc_head=_acc(pred_head[m], y_cls[m])))
        ds_ = pd.DataFrame(sub_all)
        print("\n  子群不确定性对照：")
        print(ds_[ds_["split"] == name].to_string(index=False,
                                                  float_format=lambda x: "%.4f" % x))

        # ---- 逐样本明细（供复核）----
        pd.DataFrame({
            "id": pr["ids"], "y_reg": y_reg, "score": score, "err": err,
            "y_cls": y_cls, "pred_head": pred_head, "correct": correct,
            "sigma_model": sigma_model, "sigma_ens": sigma_ens,
            "sigma_total": sigma_total, "entropy": entropy,
            "rho_text": pr["rho"][:, 0], "rho_audio": pr["rho"][:, 1],
            "rho_vision": pr["rho"][:, 2],
        }).to_csv(os.path.join(out_dir, "uncertainty_per_sample_%s.csv" % name),
                  index=False, encoding="utf-8-sig")

    # ---- 落盘 ----
    pd.concat(bins_all, ignore_index=True).to_csv(
        os.path.join(out_dir, "uncertainty_bins.csv"), index=False, encoding="utf-8-sig")
    rc = pd.concat(rc_all, ignore_index=True)
    rc.to_csv(os.path.join(out_dir, "uncertainty_risk_coverage.csv"),
              index=False, encoding="utf-8-sig")
    pd.DataFrame(sub_all).to_csv(os.path.join(out_dir, "uncertainty_subgroups.csv"),
                                 index=False, encoding="utf-8-sig")
    pd.DataFrame(rej_all).to_csv(os.path.join(out_dir, "uncertainty_rejection.csv"),
                                 index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "uncertainty_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        figs = a.figs_dir or os.path.join(os.path.dirname(out_dir), "..", "figs")
        os.makedirs(figs, exist_ok=True)
        style = {"sigma_total(升序)": ("-o", "按 σ_total 剔除"),
                 "entropy(升序)": ("-s", "按 熵 剔除"),
                 "oracle_mae": ("--", "oracle(按真实误差)"),
                 "oracle_acc": ("--", "oracle(按真实对错)")}
        n = len(splits)
        for metric, fname, ylab, bkey in (
                ("mae", "risk_coverage_%s.png", "MAE（风险）", "mae"),
                ("acc", "reject_acc_%s.png", "ACC（分类头）", "acc_head")):
            fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4), squeeze=False)
            for j, (name, _sp) in enumerate(splits):
                ax = axes[0][j]
                for key, (ls, lab) in style.items():
                    d = rc[(rc["split"] == name) & (rc["order_by"] == key)]
                    if d.empty:
                        continue
                    ax.plot(d["coverage"], d[metric], ls, ms=3, lw=1.1, label=lab)
                ax.axhline(rep["splits"][name]["base"][bkey], color="gray",
                           ls=":", lw=1, label="随机（全量）")
                ax.set_xlabel("保留比例 coverage"); ax.set_ylabel(ylab)
                ax.set_title("拒识曲线（%s）" % name)
                ax.invert_xaxis(); ax.legend(fontsize=7)
            fig.tight_layout()
            p = os.path.join(figs, fname % tag)
            fig.savefig(p, dpi=200); plt.close(fig)
            print("[FIG] %s" % p)

        fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4), squeeze=False)
        for j, (name, _sp) in enumerate(splits):
            ax = axes[0][j]
            ps = pd.read_csv(os.path.join(out_dir, "uncertainty_per_sample_%s.csv" % name),
                             encoding="utf-8-sig")
            ax.scatter(ps["sigma_total"], ps["err"], s=4, alpha=0.35)
            ax.set_xlabel("σ_total"); ax.set_ylabel("|强度误差|")
            ax.set_title("σ vs 误差（%s，Spearman=%.3f）"
                         % (name, rep["splits"][name]["corr"]["sigma_total_spearman"]))
        fig.tight_layout()
        p3 = os.path.join(figs, "uncertainty_vs_error_%s.png" % tag)
        fig.savefig(p3, dpi=200); plt.close(fig)
        print("[FIG] %s" % p3)
    except Exception as e:
        print("[WARN] 绘图失败：%s" % e)

    print("[SAVE] %s" % os.path.join(out_dir, "uncertainty_report.json"))


if __name__ == "__main__":
    main()
