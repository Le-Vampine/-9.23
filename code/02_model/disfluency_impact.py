# -*- coding: utf-8 -*-
r"""6.2 前置探针：非流利标记与预测误差的关联性检验（零成本，不训练）。

背景
----
错误归因把 "F4 口语转写噪声" 列为 top-20 中的一类失败模式。
但在投入重训之前，需要先确认**标记与误差之间是否存在真实关联**，
并排除混淆因素（标记多出现在更长的口语化转写中，而长度本身与误差相关）。

本脚本用**已算好的逐样本预测**（`runs/<tag>/uncertainty_per_sample_*.csv`）直接做关联分析。

用法
----
  python disfluency_impact.py --ckpt_dir ..\runs\ens_top2 --version aligned
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config                                  # noqa: E402
from data_utils import load_all                            # noqa: E402

MARK_RE = re.compile(r"\((?:[^()]{1,20})\)|\{[^{}]{1,20}\}")


def marker_stats(texts):
    n_mark = np.array([len(MARK_RE.findall(str(t))) for t in texts])
    return n_mark


def group_metrics(df, col="marker_grp"):
    from sklearn.metrics import accuracy_score, f1_score
    rows = []
    for g, sub in df.groupby(col, observed=True):
        rows.append(dict(group=str(g), n=len(sub), mae=sub["err"].mean(),
                         acc_head=accuracy_score(sub["y_cls"], sub["pred_head"]),
                         macro_f1=f1_score(sub["y_cls"], sub["pred_head"],
                                           average="macro", zero_division=0),
                         mean_abs_y=float(np.abs(sub["y_reg"]).mean()),
                         mean_valid_len=float(sub["n_valid"].mean()),
                         mean_err_strong=float(sub.loc[np.abs(sub["y_reg"]) > 1.5,
                                                       "err"].mean()
                                               if (np.abs(sub["y_reg"]) > 1.5).any()
                                               else np.nan)))
    return pd.DataFrame(rows)


def enrich(df, ks=(20, 50, 100)):
    base = float((df["n_mark"] > 0).mean())
    out = []
    d = df.sort_values("err", ascending=False)
    for k in ks:
        top = d.head(k)
        r = float((top["n_mark"] > 0).mean())
        out.append(dict(topK=k, marker_rate=r, base_rate=base,
                        enrichment=(r / base if base > 0 else np.nan)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--version", default="aligned")
    ap.add_argument("--data_dir", default=None)
    a = ap.parse_args()

    cfg = Config(version=a.version, out_dir=os.path.dirname(os.path.abspath(__file__)))
    if a.data_dir:
        cfg.data_dir = a.data_dir
        cfg.__post_init__()
    tr, va, te = load_all(cfg)

    for name, sp in (("valid", va), ("test", te)):
        pf = os.path.join(a.ckpt_dir, "uncertainty_per_sample_%s.csv" % name)
        if not os.path.isfile(pf):
            print("[SKIP] 缺少 %s（先跑 uncertainty_report.py）" % pf)
            continue
        P = pd.read_csv(pf, encoding="utf-8-sig")
        P["id"] = P["id"].astype(str)
        n_valid = (~sp["pad"]["text"]).sum(axis=1)
        D = pd.DataFrame({"id": np.asarray(sp["ids"]).astype(str),
                          "raw_text": [str(x) for x in np.asarray(sp["raw_text"])],
                          "n_valid": n_valid,
                          "text_len": np.array([len(str(x).split())
                                                for x in np.asarray(sp["raw_text"])])})
        D["n_mark"] = marker_stats(D["raw_text"].values)
        D["mark_frac"] = D["n_mark"] / np.maximum(D["n_valid"], 1)
        m = D.merge(P, on="id", how="inner", suffixes=("", "_p"))
        print("\n" + "=" * 78)
        print("[%s] 合并 %d 条（预测文件 %d 条）" % (name, len(m), len(P)))

        m["marker_grp"] = np.where(m["n_mark"] == 0, "0 无标记",
                                   np.where(m["n_mark"] == 1, "1 一个标记", "2+ 多个标记"))
        tab = group_metrics(m)
        print("\n  按标记数量分组：")
        print(tab.to_string(index=False, float_format=lambda x: "%.4f" % x))

        # 文本长度三分位内比较（排除"长度"混淆）
        m["len_bin"] = pd.qcut(m["text_len"].rank(method="first"), 3,
                               labels=["短", "中", "长"])
        print("\n  文本长度分层内：有/无标记 的 MAE（排除长度混淆）")
        rows = []
        for lb, sub in m.groupby("len_bin", observed=True):
            a1 = sub[sub["n_mark"] == 0]["err"]
            a2 = sub[sub["n_mark"] > 0]["err"]
            if len(a1) < 10 or len(a2) < 10:
                continue
            from scipy.stats import mannwhitneyu
            p = float(mannwhitneyu(a2, a1, alternative="greater").pvalue)
            rows.append(dict(len_bin=str(lb), n_no_mark=len(a1), n_mark=len(a2),
                             mae_no_mark=a1.mean(), mae_mark=a2.mean(),
                             d_mae=a2.mean() - a1.mean(), p_greater=p))
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: "%.4f" % x))

        print("\n  标记在 top-K 高误差样本中的富集（base = 全体含标记比例）：")
        print(enrich(m).to_string(index=False, float_format=lambda x: "%.4f" % x))

        # ---------- 关键：控制"真值强度"与"文本长度"两个混淆 ----------
        m["absy_bin"] = pd.cut(np.abs(m["y_reg"]), [-0.001, 0.5, 1.5, 3.01],
                               labels=["|y|<=0.5", "0.5<|y|<=1.5", "|y|>1.5"])
        from scipy.stats import mannwhitneyu

        def strat(col):
            rows = []
            for b, sub in m.groupby(col, observed=True):
                a1 = sub[sub["n_mark"] == 0]["err"].values
                a2 = sub[sub["n_mark"] > 0]["err"].values
                if len(a1) < 10 or len(a2) < 10:
                    rows.append(dict(stratum=str(b), n_no=len(a1), n_mark=len(a2),
                                     mae_no=np.nan, mae_mark=np.nan,
                                     d_mae=np.nan, p=np.nan))
                    continue
                p = float(mannwhitneyu(a2, a1, alternative="greater").pvalue)
                rows.append(dict(stratum=str(b), n_no=len(a1), n_mark=len(a2),
                                 mae_no=a1.mean(), mae_mark=a2.mean(),
                                 d_mae=a2.mean() - a1.mean(), p=p))
            return pd.DataFrame(rows)

        print("\n  ★ 按真值强度分层（含标记样本本身情感更强，必须控制）：")
        print(strat("absy_bin").to_string(index=False, float_format=lambda x: "%.4f" % x))

        m["cell"] = m["absy_bin"].astype(str) + " × " + m["len_bin"].astype(str)
        print("\n  ★ 强度 × 长度 双向分层：")
        print(strat("cell").to_string(index=False, float_format=lambda x: "%.4f" % x))

        # 多元回归：err ~ |y| + len + n_mark（看 n_mark 的偏效应）
        try:
            import statsmodels.formula.api as smf
            d = m.copy()
            d["absy"] = np.abs(d["y_reg"])
            mod = smf.ols("err ~ absy + text_len + n_mark", data=d).fit()
            print("\n  ★ 回归 err ~ |y| + text_len + n_mark（含标记的偏效应）：")
            print("    n_mark 系数 = %+.5f   t = %.2f   p = %.4f"
                  % (mod.params["n_mark"], mod.tvalues["n_mark"],
                     mod.pvalues["n_mark"]))
            print("    |y|    系数 = %+.5f   t = %.2f   p = %.4f"
                  % (mod.params["absy"], mod.tvalues["absy"], mod.pvalues["absy"]))
            print("    R² = %.4f" % mod.rsquared)
        except Exception as e:
            print("  [WARN] 回归失败：%s" % e)

        # 富集是否也被强度混淆？在强情感子集内重算
        strong = m[np.abs(m["y_reg"]) > 1.5]
        if len(strong) > 40:
            bs = float((strong["n_mark"] > 0).mean())
            top = strong.sort_values("err", ascending=False).head(20)
            rs = float((top["n_mark"] > 0).mean())
            print("\n  ★ 仅在 |y|>1.5 子集内（n=%d）的富集：top-20 含标记 %.3f "
                  "vs 子集基线 %.3f → 富集 %.2fx"
                  % (len(strong), rs, bs, (rs / bs if bs > 0 else np.nan)))

        from scipy.stats import spearmanr
        rho, pv = spearmanr(m["mark_frac"], m["err"])
        print("\n  Spearman(标记帧占比, |误差|) = %+.4f (p=%.3g)" % (rho, pv))
        rho2, pv2 = spearmanr(m["n_mark"], m["err"])
        print("  Spearman(标记个数,   |误差|) = %+.4f (p=%.3g)" % (rho2, pv2))

        # 结论提示
        print("\n  [判据] 若『长度分层内的 ΔMAE 不显著为正』或『top-K 富集 < 1.5x』，"
              "则不值得为该假设重训。")
        m.to_csv(os.path.join(a.ckpt_dir, "disfluency_%s.csv" % name),
                 index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
