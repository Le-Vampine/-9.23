# -*- coding: utf-8 -*-
r"""6.7 L1 探针：中性边界重判定（推理期规则，零重训）。

动机（来自本会话的三条证据）
---------------------------
1. 错误归因：|y|<=0.5 占 valid 的 45.5%（331/728），其 ACC(分类头) 仅 **0.5045**，
   接近三分类随机水平 —— 这是极性指标的最大失分源；
2. 不确定性分析：熵 H 是"中性失效"的可量化信号（极性判错样本 H=0.901 ≫ 判对 0.743），
   且 H 对回归误差无信息（AURC_MAE ≈ 随机）；
3. 有序回归 L1 探针：V4（软符号×有序期望）在弱强度样本上 mae_weak 0.402→**0.272**，
   但强样本恶化 → 有序分布**携带中性/弱样本的额外信息**，适合作辅助信号。

做法
----
在**已有的分类头 argmax 输出**之上，叠加一条"中性重判定"规则：
条件成立则改判为中性，否则保留分类头结果。所有阈值**只在 valid 上选**（按 Macro-F1），
然后在 test 上核验（不用 test 调参）。

⚠️ 合规提示
----------
题目规定"0（且仅 0）属于中性"，本探针的规则会**移动中性边界**。
按文档 07 §5 的既定边界，这类改动**不得作为提交 CSV 的主口径**，
仅可作"探索性改进"在消融/讨论小节报告，并须同时给出与题目定义一致的对照结果。
本脚本正是为此提供**可量化的收益/代价**证据。

用法
----
  python neutral_probe.py --ckpt_dir ..\runs\ens_top2 --out_dir ..\runs\ens_top2 --tag ens_top2
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


def _acc(y, p):
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(y, p))


def _mf1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro", zero_division=0))


def _perclass(y, p):
    from sklearn.metrics import f1_score
    return [float(x) for x in f1_score(y, p, average=None, zero_division=0,
                                       labels=[0, 1, 2])]


def mcnemar(y, p_new, p_old):
    """精确二项检验：p_new 与 p_old 是否显著不同。"""
    from scipy.stats import binomtest
    c_new = (np.asarray(p_new) == np.asarray(y))
    c_old = (np.asarray(p_old) == np.asarray(y))
    n01 = int(np.sum(~c_old & c_new))     # 旧错新对
    n10 = int(np.sum(c_old & ~c_new))     # 旧对新错
    n = n01 + n10
    p = float(binomtest(n01, n, 0.5).pvalue) if n > 0 else 1.0
    return dict(n01=n01, n10=n10, p=p)


def build_signals(models, sp, bs, dev):
    """一次推理取出全部可用信号。"""
    pr = runtime.predict_split(models, sp, bs, dev, aux_keys=("logits_mag",))
    prob = pr["prob"]
    mu = pr["score"]
    base_pred = prob.argmax(-1)
    S = dict(mu=mu, prob=prob, base_pred=base_pred,
             p_neu=prob[:, 1], p_neg=prob[:, 0], p_pos=prob[:, 2],
             H_head=-(prob * np.log(prob + 1e-12)).sum(-1),
             sigma=np.sqrt(np.exp(pr["logvar"]) + pr["score_std"] ** 2))
    if "logits_mag" in pr:
        pp = 1.0 / (1.0 + np.exp(-pr["logits_mag"]))       # p_j = P(k > j)
        S["E_abs"] = np.clip(pp.sum(axis=1) / 3.0, 0.0, 3.0)
        S["p0_ord"] = 1.0 - pp[:, 0]                       # P(k = 0) = P(|y| 极小)
        # 有序分布熵：P(k=j) = p_{j-1} - p_j
        K = pp.shape[1] + 1
        Pk = np.zeros((len(pp), K))
        prev = np.ones(len(pp))
        for j in range(K - 1):
            Pk[:, j] = prev - pp[:, j]
            prev = pp[:, j]
        Pk[:, K - 1] = prev
        Pk = np.clip(Pk, 1e-12, 1.0)
        S["H_ord"] = -(Pk * np.log(Pk)).sum(axis=1)
    return S, pr["y_cls"], pr["y_reg"]


def apply_rule(S, kind, params):
    """返回新的预测标签；规则不触发时保留分类头结果。"""
    pred = S["base_pred"].copy()
    if kind == "R1 p_neu>=t":
        m = S["p_neu"] >= params[0]
    elif kind == "R2 |mu|<=t":
        m = np.abs(S["mu"]) <= params[0]
    elif kind == "R3 E|y|<=t":
        m = S["E_abs"] <= params[0]
    elif kind == "R4 H_head>=t":
        m = S["H_head"] >= params[0]
    elif kind == "R5 p0_ord>=t":
        m = S["p0_ord"] >= params[0]
    elif kind == "R6 p_neu>=t1 且 H_head>=t2":
        m = (S["p_neu"] >= params[0]) & (S["H_head"] >= params[1])
    elif kind == "R7 |mu|<=t1 且 H_head>=t2":
        m = (np.abs(S["mu"]) <= params[0]) & (S["H_head"] >= params[1])
    elif kind == "R8 类别偏置 γ（中性×γ）":
        z = np.stack([S["p_neg"], S["p_neu"] * params[0], S["p_pos"]], axis=1)
        return z.argmax(-1)
    elif kind == "R9 强制离开中性 |mu|>=t":
        # 反方向：若幅值够大但仍被判为中性 → 按 sgn(μ) 改判极性
        m = (np.abs(S["mu"]) >= params[0]) & (S["base_pred"] == 1)
        pred[m] = np.where(S["mu"][m] > 0, 2, 0)
        return pred
    else:
        raise ValueError(kind)
    pred[m] = 1
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--version", default="aligned")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
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

    Sv, yv_cls, yv_reg = build_signals(models, va, a.batch_size, a.device)
    St, yt_cls, yt_reg = build_signals(models, te, a.batch_size, a.device)

    print("\n===== 基线（分类头 argmax，题目主口径）=====")
    for nm, S, y in (("valid", Sv, yv_cls), ("test", St, yt_cls)):
        pc = _perclass(y, S["base_pred"])
        print("  [%s] n=%d  ACC=%.4f  MacroF1=%.4f  F1 neg/neu/pos=%.4f/%.4f/%.4f"
              % (nm, len(y), _acc(y, S["base_pred"]), _mf1(y, S["base_pred"]),
                 pc[0], pc[1], pc[2]))
    base = dict(valid=dict(acc=_acc(yv_cls, Sv["base_pred"]),
                           mf1=_mf1(yv_cls, Sv["base_pred"]),
                           pc=_perclass(yv_cls, Sv["base_pred"])),
                test=dict(acc=_acc(yt_cls, St["base_pred"]),
                          mf1=_mf1(yt_cls, St["base_pred"]),
                          pc=_perclass(yt_cls, St["base_pred"])))

    # 中性类混淆方向诊断：是"过度判中性"还是"漏判中性"？
    print("\n  中性类混淆统计（基线 = 分类头 argmax）：")
    for nm, S, y in (("valid", Sv, yv_cls), ("test", St, yt_cls)):
        p = S["base_pred"]
        tp = int(np.sum((y == 1) & (p == 1)))
        fn = int(np.sum((y == 1) & (p != 1)))
        fp = int(np.sum((y != 1) & (p == 1)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        print("    [%s] 真中性 n=%d  预测中性 n=%d  TP=%d FP=%d FN=%d  精确率=%.4f  召回率=%.4f"
              % (nm, int((y == 1).sum()), int((p == 1).sum()), tp, fp, fn, prec, rec))

    # ---- 阈值搜索空间（全部由 valid 的分位数决定，不看 test）----
    def qgrid(x, n=31, lo=0.02, hi=0.98):
        return np.unique(np.round(np.quantile(x, np.linspace(lo, hi, n)), 4))

    one_dim = {
        "R1 p_neu>=t": qgrid(Sv["p_neu"]),
        "R2 |mu|<=t": qgrid(np.abs(Sv["mu"])),
        "R3 E|y|<=t": qgrid(Sv["E_abs"]) if "E_abs" in Sv else None,
        "R4 H_head>=t": qgrid(Sv["H_head"]),
        "R5 p0_ord>=t": qgrid(Sv["p0_ord"]) if "p0_ord" in Sv else None,
        "R9 强制离开中性 |mu|>=t": qgrid(np.abs(Sv["mu"])),
        "R8 类别偏置 γ（中性×γ）": np.round(np.linspace(0.8, 2.5, 35), 3),
    }
    two_dim = {
        "R6 p_neu>=t1 且 H_head>=t2": (qgrid(Sv["p_neu"], 9), qgrid(Sv["H_head"], 9)),
        "R7 |mu|<=t1 且 H_head>=t2": (qgrid(np.abs(Sv["mu"]), 9), qgrid(Sv["H_head"], 9)),
    }

    results = []

    def record(kind, params, pv, pt):
        results.append(dict(
            variant=kind, params=str([round(float(p), 4) for p in params]),
            valid_acc=_acc(yv_cls, pv), valid_mf1=_mf1(yv_cls, pv),
            valid_f1neg=_perclass(yv_cls, pv)[0], valid_f1neu=_perclass(yv_cls, pv)[1],
            valid_f1pos=_perclass(yv_cls, pv)[2],
            test_acc=_acc(yt_cls, pt), test_mf1=_mf1(yt_cls, pt),
            test_f1neg=_perclass(yt_cls, pt)[0], test_f1neu=_perclass(yt_cls, pt)[1],
            test_f1pos=_perclass(yt_cls, pt)[2],
            d_valid_mf1=_mf1(yv_cls, pv) - base["valid"]["mf1"],
            d_test_mf1=_mf1(yt_cls, pt) - base["test"]["mf1"],
            n_forced_neu=int(np.sum(pv == 1) - np.sum(Sv["base_pred"] == 1))))

    print("\n===== 一维阈值族（在 valid 上按 Macro-F1 选阈值）=====")
    for kind, grid in one_dim.items():
        if grid is None:
            print("  [SKIP] %s（该模型无有序头信号）" % kind)
            continue
        best, bp = None, None
        for t in grid:
            p = apply_rule(Sv, kind, [t])
            f = _mf1(yv_cls, p)
            if best is None or f > best:
                best, bp = f, [t]
        pv = apply_rule(Sv, kind, bp)
        pt = apply_rule(St, kind, bp)
        record(kind, bp, pv, pt)
        r = results[-1]
        print("  %-26s t=%-8s  valid MF1=%.4f (%+.4f)  neuF1=%.4f  |  "
              "test MF1=%.4f (%+.4f)  ACC=%.4f"
              % (kind, r["params"], r["valid_mf1"], r["d_valid_mf1"],
                 r["valid_f1neu"], r["test_mf1"], r["d_test_mf1"], r["test_acc"]))

    print("\n===== 二维阈值族 =====")
    for kind, (g1, g2) in two_dim.items():
        best, bp = None, None
        for t1 in g1:
            for t2 in g2:
                f = _mf1(yv_cls, apply_rule(Sv, kind, [t1, t2]))
                if best is None or f > best:
                    best, bp = f, [t1, t2]
        pv = apply_rule(Sv, kind, bp)
        pt = apply_rule(St, kind, bp)
        record(kind, bp, pv, pt)
        r = results[-1]
        print("  %-26s t=%-18s valid MF1=%.4f (%+.4f)  neuF1=%.4f  |  "
              "test MF1=%.4f (%+.4f)  ACC=%.4f"
              % (kind, r["params"], r["valid_mf1"], r["d_valid_mf1"],
                 r["valid_f1neu"], r["test_mf1"], r["d_test_mf1"], r["test_acc"]))

    tab = pd.DataFrame(results).sort_values("d_valid_mf1", ascending=False)
    tab.to_csv(os.path.join(out_dir, "neutral_probe.csv"), index=False, encoding="utf-8-sig")
    print("\n" + tab[["variant", "params", "valid_mf1", "d_valid_mf1", "valid_f1neu",
                      "test_mf1", "d_test_mf1", "test_f1neu", "n_forced_neu"]]
          .to_string(index=False, float_format=lambda x: "%.4f" % x))

    # ---- 最佳候选的两级判据 + McNemar ----
    print("\n" + "=" * 78)
    print("预注册判据裁决（① valid Macro-F1 提升；② test Macro-F1 同向提升）")
    print("=" * 78)
    passed = []
    for _, r in tab.iterrows():
        ok = (r["d_valid_mf1"] > 0) and (r["d_test_mf1"] > 0)
        print("  %-26s Δvalid=%+.4f  Δtest=%+.4f  %s"
              % (r["variant"], r["d_valid_mf1"], r["d_test_mf1"], "通过" if ok else "不通过"))
        if ok:
            passed.append(r)
    verdict = {"n_passed": len(passed)}
    if passed:
        best = passed[0]
        kind, params = best["variant"], [float(x) for x in
                                         best["params"].strip("[]").split(",")]
        pv = apply_rule(Sv, kind, params)
        pt = apply_rule(St, kind, params)
        mc_v = mcnemar(yv_cls, pv, Sv["base_pred"])
        mc_t = mcnemar(yt_cls, pt, St["base_pred"])
        print("\n  → 最佳：%s  params=%s" % (kind, params))
        print("     McNemar valid: n01=%d n10=%d p=%.4f" % (mc_v["n01"], mc_v["n10"],
                                                            mc_v["p"]))
        print("     McNemar test : n01=%d n10=%d p=%.4f" % (mc_t["n01"], mc_t["n10"],
                                                            mc_t["p"]))
        verdict.update(best=dict(variant=kind, params=params), mcnemar=dict(valid=mc_v,
                                                                            test=mc_t))
    else:
        print("\n  → 没有任何规则同时通过两级判据 → **不建议引入中性重判定**")

    rep = dict(tag=tag, baseline=base, results=results, verdict=verdict,
               caveat="规则移动中性边界，按文档07 §5 仅可作探索项，不得作为提交主口径")
    with open(os.path.join(out_dir, "neutral_probe.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=float)
    print("\n[SAVE] %s" % os.path.join(out_dir, "neutral_probe.csv"))
    print("[SAVE] %s" % os.path.join(out_dir, "neutral_probe.json"))


if __name__ == "__main__":
    main()
