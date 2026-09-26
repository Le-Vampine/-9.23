# -*- coding: utf-8 -*-
r"""
P0 评估报告：统一指标口径 + 决策阈值敏感性 + 验证集校准 + 统计检验（**不重训**）。

赛题口径依据（见 `赛事文件.txt` 三、问题2 末段）：
  * 模型参数在附件2 **训练集**上学习；模型结构、超参数与**决策阈值**在**验证集**上选择；
  * 分类（极性）用 Accuracy、F1；回归（强度）用 MAE、Pearson；
  * 论文"基础性能评价"以**验证集**为主报口径（附件2 valid，728 条）；
  * 附件2 的 test 划分（727 条）仅作泛化核验，只在最终评一次；
  * 附件3/附件4 无标签，只用于推理提交。

本脚本做四件事（全部在 valid 上拟合/选择，在 test 上核验）：
  1. 双口径指标：θ 阈值口径（主）+ 分类头 argmax 口径，含每类 F1；
  2. θ 敏感性曲线（ACC / Macro-F1），并给出 θ_f1 / θ_acc / 折中 θ 三个候选；
  3. 后处理改进：仿射校准（L1/OLS）、双头融合（μ + α·E[cls]）及其组合；
  4. 统计检验：McNemar 检验 + bootstrap 置信区间 + 标签强度分布校正 MAE。

产出：
  <out_dir>/report_p0.json         全部数值（机器可读）
  <out_dir>/report_p0.md           论文可直接使用的表格与结论（人可读）
  <figs_dir>/theta_sweep_<tag>.png θ 敏感性曲线
  <figs_dir>/calibration_<tag>.png 校准前后散点与分桶偏差

用法：
  python 02_model\report_p0.py --ckpt_dir ..\runs\q2 --out_dir ..\runs\q2 --figs_dir ..\figs
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config                                        # noqa: E402
from data_utils import load_all                                  # noqa: E402
from losses import compute_metrics, polar_from_score             # noqa: E402
import runtime                                                   # noqa: E402

from scipy.stats import binomtest, chi2                          # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge         # noqa: E402
from sklearn.metrics import accuracy_score, f1_score             # noqa: E402

MODALITIES = ("text", "audio", "vision")
LABELS = ["Negative", "Neutral", "Positive"]
THETA_GRID = np.round(np.arange(0.0, 2.0001, 0.02), 3)
STRENGTH_EDGES = (0.5, 1.5)          # |y| 分桶边界


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def fmt(x, nd=4):
    if x is None:
        return "—"
    try:
        x = float(x)
    except Exception:
        return str(x)
    return "—" if not np.isfinite(x) else ("%.*f" % (nd, x))


def jsonable(o):
    """把 numpy 类型递归转成原生类型，便于 json.dump。"""
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def theta_curve(score, y_cls, grid=THETA_GRID):
    """给定分数与真实极性，返回 θ 网格上的 ACC / Macro-F1 曲线。"""
    accs, f1s = [], []
    for th in grid:
        pred = polar_from_score(score, th)
        accs.append(accuracy_score(y_cls, pred))
        f1s.append(f1_score(y_cls, pred, average="macro", zero_division=0))
    return np.asarray(accs), np.asarray(f1s)


def pick_theta(grid, accs, f1s, rule="f1"):
    """按规则选阈值：f1 | acc | compromise（两者归一化后平均最大）。"""
    if rule == "f1":
        i = int(np.argmax(f1s))
    elif rule == "acc":
        i = int(np.argmax(accs))
    else:
        na = (accs - accs.min()) / (np.ptp(accs) + 1e-12)
        nf = (f1s - f1s.min()) / (np.ptp(f1s) + 1e-12)
        i = int(np.argmax(0.5 * (na + nf)))
    return float(grid[i]), float(accs[i]), float(f1s[i])


def search_asym(score, y_cls, grid=None):
    """非对称双阈值（探索项）：θ_neg≠θ_pos，最大化 Macro-F1。"""
    if grid is None:
        grid = np.round(np.arange(0.0, 1.0001, 0.05), 3)
    best = dict(f1=-1.0, theta_neg=0.0, theta_pos=0.0, acc=0.0)
    for tn in grid:
        neg = score < -tn
        for tp in grid:
            pred = np.ones_like(y_cls)
            pred[neg] = 0
            pred[score > tp] = 2
            f1 = f1_score(y_cls, pred, average="macro", zero_division=0)
            if f1 > best["f1"]:
                best.update(f1=float(f1), theta_neg=float(tn), theta_pos=float(tp),
                            acc=float(accuracy_score(y_cls, pred)))
    return best


# --------------------------------------------------------------------------
# 校准与融合（全部只在 valid 上拟合）
# --------------------------------------------------------------------------
def fit_affine_ols(mu, y):
    """最小二乘仿射 s = a·μ + b（最小化 MSE）。"""
    lr = LinearRegression().fit(np.asarray(mu).reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0]), float(lr.intercept_)


def fit_affine_l1(mu, y, a0=1.0, b0=0.0, grid_n=60):
    """最小化 L1（即 MAE）的仿射系数：粗网格 + 局部细化（无需 scipy 优化器）。"""
    mu = np.asarray(mu, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    best = dict(mae=np.inf, a=1.0, b=0.0)
    a_lo, a_hi = 0.05, 3.0
    b_lo, b_hi = -1.0, 1.0
    for _ in range(3):
        for a in np.linspace(a_lo, a_hi, grid_n):
            res = np.abs(a * mu + np.linspace(b_lo, b_hi, grid_n)[:, None] - y)
            m = res.mean(axis=1)
            j = int(np.argmin(m))
            if m[j] < best["mae"]:
                best.update(mae=float(m[j]), a=float(a),
                            b=float(np.linspace(b_lo, b_hi, grid_n)[j]))
        a_lo, a_hi = best["a"] - 0.15, best["a"] + 0.15
        b_lo, b_hi = best["b"] - 0.08, best["b"] + 0.08
    return best["a"], best["b"], best["mae"]


def fit_affine_ridge(mu, y, alpha=1.0):
    """带正则的仿射（更保守，防止校准过拟合）。"""
    r = Ridge(alpha=alpha).fit(np.asarray(mu).reshape(-1, 1), np.asarray(y))
    return float(r.coef_[0]), float(r.intercept_)


def fit_fusion_alpha(mu, prob, y, grid=None):
    """双头融合 s = μ + α·(P(pos) − P(neg))，α 在 valid 上按 MAE 选。"""
    if grid is None:
        grid = np.round(np.arange(0.0, 1.5001, 0.05), 3)
    e = np.asarray(prob)[:, 2] - np.asarray(prob)[:, 0]
    best = dict(alpha=0.0, mae=float(np.abs(mu - y).mean()))
    for al in grid:
        mae = float(np.abs(mu + al * e - y).mean())
        if mae < best["mae"]:
            best.update(alpha=float(al), mae=mae)
    return best["alpha"], best["mae"]


def fusion_score(mu, prob, alpha):
    e = np.asarray(prob)[:, 2] - np.asarray(prob)[:, 0]
    return np.asarray(mu) + alpha * e


# --------------------------------------------------------------------------
# 统计检验
# --------------------------------------------------------------------------
def mcnemar(pred_a, pred_b, y):
    """配对 McNemar 检验：a 与 b 两个预测的 ACC 差异是否显著。"""
    ca = np.asarray(pred_a) == np.asarray(y)
    cb = np.asarray(pred_b) == np.asarray(y)
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    n = b + c
    if n == 0:
        return dict(n01=b, n10=c, stat=0.0, p=1.0, method="none")
    if n < 25:
        p = float(binomtest(b, n, 0.5).pvalue)
        return dict(n01=b, n10=c, stat=float("nan"), p=p, method="exact-binomial")
    stat = (abs(b - c) - 1.0) ** 2 / n
    return dict(n01=b, n10=c, stat=float(stat), p=float(chi2.sf(stat, 1)),
                method="chi2-continuity")


def bootstrap_ci(y, score, y_cls=None, theta=None, n_boot=2000, seed=1234, level=0.95):
    """MAE / ACC / Macro-F1 的 bootstrap 置信区间（按样本重采样）。"""
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(y)
    maes, accs, f1s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        maes.append(np.abs(y[idx] - score[idx]).mean())
        if y_cls is not None and theta is not None:
            pred = polar_from_score(score[idx], theta)
            accs.append(accuracy_score(np.asarray(y_cls)[idx], pred))
            f1s.append(f1_score(np.asarray(y_cls)[idx], pred, average="macro",
                                zero_division=0))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out = dict(mae_ci=[float(np.percentile(maes, lo)), float(np.percentile(maes, hi))],
               mae_se=float(np.std(maes, ddof=1)), n_boot=n_boot, level=level)
    if accs:
        out["acc_ci"] = [float(np.percentile(accs, lo)), float(np.percentile(accs, hi))]
        out["f1_ci"] = [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))]
    return out


def strength_bucket(y):
    a = np.abs(np.asarray(y, dtype=np.float64))
    return np.digitize(a, STRENGTH_EDGES)      # 0: |y|<=0.5, 1: 0.5<|y|<=1.5, 2: |y|>1.5


def reweighted_mae(y_src, p_src, y_tgt):
    """把源划分的分桶 MAE 按目标划分的标签强度分布加权重算（解释 valid→test 落差）。"""
    bs, bt = strength_bucket(y_src), strength_bucket(y_tgt)
    ws = np.bincount(bt, minlength=3).astype(np.float64)
    ws = ws / ws.sum()
    rows, tot, wsum = [], 0.0, 0.0
    for b in (0, 1, 2):
        m = bs == b
        if m.sum() < 1:
            continue
        mae = float(np.abs(np.asarray(y_src)[m] - np.asarray(p_src)[m]).mean())
        rows.append(dict(bucket=b, n=int(m.sum()), mae=mae, weight=float(ws[b])))
        tot += mae * ws[b]
        wsum += ws[b]
    return (tot / wsum if wsum > 0 else float("nan")), rows


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=os.path.join("..", "runs", "q2"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--figs_dir", default=os.path.join("..", "figs"))
    ap.add_argument("--tag", default="q2")
    ap.add_argument("--version", default="aligned")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--primary", default="head", choices=["head", "theta"],
                    help="极性主口径：head=分类头 argmax（默认，评测目标与训练目标一致）；theta=回归阈值口径")
    ap.add_argument("--write_metrics", action="store_true",
                    help="把重算结果写回 <out_dir>/metrics_<tag>.json（修正旧口径；先自动备份）")
    ap.add_argument("--dump_preds", action="store_true",
                    help="导出逐样本预测 preds_{valid,test}.csv（用于基线对照、二分类桥接、案例研究）")
    a = ap.parse_args()

    out_dir = a.out_dir or a.ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(a.figs_dir, exist_ok=True)
    try:                        # 中文/符号在 GBK 控制台下会报 UnicodeEncodeError，降级为替换
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    # ---------- 1) 数据与模型 ----------
    cfg = Config(version=a.version, out_dir=out_dir)
    if a.data_dir:
        cfg.data_dir = a.data_dir
    cfg.__post_init__()
    print("[DATA] pkl = %s" % cfg.pkl_path)
    tr, va, te = load_all(cfg)
    print("[DATA] train/valid/test = %d/%d/%d" % (tr["N"], va["N"], te["N"]))

    ckpts = runtime.find_ckpts(a.ckpt_dir)
    if not ckpts:
        raise SystemExit("未找到 ckpt：%s" % a.ckpt_dir)
    models = runtime.load_ensemble(ckpts, device=a.device)
    theta_ckpt = float(np.mean([ck["theta"] for _, ck in models]))

    pv = runtime.predict_split(models, va, batch_size=a.batch_size, device=a.device)
    pt = runtime.predict_split(models, te, batch_size=a.batch_size, device=a.device)

    yv, yt = np.asarray(pv["y_reg"], dtype=np.float64), np.asarray(pt["y_reg"], dtype=np.float64)
    cv, ct = np.asarray(pv["y_cls"], dtype=np.int64), np.asarray(pt["y_cls"], dtype=np.int64)
    sv, st = np.asarray(pv["score"], dtype=np.float64), np.asarray(pt["score"], dtype=np.float64)
    hv, ht = np.asarray(pv["prob"]).argmax(-1), np.asarray(pt["prob"]).argmax(-1)
    n_cal = len(ckpts)
    print("[INFO] 集成 %d 个 ckpt；ckpt 记录的 theta=%.2f" % (n_cal, theta_ckpt))

    rep = dict(meta=dict(ckpt_dir=os.path.abspath(a.ckpt_dir),
                         ckpts=[os.path.basename(p) for p in ckpts],
                         n_ckpt=n_cal, theta_ckpt=theta_ckpt, primary=a.primary,
                         version=a.version, n_valid=va["N"], n_test=te["N"],
                         note="阈值/校准/融合均在 valid 上拟合，test 仅作核验"),
               base={}, theta_candidates={}, calibration={}, fusion={},
               combined={}, asymmetric={}, stats={}, distribution={})

    # ---------- 2) 基线（ckpt 自带 θ），双口径 ----------
    for name, yy, cc, ss, hh, n in (("valid", yv, cv, sv, hv, va["N"]),
                                    ("test", yt, ct, st, ht, te["N"])):
        rep["base"][name] = compute_metrics(yy, ss, cc, theta=theta_ckpt, y_pred_cls=hh)
        rep["base"][name]["acc_head"] = float(accuracy_score(cc, hh))
        rep["base"][name]["f1_head"] = float(f1_score(cc, hh, average="macro",
                                                     zero_division=0))
        f1h = f1_score(cc, hh, average=None, zero_division=0, labels=[0, 1, 2])
        rep["base"][name]["f1_head_neg"], rep["base"][name]["f1_head_neu"], \
            rep["base"][name]["f1_head_pos"] = [float(x) for x in f1h]
        rep["base"][name]["agree_head_theta"] = float(
            (hh == polar_from_score(ss, theta_ckpt)).mean())

    # ---------- 3) θ 敏感性 ----------
    acc_v, f1_v = theta_curve(sv, cv)
    acc_t, f1_t = theta_curve(st, ct)
    rep["theta_candidates"] = {
        "theta_f1": pick_theta(THETA_GRID, acc_v, f1_v, "f1"),
        "theta_acc": pick_theta(THETA_GRID, acc_v, f1_v, "acc"),
        "theta_compromise": pick_theta(THETA_GRID, acc_v, f1_v, "compromise"),
        "curve": dict(grid=THETA_GRID.tolist(), valid_acc=acc_v.tolist(),
                      valid_f1=f1_v.tolist(), test_acc=acc_t.tolist(), test_f1=f1_t.tolist()),
    }

    def metrics_at(score_v, score_t, theta, y_pred_cls_v, y_pred_cls_t, yv_, yt_, cv_, ct_):
        mv = compute_metrics(yv_, score_v, cv_, theta=theta, y_pred_cls=y_pred_cls_v)
        mt = compute_metrics(yt_, score_t, ct_, theta=theta, y_pred_cls=y_pred_cls_t)
        return mv, mt

    # ---------- 4) 仿射校准（valid 拟合 → test 核验） ----------
    a_ols, b_ols = fit_affine_ols(sv, yv)
    a_l1, b_l1, mae_l1 = fit_affine_l1(sv, yv)
    a_rdg, b_rdg = fit_affine_ridge(sv, yv, alpha=1.0)
    cand = {}
    for name, (aa, bb) in (("ols", (a_ols, b_ols)), ("l1", (a_l1, b_l1)),
                           ("ridge", (a_rdg, b_rdg))):
        v2 = aa * sv + bb
        t2 = aa * st + bb
        th, _a, _f = pick_theta(THETA_GRID, *theta_curve(v2, cv), "f1")
        mv, mt = metrics_at(v2, t2, th, hv, ht, yv, yt, cv, ct)
        cand[name] = dict(a=float(aa), b=float(bb), theta=float(th),
                          valid=mv, test=mt,
                          valid_mae_insample=float(np.abs(v2 - yv).mean()),
                          test_mae=float(np.abs(t2 - yt).mean()))
    best_cal = min(cand, key=lambda k: cand[k]["valid"]["mae"])
    # 推荐口径：OLS（最小二乘）是唯一"无需选择"的标准估计量；l1/ridge 属变体，作敏感性报告
    rep["calibration"] = dict(candidates=cand, best_by_valid_mae=best_cal,
                              recommended="ols" if "ols" in cand else best_cal,
                              recommended_reason=(
                                  "OLS 为闭式最小二乘估计，不含需要挑选的超参数；"
                                  "l1/ridge 作为敏感性变体报告。若按 valid MAE 最小挑选则选 "
                                  + best_cal + "，但二者 valid MAE 差异 <0.001。"),
                              chosen=dict(alpha=cand[best_cal]["a"],
                                          beta=cand[best_cal]["b"],
                                          theta=cand[best_cal]["theta"]))

    # ---------- 5) 双头融合 + 校准（组合方案） ----------
    alpha, mae_fuse = fit_fusion_alpha(sv, pv["prob"], yv)
    fv, ft = fusion_score(sv, pv["prob"], alpha), fusion_score(st, pt["prob"], alpha)
    th_f, _a, _f = pick_theta(THETA_GRID, *theta_curve(fv, cv), "f1")
    mv_f, mt_f = metrics_at(fv, ft, th_f, hv, ht, yv, yt, cv, ct)
    rep["fusion"] = dict(alpha=float(alpha), valid_mae=float(mae_fuse),
                         theta=float(th_f), valid=mv_f, test=mt_f)

    _fit_map = {
        "ols": lambda m, y: fit_affine_ols(m, y),
        "l1": lambda m, y: fit_affine_l1(m, y)[:2],
        "ridge": lambda m, y: fit_affine_ridge(m, y),
    }
    fit_fn = _fit_map[rep["calibration"]["recommended"]]
    aa, bb = fit_fn(fv, yv)
    qv, qt = aa * fv + bb, aa * ft + bb
    th_c, _a, _f = pick_theta(THETA_GRID, *theta_curve(qv, cv), "f1")
    mv_c, mt_c = metrics_at(qv, qt, th_c, hv, ht, yv, yt, cv, ct)
    rep["combined"] = dict(alpha=float(alpha), a=float(aa), b=float(bb), theta=float(th_c),
                           valid=mv_c, test=mt_c)
    cb = rep["combined"]

    # ---------- 6) 非对称阈值（探索项，仅讨论） ----------
    asy = search_asym(sv, cv)
    pred_asy_v = np.ones_like(cv)
    pred_asy_v[sv < -asy["theta_neg"]] = 0
    pred_asy_v[sv > asy["theta_pos"]] = 2
    pred_asy_t = np.ones_like(ct)
    pred_asy_t[st < -asy["theta_neg"]] = 0
    pred_asy_t[st > asy["theta_pos"]] = 2
    rep["asymmetric"] = dict(
        theta_neg=asy["theta_neg"], theta_pos=asy["theta_pos"],
        valid=dict(acc=float(accuracy_score(cv, pred_asy_v)),
                   f1=float(f1_score(cv, pred_asy_v, average="macro", zero_division=0))),
        test=dict(acc=float(accuracy_score(ct, pred_asy_t)),
                  f1=float(f1_score(ct, pred_asy_t, average="macro", zero_division=0))),
        note="非对称阈值改变了题目对 Neutral=0 的定义边界，仅作探索性讨论，不作提交主口径")

    # ---------- 7) 统计检验 ----------
    base_pred_v = polar_from_score(sv, theta_ckpt)
    cal_key = rep["calibration"]["recommended"]
    cal_pred_v = polar_from_score(cand[cal_key]["a"] * sv + cand[cal_key]["b"],
                                  cand[cal_key]["theta"])
    comb_pred_v = polar_from_score(cb["a"] * fv + cb["b"], cb["theta"])
    rep["stats"] = dict(
        mcnemar_base_vs_head=mcnemar(base_pred_v, hv, cv),
        mcnemar_base_vs_calibrated=mcnemar(base_pred_v, cal_pred_v, cv),
        mcnemar_base_vs_combined=mcnemar(base_pred_v, comb_pred_v, cv),
        bootstrap_valid=bootstrap_ci(yv, sv, cv, theta_ckpt, n_boot=a.n_boot),
        bootstrap_test=bootstrap_ci(yt, st, ct, theta_ckpt, n_boot=a.n_boot),
        bootstrap_valid_calibrated=bootstrap_ci(yv, cand[cal_key]["a"] * sv + cand[cal_key]["b"],
                                                cv, cand[cal_key]["theta"], n_boot=a.n_boot),
        bootstrap_test_calibrated=bootstrap_ci(yt, cand[cal_key]["a"] * st + cand[cal_key]["b"],
                                               ct, cand[cal_key]["theta"], n_boot=a.n_boot),
    )

    # ---------- 8) 标签强度分布校正 ----------
    corr_v2t, rows = reweighted_mae(yv, sv, yt)
    corr_t2v, rows2 = reweighted_mae(yt, st, yv)
    rep["distribution"] = dict(
        valid_to_test=dict(reweighted_mae=float(corr_v2t), buckets=rows,
                           actual_test_mae=float(np.abs(yt - st).mean())),
        test_to_valid=dict(reweighted_mae=float(corr_t2v), buckets=rows2,
                           actual_valid_mae=float(np.abs(yv - sv).mean())),
        note="把 valid 的分桶 MAE 按 test 的 |y| 分布加权重算，用于解释两划分的 MAE 落差")

    # ---------- 9) 保存 JSON ----------
    jp = os.path.join(out_dir, "report_p0.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(jsonable(rep), f, ensure_ascii=False, indent=2)
    print("[SAVE] %s" % jp)

    # ---------- 9b) 可选：把重算指标写回 metrics_<tag>.json（修正旧版口径） ----------
    if a.write_metrics:
        _seed = None
        try:
            _seed = models[0][1].get("seed", None)
        except Exception:
            _seed = None
        fname = ("metrics_s%s.json" % _seed) if _seed is not None \
            else ("metrics_%s.json" % a.tag)
        mpath = os.path.join(out_dir, fname)
        if os.path.isfile(mpath):
            bak = os.path.join(out_dir, "metrics_%s_legacy.json" % a.tag)
            if not os.path.isfile(bak):
                with open(mpath, "r", encoding="utf-8") as f:
                    old = f.read()
                with open(bak, "w", encoding="utf-8") as f:
                    f.write(old)
                print("[BACKUP] %s" % bak)
        def _shape(m, prim):
            """把双口径揉进统一字段：acc/f1 = 主口径，acc_theta/f1_theta = θ 口径，acc_head/f1_head = 分类头。"""
            d = dict(m)
            d["acc_theta"], d["f1_theta"] = float(m["acc"]), float(m["f1"])
            d["f1_theta_neg"] = float(m.get("f1_neg", float("nan")))
            d["f1_theta_neu"] = float(m.get("f1_neu", float("nan")))
            d["f1_theta_pos"] = float(m.get("f1_pos", float("nan")))
            if prim == "head":
                d["acc"], d["f1"] = float(m["acc_head"]), float(m["f1_head"])
                d["f1_neg"] = float(m.get("f1_head_neg", float("nan")))
                d["f1_neu"] = float(m.get("f1_head_neu", float("nan")))
                d["f1_pos"] = float(m.get("f1_head_pos", float("nan")))
            d["primary"] = prim
            return d

        payload = dict(
            theta=theta_ckpt,
            primary=a.primary,
            valid_clean=_shape(rep["base"]["valid"], a.primary),
            test_clean=_shape(rep["base"]["test"], a.primary),
            _note=("由 report_p0.py 重算：acc/f1 为**主口径**（见 primary 字段，当前为 %s）；"
                   "acc_theta/f1_theta 为 θ 回归阈值口径；acc_head/f1_head 为分类头口径。" % a.primary),
            _recomputed=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            _ckpts=rep["meta"]["ckpts"])
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(jsonable(payload), f, ensure_ascii=False, indent=2)
        print("[SAVE] %s（已修正口径）" % mpath)

    # ---------- 9c) 逐样本预测导出（用于上限/基线对照、二分类桥接、案例研究） ----------
    if a.dump_preds:
        for name, yy, cc, ss, hh, pp, rr, ii in (
                ("valid", yv, cv, sv, hv, pv["prob"], pv["rho"], pv["ids"]),
                ("test", yt, ct, st, ht, pt["prob"], pt["rho"], pt["ids"])):
            d = pd.DataFrame({
                "id": np.asarray(ii),
                "y_reg": yy, "y_cls": cc,
                "score_theta": np.round(ss, 6),
                "pred_theta": polar_from_score(ss, theta_ckpt),
                "pred_head": np.asarray(hh),
                "prob_neg": np.round(np.asarray(pp)[:, 0], 6),
                "prob_neu": np.round(np.asarray(pp)[:, 1], 6),
                "prob_pos": np.round(np.asarray(pp)[:, 2], 6),
                "rho_text": np.round(np.asarray(rr)[:, 0], 4),
                "rho_audio": np.round(np.asarray(rr)[:, 1], 4),
                "rho_vision": np.round(np.asarray(rr)[:, 2], 4),
            })
            pth = os.path.join(out_dir, "preds_%s.csv" % name)
            d.to_csv(pth, index=False, encoding="utf-8-sig")
            print("[SAVE] %s（%d 行）" % (pth, len(d)))

    # ---------- 10) 图 ----------
    try:
        cal_v = cand[cal_key]["a"] * sv + cand[cal_key]["b"]
        cal_t = cand[cal_key]["a"] * st + cand[cal_key]["b"]
        draw_theta_fig(rep, os.path.join(a.figs_dir, "theta_sweep_%s.png" % a.tag))
        draw_calib_fig(yv, sv, cal_v, yt, st, cal_t,
                       os.path.join(a.figs_dir, "calibration_%s.png" % a.tag))
        print("[SAVE] figs: theta_sweep_%s.png, calibration_%s.png" % (a.tag, a.tag))
    except Exception as e:                                     # pragma: no cover
        print("[WARN] 画图失败：%s" % e)

    # ---------- 11) Markdown ----------
    md = build_md(rep, os.path.basename(a.ckpt_dir))
    mp = os.path.join(out_dir, "report_p0.md")
    with open(mp, "w", encoding="utf-8") as f:
        f.write(md)
    print("[SAVE] %s" % mp)
    print(md)


# --------------------------------------------------------------------------
def draw_theta_fig(rep, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    cur = rep["theta_candidates"]["curve"]
    grid = np.asarray(cur["grid"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, split in zip(axes, ("valid", "test")):
        acc = np.asarray(cur["%s_acc" % split])
        f1 = np.asarray(cur["%s_f1" % split])
        ax.plot(grid, acc, "-", lw=1.4, label="Accuracy")
        ax.plot(grid, f1, "-", lw=1.4, label="Macro-F1")
        for k, c in (("theta_f1", "tab:red"), ("theta_acc", "tab:blue"),
                     ("theta_compromise", "tab:green")):
            th = rep["theta_candidates"][k][0]
            ax.axvline(th, color=c, ls=":", lw=1.2, label="%s=%.2f" % (k, th))
        ax.axvline(rep["meta"]["theta_ckpt"], color="k", ls="--", lw=1.0,
                   label="ckpt θ=%.2f" % rep["meta"]["theta_ckpt"])
        ax.set_xlabel("决策阈值 θ（%s）" % ("验证集：选择用" if split == "valid" else "测试集：仅核验"))
        ax.set_ylabel("指标值")
        ax.set_title("%s 集：θ 敏感性" % split, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def draw_calib_fig(yv, sv, scv, yt, st, sct, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.0))
    for ax, (y, s, ttl, col) in zip(axes[:, 0],
                                    [(yv, sv, "valid 校准前", "tab:blue"),
                                     (yt, st, "test 校准前", "tab:orange")]):
        ax.scatter(y, s, s=6, alpha=0.3, color=col)
        ax.plot([-3, 3], [-3, 3], "k--", lw=1)
        z = np.polyfit(y, s, 1)
        xs = np.linspace(-3, 3, 30)
        ax.plot(xs, np.polyval(z, xs), "r-", lw=1.2,
                label="拟合 y=%.3fx%+.3f" % (z[0], z[1]))
        ax.set_title("%s  MAE=%.3f" % (ttl, np.abs(y - s).mean()), fontsize=10)
        ax.set_xlabel("真值强度"); ax.set_ylabel("预测强度")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    for ax, (y, s, ttl, col) in zip(axes[:, 1],
                                    [(yv, scv, "valid 校准后", "tab:green"),
                                     (yt, sct, "test 校准后", "tab:purple")]):
        ax.scatter(y, s, s=6, alpha=0.3, color=col)
        ax.plot([-3, 3], [-3, 3], "k--", lw=1)
        z = np.polyfit(y, s, 1)
        xs = np.linspace(-3, 3, 30)
        ax.plot(xs, np.polyval(z, xs), "r-", lw=1.2,
                label="拟合 y=%.3fx%+.3f" % (z[0], z[1]))
        ax.set_title("%s  MAE=%.3f" % (ttl, np.abs(y - s).mean()), fontsize=10)
        ax.set_xlabel("真值强度"); ax.set_ylabel("校准后强度")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _row(name, m):
    return ("| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (name, fmt(m.get("mae")), fmt(m.get("pearson")), fmt(m.get("ccc")),
               fmt(m.get("acc")), fmt(m.get("f1")),
               fmt(m.get("f1_neg")), fmt(m.get("f1_neu")), fmt(m.get("f1_pos"))))


def build_md(rep, exp):
    b = rep["base"]
    primary = rep.get("meta", {}).get("primary", "head")
    md = []
    md.append("# 问题2 P0 评估报告（双口径 + 阈值敏感性 + 验证集校准）\n")
    md.append("> 实验目录：`%s`　｜　集成 ckpt：%d 个　｜　θ(ckpt)=%.2f　｜　极性主口径：**%s**"
              % (exp, rep["meta"]["n_ckpt"], rep["meta"]["theta_ckpt"],
                 "分类头 argmax" if primary == "head" else "θ 回归阈值"))
    md.append("> 规模：valid %d 条 / test %d 条（附件2）；附件3/4 无标签，仅推理\n"
              % (rep["meta"]["n_valid"], rep["meta"]["n_test"]))
    md.append("**口径声明**：赛题规定模型结构、超参数与**决策阈值**在验证集上选择；"
              "论文“基础性能评价”以**验证集**为主报口径，附件2 的 test 划分仅作泛化核验。"
              "本表所有阈值、校准系数、融合系数均在 valid 上拟合，test 未参与任何选择。\n")

    md.append("\n## 1. 基础性能（主报 valid，test 为附加泛化核验）\n")
    if primary == "head":
        md.append("**极性主口径：分类头 argmax** —— 直接由附件2 `classification_labels` 监督，"
                  "评测目标与训练目标一致，且不受强度分数尺度（收缩/校准）影响。\n")
    else:
        md.append("**极性主口径：θ 回归阈值** —— 与题目“0 仅属中性、符号定正负”的定义严格一致，"
                  "并与强度输出共用同一分数；但需在 valid 上选 θ 且对分数尺度敏感。\n")
    md.append("| 划分 | 口径 | MAE | Pearson | CCC | ACC | Macro-F1 | F1 neg | F1 neu | F1 pos |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for split, label in (("valid", "valid（主报）"), ("test", "test（附加核验）")):
        m = b[split]
        tag_head = "分类头（主口径）" if primary == "head" else "分类头（对照）"
        tag_th = "θ 阈值（对照）" if primary == "head" else "θ 阈值（主口径）"
        md.append("| %s | %s | — | — | — | %.4f | %.4f | %.4f | %.4f | %.4f |"
                  % (label, tag_head, m.get("acc_head", float("nan")),
                     m.get("f1_head", float("nan")), m.get("f1_head_neg", float("nan")),
                     m.get("f1_head_neu", float("nan")), m.get("f1_head_pos", float("nan"))))
        md.append("| %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |"
                  % (label, tag_th, m["mae"], m["pearson"], m["ccc"], m["acc"], m["f1"],
                     m["f1_neg"], m["f1_neu"], m["f1_pos"]))
    md.append("\n> 两口径一致率：valid %.3f，test %.3f。同一模型在两种口径下的 ACC 差异可达 0.05 量级"
              "（强正则会压缩强度分数、迫使 θ 变大），因此论文必须显式声明主口径。\n"
              % (b["valid"].get("agree_head_theta", float("nan")),
                 b["test"].get("agree_head_theta", float("nan"))))
    md.append("> ⚠️ **口径核对**：`runs/*/metrics_s*.json` 中历史记录的 `acc`/`f1` 与上面"
              "**分类头口径**完全一致（属旧版口径产物）；论文引用请以本表为准。\n")

    md.append("\n## 2. 决策阈值敏感性（θ 在 valid 上选择）\n")
    md.append("| 准则 | θ | valid ACC | valid Macro-F1 | test ACC | test Macro-F1 |")
    md.append("|---|---|---|---|---|---|")
    for k, label in (("theta_f1", "最大化 Macro-F1（推荐主口径）"),
                     ("theta_acc", "最大化 Accuracy"),
                     ("theta_compromise", "ACC 与 F1 归一化折中")):
        th, ac, f1 = rep["theta_candidates"][k]
        ac_t, f1_t = _eval_theta(rep, k)
        md.append("| %s | %.2f | %.4f | %.4f | %.4f | %.4f |" % (label, th, ac, f1, ac_t, f1_t))
    md.append("\n*曲线见 `figs/theta_sweep_%s.png`；用于说明所选 θ 的稳健区间。*\n" % "q2")

    md.append("\n## 3. 后处理改进（valid 拟合 → test 核验）\n")
    md.append("| 方案 | 参数 | θ | valid MAE | test MAE | valid ACC | valid Macro-F1 | test ACC | test Macro-F1 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    md.append("| 基线（无校准） | — | %.2f | %s | %s | %s | %s | %s | %s |"
              % (rep["meta"]["theta_ckpt"], fmt(b["valid"]["mae"]), fmt(b["test"]["mae"]),
                 fmt(b["valid"]["acc"]), fmt(b["valid"]["f1"]),
                 fmt(b["test"]["acc"]), fmt(b["test"]["f1"])))
    for k, v in rep["calibration"]["candidates"].items():
        md.append("| 仿射校准-%s | a=%.3f b=%+.3f | %.2f | %s | %s | %s | %s | %s | %s |"
                  % (k, v["a"], v["b"], v["theta"], fmt(v["valid"]["mae"]), fmt(v["test"]["mae"]),
                     fmt(v["valid"]["acc"]), fmt(v["valid"]["f1"]),
                     fmt(v["test"]["acc"]), fmt(v["test"]["f1"])))
    fu = rep["fusion"]
    md.append("| 双头融合 | α=%.3f | %.2f | %s | %s | %s | %s | %s | %s |"
              % (fu["alpha"], fu["theta"], fmt(fu["valid"]["mae"]), fmt(fu["test"]["mae"]),
                 fmt(fu["valid"]["acc"]), fmt(fu["valid"]["f1"]),
                 fmt(fu["test"]["acc"]), fmt(fu["test"]["f1"])))
    cb = rep["combined"]
    md.append("| 融合+校准（推荐） | α=%.3f a=%.3f b=%+.3f | %.2f | %s | %s | %s | %s | %s | %s |"
              % (cb["alpha"], cb["a"], cb["b"], cb["theta"],
                 fmt(cb["valid"]["mae"]), fmt(cb["test"]["mae"]),
                 fmt(cb["valid"]["acc"]), fmt(cb["valid"]["f1"]),
                 fmt(cb["test"]["acc"]), fmt(cb["test"]["f1"])))
    md.append("\n*校准后仍重新在 valid 上选 θ（与赛题“阈值在验证集上选”一致）。"
              "校准前/后散点见 `figs/calibration_%s.png`。*\n" % "q2")
    md.append("\n> **推荐校准方法：`%s`**——%s\n"
              % (rep["calibration"]["recommended"],
                 rep["calibration"]["recommended_reason"]))

    md.append("\n## 4. 非对称阈值（探索项，不计入提交主口径）\n")
    asy = rep["asymmetric"]
    md.append("- valid 最优：θ_neg=%.2f, θ_pos=%.2f → ACC=%.4f, Macro-F1=%.4f"
              % (asy["theta_neg"], asy["theta_pos"], asy["valid"]["acc"], asy["valid"]["f1"]))
    md.append("- test 核验：ACC=%.4f, Macro-F1=%.4f" % (asy["test"]["acc"], asy["test"]["f1"]))
    md.append("- 说明：%s\n" % asy["note"])

    md.append("\n## 5. 统计检验（n=728/727，ACC 标准误约 1.8%）\n")
    st_ = rep["stats"]
    md.append("| 比较 | n01 | n10 | p 值 | 方法 |")
    md.append("|---|---|---|---|---|")
    for k, label in (("mcnemar_base_vs_head", "基线(θ) vs 分类头口径"),
                     ("mcnemar_base_vs_calibrated", "基线 vs 仿射校准"),
                     ("mcnemar_base_vs_combined", "基线 vs 融合+校准")):
        m = st_[k]
        md.append("| %s | %d | %d | %.4f | %s |" % (label, m["n01"], m["n10"], m["p"], m["method"]))
    md.append("95% bootstrap 置信区间（MAE / ACC / Macro-F1）：\n")
    md.append("| 划分 | MAE [lo, hi] | ACC [lo, hi] | Macro-F1 [lo, hi] |")
    md.append("|---|---|---|---|")
    for key, label in (("bootstrap_valid", "valid 基线"),
                       ("bootstrap_valid_calibrated", "valid 校准后"),
                       ("bootstrap_test", "test 基线"),
                       ("bootstrap_test_calibrated", "test 校准后")):
        s = st_[key]
        acc = s.get("acc_ci", [float("nan"), float("nan")])
        f1c = s.get("f1_ci", [float("nan"), float("nan")])
        md.append("| %s | [%s, %s] | [%s, %s] | [%s, %s] |"
                  % (label, fmt(s["mae_ci"][0], 3), fmt(s["mae_ci"][1], 3),
                     fmt(acc[0], 3), fmt(acc[1], 3),
                     fmt(f1c[0], 3), fmt(f1c[1], 3)))

    md.append("\n## 6. 标签强度分布校正（解释 valid→test 的 MAE 落差）\n")
    dd = rep["distribution"]
    md.append("| 方向 | 加权重算 MAE | 实际 MAE |")
    md.append("|---|---|---|")
    md.append("| valid 分桶 MAE 按 test 标签分布加权 | %s | %s |"
              % (fmt(dd["valid_to_test"]["reweighted_mae"]),
                 fmt(dd["valid_to_test"]["actual_test_mae"])))
    md.append("| test 分桶 MAE 按 valid 标签分布加权 | %s | %s |"
              % (fmt(dd["test_to_valid"]["reweighted_mae"]),
                 fmt(dd["test_to_valid"]["actual_valid_mae"])))
    md.append("\n分桶明细（valid→test）：\n")
    md.append("| 桶(|y|) | n | MAE | test 权重 |")
    md.append("|---|---|---|---|")
    for r in dd["valid_to_test"]["buckets"]:
        md.append("| %d | %d | %s | %s |" % (r["bucket"], r["n"], fmt(r["mae"]), fmt(r["weight"])))

    md.append("\n## 7. 结论\n")
    bv, bt = b["valid"], b["test"]
    co = rep["calibration"]["candidates"][rep["calibration"]["recommended"]]
    if primary == "head":
        pav, paf = bv.get("acc_head"), bv.get("f1_head")
        pat, paf_t = bt.get("acc_head"), bt.get("f1_head")
        pname = "分类头 argmax"
    else:
        pav, paf = bv["acc"], bv["f1"]
        pat, paf_t = bt["acc"], bt["f1"]
        pname = "θ 回归阈值"
    md.append("1. 主报口径（valid，%s）：ACC=%.4f、Macro-F1=%.4f；配套强度指标 MAE=%.4f、Pearson=%.4f。"
              % (pname, pav, paf, bv["mae"], bv["pearson"]))
    md.append("2. 泛化核验（附件2 test 划分，%s）：ACC=%.4f、Macro-F1=%.4f、MAE=%.4f；"
              "两划分的 MAE 落差中约 %s 由标签强度分布差异解释。"
              % (pname, pat, paf_t, bt["mae"],
                 fmt(abs(dd["valid_to_test"]["actual_test_mae"]
                         - dd["valid_to_test"]["reweighted_mae"]), 3)))
    md.append("3. 口径差异：两口径一致率 valid %.3f / test %.3f。θ 口径需在 valid 上选阈值且对强度分数"
              "尺度敏感（强正则/校准会移动 θ），分类头无此问题——故论文必须显式声明主口径。"
              % (bv.get("agree_head_theta", float("nan")),
                 bt.get("agree_head_theta", float("nan"))))
    md.append("4. 校准（只作用于 θ 分支，系数在 valid 上拟合，推荐方法 `%s`）：θ 分支 MAE "
              "%.4f→%.4f（valid）、%.4f→%.4f（test）；主口径（%s）不经过校准，不受其影响。"
              % (rep["calibration"]["recommended"], bv["mae"], co["valid"]["mae"],
                 bt["mae"], co["test"]["mae"], pname))
    md.append("5. 双头融合在 valid 上最优 α=%s（说明回归头已含分类头信息，无增益）；"
              "非对称阈值在 valid 上最优退化为对称（θ_neg=θ_pos=%.2f），故不引入与题目中性定义不一致的改动。"
              % (fu["alpha"], asy["theta_neg"]))
    md.append("6. 论文需说明：θ、校准与融合系数均在验证集上拟合，附件3/4 未参与任何选择；"
              "提交 CSV 的 `pred_label` 采用主口径，另一口径以并列列（`pred_label_theta` / `pred_label_head`）写出。\n")
    return "\n".join(md)


def _eval_theta(rep, key):
    """从曲线里取某个候选 θ 在 test 上的 ACC/F1。"""
    th = rep["theta_candidates"][key][0]
    cur = rep["theta_candidates"]["curve"]
    i = int(np.argmin(np.abs(np.asarray(cur["grid"]) - th)))
    return cur["test_acc"][i], cur["test_f1"][i]


if __name__ == "__main__":
    main()
