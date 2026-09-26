"""决策层：把"两口径 oracle 上界"变成**可学习、可复现、只在 valid 上拟合**的决策规则。

背景（实测）：
    valid/test 上把「分类头 argmax」与「θ 阈值」逐样本取优，可得 ACC 0.709/0.725，
    这是**决策层天花板**。oracle 本身不可实现（偷看标签），但可以用"在 valid 上
    拟合一个简单决策规则"去逼近它 —— 其差距的一半左右通常可以捞回来。

规则族（全部只在 valid 上拟合/选择，test 只做核验）：
    base      分类头 argmax（无参数）
    th        对称 θ 阈值（valid 上按 Macro-F1 或 ACC 选）
    band      非对称三段带：s<-tl -> neg, s>tp -> pos, 否则 neu；可叠加 p_neu 门控
    lr        多项 logistic 回归（在分数/概率/logit 差值等 9 维特征上）
    tree      深度 3 决策树（**可解释**，规则可直接写进论文）
    rf        随机森林
    oracle    两口径逐样本取优（上界参照，用到标签，仅供对照）

选择准则：valid 内 5 折交叉验证的 Macro-F1（避免"在 valid 上单次拟合"的过拟合，
这正是之前先验校正/仿射校准失败的原因）。

用法：
    python stack_decision.py --valid runs/exp_c1/preds_valid.csv \
                             --test  runs/exp_c1/preds_test.csv  --tag c1
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(errors="replace")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    _HAS_SK = True
except Exception:                                    # pragma: no cover
    _HAS_SK = False

FEATS = ["score_theta", "prob_neg", "prob_neu", "prob_pos",
         "abs_score", "margin", "entropy", "logit_pm", "p_polar"]


# ---------------------------------------------------------------- 指标
def macro_f1(y, p):
    f1s = []
    for c in np.unique(y):
        tp = float(((y == c) & (p == c)).sum())
        fp = float(((y != c) & (p == c)).sum())
        fn = float(((y == c) & (p != c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def score_all(y, p):
    return dict(ACC=round(float((y == p).mean()), 4),
                MacroF1=round(macro_f1(y, p), 4),
                neuF1=round(_cls_f1(y, p, 1), 4))


def _cls_f1(y, p, c):
    tp = float(((y == c) & (p == c)).sum())
    fp = float(((y != c) & (p == c)).sum())
    fn = float(((y == c) & (p != c)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


# ---------------------------------------------------------------- 特征
def build_feats(df):
    s = df["score_theta"].to_numpy(dtype=float)
    P = df[["prob_neg", "prob_neu", "prob_pos"]].to_numpy(dtype=float) + 1e-9
    P = P / P.sum(axis=1, keepdims=True)
    ps = np.sort(P, axis=1)
    F = np.column_stack([
        s,
        P[:, 0], P[:, 1], P[:, 2],
        np.abs(s),
        ps[:, -1] - ps[:, -2],                                   # margin
        -(P * np.log(P)).sum(axis=1),                            # entropy
        np.log(P[:, 2]) - np.log(P[:, 0]),                       # 极性 logit 差
        P[:, 2] / (P[:, 2] + P[:, 0]),                           # 极性内归一
    ])
    return F, s, P


def band_rule(s, P, tl, tp, tau):
    """seg: s<-tl->neg, s>tp->pos, 否则 neu；tau>0 时 p_neu>tau 先判 neu。"""
    pred = np.ones_like(s, dtype=int)
    pred[s < -tl] = 0
    pred[s > tp] = 2
    if tau > 0:
        pred = np.where(P[:, 1] > tau, 1, pred)
    return pred


def fit_band(s, P, y, crit="MacroF1"):
    best, bp = -9.0, (0.0, 0.0, 0.0)
    for tl in np.arange(0.0, 1.01, 0.05):
        for tp in np.arange(0.0, 1.01, 0.05):
            for tau in (0.0, 0.36, 0.40, 0.44, 0.48, 0.52):
                p = band_rule(s, P, tl, tp, tau)
                v = macro_f1(y, p) if crit == "MacroF1" else (y == p).mean()
                if v > best:
                    best, bp = v, (float(tl), float(tp), float(tau))
    return bp, best


def fit_theta(s, y, crit="MacroF1"):
    best, bth = -9.0, 0.0
    for th in np.arange(0.0, 1.51, 0.02):
        p = np.ones_like(s, dtype=int)
        p[s < -th] = 0
        p[s > th] = 2
        v = macro_f1(y, p) if crit == "MacroF1" else (y == p).mean()
        if v > best:
            best, bth = v, float(th)
    return bth, best


# ---------------------------------------------------------------- 规则实现
def make_rule(name):
    """返回 fit(Ftr, y) -> predict(Fte) 的闭包族。"""
    if name == "base":
        return lambda F, y, s, P: (lambda F2, s2, P2: np.argmax(P2, axis=1))
    if name == "th":
        def f(F, y, s, P):
            th, _ = fit_theta(s, y, "MacroF1")
            def pr(F2, s2, P2):
                q = np.ones_like(s2, dtype=int)
                q[s2 < -th] = 0
                q[s2 > th] = 2
                return q
            return pr
        return f
    if name == "band":
        def f(F, y, s, P):
            (tl, tp, tau), _ = fit_band(s, P, y, "MacroF1")
            return lambda F2, s2, P2: band_rule(s2, P2, tl, tp, tau)
        return f
    if name in ("lr", "tree", "rf"):
        def f(F, y, s, P):
            sc = StandardScaler().fit(F)
            X = sc.transform(F)
            if name == "lr":
                m = LogisticRegression(max_iter=3000, C=1.0,
                                       class_weight="balanced")
            elif name == "tree":
                m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20,
                                           class_weight="balanced",
                                           random_state=0)
            else:
                m = RandomForestClassifier(n_estimators=400, max_depth=6,
                                           min_samples_leaf=8,
                                           class_weight="balanced_subsample",
                                           random_state=0, n_jobs=-1)
            m.fit(X, y)
            return lambda F2, s2, P2: m.predict(sc.transform(F2))
        return f
    raise ValueError(name)


RULES = ["base", "th", "band", "lr", "tree", "rf"]


def cv_eval(name, F, y, s, P, n_fold=5, seed=0):
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(F, y):
        pr = make_rule(name)(F[tr], y[tr], s[tr], P[tr])
        oof[te] = pr(F[te], s[te], P[te])
    return oof, macro_f1(y, oof), float((y == oof).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_fold", type=int, default=5)
    ap.add_argument("--crit", default="MacroF1", choices=["MacroF1", "ACC"],
                    help="valid 上选择规则的准则")
    args = ap.parse_args()

    dv = pd.read_csv(args.valid)
    dt = pd.read_csv(args.test)
    yv = dv["y_cls"].to_numpy().astype(int)
    yt = dt["y_cls"].to_numpy().astype(int)
    Fv, sv, Pv = build_feats(dv)
    Ft, st, Pt = build_feats(dt)

    # 上界参照：两口径逐样本取优
    orc_v = ((dv["pred_head"].to_numpy() == yv) |
             (dv["pred_theta"].to_numpy() == yv)).mean()
    orc_t = ((dt["pred_head"].to_numpy() == yt) |
             (dt["pred_theta"].to_numpy() == yt)).mean()

    rows, best = [], None
    for name in RULES:
        oof, cvf1, cvacc = cv_eval(name, Fv, yv, sv, Pv, args.n_fold)
        sel = cvf1 if args.crit == "MacroF1" else cvacc
        # 用全部 valid 重新拟合后在 test 上核验（诚实口径）
        pr = make_rule(name)(Fv, yv, sv, Pv)
        pv = pr(Fv, sv, Pv)
        pt = pr(Ft, st, Pt)
        r = dict(rule=name, cv_sel=round(float(sel), 4),
                 valid=score_all(yv, pv), test=score_all(yt, pt))
        rows.append(r)
        if best is None or sel > best[1]:
            best = (name, sel, pv, pt)

    # 树规则导出（可读规则，便于写进论文）
    tree_txt = ""
    if _HAS_SK:
        m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20,
                                   class_weight="balanced", random_state=0)
        m.fit(StandardScaler().fit_transform(Fv), yv)
        tree_txt = export_text(m, feature_names=FEATS, decimals=3)

    out = dict(tag=args.tag, n_valid=int(len(yv)), n_test=int(len(yt)),
               oracle=dict(valid=round(float(orc_v), 4), test=round(float(orc_t), 4)),
               chosen=best[0], chosen_cv=round(float(best[1]), 4),
               rows=rows, tree_rules=tree_txt)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.valid))
    with open(os.path.join(out_dir, f"stack_decision_{args.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    md = [f"# 决策层规则对比（{args.tag}）", "",
          f"- valid {len(yv)} / test {len(yt)}；选择准则 = valid 内 {args.n_fold} 折 CV 的 {args.crit}",
          f"- **决策层 oracle 上界**：valid {orc_v:.4f} / test {orc_t:.4f}", "",
          "| 规则 | CV 选择值 | valid ACC | valid MacroF1 | valid neuF1 | test ACC | test MacroF1 | test neuF1 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['rule']} | {r['cv_sel']:.4f} | {r['valid']['ACC']:.4f} | "
                  f"{r['valid']['MacroF1']:.4f} | {r['valid']['neuF1']:.4f} | "
                  f"**{r['test']['ACC']:.4f}** | **{r['test']['MacroF1']:.4f}** | "
                  f"{r['test']['neuF1']:.4f} |")
    md += ["", f"**选中规则：`{best[0]}`**（CV {best[1]:.4f}）", "",
           "## 可解释树规则", "```", tree_txt.strip(), "```", ""]
    with open(os.path.join(out_dir, f"stack_decision_{args.tag}.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
