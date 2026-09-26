"""逐类诊断：混淆矩阵 / 每类召回 / 中性类专项 / 三分-二分对照 / 难度分层。

用法：
  python perclass_report.py --preds runs/exp_c1/preds_valid.csv --tag c1_valid
  python perclass_report.py --preds a.csv:tagA b.csv:tagB --out_dir runs/diag
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(errors="replace")

CLS_NAME = {0: "neg", 1: "neu", 2: "pos"}


def load(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["y_cls", "pred_head"])
    df["y_cls"] = df["y_cls"].astype(int)
    return df


def per_class(df, col):
    y = df["y_cls"].to_numpy()
    p = df[col].to_numpy().astype(int)
    cm = np.zeros((3, 3), dtype=int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    rows = []
    for c in range(3):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(cls=CLS_NAME[c], n=int(cm[c].sum()),
                         precision=round(prec, 4), recall=round(rec, 4),
                         f1=round(f1, 4)))
    acc = cm.diagonal().sum() / cm.sum()
    return cm, rows, acc, np.mean([r["f1"] for r in rows])


def binary_views(df, col):
    """三种二分口径：neu并入neg / neu并入pos / 剔除neu(仅正负)。"""
    y = df["y_cls"].to_numpy()
    p = df[col].to_numpy().astype(int)
    out = {}
    # (a) neu -> neg  (neg/neu vs pos)
    yt = (y == 2).astype(int)
    pt = (p == 2).astype(int)
    out["neu并入neg(非正 vs 正)"] = dict(acc=round(float((yt == pt).mean()), 4),
                                  n=int(len(y)))
    # (b) neu -> pos  (neg vs neu/pos)
    yt = (y == 0).astype(int)
    pt = (p == 0).astype(int)
    out["neu并入pos(负 vs 非负)"] = dict(acc=round(float((yt == pt).mean()), 4),
                                  n=int(len(y)))
    # (c) 剔除 neu
    m = y != 1
    if m.sum():
        out["剔除neu(仅neg vs pos)"] = dict(
            acc=round(float(((y[m] == 2) == (p[m] == 2)).mean()), 4), n=int(m.sum()))
    return out


def neutral_binary(df):
    """中性 vs 非中性 二分类质量（用 prob_neu 与 -|score| 两种打分）。"""
    y = df["y_cls"].to_numpy()
    t = (y == 1).astype(int)
    res = {}
    if "prob_neu" in df.columns:
        s = df["prob_neu"].to_numpy(dtype=float)
        res["head_prob_neu_AUC"] = round(roc_auc(t, s), 4)
        res["head_prob_neu_阈值扫描"] = best_thr_acc(t, s)
    if "score_theta" in df.columns:
        s = -np.abs(df["score_theta"].to_numpy(dtype=float))
        res["|theta|_AUC"] = round(roc_auc(t, s), 4)
        res["|theta|_阈值扫描"] = best_thr_acc(t, s)
    res["预测为neu的比例"] = round(float((df["pred_head"] == 1).mean()), 4)
    nz = t == 0
    if nz.sum():
        y2 = y[nz]
        p2 = df["pred_head"].to_numpy()[nz]
        res["非中性样本极性ACC"] = round(float(((y2 == 2) == (p2 == 2)).mean()), 4)
    res["中性样本占比"] = round(float(t.mean()), 4)
    return res


def roc_auc(t, s):
    order = np.argsort(s)
    r = np.empty(len(s), dtype=float)
    r[order] = np.arange(1, len(s) + 1)
    n1 = t.sum()
    n0 = len(t) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_thr_acc(t, s):
    """在 s 上取阈值判 t 的正类；同时给出该阈值下的平衡准确率（不受多数类支配）。"""
    cand = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 199)))
    b_acc = (0.0, None)
    b_bal = (0.0, None)
    for c in cand:
        pred = s >= c
        acc = float((pred == (t == 1)).mean())
        tp = float((pred & (t == 1)).sum())
        tn = float((~pred & (t == 0)).sum())
        sens = tp / max(t.sum(), 1)
        spec = tn / max((t == 0).sum(), 1)
        bal = 0.5 * (sens + spec)
        if acc > b_acc[0]:
            b_acc = (acc, float(c))
        if bal > b_bal[0]:
            b_bal = (bal, float(c))
    return dict(最优阈值ACC=round(b_acc[0], 4), 该阈值=round(b_acc[1], 4) if b_acc[1]
                is not None else None, 最优平衡ACC=round(b_bal[0], 4),
                平衡阈值=round(b_bal[1], 4) if b_bal[1] is not None else None)


def qwk(y, p, n_cls=3):
    """二次加权 Kappa（有序性度量；ACC 不区分跨类距离，QWK 区分）。"""
    W = np.zeros((n_cls, n_cls))
    for i in range(n_cls):
        for j in range(n_cls):
            W[i, j] = ((i - j) ** 2) / ((n_cls - 1) ** 2)
    O = np.zeros((n_cls, n_cls))
    for t, q in zip(y, p):
        O[t, q] += 1
    O = O / O.sum()
    hr = O.sum(axis=1)
    hc = O.sum(axis=0)
    E = np.outer(hr, hc)
    return float(1 - (W * O).sum() / (W * E).sum())



def strata(df):
    """按 |y| 的离散层级（1/3 粒度）分层看 ACC。"""
    y = df["y_cls"].to_numpy()
    a = np.abs(df["y_reg"].to_numpy(dtype=float))
    p = df["pred_head"].to_numpy().astype(int)
    lv = np.round(a * 3).astype(int)
    out = []
    for k in sorted(set(lv.tolist())):
        m = lv == k
        out.append(dict(强度=f"|y|={k/3:.3f}" if k else "|y|=0(中性)",
                        n=int(m.sum()),
                        ACC=round(float((y[m] == p[m]).mean()), 4),
                        对应类别="_".join(sorted(set(
                            CLS_NAME[c] for c in y[m]))),
                        占全样本比=round(float(m.mean()), 4)))
    return out


def acc2_views(df):
    """Acc-2 报法（**固化**，2026-09-24）：

    文献口径两种设置：Acc-2(负/正) 只在 |y|>0 上判正负；Acc-2(负/非负) 判 y<0。
    实测（B 集成 / test）：
        回归分数 θ=0            : 0.8489 / 0.8074
        valid 选优 θ(−0.26)     : 0.8489 / 0.8281
        **分类头概率构造**         : --- / **0.8514**   ← 主报口径
    结论：Acc-2 主报“分类头概率构造”（免调参、稳健），回归分数口径作对照。
    """
    y = df["y_reg"].to_numpy(dtype=float)
    s = df["score_theta"].to_numpy(dtype=float)
    m = np.abs(y) > 1e-6
    out = {}
    out["基线(负/正)"] = round(float(max((y[m] > 0).mean(), (y[m] < 0).mean())), 4)
    out["基线(负/非负)"] = round(float(max((y < 0).mean(), (y >= 0).mean())), 4)
    out["回归θ0(负/正)"] = round(float(((y[m] > 0) == (s[m] > 0)).mean()), 4)
    out["回归θ0(负/非负)"] = round(float(((y < 0) == (s < 0)).mean()), 4)
    if {"prob_neg", "prob_neu", "prob_pos"} <= set(df.columns):
        p = df[["prob_neg", "prob_neu", "prob_pos"]].to_numpy(dtype=float)
        out["分类头(负/正)"] = round(float(((y[m] > 0) == (p[m, 2] > p[m, 0])).mean()), 4)
        out["分类头(负/非负)"] = round(
            float(((y < 0) == (p[:, 0] > p[:, 1] + p[:, 2])).mean()), 4)
    return out


def y_map(df):
    """连续标签 -> 分类标签 的映射核对（确认 类=符号(y)）。"""
    y = df["y_cls"].to_numpy()
    a = df["y_reg"].to_numpy(dtype=float)
    lv = np.round(a * 3).astype(int)
    rows = []
    for k in sorted(set(lv.tolist()), key=abs):
        m = lv == k
        cnt = {CLS_NAME[c]: int((y[m] == c).sum()) for c in range(3)}
        rows.append(dict(y值=round(k / 3, 4), n=int(m.sum()), **cnt))
    return rows



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True,
                    help="csv 或 csv:tag 形式，可多个")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    allout = {}
    for spec in args.preds:
        # 允许 "path:tag"；盘符里的冒号（D:\）要跳过
        start = 2 if len(spec) > 1 and spec[1] == ":" else 0
        i = spec.find(":", start)
        if i > 0:
            path = spec[:i]
            tag = spec[i + 1:]
        else:
            path, tag = spec, os.path.splitext(os.path.basename(spec))[0]
        df = load(path)
        o = dict(n=int(len(df)))
        cm_h, rows_h, acc_h, f1_h = per_class(df, "pred_head")
        o["head"] = dict(acc=round(float(acc_h), 4), macro_f1=round(float(f1_h), 4),
                         per_class=rows_h, cm=cm_h.tolist())
        if "pred_theta" in df.columns:
            cm_t, rows_t, acc_t, f1_t = per_class(df, "pred_theta")
            o["theta"] = dict(acc=round(float(acc_t), 4), macro_f1=round(float(f1_t), 4),
                              per_class=rows_t, cm=cm_t.tolist())
        y = df["y_cls"].to_numpy()
        p = df["pred_head"].to_numpy().astype(int)
        o["head"]["QWK"] = round(qwk(y, p), 4)
        o["head"]["近邻正确率(误差<=1类)"] = round(float((np.abs(y - p) <= 1).mean()), 4)
        cm = np.array(o["head"]["cm"])
        off = int(cm.sum() - cm.diagonal().sum())
        far = int(cm[0, 2] + cm[2, 0])
        o["head"]["错误总数"] = off
        o["head"]["跨类错误(neg<->pos)"] = far
        o["head"]["跨类错误占比"] = round(float(far / off), 4) if off else 0.0
        if "pred_theta" in df.columns:
            p2 = df["pred_theta"].to_numpy().astype(int)
            o["theta"]["QWK"] = round(qwk(y, p2), 4)
            hit = ((p == y) | (p2 == y))
            o["head"]["两口径oracle上界ACC"] = round(float(hit.mean()), 4)
        o["二分对照"] = binary_views(df, "pred_head")
        o["中性专项"] = neutral_binary(df)
        o["强度分层"] = strata(df)
        o["强度标签映射"] = y_map(df)
        o["Acc2"] = acc2_views(df)
        o["类别分布"] = {CLS_NAME[c]: int((df["y_cls"] == c).sum()) for c in range(3)}
        allout[tag] = o

    txt = json.dumps(allout, ensure_ascii=False, indent=2)
    print(txt)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for tag, o in allout.items():
            with open(os.path.join(args.out_dir, f"perclass_{tag}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(o, f, ensure_ascii=False, indent=2)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(txt)


if __name__ == "__main__":
    main()
