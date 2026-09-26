# -*- coding: utf-8 -*-
r"""S6-a：典型样本解释卡（JSON + 三联图）。

三联图内容（论文 section5.3 典型样本解释卡）：
  (1) 三模态时间重要性条带（含有效位边界与 top 片段标注）；
  (2) 文本词着色条（按词级重要性深浅着色，凸显关键证据词）；
  (3) 关键帧缩略图 + 音频能量曲线（高亮证据时段）。

用法：
  python explain_card.py --name att4 --out_dir ..\..\runs\q3 --fig_dir ..\..\figs
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.append(HERE)
from style import apply_style, COLORS, MOD_COLORS, LABEL_COLORS   # noqa: E402
from xai_common import MODALITIES, save_json, ROOT                # noqa: E402

MOD_CN = {"text": "文本", "audio": "语音", "vision": "视觉"}


def load_all(out_dir, name):
    fused = np.load(os.path.join(out_dir, "q3_temporal_fused_%s.npz" % name), allow_pickle=True)
    shap = np.load(os.path.join(out_dir, "q3_shapley_%s.npz" % name), allow_pickle=True)
    ev = json.load(open(os.path.join(out_dir, "q3_evidence_%s.json" % name), encoding="utf-8"))
    return fused, shap, ev


def card_figure(sid, W, valid, rec, pi, score, prob, theta, energy, out_png,
                label_cn="", meta=None):
    plt = apply_style()
    fig = plt.figure(figsize=(9.2, 6.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.75, 1.15], hspace=0.55, wspace=0.22)

    # ---- (1) 三模态时间重要性条带 ----
    ax1 = fig.add_subplot(gs[0, :])
    for k, m in enumerate(MODALITIES):
        w = np.asarray(W[k], np.float64)
        n = int(valid[k].sum())
        ax1.bar(np.arange(len(w)) + (k - 1) * 0.0, w, width=0.62, bottom=k, color=MOD_COLORS[k],
                label="%s（pi=%.2f）" % (MOD_CN[m], pi[k]))
        ax1.axvline(n - 0.5, color="#666666", lw=0.8, ls=":")
        top = np.argsort(-w[:n])[:3]
        for p in top:
            ax1.text(p, k + w[p] + 0.06, "*", ha="center", va="bottom",
                     fontsize=6.5, color="#333333")
    ax1.set_yticks([0.5, 1.5, 2.5])
    ax1.set_yticklabels([MOD_CN[m] for m in MODALITIES])
    ax1.set_xlabel("对齐段号 $k$（文本为词元位置；虚线为该模态有效长度上界）")
    ax1.set_title("(a) 三模态时间重要性 $w^m_k$（*=top-3 关键片段）", loc="left")
    ax1.legend(loc="upper right", ncol=3)

    # ---- (2) 文本词着色 ----
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis("off")
    te = rec.get("text_evidence", {})
    words = te.get("top_words", [])
    phrase = te.get("phrase", "")
    span = te.get("phrase_char_span")
    raw = (meta or {}).get("raw_text", "")
    ax2.set_title("(b) 文本证据：高亮为模型关注词，方框为可回看的字符区间", loc="left")
    txt = raw if raw else phrase
    if txt:
        hi = {w["word"].lower() for w in words if w.get("word")}
        # 用等宽示意图：把原句逐词着色（步长按字符数估算，换行留出边距）
        import re
        x, y = 0.0, 0.74
        for tok in re.finditer(r"\S+", txt):
            word = tok.group(0)
            key = re.sub(r"[^A-Za-z0-9']", "", word).lower()
            hot = key in hi
            ax2.text(x, y, word, fontsize=9.5, va="center",
                     color=(COLORS["accent"] if hot else "#333333"),
                     weight=("bold" if hot else "normal"),
                     bbox=(dict(facecolor="#FBEAD6", edgecolor="none", pad=1.2) if hot else None),
                     transform=ax2.transAxes)
            x += 0.0092 * (len(word) + 1.35)
            if x > 0.995:
                x, y = 0.0, y - 0.26
    else:
        ax2.text(0.01, 0.5, "（该样本无 raw_text，无法回看原文）", transform=ax2.transAxes,
                 fontsize=9)
    if span:
        ax2.text(0.0, 0.06, "字符区间 [%d, %d)：%s" % (span[0], span[1], phrase),
                 transform=ax2.transAxes, fontsize=8.5, color=COLORS["accent"])

    # ---- (3) 关键帧 + 音频能量 ----
    ax3 = fig.add_subplot(gs[2, 0])
    kf = (rec.get("vision_evidence") or {}).get("keyframe_file")
    if kf and os.path.isfile(os.path.join(ROOT, kf)):
        img = plt.imread(os.path.join(ROOT, kf))
        ax3.imshow(img)
        ax3.set_title("(c) 视觉关键帧 t=%.2fs" % (rec["vision_evidence"].get("keyframe_time_s") or 0.0),
                      loc="left")
    else:
        ax3.text(0.5, 0.5, "无关键帧", ha="center", transform=ax3.transAxes)
        ax3.set_title("(c) 视觉关键帧", loc="left")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.grid(False)

    ax4 = fig.add_subplot(gs[2, 1])
    if energy is not None:
        t, rms = energy["t"], energy["rms"]
        ax4.plot(t, rms, color=COLORS["audio"], lw=1.2)
        ae = rec.get("audio_evidence") or {}
        if ae.get("start_s") is not None:
            ax4.axvspan(ae["start_s"], ae["end_s"], color=COLORS["audio"], alpha=0.18)
        ax4.set_xlabel("时间 / s")
        ax4.set_ylabel("音频能量")
        ax4.set_title("(d) 音频能量与证据时段（阴影）", loc="left")
    else:
        ax4.text(0.5, 0.5, "无音频能量", ha="center", transform=ax4.transAxes)
        ax4.set_title("(d) 音频能量", loc="left")

    sup = "预测：%s（强度 %+.3f，theta=%.3f，主模态 %s）" % (
        label_cn, score, theta, MOD_CN[MODALITIES[int(np.argmax(pi))]])
    fig.suptitle("解释卡 . 样本 %s %s" % (sid, sup), fontsize=11.5, y=0.985)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="att4")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "runs", "q3"))
    ap.add_argument("--fig_dir", default=os.path.join(ROOT, "figs"))
    ap.add_argument("--theta", type=float, default=0.325)
    ap.add_argument("--ids", default=None, help="逗号分隔；默认自动挑（正/中/负各一）")
    a = ap.parse_args()

    fused, shap, ev = load_all(a.out_dir, a.name)
    W, valid = fused["w_fused"], fused["valid"]
    ids = [str(x) for x in fused["ids"]]
    pi_all = shap["pi_soft"]
    score = shap["score"]
    prob = shap["prob"]
    LAB = ("Negative", "Neutral", "Positive")
    CN = {"Negative": "负向", "Neutral": "中性", "Positive": "正向"}
    pred = [LAB[i] for i in prob.argmax(1)]
    meta = {r["id"]: r for r in
            json.load(open(os.path.join(ROOT, "data_att", "att_meta.json"),
                           encoding="utf-8"))[a.name]}

    if a.ids:
        pick = [s.strip() for s in a.ids.split(",")]
    else:   # 自动挑：正/中/负各一条（优先解释指标最有说服力的：置信度高 + pi 集中）
        pick = []
        for lab in ("Positive", "Neutral", "Negative"):
            cand = [i for i, p in enumerate(pred) if p == lab]
            if not cand:
                continue
            best = max(cand, key=lambda i: (float(prob[i].max()) * float(np.max(pi_all[i])
                                                                          - np.min(pi_all[i]))))
            pick.append(ids[best])
    print("[PICK] %s" % pick)

    made = []
    cards = {}
    for sid in pick:
        i = ids.index(sid)
        energy = None
        ep = os.path.join(a.out_dir, "_audio", "%s_energy.npz" % sid)
        if os.path.isfile(ep):
            with np.load(ep) as z:
                energy = dict(t=z["t"], rms=z["rms"])
        card = dict(id=sid, pred_label=pred[i], pred_score=round(float(score[i]), 4),
                    prob={LAB[k]: round(float(prob[i, k]), 4) for k in range(3)},
                    confidence=round(float(prob[i].max()), 4), theta=a.theta,
                    main_modality=MODALITIES[int(np.argmax(pi_all[i]))],
                    modality_contribution={MODALITIES[k]: round(float(pi_all[i, k]), 4)
                                           for k in range(3)},
                    shapley_raw={MODALITIES[k]: round(float(shap["phi_mu_miss"][i, k]), 4)
                                 for k in range(3)},
                    temporal_importance={MODALITIES[k]: np.round(W[i, k], 4).tolist()
                                         for k in range(3)},
                    evidence=ev.get(sid, {}))
        png = os.path.join(a.fig_dir, "Q3_解释卡_%s_%s_组合图.png" % (sid, pred[i]))
        card_figure(sid, W[i], valid[i], ev.get(sid, {}), pi_all[i], float(score[i]),
                    prob[i], a.theta, energy, png, CN[pred[i]], meta.get(sid, {}))
        cards[sid] = card
        made.append(os.path.relpath(png, ROOT))
        print("    [CARD] %s -> %s" % (sid, os.path.relpath(png, ROOT)))

    save_json(os.path.join(a.out_dir, "q3_cards_%s.json" % a.name), cards)
    print("[SAVE] %d 张解释卡: %s" % (len(made), made))
    print("CARD_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
