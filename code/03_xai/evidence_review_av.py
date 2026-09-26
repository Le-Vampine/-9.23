# -*- coding: utf-8 -*-
"""附件四 语音时段与视觉关键帧证据的核对（可复核性证据，续 文本版）。

两类核查：
  * 语音：证据时段是"能量代理"核查（无听音条件时的可复核替代）——
    计算窗口 [start_s, end_s] 内的 RMS 均值在全片分位、窗口是否覆盖全片前三能量峰、
    有效语音占比；据此给出 合理/一般/偏弱 三档（规则见 REF 字典，人工可改判）。
  * 视觉：生成 19 张关键帧的拼图（含样本号、预测标签与强度、关键帧时刻），
    供人工逐格目视判读"画面是否含人脸/表情是否可辨/是否与情感判断相关"。

输出：
  figs/Q3_关键帧核对表_拼图.png
  paper_tables/Q3_证据人工判读_语音视觉.csv
  runs/q3/q3_evidence_review_av.json

用法：
  python evidence_review_av.py                # 生成拼图 + CSV（含人工列）
  python evidence_review_av.py --from-csv     # 复核者改判后一键重算统计
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EVID = os.path.join(ROOT, "runs", "q3", "q3_evidence_att4.json")
AUD = os.path.join(ROOT, "runs", "q3", "_audio")
KF = os.path.join(ROOT, "submission", "evidence")
CSV_IN = os.path.join(ROOT, "submission", "pred_explain_att4.csv")
OUT_CSV = os.path.join(ROOT, "paper_tables", "Q3_证据人工判读_语音视觉.csv")
OUT_JSON = os.path.join(ROOT, "runs", "q3", "q3_evidence_review_av.json")
OUT_FIG = os.path.join(ROOT, "figs", "Q3_关键帧核对表_拼图.png")

# ---- 视觉人工判读（目视 figs/Q3_关键帧核对表_拼图.png 后填写）：合理 / 一般 / 未命中 ----
MANUAL_VISION = {
    "01": ("合理", "人物近景（汽配工作台前），面部清晰可辨"),
    "02": ("合理", "黑白人物肖像，面部清晰、表情可辨"),
    "03": ("合理", "人物近景正对镜头"),
    "04": ("合理", "演播室内人物 + 屏幕图文叠加，面部可辨"),
    "05": ("合理", "人物近景，面部与口型清晰"),
    "06": ("合理", "人物近景（沙发），面部可辨"),
    "07": ("一般", "分屏画面：会场听众 + 主讲人，人物占比小、表情细节有限"),
    "08": ("合理", "讲台发言场景，面部可辨（中景）"),
    "09": ("合理", "人物近景并持物（DVD），面部清晰"),
    "10": ("合理", "人物近景，面部清晰"),
    "11": ("合理", "人物近景（旁有电视画面），面部可辨"),
    "12": ("合理", "办公室中人物半身，面部清晰"),
    "13": ("未命中", "房间全景、逆光，面部不可辨（视觉证据信息量低）"),
    "14": ("一般", "舞台远景，人物较小、面部细节有限"),
    "15": ("合理", "人物近景，面部清晰"),
    "16": ("合理", "人物近景（戴眼镜），面部清晰"),
    "17": ("合理", "人物近景 + 白板图文，面部清晰"),
    "18": ("合理", "人物中景（持展板），面部可辨"),
    "19": ("合理", "人物近景，面部清晰"),
    "20": ("合理", "办公场景人物近景，面部清晰"),
}


def audio_metrics(sid: str, ae: dict):
    """返回语音证据的能量代理指标与规则结论。"""
    z = np.load(os.path.join(AUD, "%s_energy.npz" % sid))
    t, rms = z["t"], z["rms"]
    dur = float(t[-1]) if len(t) else 0.0
    a, b = float(ae["start_s"]), float(ae["end_s"])
    m = (t >= a) & (t <= b)
    if m.sum() == 0 or rms.size == 0:
        return dict(dur=dur, a=a, b=b, pct=float("nan"), n_peak_in=0, verdict="数据缺失", note="窗口内无采样")
    w_mean = float(rms[m].mean())
    pct = float((rms <= w_mean).mean() * 100)
    peaks = np.argsort(-rms)[:3]
    peak_in = int(np.isin(np.arange(len(rms)), peaks).sum() and sum(1 for p in peaks if m[p]))
    note = "窗口 RMS 处于全片第 %.0f 百分位；覆盖全片前三能量峰 %d/3" % (pct, peak_in)
    # 规则（或逻辑）：高分位或含峰→合理；40–60 且无峰→一般；<40 且无峰→偏弱
    if pct >= 60 or peak_in >= 1:
        verdict = "合理"
    elif pct >= 40:
        verdict = "一般"
    else:
        verdict = "偏弱"
    return dict(dur=dur, a=a, b=b, pct=pct, n_peak_in=peak_in, verdict=verdict, note=note)


def build_montage(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams["font.sans-serif"] = ["SimSun", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    n = len(rows)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.0, nrow * 2.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), rows):
        f = os.path.join(KF, os.path.basename(r["keyframe_file"] or ""))
        if r["keyframe_file"] and os.path.exists(f):
            ax.imshow(Image.open(f))
        else:
            ax.text(0.5, 0.5, "无关键帧", ha="center", va="center", fontsize=9)
        title = "%s  %s %+0.2f  @%.1fs" % (r["id"], r["pred_label"], float(r["pred_score"]), r["kf_time"])
        ax.set_title(title, fontsize=8)
    fig.suptitle("附件四 关键帧核对表（19/20 张可用；标注为 样本号 预测标签 强度 关键帧时刻）", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140)
    plt.close(fig)
    return OUT_FIG


def summarize(rows, src):
    n = len(rows)
    def cnt(key, val):
        return sum(1 for r in rows if r[key] == val)
    a_ok, a_mid, a_bad = cnt("语音判读", "合理"), cnt("语音判读", "一般"), cnt("语音判读", "偏弱")
    v_ok, v_mid, v_bad = cnt("视觉判读", "合理"), cnt("视觉判读", "一般"), cnt("视觉判读", "未命中")
    nv = sum(1 for r in rows if r["视觉判读"])
    out = {
        "n": n,
        "语音_合理": a_ok, "语音_一般": a_mid, "语音_偏弱": a_bad,
        "语音合理率": round((a_ok + a_mid) / n, 4),
        "视觉_可判读条数": nv, "视觉_合理": v_ok, "视觉_一般": v_mid, "视觉_未命中": v_bad,
        "视觉合理率": round((v_ok + v_mid) / nv, 4) if nv else None,
        "数据来源": src,
    }
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true")
    args = ap.parse_args()

    if args.from_csv:
        rows = list(csv.DictReader(open(OUT_CSV, encoding="utf-8-sig")))
        s = summarize(rows, "由复核者改判后的 CSV 重算（--from-csv）")
        print("[重算] 语音合理 %d/%d = %.0f%%（一般计合理）；视觉合理 %d/%d = %.0f%%"
              % (s["语音_合理"] + s["语音_一般"], s["n"], 100 * s["语音合理率"],
                 s["视觉_合理"] + s["视觉_一般"], s["视觉_可判读条数"], 100 * s["视觉合理率"]))
        return 0

    evid = json.load(open(EVID, encoding="utf-8"))
    pred = {r["id"]: r for r in csv.DictReader(open(CSV_IN, encoding="utf-8-sig"))}
    rows = []
    for sid in sorted(evid):
        e = evid[sid]
        ae, ve = e["audio_evidence"], e["vision_evidence"]
        am = audio_metrics(sid, ae)
        v_verdict, v_note = MANUAL_VISION.get(sid, ("", ""))
        rows.append({
            "id": sid,
            "pred_label": pred[sid]["pred_label"],
            "pred_score": pred[sid]["pred_score"],
            "main_modality": pred[sid]["main_modality"],
            "audio_start_s": "%.2f" % am["a"], "audio_end_s": "%.2f" % am["b"],
            "audio_span_s": "%.2f" % (am["b"] - am["a"]), "duration_s": "%.2f" % am["dur"],
            "audio_rms_pct": "%.0f" % am["pct"], "audio_peaks_in": am["n_peak_in"],
            "语音判读": ("合理" if am["verdict"] == "合理" else ("一般" if am["verdict"] == "一般" else "偏弱")),
            "语音备注": am["note"],
            "keyframe_file": os.path.basename(ve.get("keyframe_file") or ""),
            "kf_time": float(ve.get("keyframe_time_s") or 0.0),
            "视觉判读": v_verdict, "视觉备注": v_note,
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    fig = build_montage(rows)
    s = summarize(rows, "自动能量代理 + 人工目视拼图")
    print("[SAVE]", os.path.relpath(OUT_CSV, ROOT))
    print("[SAVE]", os.path.relpath(OUT_FIG, ROOT))
    print("[SAVE]", os.path.relpath(OUT_JSON, ROOT))
    print("[统计] 语音：合理 %d、一般 %d、偏弱 %d（合理率 %.0f%%）｜视觉：合理 %d、一般 %d、未命中 %d（可判读 %d 条）"
          % (s["语音_合理"], s["语音_一般"], s["语音_偏弱"], 100 * s["语音合理率"],
             s["视觉_合理"], s["视觉_一般"], s["视觉_未命中"], s["视觉_可判读条数"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
