# -*- coding: utf-8 -*-
r"""S6-b：主入口 —— 生成附件4 的"预测 + 解释"提交文件与解释卡。

产物：
  submission/pred_explain_att4.csv        题目要求的附件4 预测与解释结果（每行一条样本）
  submission/explain_cards/{id}.json      逐样本完整解释卡（含 50 步时间重要性）
  submission/evidence/{id}_kf.jpg         关键帧（由 evidence.py 生成）

内建自查断言（任一不过即报错退出）：
  1. 行数与 id 顺序与附件4 严格一致；
  2. pred_label / pred_score 与 submission/pred_att4.csv 逐位一致（同一骨干、同一 θ）；
  3. 解释数值有限、π 和为 1、时间重要性非负；
  4. 证据区间合法（0 ≤ start < end ≤ 有效长度；秒 ≤ 视频时长）。

用法：
  python infer_explain_att4.py --out_dir ..\..\runs\q3 --sub_dir ..\..\submission
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)
from xai_common import (MODALITIES, load_backbone, load_split, make_batch,      # noqa: E402
                        predict_batch, save_json, q3_meta, ROOT)

LABELS = ("Negative", "Neutral", "Positive")
MOD_CN = {"text": "文本", "audio": "语音", "vision": "视觉"}


def probe_fps(video, cache_path):
    """用 ffmpeg 的 stderr 解析帧率（缓存复用）。"""
    cache = {}
    if os.path.isfile(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception:
            cache = {}
    key = os.path.basename(video)
    if key in cache:
        return cache[key]
    import imageio_ffmpeg
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", video],
                       capture_output=True)
    txt = (r.stderr or b"").decode("utf-8", "replace")
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", txt)
    fps = float(m.group(1)) if m else None
    cache[key] = fps
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    except OSError:
        pass
    return fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=os.path.join(ROOT, "data_att", "att4_aligned.pkl"))
    ap.add_argument("--ckpt_dir", default=os.path.join(ROOT, "runs", "ens_top2"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "runs", "q3"))
    ap.add_argument("--sub_dir", default=os.path.join(ROOT, "submission"))
    ap.add_argument("--theta", type=float, default=0.325)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    models, stats, theta_ck = load_backbone(a.ckpt_dir, a.device, verbose=False)
    split, key = load_split(a.pkl, stats)
    N = split["N"]
    ids = [str(x) for x in split["ids"]]
    print("[DATA] att4 N=%d  (ckpt θ 均值 %.3f，提交用 θ=%.3f)" % (N, theta_ck, a.theta))

    # ---------- 1) 复现预测 ----------
    b = make_batch(split, list(range(N)), a.device)
    out = predict_batch(models, b, target="mu")
    score = out["score"].detach().cpu().numpy()
    prob = out["prob"].detach().cpu().numpy()
    pred = [LABELS[i] for i in prob.argmax(1)]
    ref_path = os.path.join(a.sub_dir, "pred_att4.csv")
    assert os.path.isfile(ref_path), "缺少 %s（先跑 d1_recalib.py）" % ref_path
    ref = pd.read_csv(ref_path, dtype={"id": str})
    ref_ids = [str(x).zfill(2) if str(x).isdigit() else str(x) for x in ref["id"]]
    assert ref_ids == ids, "附件4 id 顺序不一致：%s" % ref_ids[:4]
    lab_eq = float((np.array(pred) == ref["pred_label"].values).mean())
    sc_diff = float(np.abs(score - ref["pred_score"].values).max())
    print("[CHECK] 与 pred_att4.csv：标签一致率 %.3f；分数最大差 %.1e" % (lab_eq, sc_diff))
    assert lab_eq == 1.0, "标签与 pred_att4.csv 不一致，禁止输出"
    assert sc_diff < 1e-3, "分数与 pred_att4.csv 不一致（差 %.4f）" % sc_diff

    # ---------- 2) 读解释与证据 ----------
    fused = np.load(os.path.join(a.out_dir, "q3_temporal_fused_att4.npz"), allow_pickle=True)
    shap = np.load(os.path.join(a.out_dir, "q3_shapley_att4.npz"), allow_pickle=True)
    met = np.load(os.path.join(a.out_dir, "q3_temporal_occ_att4.npz"), allow_pickle=True)
    ev = json.load(open(os.path.join(a.out_dir, "q3_evidence_att4.json"), encoding="utf-8"))
    faith_p = os.path.join(a.out_dir, "q3_faithfulness_att4.csv")
    faith = pd.read_csv(faith_p, dtype={"id": str}) if os.path.isfile(faith_p) else None
    meta = {r["id"]: r for r in json.load(
        open(os.path.join(ROOT, "data_att", "att_meta.json"), encoding="utf-8"))["att4"]}

    W = fused["w_fused"]
    W_att, W_ig, W_occ = fused["w_att"], fused["w_ig"], fused["w_occ"]
    valid = fused["valid"]
    pi = shap["pi_soft"]
    pi_relu = shap["pi_relu"]
    phi = shap["phi_mu_miss"]
    occ = met["w_occ"]
    fps_cache = os.path.join(a.out_dir, "video_fps.json")

    rows, cards = [], {}
    for i, sid in enumerate(ids):
        n_text = int(valid[i, 0].sum())
        n_audio = int(valid[i, 1].sum())
        n_vision = int(valid[i, 2].sum())
        e = ev.get(sid, {})
        m = meta.get(sid, {})
        # 断言：π 归一化与时间重要性合法性
        assert abs(float(pi[i].sum()) - 1.0) < 1e-6, "π 未归一化：%s" % sid
        assert np.isfinite(W[i]).all() and (W[i] >= -1e-9).all(), "时间重要性非法：%s" % sid

        te = e.get("text_evidence", {}) or {}
        tr = te.get("seg_runs") or [[None, None]]
        ae = e.get("audio_evidence", {}) or {}
        ar = ae.get("seg_runs") or [[None, None]]
        ve = e.get("vision_evidence", {}) or {}
        vr = ve.get("seg_runs") or [[None, None]]
        kf_t = ve.get("keyframe_time_s")
        fps = probe_fps(m["video"], fps_cache) if m.get("video") else None
        frame_idx = int(round(kf_t * fps)) if (kf_t is not None and fps) else None

        # 证据合法性断言
        if tr and tr[0][0] is not None:
            assert 0 <= tr[0][0] < max(1, n_text), "文本证据段号非法：%s" % sid
        if ae.get("end_s") is not None and m.get("duration_s"):
            assert ae["end_s"] <= m["duration_s"] + 0.05, "语音证据超视频时长：%s" % sid

        # 忠实性/充分性/稳定性/稀疏性（来自 metrics_xai.py，取融合解释那一行）
        compr = suff = stab = spar = None
        if faith is not None:
            sub = faith[(faith["id"] == sid) & (faith["source"] == "fused")]
            if len(sub):
                compr = float(sub["compr_mean"].iloc[0])
                suff = float(sub["suff_absmean"].iloc[0])
                spar = float(sub["sparsity_mean"].iloc[0])
        rep = json.load(open(os.path.join(a.out_dir, "q3_faithfulness_att4_report.json"),
                             encoding="utf-8")) if os.path.isfile(
            os.path.join(a.out_dir, "q3_faithfulness_att4_report.json")) else {}
        stab = (rep.get("stability", {}).get(sid, {}) or {}).get("stability_fused")

        phrase = te.get("phrase", "")
        row = dict(
            id=sid, pred_label=pred[i], pred_score=round(float(score[i]), 4),
            prob_neg=round(float(prob[i, 0]), 4), prob_neu=round(float(prob[i, 1]), 4),
            prob_pos=round(float(prob[i, 2]), 4),
            confidence=round(float(prob[i].max()), 4), theta=a.theta,
            main_modality=MODALITIES[int(np.argmax(pi[i]))],
            w_text=round(float(pi[i, 0]), 4), w_audio=round(float(pi[i, 1]), 4),
            w_vision=round(float(pi[i, 2]), 4),
            w_text_relu=round(float(pi_relu[i, 0]), 4),
            w_audio_relu=round(float(pi_relu[i, 1]), 4),
            w_vision_relu=round(float(pi_relu[i, 2]), 4),
            phi_text=round(float(phi[i, 0]), 4), phi_audio=round(float(phi[i, 1]), 4),
            phi_vision=round(float(phi[i, 2]), 4),
            text_evidence=phrase or te.get("note", ""),
            text_seg_start=tr[0][0], text_seg_end=tr[0][1],
            text_char_start=(te.get("phrase_char_span") or [None, None])[0],
            text_char_end=(te.get("phrase_char_span") or [None, None])[1],
            audio_start_s=ae.get("start_s"), audio_end_s=ae.get("end_s"),
            audio_seg_start=ar[0][0], audio_seg_end=ar[0][1],
            vision_seg_start=vr[0][0], vision_seg_end=vr[0][1],
            vision_frame_idx=frame_idx, vision_time_s=kf_t,
            vision_keyframe=ve.get("keyframe_file", ""),
            compr=None if compr is None else round(compr, 4),
            suff=None if suff is None else round(suff, 4),
            stability=stab, sparsity=None if spar is None else round(spar, 4),
        )
        rows.append(row)
        cards[sid] = dict(
            id=sid, pred_label=pred[i], pred_score=round(float(score[i]), 4),
            prob={LABELS[k]: round(float(prob[i, k]), 4) for k in range(3)},
            confidence=round(float(prob[i].max()), 4), theta=a.theta,
            main_modality=row["main_modality"],
            modality_contribution={MODALITIES[k]: round(float(pi[i, k]), 4) for k in range(3)},
            modality_contribution_relu={MODALITIES[k]: round(float(pi_relu[i, k]), 4)
                                        for k in range(3)},
            shapley_raw={MODALITIES[k]: round(float(phi[i, k]), 4) for k in range(3)},
            temporal_importance_fused={MODALITIES[k]: np.round(W[i, k], 4).tolist()
                                       for k in range(3)},
            temporal_importance_att={MODALITIES[k]: np.round(W_att[i, k], 4).tolist()
                                     for k in range(3)},
            temporal_importance_ig={MODALITIES[k]: np.round(W_ig[i, k], 4).tolist()
                                    for k in range(3)},
            temporal_importance_occ={MODALITIES[k]: np.round(W_occ[i, k], 4).tolist()
                                     for k in range(3)},
            gamma=fused["gamma"].tolist(),
            evidence=dict(text=te, audio=ae, vision=ve),
            raw_text=m.get("raw_text", ""), video=os.path.basename(m.get("video", "") or ""),
            duration_s=m.get("duration_s"),
            faithfulness=dict(comprehensiveness=row["compr"], sufficiency=row["suff"],
                              stability=stab, sparsity=row["sparsity"]))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(a.sub_dir, "pred_explain_att4.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    os.makedirs(os.path.join(a.sub_dir, "explain_cards"), exist_ok=True)
    for sid, c in cards.items():
        with open(os.path.join(a.sub_dir, "explain_cards", "%s.json" % sid), "w",
                  encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False, indent=1)
    save_json(os.path.join(a.out_dir, "q3_explain_meta.json"),
              dict(meta=q3_meta(a.ckpt_dir, models, a.theta, dict(pkl=os.path.relpath(a.pkl, ROOT))),
                   csv=os.path.relpath(out_csv, ROOT), n=N,
                   columns=list(df.columns)))
    print("[SAVE] %s（%d 行 × %d 列）" % (os.path.relpath(out_csv, ROOT), len(df), df.shape[1]))
    print("[SAVE] submission/explain_cards/*.json（%d 个）" % len(cards))
    cnt = df["main_modality"].value_counts().to_dict()
    print("[STAT] 主参考模态分布：%s" % cnt)
    print("INFER_EXPLAIN_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
