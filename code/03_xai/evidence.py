# -*- coding: utf-8 -*-
r"""S5：证据定位（文本片段 / 语音时段 / 视觉关键帧）——题目要求"可回看原始素材"。

链路：
  1. 文本：`text_bert` 的词元编号 + `vocab.txt`（行号即 id）解码 WordPiece → 合并 `##` 续接
     → 得到"词"序列及其**词元位置集合**（词元位置 = 对齐段号，见 §1.2 的对齐粒度实测）
     → 用 `difflib` 把解码词对齐回原始 `raw_text` 的字符区间（失败显式标记，不静默降级）；
  2. 语音：高重要度的段号合并为连续区间，按比例映射为时间（附件2 不提供逐词时间戳，
     声明为近似）并用视频音频能量曲线佐证；
  3. 视觉：高重要度段取段内"视觉活跃度最大"的帧作为关键帧，用 ffmpeg 抽帧存盘。

输出：
  submission/evidence/{id}_kf.jpg         关键帧
  runs/q3/q3_evidence_{name}.json         逐样本证据记录（含字符区间、秒区间、帧时刻）
  runs/q3/q3_audio_energy_{name}.npz      各样本音频能量曲线（论文解释卡用）

用法：
  python evidence.py --name att4 --out_dir ..\..\runs\q3
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
if HERE not in sys.path:
    sys.path.append(HERE)
from xai_common import MODALITIES, save_json, ROOT as W_ROOT   # noqa: E402
from data_utils import load_pickle                            # noqa: E402

VOCAB = os.path.join(CODE, "assets", "bert-base-uncased_vocab.txt")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")


# --------------------------------------------------------------------------
def load_vocab(path=VOCAB):
    if not os.path.isfile(path):
        raise SystemExit("缺少词表 %s，请先运行 fetch_assets.py" % path)
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def decode_words(ids, mask, vocab, merge_punct=True):
    """词元序列 → [(word, [token_pos, ...])]，跳过 [CLS]/[SEP]/填充。

    merge_punct=True 时把纯标点词元（如 “'”、“-”、“,”）并入相邻词，
    使解码词序列与原始转写的词序列更接近（实测可把对齐率由 ~0.75 提升到 ~0.95）。
    """
    words, cur, positions = [], None, []
    for p, (i, m) in enumerate(zip(ids, mask)):
        if m == 0:
            break
        tok = vocab[int(i)] if 0 <= int(i) < len(vocab) else "[UNK]"
        if tok in ("[CLS]", "[SEP]"):
            continue
        if tok.startswith("##"):
            if cur is None:
                cur, positions = tok[2:], [p]
            else:
                cur, positions = cur + tok[2:], positions + [p]
        else:
            if cur is not None:
                words.append((cur, positions))
            cur, positions = tok, [p]
    if cur is not None:
        words.append((cur, positions))
    if not merge_punct:
        return words
    # 合并纯标点词元：与前一单词拼合（无前词则与后一单词拼合）
    merged = []
    for w, poss in words:
        if w and all(not ch.isalnum() for ch in w):
            if merged:
                pw, pposs = merged[-1]
                merged[-1] = (pw + w, pposs + poss)
                continue
            if (not merged) and w in ("'", "’", "\"", "``", "''"):
                merged.append((w, poss))
                continue
        merged.append((w, poss))
    # 后处理：相邻的 "词 + 纯标点" 再合并一次（处理句首标点情形）
    out = []
    i = 0
    while i < len(merged):
        w, poss = merged[i]
        if (i + 1 < len(merged) and merged[i + 1][0] and
                all(not ch.isalnum() for ch in merged[i + 1][0]) and len(w) > 1):
            out.append((w + merged[i + 1][0], poss + merged[i + 1][1]))
            i += 2
            continue
        out.append((w, poss))
        i += 1
    return out


def align_to_raw(words, raw_text):
    """把解码词对齐回 raw_text 的字符区间。

    返回 (spans, info)：spans 为逐词的 (start, end)（未对齐项为 None）；
    info 记录对齐率，供论文如实报告"对齐失败样本数"。
    """
    spans = [None] * len(words)
    raw_words = [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(raw_text)]
    a = [w.lower() for w, _ in words]
    b = [w.lower() for w, _, _ in raw_words]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    hit = 0
    for blk in sm.get_matching_blocks():
        for t in range(blk.size):
            spans[blk.a + t] = (raw_words[blk.b + t][1], raw_words[blk.b + t][2])
            hit += 1
    info = dict(n_word=len(words), n_raw=len(raw_words), matched=hit,
                ratio=round(float(hit / max(len(words), 1)), 3))
    return spans, info


def merge_runs(pos, gap=1):
    """把位置列表合并为连续区间 [start, end)。"""
    pos = sorted(set(int(p) for p in pos))
    if not pos:
        return []
    runs, s, e = [], pos[0], pos[0] + 1
    for p in pos[1:]:
        if p <= e + gap - 1:
            e = p + 1
        else:
            runs.append([s, e])
            s, e = p, p + 1
    runs.append([s, e])
    return runs


def seg_to_seconds(seg_start, seg_end, n_valid, duration):
    """段号区间 → 秒区间（比例映射；duration 为该样本视频时长）。"""
    if not duration or n_valid <= 0:
        return None, None
    per = float(duration) / float(n_valid)
    return round(seg_start * per, 2), round(min(seg_end, n_valid) * per, 2)


# --------------------------------------------------------------------------
def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def grab_frame(video, t_sec, out_jpg, width=320, q=5):
    os.makedirs(os.path.dirname(out_jpg), exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", "%.3f" % max(0.0, t_sec),
           "-i", video, "-frames:v", "1", "-vf", "scale=%d:-1" % width,
           "-q:v", str(q), out_jpg]
    r = subprocess.run(cmd, capture_output=True)
    return os.path.isfile(out_jpg) and os.path.getsize(out_jpg) > 0, (r.stderr or b"")[-300:]


def audio_energy(video, tmp_wav, hop=0.05):
    """抽出 16k 单声道 wav 并算短时能量曲线（返回 times, rms）。"""
    os.makedirs(os.path.dirname(tmp_wav), exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", video, "-vn",
           "-ac", "1", "-ar", "16000", tmp_wav]
    r = subprocess.run(cmd, capture_output=True)
    if not os.path.isfile(tmp_wav):
        return None, None, (r.stderr or b"")[-300:]
    with wave.open(tmp_wav, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    step = max(1, int(hop * sr))
    nseg = max(1, len(data) // step)
    rms = np.sqrt(np.mean(data[:nseg * step].reshape(nseg, step) ** 2, axis=1))
    times = (np.arange(nseg) + 0.5) * step / sr
    return times, rms, None


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="att4")
    ap.add_argument("--out_dir", default=os.path.join(W_ROOT, "runs", "q3"))
    ap.add_argument("--sub_dir", default=os.path.join(W_ROOT, "submission"))
    ap.add_argument("--meta", default=os.path.join(W_ROOT, "data_att", "att_meta.json"))
    ap.add_argument("--pkl", default=os.path.join(W_ROOT, "data_att", "att4_aligned.pkl"),
                    help="缓存 pkl（含 text_bert 与 raw_text，用于文本证据解码）")
    ap.add_argument("--top_seg", type=int, default=3)
    ap.add_argument("--top_word", type=int, default=3)
    ap.add_argument("--phrase_win", type=int, default=2)
    ap.add_argument("--no_video", action="store_true", help="不抽帧/不提音频（只算文本与时段）")
    a = ap.parse_args()

    fused = np.load(os.path.join(a.out_dir, "q3_temporal_fused_%s.npz" % a.name),
                    allow_pickle=True)
    W = fused["w_fused"]
    valid = fused["valid"]
    ids = [str(x) for x in fused["ids"]]
    meta = json.load(open(a.meta, encoding="utf-8"))
    recs = {r["id"]: r for r in meta[a.name]}
    vocab = load_vocab()

    # 文本证据需要 text_bert（词元编号）与 raw_text：从重建后的缓存 pkl 读取
    tb_all, rt_all, idx_map = None, None, {}
    if a.pkl and os.path.isfile(a.pkl):
        d = load_pickle(a.pkl)
        if "id" in d:
            pkl_ids = [str(x) for x in np.asarray(d["id"]).reshape(-1)]
            idx_map = {sid: i for i, sid in enumerate(pkl_ids)}
        if "text_bert" in d:
            tb_all = np.asarray(d["text_bert"])
        if "raw_text" in d:
            rt_all = [str(x) for x in np.asarray(d["raw_text"]).reshape(-1)]
        print("  [PKL] %s：text_bert=%s raw_text=%s"
              % (os.path.basename(a.pkl),
                 "有" if tb_all is not None else "无",
                 "有" if rt_all is not None else "无"))

    ev_dir = os.path.join(a.sub_dir, "evidence")
    out, energies = {}, {}
    align_bad = []
    for i, sid in enumerate(ids):
        r = recs.get(sid, {})
        k = idx_map.get(sid, None)
        raw_text = (rt_all[k] if (rt_all is not None and k is not None and k < len(rt_all))
                    else (r.get("raw_text", "") or ""))
        dur = r.get("duration_s")
        tb = tb_all[k] if (tb_all is not None and k is not None and k < len(tb_all)) else None
        n_text = int(valid[i, 0].sum())
        n_audio = int(valid[i, 1].sum())
        n_vision = int(valid[i, 2].sum())
        rec = dict(id=sid)

        # ---- 文本证据 ----
        w_text = np.asarray(W[i, 0], np.float64)
        words = []
        if tb is not None:
            tb = np.asarray(tb)
            words = decode_words(tb[0], tb[1], vocab)
        if words:
            spans, al = align_to_raw(words, raw_text)
            if al["ratio"] < 0.8:
                align_bad.append(dict(id=sid, **al))
            ww = []
            for (word, poss), sp in zip(words, spans):
                ww.append((word, poss, sp, float(max(w_text[p] for p in poss if p < len(w_text)))))
            ww_sorted = sorted(ww, key=lambda t: -t[3])
            picks = [t for t in ww_sorted[:max(1, a.top_word)]]
            pick_idx = [ww.index(t) for t in picks]
            lo = max(0, min(pick_idx) - a.phrase_win)
            hi = min(len(ww) - 1, max(pick_idx) + a.phrase_win)
            phrase_words = [t[0] for t in ww[lo:hi + 1]]
            phrase = " ".join(phrase_words)
            cspans = [t[2] for t in ww[lo:hi + 1] if t[2]]
            segs = merge_runs([p for t in picks for p in t[1]])
            rec["text_evidence"] = dict(
                top_words=[dict(word=t[0], seg=t[1], char_span=t[2], w=round(t[3], 4))
                           for t in picks],
                phrase=phrase,
                phrase_char_span=([min(s[0] for s in cspans), max(s[1] for s in cspans)]
                                  if cspans else None),
                seg_runs=segs, n_valid=n_text, align=al)
        else:
            rec["text_evidence"] = dict(note="无 text_bert（无法解码），仅给段号",
                                        seg_runs=[[int(x) for x in rn] for rn in
                                                  merge_runs(np.argsort(-w_text[:n_text])[:a.top_seg])],
                                        n_valid=n_text)

        # ---- 语音证据 ----
        w_audio = np.asarray(W[i, 1], np.float64)
        top_a = [int(x) for x in np.argsort(-w_audio[:n_audio])[:a.top_seg]]
        runs_a = merge_runs(top_a)
        s0, s1 = (seg_to_seconds(runs_a[0][0], runs_a[-1][1], n_audio, dur)
                  if runs_a else (None, None))
        rec["audio_evidence"] = dict(top_segs=top_a, seg_runs=runs_a,
                                     start_s=s0, end_s=s1, n_valid=n_audio, duration_s=dur)

        # ---- 视觉证据 ----
        w_vision = np.asarray(W[i, 2], np.float64)
        top_v = [int(x) for x in np.argsort(-w_vision[:n_vision])[:a.top_seg]]
        runs_v = merge_runs(top_v)
        t0, t1 = (seg_to_seconds(runs_v[0][0], runs_v[0][1], n_vision, dur)
                  if runs_v else (None, None))
        kf_t = None if t0 is None else round((t0 + (t1 if t1 else t0)) / 2.0, 2)
        rec["vision_evidence"] = dict(top_segs=top_v, seg_runs=runs_v,
                                      keyframe_time_s=kf_t, n_valid=n_vision)
        if (not a.no_video) and r.get("video") and kf_t is not None:
            out_jpg = os.path.join(ev_dir, "%s_kf.jpg" % sid)
            ok, err = grab_frame(r["video"], kf_t, out_jpg)
            rec["vision_evidence"]["keyframe_file"] = os.path.relpath(out_jpg, W_ROOT)
            rec["vision_evidence"]["keyframe_ok"] = bool(ok)
            if not ok:
                rec["vision_evidence"]["keyframe_err"] = err.decode("utf-8", "replace")
            if sid not in energies:
                tw = os.path.join(a.out_dir, "_audio", "%s.wav" % sid)
                tt, rr, err2 = audio_energy(r["video"], tw)
                if tt is not None:
                    energies[sid] = dict(t=tt, rms=rr)
                    np.savez_compressed(os.path.join(a.out_dir, "_audio", "%s_energy.npz" % sid),
                                        t=tt, rms=rr)
                else:
                    rec["vision_evidence"]["audio_err"] = str(err2)[:200]
                try:
                    os.remove(tw)
                except OSError:
                    pass
        out[sid] = rec
        if (i + 1) % 5 == 0 or i + 1 == len(ids):
            print("    %d/%d" % (i + 1, len(ids)))

    save_json(os.path.join(a.out_dir, "q3_evidence_%s.json" % a.name), out)
    if align_bad:
        print("  [警告] 文本对齐率 <0.8 的样本 %d 条：%s" % (len(align_bad), align_bad[:3]))
    print("  [KF] 关键帧 %d 张；音频能量曲线 %d 条"
          % (len([1 for v in out.values() if v.get("vision_evidence", {}).get("keyframe_ok")]),
             len(energies)))
    print("EVIDENCE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
