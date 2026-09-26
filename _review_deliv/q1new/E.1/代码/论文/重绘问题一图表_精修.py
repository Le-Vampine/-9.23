"""Redraw Problem 1 figures from frozen E.1 evidence and original media.

The five stills in each case span the complete clip. Frames are decoded from
the corresponding MP4 near actual sampled PTS; they are not generated images.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import subprocess
import textwrap
import wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "论文" / "图表精修" / "论文插图"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "运行环境" / "matplotlib_cache_quality"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import numpy as np
from PIL import Image

font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
font_manager.fontManager.addfont(str(font_path))
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

C = {
    "ink": "#203645", "muted": "#556C7B", "line": "#DAE2E6",
    "blue": "#2E628A", "blue_light": "#EDF4F8",
    "teal": "#20776F", "teal_light": "#EBF6F2",
    "amber": "#AB6E32", "amber_light": "#FAF2E9",
    "red": "#B0534B", "red_light": "#F9EEEC",
    "slate": "#A8B6BF", "pale": "#E9EEF1", "purple": "#756994",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


SAMPLES = json.loads((ROOT / "结果" / "10_全量质量检查与论文图表" / "全量样本统计.json").read_text(encoding="utf-8"))
STEP4 = rows(ROOT / "日志" / "第4步文本一致性检查" / "原文词级时间_MFA候选及最终采用.csv")
ALIGNED = rows(ROOT / "结果" / "08_原文词级跨模态对齐" / "逐词对齐索引.csv")
MANIFEST = json.loads((ROOT / "输入" / "样本清单" / "样本清单.json").read_text(encoding="utf-8"))
BY_ID = {row["sample_id"]: row for row in SAMPLES}
TEXT_BY_ID = {row["sample_id"]: row["text"] for row in MANIFEST}


def clean_axes(ax, grid_axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(C["line"])
    ax.spines["bottom"].set_color(C["line"])
    ax.grid(axis=grid_axis, color=C["line"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=C["muted"], labelsize=9)


def save(fig, name):
    fig.savefig(OUT / name, dpi=320, bbox_inches="tight", pad_inches=0.12)
    fig.savefig((OUT / name).with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def box(ax, xy, wh, number, title, detail, fill):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.025",
                                facecolor=fill, edgecolor=C["line"], lw=0.9))
    ax.text(x + 0.027, y + h * 0.69, number, color=C["teal"], fontsize=10, weight="bold", va="center")
    ax.text(x + 0.075, y + h * 0.69, title, color=C["ink"], fontsize=11.2, weight="bold", va="center")
    ax.text(x + 0.027, y + h * 0.30, detail, color=C["muted"], fontsize=8.7, va="center")


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                 lw=1.15, color=C["slate"], connectionstyle="arc3,rad=0"))


def figure_flow():
    fig, ax = plt.subplots(figsize=(11.3, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, (.035, .78), (.93, .16), "输入", "附件1：100条原始视频＋label-100.xlsx",
        "核对编号、路径、text与标签；保留原视频和完整音频的时间轴", C["blue_light"])
    # The three lanes show that the official text and audio-derived transcript have different roles.
    box(ax, (.035, .48), (.28, .17), "①", "原文词与文本表示", "text → BERT → 每词768维", C["blue_light"])
    box(ax, (.36, .48), (.28, .17), "②", "语音识别与词定位", "WAV → Whisper → MFA → 原文词时间", C["amber_light"])
    box(ax, (.685, .48), (.28, .17), "③", "连续声画特征", "WAV 74维/10ms；视频52维/约10fps", C["teal_light"])
    for x in (.175, .50, .825):
        arrow(ax, (x, .78), (x, .66))
    box(ax, (.18, .18), (.64, .17), "④", "以原文词区间汇合三模态", "保存T768/A74/V52、真实时间、有效长度及掩码", C["pale"])
    for x, target in ((.175, .30), (.50, .50), (.825, .70)):
        arrow(ax, (x, .48), (target, .36))
    ax.text(.5, .075, "时间仅决定声画帧的归属；BERT始终编码题目原文text",
            ha="center", va="center", fontsize=10, color=C["muted"])
    save(fig, "fig01_flow.png")


def figure_evidence():
    stat = Counter(r["timestamp_status"] for r in STEP4)
    labels = ["A层  全文同词", "B层  局部唯一同词", "边界分歧＞0.5 s", "未匹配或映射不唯一", "MFA无结果", "Whisper空转写"]
    keys = ["mfa_primary_exact_transcript_auto_qc_pass", "mfa_primary_unique_word_overlap_auto_qc_pass",
            "mfa_whisper_discrepancy_over_0_5s_null_primary", "official_word_unmatched_or_ambiguous_in_asr",
            "mfa_alignment_output_missing", "asr_empty_no_timestamp"]
    values = [stat[k] for k in keys]
    colors = [C["blue"], C["teal"], C["amber"], C["slate"], "#B9C4CC", C["red"]]
    diffs = np.array([float(r["mfa_whisper_max_edge_difference_s"]) for r in STEP4
                      if r["mfa_candidate_start_time_wav_s"] and r["mfa_whisper_max_edge_difference_s"]], dtype=float)
    fig = plt.figure(figsize=(11.2, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], left=.06, right=.98, top=.88, bottom=.18, wspace=.31)
    ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])
    yp = np.arange(len(labels))
    ax.barh(yp, values, color=colors, height=.54)
    ax.set_yticks(yp, labels, fontsize=9.2, color=C["ink"]); ax.invert_yaxis()
    ax.set_xlim(0, 1010); ax.set_xlabel("原文词数", color=C["muted"], fontsize=9)
    for y, n in zip(yp, values):
        ax.text(n + 10, y, str(n), va="center", color=C["ink"], fontsize=9.3, weight="bold")
    ax.set_title("(a) 原文词时间的互斥状态", loc="left", fontsize=11.5, color=C["ink"], pad=10)
    clean_axes(ax)
    xs = np.sort(diffs)
    ys = np.arange(1, len(xs) + 1) / len(xs) if len(xs) else []
    bx.fill_between(xs, ys, color=C["blue_light"], alpha=.75)
    bx.plot(xs, ys, color=C["blue"], lw=2.25)
    bx.axvline(.5, color=C["red"], lw=1.3, ls="--")
    within = np.mean(diffs <= .5)
    bx.text(.52, .14, f"0.5 s筛查线\n候选词中≤阈值 {within:.1%}", color=C["red"], fontsize=8.8, linespacing=1.4)
    bx.set_xlim(0, min(2, max(1.25, np.quantile(xs, .995) if len(xs) else 1.25)))
    bx.set_ylim(0, 1.02)
    bx.set_xlabel("MFA与Whisper任一词边界的最大差值 / s", fontsize=9, color=C["muted"])
    bx.set_ylabel("候选词累计比例", fontsize=9, color=C["muted"])
    bx.set_title("(b) MFA与Whisper的最大边界差", loc="left", fontsize=11.5, color=C["ink"], pad=10)
    clean_axes(bx, "both")
    fig.text(.06, .965, "1,421 / 1,934 词获得主时间（73.47%）", fontsize=13, weight="bold", color=C["ink"])
    save(fig, "fig02_evidence.png")


def figure_coverage():
    fig, ax = plt.subplots(figsize=(11.2, 3.15))
    fig.subplots_adjust(left=.18, right=.96, top=.84, bottom=.22)
    vals = [[1934, 0, 0], [1421, 0, 513], [1228, 193, 513]]
    names = ["文本向量", "主时间与声学", "区间内有效视觉"]
    y = np.arange(3)
    for i, (good, noface, notime) in enumerate(vals):
        ax.barh(i, good, color=C["teal"], height=.59)
        if noface: ax.barh(i, noface, left=good, color=C["amber"], height=.59)
        if notime: ax.barh(i, notime, left=good+noface, color=C["pale"], height=.59)
        ax.text(good/2, i, f"{good:,}  ({good/1934:.1%})", ha="center", va="center",
                color="white", fontsize=10, weight="bold")
        if noface: ax.text(good+noface/2, i, str(noface), ha="center", va="center", color=C["ink"], fontsize=9)
        if notime: ax.text(good+noface+notime/2, i, str(notime), ha="center", va="center", color=C["muted"], fontsize=9)
    ax.set_yticks(y, names, color=C["ink"]); ax.invert_yaxis()
    ax.set_xlim(0, 1934); ax.set_xticks([0, 500, 1000, 1500, 1934]); ax.set_xlabel("原文词数", color=C["muted"])
    clean_axes(ax)
    fig.legend(handles=[Patch(color=C["teal"], label="可用"), Patch(color=C["amber"], label="有时间、无有效视觉"),
                        Patch(color=C["pale"], label="无主时间")],
               loc="upper center", bbox_to_anchor=(.59,.98), ncol=3, frameon=False, fontsize=8.6)
    save(fig, "fig03_coverage.png")


def figure_distribution():
    fig = plt.figure(figsize=(11.2, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, .92], left=.09, right=.97, top=.88, bottom=.16, wspace=.31)
    ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])
    types = {"exact_text": (C["blue"], "全文同词"), "text_mismatch": (C["teal"], "转写不一致"),
             "empty_asr": (C["red"], "空转写")}
    for kind, (color, label) in types.items():
        subset = [r for r in SAMPLES if r["transcript_group"] == kind]
        xx = [r["timed_words"]/r["word_count"] for r in subset]
        yy = [r["visual_words"]/r["word_count"] for r in subset]
        ax.scatter(xx, yy, s=[25+1.25*r["word_count"] for r in subset], c=color,
                   alpha=.62, edgecolor="white", linewidth=.7, label=f"{label}（{len(subset)}）")
    ax.plot([0,1],[0,1], color=C["line"], lw=1.0, ls="--", zorder=0)
    for short, pre, suf, dx, dy in [("A", "-vxjVxOeScU", "4", -.13, -.09),
                                    ("B", "-UuX1xuaiiE", "6", -.10, -.13),
                                    ("C", "-HwX2H8Z4hY", "9", .015, .035),
                                    ("D", "-yRb-Jum7EQ", "1", .025, .04)]:
        row = next(r for r in SAMPLES if r["sample_id"].startswith(pre) and r["sample_id"].endswith(suf))
        x, y = row["timed_words"]/row["word_count"], row["visual_words"]/row["word_count"]
        ax.annotate(short, (x,y), xytext=(x+dx,y+dy), fontsize=9, color=C["ink"], weight="bold",
                    arrowprops={"arrowstyle":"-", "color":C["muted"], "lw":.7})
    ax.set_xlim(-.03,1.06); ax.set_ylim(-.03,1.06)
    ax.set_xlabel("主时间词数 / 原文词数", color=C["muted"], fontsize=9)
    ax.set_ylabel("有效视觉词数 / 原文词数", color=C["muted"], fontsize=9)
    ax.set_title("(a) 100条样本的双模态词覆盖", loc="left", fontsize=11.5, color=C["ink"], pad=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False, markerscale=.7)
    clean_axes(ax, "both")
    time_ratio=np.array([r["timed_words"]/r["word_count"] for r in SAMPLES])
    vis_ratio=np.array([r["visual_words"]/r["word_count"] for r in SAMPLES])
    for ratio, color, label in [(time_ratio,C["blue"],"主时间"),(vis_ratio,C["teal"],"有效视觉")]:
        xs=np.r_[0,np.sort(ratio),1]
        ys=np.r_[0,np.arange(1,len(ratio)+1)/len(ratio),1]
        bx.step(xs,ys,where="post",color=color,lw=2.2,label=label)
    bx.set_xlim(0,1); bx.set_ylim(0,1.02)
    bx.set_xticks(np.linspace(0,1,6)); bx.set_yticks(np.linspace(0,1,6))
    bx.set_xlabel("原文词覆盖率", color=C["muted"], fontsize=9)
    bx.set_ylabel("累计样本比例", color=C["muted"], fontsize=9)
    bx.set_title("(b) 覆盖率的样本累积分布", loc="left", fontsize=11.5, color=C["ink"], pad=10)
    bx.legend(loc="upper left", fontsize=8.8, frameon=False)
    clean_axes(bx,"both")
    save(fig,"fig04_distribution.png")


def wav_envelope(path: Path):
    with wave.open(str(path), "rb") as w:
        sr=w.getframerate(); n=w.getnframes(); signal=np.frombuffer(w.readframes(n),dtype="<i2").astype(np.float32)/32768
    stride=max(1,int(sr*.004)); end=(len(signal)//stride)*stride
    chunks=signal[:end].reshape(-1,stride)
    t=(np.arange(len(chunks))+.5)*stride/sr
    return t,chunks.min(axis=1),chunks.max(axis=1),len(signal)/sr


def video_path(sample_id):
    group, clip=sample_id.rsplit("$_$",1)
    return ROOT/"输入"/"原始视频"/group/(clip+".mp4")


def frame_image(path: Path,t:float):
    cmd=["ffmpeg","-hide_banner","-loglevel","error","-i",str(path),"-ss",f"{max(0,t):.3f}",
         "-frames:v","1","-vf","scale=520:-2","-f","image2pipe","-vcodec","png","-"]
    out=subprocess.run(cmd,capture_output=True,check=True,timeout=30).stdout
    if not out: raise RuntimeError(f"No decoded frame: {path}, {t}")
    return Image.open(io.BytesIO(out)).convert("RGB")


def pick_frames(sample_id, aligned_words, face_npz, count=5, require_word=False, max_time_s=None):
    time=face_npz["frame_time_wav_s"]; source=face_npz["frame_time_source_s"]
    face=face_npz["face_valid_mask"].astype(bool)
    if require_word:
        candidates=[w for w in aligned_words if w["word_start_wav_s"] and int(w["visual_valid_mask"])==1]
        selected=[candidates[round(i*(len(candidates)-1)/(count-1))] for i in range(count)]
        result=[]
        for w in selected:
            a,b=float(w["word_start_wav_s"]),float(w["word_end_wav_s"])
            idx=np.flatnonzero((time>=a)&(time<b)&face)
            if not len(idx): raise ValueError(f"No valid face in selected word: {sample_id} {w['official_word']}")
            pick=idx[np.argmin(abs(time[idx]-(a+b)/2))]
            result.append((float(source[pick]), f"#{w['official_word_index']} {w['official_word']}", bool(face[pick])))
        return result
    eligible=np.flatnonzero(time < (max_time_s-.03 if max_time_s is not None else np.inf))
    if len(eligible)<count: raise ValueError(f"Too few video frames inside WAV span: {sample_id}")
    idx=eligible[np.linspace(.05*(len(eligible)-1),.95*(len(eligible)-1),count).round().astype(int)]
    return [(float(source[i]),"源视频帧",bool(face[i])) for i in idx]


def figure_case(sample_id, name, case_type):
    info=BY_ID[sample_id]
    ws=[r for r in ALIGNED if r["sample_id"]==sample_id]
    wp=ROOT/"输入"/"音频WAV"/(sample_id+".wav")
    vp=video_path(sample_id)
    vis=np.load(ROOT/"结果"/"07_FaceLandmarker52"/"逐样本"/(sample_id+".npz"))
    t,lo,hi,duration=wav_envelope(wp)
    chosen=pick_frames(sample_id,ws,vis,require_word=case_type in {"A","B"},max_time_s=duration)
    fig=plt.figure(figsize=(11.4,6.9))
    gs=fig.add_gridspec(4,1,height_ratios=[1.02,.68,.72,1.62],left=.09,right=.985,top=.865,bottom=.145,hspace=.50)
    ax=fig.add_subplot(gs[0]); bx=fig.add_subplot(gs[1],sharex=ax)
    mx=fig.add_subplot(gs[2]); frames_gs=gs[3].subgridspec(1,5,wspace=.12)
    stats=f"原文 {info['word_count']}词    主时间 {info['timed_words']}词    区间视觉 {info['visual_words']}词"
    title={"A":"A层  整句时间与视觉均有效", "B":"B层  转写不一致句的局部恢复",
           "visual_missing":"词时间有效、视觉无效", "empty":"空转写样本的时间缺失"}[case_type]
    fig.text(.075,.968,title,fontsize=13.4,weight="bold",color=C["ink"])
    fig.text(.985,.968,sample_id.replace("$", r"\$"),fontsize=8.9,color=C["muted"],ha="right")
    fig.text(.075,.925,stats,fontsize=9.2,color=C["muted"])

    ax.fill_between(t,lo,hi,color="#AFC5D1",lw=0)
    ax.axhline(0,color=C["muted"],lw=.7)
    ax.set_xlim(0,duration)
    ax.set_ylabel("WAV幅值",color=C["muted"],fontsize=9)
    ax.set_title("(a) 完整WAV波形与抽样视频时刻",loc="left",fontsize=10.6,color=C["ink"],pad=6)
    ax.tick_params(labelbottom=False)
    clean_axes(ax)
    for i,(pt,_,_) in enumerate(chosen,1):
        ax.axvline(pt,color=C["teal"],lw=.8,ls="--",alpha=.65)
        ax.text(pt,ax.get_ylim()[1]*.88,str(i),ha="center",va="center",color=C["teal"],fontsize=8.4,
                bbox={"boxstyle":"circle,pad=.12","fc":"white","ec":C["teal"],"lw":.8})

    bx.set_ylim(0,1);bx.set_yticks([])
    bx.set_title("(b) 原文词的MFA主时间区间" if case_type!="empty"
                 else "(b) 原文词的主时间状态",loc="left",fontsize=10.6,color=C["ink"],pad=6)
    clean_axes(bx)
    if case_type=="empty":
        bx.text(duration/2,.52,"Whisper空转写  ·  49个原文词均无主时间",
                ha="center",va="center",fontsize=10.7,color=C["red"],weight="bold")
    else:
        for w in ws:
            if not w["word_start_wav_s"]: continue
            a,b=float(w["word_start_wav_s"]),float(w["word_end_wav_s"])
            tier=w["timestamp_evidence_tier"]
            color=C["blue"] if tier=="A_exact_transcript" else C["amber"]
            bx.add_patch(Rectangle((a,.26),b-a,.47,facecolor=color,edgecolor="white",lw=.35))
            if b-a>=.14:
                label=w["official_word"] if b-a>=.38 and len(w["official_word"])<=11 else w["official_word_index"]
                bx.text((a+b)/2,.495,label,ha="center",va="center",fontsize=7.2,color="white",weight="bold",clip_on=True)
        missing=[f"{w['official_word_index']} {w['official_word']}" for w in ws if not w["word_start_wav_s"]]
        if missing:
            bx.text(.02*duration,.91,"无主时间："+"、".join(missing[:6])+(" 等" if len(missing)>6 else ""),
                    fontsize=8.4,color=C["red"],va="center")
    for pt,_,_ in chosen: bx.axvline(pt,color=C["teal"],lw=.7,ls="--",alpha=.55)

    keys=[("文本","text_valid_mask",C["blue"]),("时间","word_time_valid_mask",C["teal"]),
          ("声学","audio_valid_mask",C["amber"]),("视觉","visual_valid_mask",C["purple"])]
    if case_type=="empty":
        mx.set_xlim(0,len(ws)*1.08); mx.set_ylim(-.5,3.5)
        for i,(label,key,color) in enumerate(keys):
            n=sum(int(w[key]) for w in ws); y=3-i
            mx.barh(y,len(ws),height=.65,color=C["pale"])
            if n: mx.barh(y,n,height=.65,color=color)
            mx.text(len(ws)*1.01,y,f"{n}/{len(ws)}",va="center",fontsize=8.6,
                    color=C["muted"])
        mx.set_yticks([3,2,1,0],[x[0] for x in keys],fontsize=8.6,color=C["muted"])
        mx.set_xticks([])
        mx.tick_params(axis="y",length=0)
        mx.spines[:].set_visible(False)
    else:
        mx.set_xlim(-.6,len(ws)-.4);mx.set_ylim(-.6,3.6)
        for j,w in enumerate(ws):
            for i,(_,key,color) in enumerate(keys):
                good=int(w[key])==1
                mx.add_patch(Rectangle((j-.45,3-i-.40),.90,.80,facecolor=color if good else C["pale"],
                                       edgecolor="white",lw=.45))
        mx.set_yticks(range(4),[x[0] for x in keys][::-1],fontsize=8.6,color=C["muted"])
        mx.set_xticks(np.arange(len(ws)),[w["official_word_index"] for w in ws],fontsize=7.2)
        mx.tick_params(axis="both",length=0)
        mx.spines[:].set_visible(False)
    mx.set_title("(c) 逐原文词有效性：有色为有效，浅灰为缺失",loc="left",fontsize=10.5,color=C["ink"],pad=5)
    for j,(pt,label,face) in enumerate(chosen):
        fx=fig.add_subplot(frames_gs[j])
        fx.imshow(frame_image(vp,pt));fx.axis("off")
        fx.set_title(f"{j+1}  t≈{pt:.2f} s",fontsize=8.8,color=C["ink"],pad=4)
        tag=label if case_type in {"A","B"} else ("人脸有效" if face else "人脸无效")
        if case_type=="empty": tag="无词级归属"
        fx.text(.5,-.065,tag,transform=fx.transAxes,ha="center",va="top",fontsize=8.2,
                color=C["teal"] if face else C["red"])
    if case_type=="empty":
        excerpt=" ".join(w["official_word"] for w in ws[:14])+" …"
        source_line=f"原文词序列节选（前14/49词，仅文本）：{excerpt}"
    else:
        source_line="原文词序列："+"  ".join(f"{w['official_word_index']} {w['official_word']}" for w in ws)
    wrapped=textwrap.wrap(source_line,width=112,break_long_words=False,break_on_hyphens=False)
    fig.text(.075,.055,"\n".join(wrapped[:3]),fontsize=8.0,color=C["muted"],va="bottom",linespacing=1.3)
    save(fig,name)


def main():
    if len(SAMPLES)!=100 or len(STEP4)!=1934 or len(ALIGNED)!=1934:
        raise ValueError("Audited E.1 rows missing")
    flow = ROOT / "论文" / "图表精修" / "流程图_可编辑" / "preview.png"
    if not flow.is_file():
        raise FileNotFoundError(flow)
    shutil.copy2(flow, OUT / "fig01_flow.png")
    figure_evidence(); figure_coverage(); figure_distribution()
    cases=[("-vxjVxOeScU$_$4","fig05_case_A.png","A"),
           ("-UuX1xuaiiE$_$6","fig06_case_B.png","B"),
           ("-HwX2H8Z4hY$_$9","fig07_case_visual_missing.png","visual_missing"),
           ("-yRb-Jum7EQ$_$1","fig08_case_asr_empty.png","empty")]
    for sample_id,name,kind in cases:
        figure_case(sample_id,name,kind)
    print(json.dumps({"output":str(OUT),"figures":8,"sample_frames_per_case":5},ensure_ascii=False))


if __name__=="__main__":
    main()
