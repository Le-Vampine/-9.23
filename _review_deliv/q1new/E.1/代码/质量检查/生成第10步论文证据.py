"""Audit all 100 E.1 samples and prepare reproducible evidence for Problem 1.

Outputs a complete 100-sample table, machine-readable statistics, five figures,
and a typical-sample word/frame mapping. No model result is called ground truth.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import wave
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "结果" / "10_全量质量检查与论文图表"
FIG = OUT / "图"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "运行环境" / "matplotlib_cache_quality"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


MANIFEST = ROOT / "输入" / "样本清单" / "样本清单.json"
INVENTORY = ROOT / "输入" / "样本清单" / "核查报告.json"
MEDIA = ROOT / "日志" / "第2步视频检查与音频分离" / "第2步视频与音频分离核验.csv"
WORD_TIMES = ROOT / "日志" / "第4步文本一致性检查" / "原文词级时间_MFA候选及最终采用.csv"
TEXT_INDEX = ROOT / "结果" / "05_BERT_base_uncased" / "原文逐词索引.csv"
TEXT_MATRIX = ROOT / "结果" / "05_BERT_base_uncased" / "原文逐词BERT特征.npy"
AUDIO_SUMMARY = ROOT / "结果" / "06_Librosa74" / "逐样本统计.csv"
VISUAL_SUMMARY = ROOT / "结果" / "07_FaceLandmarker52" / "逐样本统计.csv"
ALIGN_SUMMARY = ROOT / "结果" / "08_原文词级跨模态对齐" / "逐样本对齐统计.csv"
ALIGN_INDEX = ROOT / "结果" / "08_原文词级跨模态对齐" / "逐词对齐索引.csv"
AUDIO_FRAMES = ROOT / "结果" / "06_Librosa74" / "逐样本"
VISUAL_FRAMES = ROOT / "结果" / "07_FaceLandmarker52" / "逐样本"
ALIGNED = ROOT / "结果" / "08_原文词级跨模态对齐" / "逐样本"
TYPICAL_PREFIX = "-vxjVxOeScU"
COLORS = {"ink": "#17243A", "blue": "#2176AE", "teal": "#1C947E", "orange": "#E29A32",
          "red": "#BF5547", "pale": "#EDF3F7", "gray": "#788594"}
FONT = font_manager.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = font_manager.FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def index_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result = {row["sample_id"]: row for row in rows}
    if len(result) != 100 or len(rows) != 100:
        raise ValueError("Expected 100 unique sample rows")
    return result


def group_by_id(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["sample_id"]].append(row)
    return groups


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptive(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    return {"min": float(values[0]), "median": float(statistics.median(values)),
            "mean": float(statistics.mean(values)), "max": float(values[-1])}


def set_style() -> None:
    plt.rcParams.update({"font.family": FONT.get_name(), "axes.unicode_minus": False,
                         "font.size": 10, "axes.labelcolor": COLORS["ink"],
                         "text.color": COLORS["ink"], "axes.edgecolor": "#B9C6D0",
                         "savefig.facecolor": "white", "figure.facecolor": "white"})


def save_figure(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_flow() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    def box(x, y, w, h, title, subtitle, color):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.13",
                               facecolor=color, edgecolor="none")
        ax.add_patch(patch)
        ax.text(x + w/2, y + h*0.63, title, ha="center", va="center", fontproperties=FONT_BOLD, fontsize=12)
        ax.text(x + w/2, y + h*0.25, subtitle, ha="center", va="center", fontproperties=FONT, fontsize=8.8)
    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15,
                                     color="#71879C", lw=1.6, connectionstyle="arc3,rad=0"))
    box(0.2, 3.65, 2.1, 1.15, "附件1原始输入", "100视频＋原文text＋标签", "#E7F0F8")
    box(3.0, 3.65, 2.25, 1.15, "完整音视频时间轴", "WAV首采样点与视频PTS", "#E7F0F8")
    box(6.0, 3.65, 2.3, 1.15, "词时间自动筛查", "Whisper→文本比对→MFA", "#FCEFD9")
    box(9.0, 3.65, 2.7, 1.15, "原文词序列", "1,934词；1,421个主时间", "#FCEFD9")
    box(0.55, 1.45, 2.8, 1.2, "文本分支", "bert-base-uncased · 768维", "#E7F0F8")
    box(4.0, 1.45, 2.8, 1.2, "音频分支", "Librosa · 74维 · 10 ms", "#E5F4F0")
    box(7.45, 1.45, 3.05, 1.2, "视觉分支", "Face Landmarker · 52维 · 约10 fps", "#E5F4F0")
    box(3.55, 0.1, 5.0, 0.65, "按词区间聚合 → 三模态矩阵、原始时间、有效长度和掩码", "", "#E8EDF3")
    for a,b in [((2.35,4.23),(2.9,4.23)),((5.3,4.23),(5.9,4.23)),((8.35,4.23),(8.9,4.23)),
                ((10.4,3.55),(1.95,2.75)),((10.5,3.55),(5.4,2.75)),((10.6,3.55),(8.95,2.75)),
                ((1.95,1.35),(4.1,0.85)),((5.4,1.35),(5.8,0.85)),((8.95,1.35),(7.9,0.85))]:
        arrow(a,b)
    ax.text(6,5.13,"问题一原始素材到词级特征的处理链",ha="center",fontproperties=FONT_BOLD,fontsize=15)
    save_figure(fig,"图01_问题一方法流程.png")


def figure_coverage(stats: dict) -> None:
    total = stats["word_count"]
    values = [
        ("原文文本", [total,0,0]),
        ("主时间与音频", [stats["audio_valid_words"],0,total-stats["audio_valid_words"]]),
        ("区间内有效视觉", [stats["visual_valid_words"],
                            stats["audio_valid_words"]-stats["visual_valid_words"],
                            total-stats["audio_valid_words"]]),
    ]
    fig,ax=plt.subplots(figsize=(10.5,4.2))
    for i,(name,parts) in enumerate(values):
        left=0
        for count,color in zip(parts,[COLORS["teal"],COLORS["orange"],"#D7DEE5"]):
            if count:
                ax.barh(i,count,left=left,height=0.55,color=color,edgecolor="white",lw=1)
                if count>90:
                    ax.text(left+count/2,i,f"{count}",ha="center",va="center",color="white" if color==COLORS["teal"] else COLORS["ink"],fontsize=10,weight="bold")
                left+=count
    ax.set_yticks(range(3),[x[0] for x in values],fontproperties=FONT)
    ax.invert_yaxis();ax.set_xlim(0,total);ax.set_xlabel("原文词数",fontproperties=FONT)
    ax.set_title("1,934个原文词的三模态主结果覆盖",fontproperties=FONT_BOLD,fontsize=14,pad=14)
    ax.text(total*0.98,2.77,"绿色：有效  橙色：有词时间但无有效视觉  灰色：无主词时间",ha="right",fontproperties=FONT,fontsize=8.5)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(axis="x",alpha=0.15);ax.set_axisbelow(True)
    fig.tight_layout();save_figure(fig,"图02_三模态主结果覆盖.png")


def figure_sample_coverage(rows: list[dict]) -> None:
    time_rate=np.array([x["timed_words"]/x["word_count"] for x in rows])
    visual_rate=np.array([x["visual_words"]/x["word_count"] for x in rows])
    order=np.argsort(time_rate)[::-1]
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(11,6.2),gridspec_kw={"height_ratios":[2.1,1]},sharex=False)
    x=np.arange(1,len(rows)+1)
    ax1.plot(x,time_rate[order],color=COLORS["blue"],lw=1.6,label="有主词时间/原文词")
    ax1.scatter(x,visual_rate[order],s=14,color=COLORS["teal"],alpha=0.8,label="有区间内有效视觉/原文词")
    ax1.set_ylim(-0.04,1.04);ax1.set_xlim(1,100);ax1.set_ylabel("逐样本词比例",fontproperties=FONT)
    ax1.set_title("100条样本的词级时间与视觉覆盖",fontproperties=FONT_BOLD,fontsize=14,pad=12)
    ax1.legend(prop=FONT,frameon=False,ncol=2,loc="lower left");ax1.grid(axis="y",alpha=0.18)
    bins=[0,0.01,0.25,0.5,0.75,0.999,1.001]
    labels=["0","(0,.25]","(.25,.5]","(.5,.75]","(.75,1)","1"]
    h1=np.histogram(time_rate,bins=bins)[0];h2=np.histogram(visual_rate,bins=bins)[0]
    xx=np.arange(len(labels));w=0.38
    ax2.bar(xx-w/2,h1,w,color=COLORS["blue"],label="主时间覆盖")
    ax2.bar(xx+w/2,h2,w,color=COLORS["teal"],label="视觉覆盖")
    ax2.set_xticks(xx,labels);ax2.set_ylabel("样本数",fontproperties=FONT);ax2.grid(axis="y",alpha=0.18)
    ax2.set_axisbelow(True);ax2.legend(prop=FONT,frameon=False,ncol=2)
    fig.tight_layout();save_figure(fig,"图03_100样本词级覆盖分布.png")


def figure_evidence(status_counts: Counter, word_rows: list[dict[str,str]]) -> None:
    names=["A层完整同词","B层唯一同词","边界差异>0.5s","未唯一映射","MFA无输出","ASR无转写"]
    counts=[520,901,status_counts["mfa_whisper_discrepancy_over_0_5s_null_primary"],
            status_counts["official_word_unmatched_or_ambiguous_in_asr"],
            status_counts["mfa_alignment_output_missing"],status_counts["asr_empty_no_timestamp"]]
    diffs=[float(r["mfa_whisper_max_edge_difference_s"]) for r in word_rows if r["mfa_whisper_max_edge_difference_s"]]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={"width_ratios":[1.3,1]})
    colors=[COLORS["teal"],COLORS["blue"],COLORS["red"],COLORS["orange"],"#A7B2BC","#BFC8D0"]
    ax1.barh(range(6),counts,color=colors)
    ax1.set_yticks(range(6),names,fontproperties=FONT);ax1.invert_yaxis();ax1.set_xlabel("词数",fontproperties=FONT)
    for i,c in enumerate(counts):ax1.text(c+8,i,str(c),va="center",fontsize=9)
    ax1.set_xlim(0,max(counts)*1.18);ax1.set_title("主时间证据与留空原因",fontproperties=FONT_BOLD,fontsize=12)
    ax1.spines[["top","right"]].set_visible(False)
    ax2.hist(diffs,bins=np.linspace(0,min(max(diffs),2),31),color=COLORS["blue"],alpha=0.78)
    ax2.axvline(0.5,color=COLORS["red"],ls="--",lw=1.7,label="自动筛查阈值0.5s")
    ax2.set_xlabel("MFA与Whisper的最大词边界差（秒）",fontproperties=FONT)
    ax2.set_ylabel("可比词数",fontproperties=FONT);ax2.set_title("两种模型的边界差分布",fontproperties=FONT_BOLD,fontsize=12)
    ax2.legend(prop=FONT,frameon=False);ax2.text(0.02,0.94,"差异不是人工真值误差",transform=ax2.transAxes,fontproperties=FONT,fontsize=8.5,va="top")
    fig.tight_layout();save_figure(fig,"图04_词时间证据与模型边界差.png")


def extract_video_nearest_frames(video: Path, targets: list[float]) -> list[tuple[float,np.ndarray]]:
    found=[None]*len(targets)
    distances=[float("inf")]*len(targets)
    with av.open(str(video)) as container:
        stream=next(s for s in container.streams if s.type=="video")
        for frame in container.decode(video=stream.index):
            if frame.pts is None or frame.time_base is None:continue
            t=float(frame.pts*frame.time_base)
            for k,target in enumerate(targets):
                d=abs(t-target)
                if d<distances[k]:
                    distances[k]=d
                    found[k]=(t,frame.to_ndarray(format="rgb24"))
    if any(x is None for x in found):raise ValueError("Cannot find video frame for typical sample")
    return found


def figure_typical(item: dict, word_rows: list[dict[str,str]]) -> list[dict]:
    sid=item["sample_id"]
    segment=[r for r in word_rows if 11<=int(r["official_word_index"])<=19]
    if len(segment)!=9 or any(r["timestamp_evidence_tier"]!="A_exact_transcript" or r["visual_valid_mask"]!="1" for r in segment):
        raise ValueError("Typical sample must have nine A-tier words with valid faces")
    t0=float(segment[0]["word_start_wav_s"])-0.18;t1=float(segment[-1]["word_end_wav_s"])+0.18
    wav=ROOT/"输入"/"音频WAV"/f"{sid}.wav"
    with wave.open(str(wav),"rb") as handle:
        sr=handle.getframerate();audio=np.frombuffer(handle.readframes(handle.getnframes()),dtype="<i2").astype(np.float32)/32768
    wave_time=np.arange(len(audio))/sr
    with np.load(AUDIO_FRAMES/f"{sid}.npz") as aa, np.load(VISUAL_FRAMES/f"{sid}.npz") as vv, np.load(ALIGNED/f"{sid}.npz") as al:
        a_time=aa["frame_time_wav_s"];rms=aa["features"][:,60];v_time=vv["frame_time_wav_s"]
        v_valid=vv["face_valid_mask"].astype(bool);v_feat=vv["features"]
        text=al["text"];af=al["audio"];vf=al["visual"]
    v_names=json.loads((ROOT/"结果"/"07_FaceLandmarker52"/"特征定义.json").read_text(encoding="utf-8"))["feature_names_in_column_order"]
    smile_idx=v_names.index("mouthSmileLeft");jaw_idx=v_names.index("jawOpen")
    fig=plt.figure(figsize=(12.4,8.5))
    gs=fig.add_gridspec(4,1,height_ratios=[0.7,2.1,1.35,2.35],hspace=0.34)
    axw=fig.add_subplot(gs[0]);axwav=fig.add_subplot(gs[1],sharex=axw);axfeat=fig.add_subplot(gs[2],sharex=axw)
    axthumb=fig.add_subplot(gs[3]);axthumb.axis("off")
    for i,row in enumerate(segment):
        s=float(row["word_start_wav_s"]);e=float(row["word_end_wav_s"])
        for ax in (axw,axwav,axfeat):ax.axvspan(s,e,color="#D9ECF4" if i%2==0 else "#F5E8CF",alpha=0.85,lw=0)
        axw.text((s+e)/2,0.5,row["official_word"],ha="center",va="center",fontsize=8.5,rotation=25)
    axw.set_xlim(t0,t1);axw.set_ylim(0,1);axw.set_yticks([]);axw.set_title("典型样本：A层同词文本的词—语音—视频对应",fontproperties=FONT_BOLD,fontsize=14,pad=16)
    region=(wave_time>=t0)&(wave_time<=t1)
    axwav.plot(wave_time[region],audio[region],color=COLORS["blue"],lw=0.65)
    axwav.set_ylabel("WAV波形",fontproperties=FONT);axwav.grid(axis="x",alpha=0.2)
    ar=(a_time>=t0)&(a_time<=t1);vr=(v_time>=t0)&(v_time<=t1)
    axfeat.plot(a_time[ar],rms[ar],color=COLORS["orange"],lw=1.6,label="声学RMS")
    ax2=axfeat.twinx();ax2.plot(v_time[vr&v_valid],v_feat[vr&v_valid,smile_idx],"o-",ms=3,lw=1.3,color=COLORS["teal"],label="嘴角上扬系数")
    ax2.scatter(v_time[vr&(~v_valid)],np.zeros((vr&(~v_valid)).sum()),marker="x",color=COLORS["red"],s=25,label="无效人脸帧")
    axfeat.set_ylabel("RMS",fontproperties=FONT);ax2.set_ylabel("mouthSmileLeft",fontproperties=FONT)
    axfeat.set_xlabel("原始WAV时间 / 秒（与视频PTS同轴）",fontproperties=FONT)
    axfeat.grid(axis="x",alpha=0.2)
    targets=[5.95,6.15,6.35]
    video=ROOT/"输入"/"原始视频"/Path(item["video_relative_path"])
    frames=extract_video_nearest_frames(video,targets)
    for k,(t,img) in enumerate(frames):
        sub=axthumb.inset_axes([0.01+k*0.33,0.1,0.3,0.78]);sub.imshow(img);sub.axis("off")
        axthumb.text(0.16+k*0.33,0.04,f"原视频帧 PTS={t:.3f}s",ha="center",va="top",transform=axthumb.transAxes,fontproperties=FONT,fontsize=8)
    axthumb.text(0.5,0.98,"“amazing”词区间内的真实视频帧（所示缩略图来自原始MP4）",ha="center",va="top",transform=axthumb.transAxes,fontproperties=FONT,fontsize=9.5)
    fig.text(0.5,0.015,"同一词的文本向量为768维，区间声学均值为74维，区间内有效表情均值为52维；图线仅显示可解释的代表坐标。",ha="center",fontproperties=FONT,fontsize=8.5)
    save_figure(fig,"图05_典型样本原始时间轴与视频帧.png")
    detail=[]
    for row in segment:
        i=int(row["official_word_index"])-1
        detail.append({"word_index":i+1,"word":row["official_word"],
                       "start_s":float(row["word_start_wav_s"]),"end_s":float(row["word_end_wav_s"]),
                       "audio_frames":int(row["audio_frame_count"]),"visual_valid_frames":int(row["visual_valid_face_frame_count"]),
                       "text_norm":float(np.linalg.norm(text[i])),"audio_rms_mean":float(af[i,60]),
                       "visual_mouth_smile_left_mean":float(vf[i,smile_idx]),
                       "visual_jaw_open_mean":float(vf[i,jaw_idx])})
    return detail


def make_markdown_table(rows: list[dict]) -> None:
    headings=["序号","样本编号","视频(s)","WAV(s)","词数","A层词","B层词","主时间/音频","有效视觉","无主时间","原文转写组"]
    lines=["# 附件1全部100条样本的特征提取与时序对齐汇总","", "维度统一为文本768、语音74、视觉52；粒度为原文词区间。主时间/音频、有效视觉是有效位置数；无主时间的词仍保留文本向量。","", "|"+"|".join(headings)+"|","|"+"|".join(["---"]*len(headings))+"|"]
    for r in rows:
        cells=[str(r["order"]),r["sample_id"],f"{r['video_duration_s']:.3f}",f"{r['wav_duration_s']:.3f}",
               str(r["word_count"]),str(r["tier_a_words"]),str(r["tier_b_words"]),str(r["timed_words"]),str(r["visual_words"]),str(r["null_time_words"]),r["transcript_group"]]
        lines.append("|"+"|".join(cells)+"|")
    (OUT/"100条样本全量汇总表.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
    set_style()
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    inventory=json.loads(INVENTORY.read_text(encoding="utf-8"))
    if not inventory["inventory_pass"] or not inventory["label_copy_matches_original_attachment"]:
        raise ValueError("Step 1 input integrity failed")
    media=index_by_id(csv_rows(MEDIA));audio=index_by_id(csv_rows(AUDIO_SUMMARY));vision=index_by_id(csv_rows(VISUAL_SUMMARY));aligned=index_by_id(csv_rows(ALIGN_SUMMARY))
    word_rows=csv_rows(WORD_TIMES);word_groups=group_by_id(word_rows)
    align_rows=csv_rows(ALIGN_INDEX);align_groups=group_by_id(align_rows)
    text_rows=csv_rows(TEXT_INDEX);text_groups=group_by_id(text_rows)
    text_matrix=np.load(TEXT_MATRIX,mmap_mode="r")
    if len(manifest)!=100 or len({x["sample_id"] for x in manifest})!=100 or text_matrix.shape!=(1934,768):
        raise ValueError("Input sample or text matrix count mismatch")
    if len(word_rows)!=1934 or len(text_rows)!=1934 or len(align_rows)!=1934:
        raise ValueError("Global word count mismatch")
    ids={x["sample_id"] for x in manifest}
    if any(ids!=set(d) for d in (media,audio,vision,aligned,word_groups,align_groups,text_groups)):
        raise ValueError("Sample ID sets do not reconcile")
    for folder in (AUDIO_FRAMES,VISUAL_FRAMES,ALIGNED):
        if len(list(folder.glob("*.npz")))!=100:
            raise ValueError(f"Expected exactly 100 NPZ files in {folder}")
    full=[];status_counts=Counter();tier_counts=Counter();groups=Counter();checks=Counter()
    for order,item in enumerate(manifest,1):
        sid=item["sample_id"];m=media[sid];a=audio[sid];v=vision[sid];al=aligned[sid]
        words=word_groups[sid];indices=align_groups[sid];tx=text_groups[sid]
        n=len(words)
        if n!=len(indices) or n!=len(tx) or n!=int(al["word_count"]):
            raise ValueError(f"Word count mismatch: {sid}")
        for j,(w,ix,t) in enumerate(zip(words,indices,tx),1):
            if int(w["official_word_index"])!=j or w["official_word"]!=ix["official_word"] or w["official_word"]!=t["official_word"]:
                raise ValueError(f"Word order mismatch: {sid} {j}")
            if int(ix["text_feature_row_0based"])!=int(t["feature_row_0based"]):
                raise ValueError(f"Text matrix index mismatch: {sid} {j}")
            status_counts[w["timestamp_status"]]+=1
            if w["final_primary_valid_mask"].strip().lower()=="true" and w["timestamp_evidence_tier"]:
                tier_counts[w["timestamp_evidence_tier"]]+=1
        with np.load(ALIGNED/f"{sid}.npz") as z:
            text=z["text"];ac=z["audio"];vi=z["visual"];start=z["word_start_wav_s"];end=z["word_end_wav_s"]
            tm=z["word_time_valid_mask"].astype(bool);am=z["audio_valid_mask"].astype(bool);vm=z["visual_valid_mask"].astype(bool);fm=z["audio_f0_valid_mask"].astype(bool)
            if (text.shape,ac.shape,vi.shape)!=((n,768),(n,74),(n,52)) or int(z["valid_length"])!=n:
                raise ValueError(f"Feature dimensions / valid length mismatch: {sid}")
            if not all(np.isfinite(q).all() for q in (text,ac,vi)):
                raise ValueError(f"Non-finite feature: {sid}")
            if not np.array_equal(np.isnan(start),~tm) or not np.array_equal(np.isnan(end),~tm):
                raise ValueError(f"Missing word times and mask disagree: {sid}")
            if np.any(am&~tm) or np.any(vm&~tm) or np.any(fm&~am):
                raise ValueError(f"Invalid mask dependency: {sid}")
            if not np.all(ac[~am]==0) or not np.all(vi[~vm]==0):
                raise ValueError(f"Unmasked zero placeholder contract violated: {sid}")
            rows=np.array([int(t["feature_row_0based"]) for t in tx],dtype=np.int32)
            if not np.allclose(text,text_matrix[rows],atol=1e-6):
                raise ValueError(f"Text matrix differs from Step 5: {sid}")
            if (int(tm.sum()),int(am.sum()),int(vm.sum()),int(fm.sum()))!=(int(al["primary_time_word_count"]),int(al["audio_valid_word_count"]),int(al["visual_valid_word_count"]),int(al["audio_f0_valid_word_count"])):
                raise ValueError(f"Summary vs NPZ masks mismatch: {sid}")
            if np.any(start[tm]<0) or np.any(end[tm]<=start[tm]):
                raise ValueError(f"Invalid accepted word intervals: {sid}")
            wav_d=float(m["wav_duration_s"])
            if np.any(end[tm]>wav_d+0.05):
                raise ValueError(f"Accepted word outside audio duration: {sid}")
            if int((tm&~vm).sum())!=int(al["timed_words_with_no_visual_frame"])+int(al["timed_words_with_frames_but_no_valid_face"]):
                raise ValueError(f"Visual missingness partition mismatch: {sid}")
        group=words[0]["transcript_group"];groups[group]+=1
        full.append({"order":order,"sample_id":sid,"video_duration_s":float(m["video_duration_s"]),
                     "wav_duration_s":float(m["wav_duration_s"]),"word_count":n,
                     "tier_a_words":sum(w["final_primary_valid_mask"].strip().lower()=="true" and w["timestamp_evidence_tier"]=="A_exact_transcript" for w in words),
                     "tier_b_words":sum(w["final_primary_valid_mask"].strip().lower()=="true" and w["timestamp_evidence_tier"]=="B_unique_word_overlap_in_mismatched_transcript" for w in words),
                     "timed_words":int(al["primary_time_word_count"]),"audio_words":int(al["audio_valid_word_count"]),
                     "visual_words":int(al["visual_valid_word_count"]),"null_time_words":int(al["words_without_primary_time"]),
                     "audio_frames":int(a["frame_count"]),"visual_sampled_frames":int(v["sampled_frame_count"]),
                     "visual_valid_frames":int(v["face_valid_frame_count"]),
                     "transcript_group":group,"annotation":item["annotation"],"label":item["label"],
                     "feature_file":al["feature_file"]})
    stats={"sample_count":100,"word_count":1934,"tier_a_words":tier_counts["A_exact_transcript"],
           "tier_b_words":tier_counts["B_unique_word_overlap_in_mismatched_transcript"],
           "audio_valid_words":sum(r["audio_words"] for r in full),"visual_valid_words":sum(r["visual_words"] for r in full),
           "null_time_words":sum(r["null_time_words"] for r in full),
           "audio_frame_count":sum(r["audio_frames"] for r in full),"visual_sampled_frame_count":sum(r["visual_sampled_frames"] for r in full),
           "samples_without_primary_time":sum(r["timed_words"]==0 for r in full),
           "samples_without_primary_visual":sum(r["visual_words"]==0 for r in full),
           "samples_without_any_valid_face":sum(r["visual_valid_frames"]==0 for r in full),
           "word_count_distribution":descriptive([r["word_count"] for r in full]),
           "video_duration_distribution_s":descriptive([r["video_duration_s"] for r in full]),
           "wav_duration_distribution_s":descriptive([r["wav_duration_s"] for r in full]),
           "transcript_groups":dict(groups),"timestamp_status_counts":dict(status_counts),
           "annotation_counts":dict(Counter(r["annotation"] for r in full)),
           "complete_file_and_mask_checks":True,
           "timestamp_accuracy_requires_gold_boundaries":True,
           "no_gold_boundaries_available":True,
           "visual_sampling_approx_fps":10,
           "source_hashes":{"manifest":file_hash(MANIFEST),"word_times":file_hash(WORD_TIMES),"alignment_index":file_hash(ALIGN_INDEX)}}
    if stats["audio_valid_words"]!=1421 or stats["visual_valid_words"]!=1228 or stats["null_time_words"]!=513 or stats["tier_a_words"]!=520 or stats["tier_b_words"]!=901:
        raise ValueError("Expected reported coverage did not reconcile")
    typical=next(x for x in manifest if x["sample_id"].startswith(TYPICAL_PREFIX))
    typical_detail=figure_typical(typical,align_groups[typical["sample_id"]])
    stats["typical_sample"]={"sample_id":typical["sample_id"],"annotation":typical["annotation"],
                             "label":typical["label"],"text":typical["text"],"detail":typical_detail}
    figure_flow();figure_coverage(stats);figure_sample_coverage(full);figure_evidence(status_counts,word_rows)
    make_markdown_table(full)
    (OUT/"全量样本统计.json").write_text(json.dumps(full,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"全量核验与论文数据.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in stats.items() if k!="typical_sample"},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
