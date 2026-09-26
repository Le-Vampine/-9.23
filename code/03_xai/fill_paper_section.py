# -*- coding: utf-8 -*-
r"""把 runs/q3 与 paper_tables/Q3_* 的正式结果回填进论文 §5.3 片段。

流程：读 `05_paper/_q3_section_draft.tex`（含 <<PLACEHOLDER>>）→ 用本文件 compute() 的
      取值表替换 → 写出 `05_paper/_q3_section_filled.tex`；
      `--apply` 时进一步把 main.tex 中 `\subsection{问题三：可解释性情感预测}` 那一小节
      （占位段）整体替换为填充后的内容（原文件先备份为 main.tex.bak_q3）。

用法：
  python fill_paper_section.py           # 只生成 _q3_section_filled.tex
  python fill_paper_section.py --apply   # 并写回 main.tex
"""
import argparse
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
PAPER = os.path.join(ROOT, "05_paper")
RUNS = os.path.join(ROOT, "runs", "q3")
TAB = os.path.join(ROOT, "paper_tables")
SUB = os.path.join(ROOT, "submission")
DRAFT = os.path.join(PAPER, "_q3_section_draft.tex")
FILLED = os.path.join(PAPER, "_q3_section_filled.tex")
MAIN = os.path.join(PAPER, "main.tex")

MOD_CN = {"text": "文本", "audio": "语音", "vision": "视觉"}


def j(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def f4(x):
    return "%.4f" % float(x)


def compute():
    v = {}
    # ---- Shapley / 融合 ----
    sv = j(os.path.join(RUNS, "q3_shapley_valid_report.json"))
    g = sv.get("global", {})
    phi = g.get("phi_negmae", [0, 0, 0])
    v["PHI_GLOBAL_TEXT"], v["PHI_GLOBAL_AUDIO"], v["PHI_GLOBAL_VISION"] = (
        f4(phi[0]), f4(phi[1]), f4(phi[2]))
    gm = j(os.path.join(RUNS, "q3_gamma_valid.json"))
    v["GAMMA"] = "(%s)" % ", ".join("%.1f" % x for x in gm.get("gamma", [0, 0, 0]))
    ag = gm.get("agreement_with_occ", {})
    v["ATT_OCC_SPEARMAN"] = f4(ag.get("att", 0))
    grid = gm.get("grid", [])
    v["FUSED_OCC_SPEARMAN"] = f4(grid[0]["spearman_occ"]) if grid else "0.0000"

    # ---- 证据头 ----
    eh = j(os.path.join(RUNS, "q3_evidence_head_report.json"))
    args = eh.get("args", {})
    v["ALPHA_F"] = "%g" % args.get("alpha_f", 1.0)
    v["ALPHA_1"] = "10^{-3}" if abs(args.get("alpha1", 1e-3) - 1e-3) < 1e-9 else "%g" % args.get("alpha1", 0)
    v["ALPHA_2"] = "10^{-3}" if abs(args.get("alpha2", 1e-3) - 1e-3) < 1e-9 else "%g" % args.get("alpha2", 0)
    v["EPOCHS"] = "%d" % args.get("epochs", 12)
    hist = eh.get("history", [])
    v["HEAD_EP1"] = ("$%+.4f$" % hist[0]["score"]) if hist else "---"
    v["HEAD_BEST_EP"] = "%d" % eh.get("best_epoch", 0)
    v["HEAD_BEST_SC"] = ("$%+.4f$" % eh.get("best_score", 0.0))

    # ---- 八子集反事实 ----
    rows = []
    mae = g.get("v_negmae", {})
    f1 = g.get("v_macrof1", {})
    order = ["none", "text", "audio", "vision", "text+audio", "text+vision",
             "audio+vision", "all"]
    phi_map = {"text": phi[0], "audio": phi[1], "vision": phi[2]}
    cn = {"none": "无（全屏蔽）", "text": "文本", "audio": "语音", "vision": "视觉",
          "text+audio": "文本+语音", "text+vision": "文本+视觉",
          "audio+vision": "语音+视觉", "all": "全模态"}
    for k in order:
        if k not in mae:
            continue
        p = ("$%+.4f$" % phi_map[k]) if k in phi_map else "---"
        rows.append("%s & %.4f & %.4f & %s \\\\" % (cn[k], -float(mae[k]), float(f1.get(k, np.nan)), p))
    v["SUBSET_ROWS"] = "\n".join(rows)

    # ---- 模态忠实性与一致性 ----
    va = j(os.path.join(RUNS, "q3_valid_analysis.json"))
    ds = va.get("modality_faithfulness", {}).get("delta_sample", {})
    v["DELTA_MAE_TEXT"] = f4(ds.get("text", 0))
    v["DELTA_MAE_AUDIO"] = f4(ds.get("audio", 0))
    v["DELTA_MAE_VISION"] = f4(ds.get("vision", 0))
    v["CONSIST_SPEARMAN"] = "%.3f" % float(va.get("consistency_with_q2", {}).get("spearman", 0))
    mf = va.get("modality_faithfulness", {}).get("rows", [])
    v["PI_DELTA_SPEARMAN"] = "%.3f" % float(mf[0]["与样本级ΔMAE秩相关"]) if mf else "0.000"

    # ---- 分层作用度 ----
    try:
        tb = pd.read_csv(os.path.join(TAB, "Q3_模态作用度_按真值极性.csv"))
        vals = tb[["π_文本", "π_语音", "π_视觉"]].values
        v["PI_BY_LABEL"] = "/".join("%.3f" % x for x in vals[:, 0])
        v["PI_MODAL_GAP"] = "%.3f" % float(np.max(vals[:, 1:]) - np.min(vals[:, 1:]))
        rws = []
        for _, r in tb.iterrows():
            rws.append("%s & %d & %.4f & %.4f & %.4f \\\\"
                       % (r["极性"], r["样本数"], r["π_文本"], r["π_语音"], r["π_视觉"]))
        v["PI_BY_LABEL_ROWS"] = "\n".join(rws)
    except Exception as e:
        print("  [警告] 分层作用度表读取失败：%s" % e)

    # ---- 缺失率机制验证 ----
    mc = va.get("missing_rate_consistency", [])
    if mc:
        a3 = [r for r in mc if r["数据集"] == "att3"]
        n = sum(int(r["样本数"]) * 3 for r in mc)
        v["MISS_N"] = "%d" % n
        v["PI_MISS_SPEARMAN"] = "%.4f" % float(a3[0]["π与缺失率秩相关"]) if a3 else "---"
        v["PI_MISS_WITH"] = f4(a3[0]["缺失模态的平均π"]) if a3 else "---"
        v["PI_MISS_WITHOUT"] = f4(a3[0]["无缺失模态的平均π"]) if a3 else "---"

    # ---- 解释质量 ----
    try:
        q = pd.read_csv(os.path.join(TAB, "Q3_解释质量对比.csv"))
        qa = q[q["数据集"] == "att4"]
        rws = []
        for _, r in qa.iterrows():
            rws.append("%s & $%+.4f$ & %.4f & %.4f & %.4f \\\\"
                       % (r["解释源"], r["忠实性_符号均值"], r["忠实性_绝对均值"],
                          r["充分性偏差_绝对均值"], r["稀疏性"]))
        v["FAITH_ROWS"] = "\n".join(rws)
        fu = qa[qa["解释源"] == "fused"]
        v["COMPR_ABS"] = f4(fu["忠实性_绝对均值"].iloc[0]) if len(fu) else "---"
        v["T_STD"] = f4(fu["预测标准差"].iloc[0]) if len(fu) else "---"
    except Exception as e:
        print("  [警告] 解释质量表读取失败：%s" % e)

    # ---- 错误归因 ----
    try:
        ea = pd.read_csv(os.path.join(TAB, "Q3_错误归因.csv"))
        rws = []
        for _, r in ea.iterrows():
            rws.append("%s & %d & %.4f & %.4f & %.4f \\\\"
                       % (r["分组"], r["样本数"], r["π熵"], r["π_文本"], r["置信度"]))
        v["ERR_ROWS"] = "\n".join(rws)
    except Exception as e:
        print("  [警告] 错误归因表读取失败：%s" % e)

    # ---- 附件4 汇总 ----
    dfx = pd.read_csv(os.path.join(SUB, "pred_explain_att4.csv"), dtype={"id": str})
    ref = pd.read_csv(os.path.join(SUB, "pred_att4.csv"), dtype={"id": str})
    miss_any = ((ref[["missing_text_ratio", "missing_audio_ratio",
                       "missing_vision_ratio"]] > 0).any(axis=1))
    v["ATT4_NEG"] = "%d" % int((dfx["pred_label"] == "Negative").sum())
    v["ATT4_NEU"] = "%d" % int((dfx["pred_label"] == "Neutral").sum())
    v["ATT4_POS"] = "%d" % int((dfx["pred_label"] == "Positive").sum())
    v["ATT4_MAINTEXT"] = "%d" % int((dfx["main_modality"] == "text").sum())
    v["ATT4_NMISS"] = "%d" % int(miss_any.sum())
    v["ATT4_NNOMISS"] = "%d" % int((~miss_any).sum())
    for tag, col, src in (("MSCORE", "pred_score", ref), ("MABS", "pred_score", ref),
                          ("MCONF", "confidence", ref)):
        s = src[col].abs() if tag == "MABS" else src[col]
        v["ATT4_%s" % tag] = "$%+.4f$" % s.mean()
        v["ATT4_%s_M" % tag] = "$%+.4f$" % s[miss_any.values].mean() if miss_any.any() else "---"
        v["ATT4_%s_N" % tag] = "$%+.4f$" % s[(~miss_any).values].mean()
    v["ATT4_AGREE"] = f4((ref["pred_label"] == ref["pred_label_theta"]).mean())

    # ---- 样本 16 / 13 的细节 ----
    try:
        sh = pd.read_csv(os.path.join(RUNS, "q3_shapley_att4.csv"), dtype={"id": str})
        sh["id"] = sh["id"].str.zfill(2)
        v["PI16_TEXT"] = f4(sh.loc[sh["id"] == "16", "pi_text"].iloc[0])
    except Exception:
        v["PI16_TEXT"] = "---"
    return v


def fill(text, vals):
    miss = sorted(set(re.findall(r"<<([A-Z0-9_]+)>>", text)) - set(vals))
    if miss:
        print("  [警告] 未定义的占位符：%s" % miss)
    for k, vv in vals.items():
        text = text.replace("<<%s>>" % k, str(vv))
    left = re.findall(r"<<[A-Z0-9_]+>>", text)
    if left:
        print("  [警告] 仍残留占位符：%s" % sorted(set(left)))
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="写回 main.tex")
    a = ap.parse_args()
    vals = compute()
    print("  [INFO] 回填 %d 个字段" % len(vals))
    txt = open(DRAFT, encoding="utf-8").read()
    out = fill(txt, vals)
    open(FILLED, "w", encoding="utf-8").write(out)
    print("[SAVE] %s" % os.path.relpath(FILLED, ROOT))
    if a.apply:
        main = open(MAIN, encoding="utf-8").read()
        pat = re.compile(r"\\subsection\{问题三：可解释性情感预测\}.*?(?=\n%% -+ 6)",
                         re.S)
        if not pat.search(main):
            raise SystemExit("未在 main.tex 中找到问题三小节（锚点不匹配）")
        shutil.copy2(MAIN, MAIN + ".bak_q3")
        main2 = pat.sub(lambda m: out.rstrip() + "\n", main, count=1)
        open(MAIN, "w", encoding="utf-8").write(main2)
        print("[APPLY] 已写回 main.tex（备份 main.tex.bak_q3）")
    print("FILL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
