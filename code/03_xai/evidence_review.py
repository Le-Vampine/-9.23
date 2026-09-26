# -*- coding: utf-8 -*-
"""附件四文本证据的人工判读表（可复核性证据）。

背景：问题三要求"关键证据需可对应至原始文本片段"。本脚本把 20 条附件四样本的
模型给出文本证据（text_evidence + 字符区间）与原始转写并排列出，并给出两种判读：

  A. 自动线索检查：证据段内是否含 情感词 / 否定词 / 程度词（内置小词表，纯规则）；
  B. 人工判读：逐条判断"证据段落是否承载该样本的文本情感线索"，结论三档：
     - 命中：证据含与预测方向一致的情感/否定/程度线索；
     - 中性-主题：样本为中性，证据为事实性或主题性描述（与判断一致，合理）；
     - 未命中：证据为寒暄/开场/收尾等无语篇情感的词，或与预测方向相反。

人工结论 MANUAL 由复核者（本项目内逐条阅读原文后）填写，脚本只负责复现统计，
保证论文中的 x/y 可被读者按同一数据重算。

输出：paper_tables/Q3_证据人工判读.csv 与 runs/q3/q3_evidence_review.json
用法：python evidence_review.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV_IN = os.path.join(ROOT, "submission", "pred_explain_att4.csv")
CARDS = os.path.join(ROOT, "submission", "explain_cards")
OUT_CSV = os.path.join(ROOT, "paper_tables", "Q3_证据人工判读.csv")
OUT_JSON = os.path.join(ROOT, "runs", "q3", "q3_evidence_review.json")

# ---- 规则词表（仅用于自动线索标记，不参与人工结论）----
POS = {"happiness", "happy", "like", "love", "recommend", "essential", "easy", "power",
       "enthusiastically", "fan", "forgiving", "good", "great", "best", "wonderful",
       "impressive", "simple", "transform", "guru", "enthusiastic", "enjoy"}
NEG = {"misery", "terrible", "ashamed", "worse", "worst", "awful", "bad", "hate",
       "dislike", "unfortunately", "haters", "not", "no", "never", "refuse", "denied"}
INT = {"really", "very", "absolutely", "surprisingly", "quite", "so", "too", "extremely"}

# ---- 人工判读（复核者逐条阅读后填写）----
MANUAL = {
    "01": ("中性-主题", "中性样本；证据为'essential to ensuring…'功能性描述，与'中性'一致"),
    "02": ("命中", "'happiness' 与 'not…misery' 否定+情感对照，方向与正向一致"),
    "03": ("中性-主题", "证据为机构名与事实陈述，中性判断合理"),
    "04": ("命中", "'kiss my chances…goodbye?] Absolutely not' 否定+习语，方向与负向一致"),
    "05": ("未命中", "证据为自我介绍（'my name is Chloe'），无语篇情感；预测偏向由语音/视觉支撑"),
    "06": ("命中", "'a fan of dancing' 表达偏好，方向与正向一致"),
    "07": ("命中", "'enthusiastically discussed' 程度+情感副词，方向与正向一致"),
    "08": ("未命中", "证据为节目过渡句（'That brings us to tonight…'），属话题词而非情感词"),
    "09": ("命中", "'would not recommend it' 否定+评价动词，方向与负向一致"),
    "10": ("未命中", "证据截取到'I really do like to see'（正向），与样本负向判断方向相反，属边界失败案例"),
    "11": ("命中", "'ashamed to have made this film' 明确负向自我评价"),
    "12": ("命中", "'Or worse, …' 负向框架词，与负向方向一致"),
    "13": ("中性-主题", "证据为举例说明（'For example, I could take a set of data…'），中性合理"),
    "14": ("中性-主题", "证据为姓名与职务，属主题性描述，中性合理"),
    "15": ("命中", "'its power to transform lives' 正向价值表达"),
    "16": ("命中", "'this is a terrible movie' 强负向评价"),
    "17": ("命中", "'easy and will make people think you…' 正向体验描述"),
    "18": ("中性-主题", "证据为渔业认证的事实性新闻语句，中性合理"),
    "19": ("未命中", "证据止于'surprisingly forgiving brands'，情感载体（'haters'/'unfortunately'）落在区间之外，属截断问题"),
    "20": ("未命中", "证据为视频收尾引导语（'click in the link…'），无语篇情感"),
}


def clue(seg: str):
    toks = set(re.findall(r"[a-z']+", seg.lower()))
    return {
        "情感词": sorted(toks & (POS | NEG)),
        "否定词": sorted(toks & {"not", "no", "never", "don't", "would"}),
        "程度词": sorted(toks & INT),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true",
                    help="不重新生成，只读取现有 CSV 的『人工判读』列重算统计（供复核者改判后一键更新 x/y）")
    args = ap.parse_args()

    if args.from_csv:
        rows = list(csv.DictReader(open(OUT_CSV, encoding="utf-8-sig")))
        n = len(rows)
        hit = sum(1 for r in rows if r["人工判读"] == "命中")
        neu = sum(1 for r in rows if r["人工判读"] == "中性-主题")
        miss = sum(1 for r in rows if r["人工判读"] == "未命中")
        pol = [r for r in rows if r["pred_label"] != "Neutral"]
        pol_hit = sum(1 for r in pol if r["人工判读"] == "命中")
        summary = {
            "n": n, "命中": hit, "中性-主题": neu, "未命中": miss,
            "合理率_命中加中性主题": round((hit + neu) / n, 4),
            "极性样本数": len(pol), "极性样本命中": pol_hit,
            "极性样本命中率": round(pol_hit / len(pol), 4),
            "未命中样本": [r["id"] for r in rows if r["人工判读"] == "未命中"],
            "数据来源": "由复核者改判后的 CSV 重算（--from-csv）",
        }
        json.dump(summary, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[重算] 合理 %d/%d = %.0f%%；极性 %d/%d = %.0f%%；未命中 %s"
              % (hit + neu, n, 100 * (hit + neu) / n,
                 pol_hit, len(pol), 100 * pol_hit / len(pol),
                 summary["未命中样本"]))
        print("[SAVE]", os.path.relpath(OUT_JSON, ROOT))
        return 0

    rows = list(csv.DictReader(open(CSV_IN, encoding="utf-8-sig")))
    out = []
    for r in rows:
        sid = r["id"]
        card = json.load(open(os.path.join(CARDS, "%s.json" % sid), encoding="utf-8"))
        txt = card.get("raw_text", "")
        a, b = int(r["text_char_start"] or 0), int(r["text_char_end"] or 0)
        seg = txt[a:b]
        c = clue(seg)
        verdict, note = MANUAL.get(sid, ("未判读", ""))
        out.append({
            "id": sid, "pred_label": r["pred_label"], "pred_score": r["pred_score"],
            "main_modality": r["main_modality"], "text_evidence": r["text_evidence"],
            "char_span": "%d-%d" % (a, b), "evidence_segment": seg,
            "自动线索_情感词": "/".join(c["情感词"]), "自动线索_否定词": "/".join(c["否定词"]),
            "自动线索_程度词": "/".join(c["程度词"]), "人工判读": verdict, "备注": note,
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    n = len(out)
    hit = sum(1 for o in out if o["人工判读"] == "命中")
    neu = sum(1 for o in out if o["人工判读"] == "中性-主题")
    miss = sum(1 for o in out if o["人工判读"] == "未命中")
    pol = [o for o in out if o["pred_label"] != "Neutral"]
    pol_hit = sum(1 for o in pol if o["人工判读"] == "命中")
    summary = {
        "n": n, "命中": hit, "中性-主题": neu, "未命中": miss,
        "合理率_命中加中性主题": round((hit + neu) / n, 4),
        "极性样本数": len(pol), "极性样本命中": pol_hit,
        "极性样本命中率": round(pol_hit / len(pol), 4),
        "未命中样本": [o["id"] for o in out if o["人工判读"] == "未命中"],
        "说明": "自动线索为规则词表标记，人工判读为逐条阅读原文后的结论；两栏均在同一 CSV 中可复核",
    }
    json.dump(summary, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[SAVE]", os.path.relpath(OUT_CSV, ROOT))
    print("[SAVE]", os.path.relpath(OUT_JSON, ROOT))
    print("[统计] 合理（命中+中性-主题）= %d/%d = %.0f%%；极性样本命中 %d/%d = %.0f%%；未命中 %s"
          % (hit + neu, n, 100 * (hit + neu) / n, pol_hit, len(pol),
             100 * pol_hit / len(pol), summary["未命中样本"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
