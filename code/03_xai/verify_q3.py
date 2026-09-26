# -*- coding: utf-8 -*-
r"""S9：问题3 交付自查（任一断言不过即失败）。

检查项：
  1. pred_explain_att4.csv 行数 = 20、id 顺序与附件4 文件名一致、列齐全；
  2. 其 pred_label / pred_score 与 submission/pred_att4.csv 逐位一致；
  3. 每行模态作用度之和为 1；时间重要性数值合法（在卡片 JSON 中抽检）；
  4. 证据区间合法（段号、秒区间不超过视频时长）；
  5. 关键帧文件存在且非空；
  6. 解释卡 JSON 每个 id 齐全；
  7. submission/ 总体积 ≤ 50 MB；
  8. 匿名性：CSV/JSON 中不含单位/队号/学号/手机号/邮箱等身份信息。

用法：
  python verify_q3.py --sub_dir ..\..\submission --out_dir ..\..\runs\q3
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)
from xai_common import save_json, ROOT        # noqa: E402

BANNED = [r"参赛队", r"队号", r"学号", r"指导教[师]?", r"大[学]?[\u4e00-\u9fff]{0,6}学院",
          r"1[3-9]\d{9}", r"[\w.\-]+@[\w.\-]+\.\w+", r"[\u4e00-\u9fff]{2,4}(同学|老师)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub_dir", default=os.path.join(ROOT, "submission"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "runs", "q3"))
    ap.add_argument("--max_mb", type=float, default=50.0)
    a = ap.parse_args()
    checks, fails = [], []

    def ck(name, ok, detail=""):
        checks.append(dict(check=name, ok=bool(ok), detail=str(detail)))
        if not ok:
            fails.append("%s：%s" % (name, detail))
        print("  [%s] %-34s %s" % ("OK " if ok else "FAIL", name, detail))

    # 1) 行数 / id / 列
    path = os.path.join(a.sub_dir, "pred_explain_att4.csv")
    ck("文件存在", os.path.isfile(path), os.path.relpath(path, ROOT))
    df = pd.read_csv(path, dtype={"id": str})
    df["id"] = [str(x).zfill(2) if str(x).isdigit() else str(x) for x in df["id"]]
    meta4 = json.load(open(os.path.join(ROOT, "data_att", "att_meta.json"),
                           encoding="utf-8"))
    ids = [r["id"] for r in meta4["att4"]]
    ck("行数=20", len(df) == 20, "实为 %d" % len(df))
    ck("id 顺序与附件4 一致", list(df["id"]) == ids, "%s" % list(df["id"])[:4])
    need = ["id", "pred_label", "pred_score", "prob_neg", "prob_neu", "prob_pos",
            "main_modality", "w_text", "w_audio", "w_vision", "text_evidence",
            "text_seg_start", "text_seg_end", "audio_start_s", "audio_end_s",
            "vision_frame_idx", "vision_time_s", "compr", "suff", "stability", "sparsity"]
    miss = [c for c in need if c not in df.columns]
    ck("必需列齐全", not miss, "缺 %s" % miss)

    # 2) 与 pred_att4.csv 一致
    ref = pd.read_csv(os.path.join(a.sub_dir, "pred_att4.csv"), dtype={"id": str})
    ck("标签与 pred_att4.csv 一致",
       float((df["pred_label"].values == ref["pred_label"].values).mean()) == 1.0)
    ck("分数与 pred_att4.csv 一致",
       float(np.abs(df["pred_score"].values - ref["pred_score"].values).max()) < 1e-3,
       "最大差 %.2e" % float(np.abs(df["pred_score"].values - ref["pred_score"].values).max()))

    # 3) π 归一化
    s = df[["w_text", "w_audio", "w_vision"]].values.sum(1)
    ck("模态作用度之和=1", float(np.abs(s - 1).max()) < 2e-3, "最大偏差 %.2e"
       % float(np.abs(s - 1).max()))
    ck("主模态取值合法",
       set(df["main_modality"]) <= {"text", "audio", "vision"},
       "%s" % sorted(set(df["main_modality"])))

    # 4) 证据合法性
    bad_seg = df[(df["text_seg_start"].notna()) & (df["text_seg_end"].notna()) &
                 (~((df["text_seg_start"] >= 0) & (df["text_seg_end"] <= 50) &
                    (df["text_seg_start"] < df["text_seg_end"])))]
    ck("文本段号合法", len(bad_seg) == 0, "越界 %d 行" % len(bad_seg))
    dur = {r["id"]: r.get("duration_s") for r in meta4["att4"]}
    bad_t = [i for i, r in df.iterrows()
             if pd.notna(r["audio_end_s"]) and dur.get(r["id"]) and
             r["audio_end_s"] > dur[r["id"]] + 0.05]
    ck("语音时段不超视频时长", len(bad_t) == 0, "越界 %d 行" % len(bad_t))
    ck("含文本证据的样本数>0", int(df["text_evidence"].notna().sum()) > 0,
       "%d/20" % int(df["text_evidence"].notna().sum()))

    # 5) 关键帧
    kf = [os.path.join(ROOT, p) for p in df["vision_keyframe"].dropna().astype(str) if p]
    ok_kf = [p for p in kf if os.path.isfile(p) and os.path.getsize(p) > 500]
    ck("关键帧文件有效", len(ok_kf) >= max(1, int(0.8 * len(kf))),
       "%d/%d 张可用" % (len(ok_kf), len(kf)))

    # 6) 解释卡
    cdir = os.path.join(a.sub_dir, "explain_cards")
    have = sorted(os.path.splitext(f)[0] for f in os.listdir(cdir)) if os.path.isdir(cdir) else []
    ck("解释卡 JSON 齐全", set(have) == set(ids), "%d/20" % len(have))
    if set(have) >= set(ids):
        c = json.load(open(os.path.join(cdir, ids[0] + ".json"), encoding="utf-8"))
        ti = c["temporal_importance_fused"]["text"]
        ck("时间重要性长度=50", len(ti) == 50, "len=%d" % len(ti))
        ck("时间重要性非负且有界", bool(np.min(ti) >= -1e-6 and np.max(ti) <= 1 + 1e-6))

    # 7) 体积
    tot = 0
    for r, _, fs in os.walk(a.sub_dir):
        if "_backup" in r:
            continue
        for f in fs:
            tot += os.path.getsize(os.path.join(r, f))
    mb = tot / 1024 / 1024
    ck("submission 体积 ≤ %.0f MB" % a.max_mb, mb <= a.max_mb, "%.2f MB" % mb)

    # 8) 匿名性
    hits = []
    for r, _, fs in os.walk(a.sub_dir):
        if "_backup" in r:
            continue
        for f in fs:
            if not f.lower().endswith((".csv", ".json", ".md", ".txt")):
                continue
            p = os.path.join(r, f)
            try:
                txt = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            for pat in BANNED:
                for m in re.finditer(pat, txt):
                    hits.append("%s :: %s" % (os.path.relpath(p, ROOT), m.group(0)))
    ck("无身份信息", not hits, "%d 处命中" % len(hits))
    if hits:
        print("       %s" % hits[:5])

    rep = dict(n_check=len(checks), n_fail=len(fails), checks=checks, fails=fails,
               submission_mb=round(mb, 2))
    save_json(os.path.join(a.out_dir, "verify_q3.json"), rep)
    print("\n%s（%d 项检查，%d 项失败）" % ("VERIFY_PASS" if not fails else "VERIFY_FAIL",
                                          len(checks), len(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
