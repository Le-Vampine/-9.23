# -*- coding: utf-8 -*-
r"""
附件3/附件4 缺失分布标定：统计专项测试集的缺失形态，并反推训练时的模拟参数。

回答题目"缺失模态类型、缺失位置、缺失时长"三要素在**测试集真实分布**下的取值，
避免"训练模拟分布"与"测试分布"错配导致性能虚高/虚低。

用法：
  python calibrate_missing.py --pkl <att3.pkl> --out_dir runs\q2
  python calibrate_missing.py --version aligned            # 自动定位（会先找到附件2，需显式指定附件3）
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import load_any, missing_regions      # noqa: E402

MODALITIES = ("text", "audio", "vision")
COMBOS = ["text", "audio", "vision", "text+audio", "text+vision", "audio+vision",
          "text+audio+vision", "none"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out_dir", default=r"E:\数学建模\runs\q2")
    ap.add_argument("--min_zero_len", type=int, default=2)
    a = ap.parse_args()

    splits = load_any(a.pkl, min_zero_len=a.min_zero_len)
    rep = {"pkl": a.pkl, "splits": {}}

    for name, sp in splits.items():
        N = sp["N"]
        print("=" * 78)
        print("[SPLIT %s] N=%d  L=%s" % (name, N, sp["L"]))
        info = {"N": N}
        per_sample_regions = []
        combo_count = {c: 0 for c in COMBOS}
        int_len_all, int_ratio_all, start_frac_all = [], [], []

        for m in MODALITIES:
            valid_len = (~sp["pad"][m]).sum(axis=1)
            has = 0
            ratios = []
            for i in range(N):
                regs = missing_regions(sp["miss"][m][i], a.min_zero_len)
                if regs:
                    has += 1
                vlen = max(int(valid_len[i]), 1)
                ratios.append(sum(e - s for s, e in regs) / vlen)
                for s, e in regs:
                    int_len_all.append(e - s)
                    int_ratio_all.append((e - s) / vlen)
                    start_frac_all.append(s / vlen)
            ratios = np.asarray(ratios)
            info[m] = dict(n_with_missing=int(has),
                           frac_with_missing=float(has / N),
                           mean_missing_ratio=float(ratios.mean()),
                           p50=float(np.percentile(ratios, 50)),
                           p90=float(np.percentile(ratios, 90)),
                           p99=float(np.percentile(ratios, 99)),
                           max=float(ratios.max()))
            print("  %-6s 含缺失样本 %d/%d (%.1f%%)  缺失率 mean=%.3f p50=%.3f p90=%.3f max=%.3f"
                  % (m, has, N, 100 * has / N, ratios.mean(),
                     info[m]["p50"], info[m]["p90"], info[m]["max"]))

        for i in range(N):
            mods = [m for m in MODALITIES if len(missing_regions(sp["miss"][m][i], a.min_zero_len)) > 0]
            combo_count["none" if not mods else "+".join(mods)] += 1
            per_sample_regions.append(
                [[m, s, e] for m in MODALITIES
                 for s, e in missing_regions(sp["miss"][m][i], a.min_zero_len)])
        info["combo"] = combo_count
        print("  缺失模态组合: %s" % {k: v for k, v in combo_count.items() if v})

        if int_len_all:
            il = np.asarray(int_len_all)
            rf = np.asarray(start_frac_all)
            info["interval"] = dict(count=len(il), len_min=int(il.min()),
                                    len_mean=float(il.mean()), len_p50=float(np.percentile(il, 50)),
                                    len_max=int(il.max()),
                                    head=float((rf < 0.2).mean()),
                                    middle=float(((rf >= 0.2) & (rf <= 0.8)).mean()),
                                    tail=float((rf > 0.8).mean()))
            print("  缺失区间数=%d  长度(步) min/mean/p50/max=%d/%.1f/%.0f/%d"
                  % (len(il), il.min(), il.mean(), np.percentile(il, 50), il.max()))
            print("  区间起始位置分布: 头部(<20%%)=%.2f 中部=%.2f 尾部(>80%%)=%.2f"
                  % (info["interval"]["head"], info["interval"]["middle"], info["interval"]["tail"]))
            info["interval"]["ratio_p5"] = float(np.percentile(int_ratio_all, 5))
            info["interval"]["ratio_p95"] = float(np.percentile(int_ratio_all, 95))

        # ---- 反推训练模拟参数建议 ----
        n_mods = [len(m.split("+")) for m in combo_count if m != "none"]
        if int_len_all:
            sug = dict(
                rho_min=round(float(np.percentile(int_ratio_all, 5)), 3),
                rho_max=round(float(np.percentile(int_ratio_all, 95)), 3),
                mode=("double" if n_mods and np.mean(n_mods) > 1.5 else "mixed"),
                pos_modes_weights=dict(head=round(info["interval"]["head"], 2),
                                       middle=round(info["interval"]["middle"], 2),
                                       tail=round(info["interval"]["tail"], 2)),
                note="把 rho_min/rho_max 覆盖到测试集缺失率区间，"
                     "curric 第三阶段的 rho 上限应不低于测试集 p95")
            info["suggest_train_params"] = sug
            print("  [建议训练模拟参数] %s" % json.dumps(sug, ensure_ascii=False))

        rep["splits"][name] = info

        import pandas as pd
        pd.DataFrame({"id": sp["ids"],
                      "missing_regions": [json.dumps(r, ensure_ascii=False) for r in per_sample_regions]}
                     ).to_csv(os.path.join(a.out_dir, "att3_missing_per_sample.csv"),
                              index=False, encoding="utf-8-sig")

    os.makedirs(a.out_dir, exist_ok=True)
    p = os.path.join(a.out_dir, "att3_missing_report.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("\n[SAVE] %s" % p)


if __name__ == "__main__":
    main()
