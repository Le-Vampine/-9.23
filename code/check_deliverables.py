# -*- coding: utf-8 -*-
"""
交付物与数据资产完成度检查（问题2）。

用法：python check_deliverables.py
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # e:\数学建模
CODE = os.path.join(ROOT, "code")


def hr(t):
    print("\n" + "=" * 78)
    print("== %s" % t)
    print("=" * 78)


def size_mb(p):
    return os.path.getsize(p) / 1e6


def main():
    hr("1. 数据资产")
    want = {
        "附件2 / aligned_50.pkl": None,
        "附件2 / unaligned_50.pkl": None,
        "附件2 / label.xlsx": None,
        "附件3 (模态缺失测试集)": None,
        "附件4 (可解释测试集)": None,
    }
    found = {}
    for p in glob.glob(os.path.join(ROOT, "**", "*.pkl"), recursive=True):
        if "data_dummy" in p:
            continue
        found.setdefault(os.path.basename(p), []).append(p)
    for p in glob.glob(os.path.join(ROOT, "**", "*.xlsx"), recursive=True):
        found.setdefault(os.path.basename(p), []).append(p)

    for name in ("aligned_50.pkl", "unaligned_50.pkl"):
        hit = found.get(name)
        print("  %-26s %s" % (name, ("%.1f MB  %s" % (size_mb(hit[0]), hit[0])) if hit else "**缺失**"))
    for name in ("label.xlsx",):
        hit = found.get(name)
        print("  %-26s %s" % (name, ("%.2f MB  %s" % (size_mb(hit[0]), hit[0])) if hit else "**缺失**"))

    others = {k: v for k, v in found.items()
              if k not in ("aligned_50.pkl", "unaligned_50.pkl", "label.xlsx")}
    print("  其它 pkl/xlsx: %s" % (list(others.keys()) if others else "无"))
    print("  ⚠ 附件3 / 附件4 是否已导入: %s" % ("是" if others else "**否**"))

    hr("2. 代码模块")
    mods = ["config.py", "data_utils.py", "missing_sim.py", "model.py", "losses.py",
            "train.py", "infer_att3.py", "analyze_sweep.py"]
    for m in mods:
        p = os.path.join(CODE, "02_model", m)
        n = len(open(p, encoding="utf-8").readlines()) if os.path.isfile(p) else 0
        print("  %-20s %s (%d 行)" % (m, "OK" if n else "缺失", n))
    for m in ["inspect_data.py", "make_dummy_data.py", "run_smoke_test.cmd",
              "requirements.txt", "README.md"]:
        p = os.path.join(CODE, m)
        print("  %-20s %s" % (m, "OK" if os.path.isfile(p) else "缺失"))

    # 论文/实验脚本（本轮已补齐）
    print("  -- 实验与论文配套脚本 --")
    for m in ["runtime.py (集成推理运行时)", "calibrate_missing.py (附件3 缺失分布标定)",
              "viz_results.py (混淆矩阵/散点/缺失可视化/曲线)", "error_analysis.py (分层错误归因)",
              "analyze_sweep.py (ANOVA + 退化曲线)", "tune_hyperparams.py (valid 上选超参)"]:
        f = m.split(" ")[0]
        p = os.path.join(CODE, "02_model", f)
        print("     [%s] %s" % ("x" if os.path.isfile(p) else " ", m))
    print("  -- 仍待补充 --")
    for m in ["make_tables.py (论文表格汇总导出)", "ablation 开关统一化 (config 中加入 ablate_*)",
              "requirements_lock.txt (pip freeze 复现清单)"]:
        print("     [ ] %s" % m)

    hr("3. 训练产物与模型体积")
    cks = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "student_*.pt")))
    if not cks:
        print("  尚无 checkpoint")
    for c in cks:
        print("  %-46s %.2f MB" % (os.path.relpath(c, ROOT), size_mb(c)))
    tot = sum(size_mb(c) for c in cks)
    print("  checkpoint 合计 %.2f MB（提交预算 ≤50 MB，建议只留最终 3~5 个）" % tot)

    hr("4. 指标结果")
    for j in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "metrics_*.json"))):
        try:
            d = json.load(open(j, encoding="utf-8"))
            v = d.get("valid_clean", {})
            t = d.get("test_clean", {})
            print("  %-34s valid MAE=%.4f ACC=%.4f | test MAE=%.4f ACC=%.4f"
                  % (os.path.relpath(j, ROOT), v.get("mae", -1), v.get("acc", -1),
                     t.get("mae", -1), t.get("acc", -1)))
        except Exception as e:
            print("  %s 读取失败 %r" % (j, e))

    hr("5. 提交文件")
    for c in sorted(glob.glob(os.path.join(ROOT, "submission", "*.csv"))):
        with open(c, encoding="utf-8-sig") as f:
            n = sum(1 for _ in f) - 1
        print("  %-40s %d 行  %.3f MB" % (os.path.relpath(c, ROOT), n, size_mb(c)))


if __name__ == "__main__":
    main()
