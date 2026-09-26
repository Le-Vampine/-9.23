"""对比两版提交 CSV（行数/列名/标签变化/分数漂移）。"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(errors="replace")


def main():
    a, b = sys.argv[1], sys.argv[2]
    da, db = pd.read_csv(a), pd.read_csv(b)
    print("=" * 72)
    print("旧: %s   行数=%d  列数=%d" % (a, len(da), da.shape[1]))
    print("新: %s   行数=%d  列数=%d" % (b, len(db), db.shape[1]))
    if list(da.columns) != list(db.columns):
        print("!! 列名不同：")
        print("   旧:", list(da.columns))
        print("   新:", list(db.columns))
    if "id" in da.columns and "id" in db.columns:
        same = (da["id"].astype(str).tolist() == db["id"].astype(str).tolist())
        print("id 顺序一致: %s" % same)
    for col in ("pred_label", "pred_label_head"):
        if col in da.columns and col in db.columns:
            n = int((da[col].astype(str) != db[col].astype(str)).sum())
            print("  %-18s 变化 %d / %d 行" % (col, n, len(da)))
            if n:
                print("    旧分布:", da[col].value_counts().to_dict())
                print("    新分布:", db[col].value_counts().to_dict())
    for col in ("pred_score", "pred_score_raw", "score", "score_theta"):
        if col in da.columns and col in db.columns:
            x = da[col].to_numpy(dtype=float)
            y = db[col].to_numpy(dtype=float)
            print("  %-18s 均值 %.4f → %.4f ；平均|Δ|=%.4f ；最大|Δ|=%.4f"
                  % (col, x.mean(), y.mean(), np.abs(x - y).mean(), np.abs(x - y).max()))
    print("\n新版前 3 行：")
    print(db.head(3).to_string())


if __name__ == "__main__":
    main()
