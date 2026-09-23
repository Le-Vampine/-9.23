# -*- coding: utf-8 -*-
"""
label.xlsx 核查：字段、规模、标签分布，以及与附件2 pkl 的 id 对应关系。

用法：python inspect_label.py
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    xs = [p for p in glob.glob(os.path.join(ROOT, "**", "*.xlsx"), recursive=True)
          if "data_dummy" not in p]
    print("\n" + "=" * 78)
    print("[XLSX] 找到 %s" % xs)
    for p in xs:
        print("\n--- %s (%.2f MB) ---" % (p, os.path.getsize(p) / 1e6))
        book = pd.read_excel(p, sheet_name=None)
        for sheet, df in book.items():
            print("[sheet] %s  shape=%s" % (sheet, df.shape))
            print("  columns: %s" % list(df.columns))
            print("  dtypes : %s" % dict(df.dtypes.astype(str)))
            with pd.option_context("display.max_colwidth", 60, "display.width", 200):
                print(df.head(5).to_string())
            for c in df.columns:
                if df[c].dtype == object or df[c].nunique() <= 12:
                    vc = df[c].value_counts().head(12)
                    print("  value_counts[%s]: %s" % (c, dict(vc)))
        # 与 pkl 的 id 对照
        try:
            sys_path = os.path.join(ROOT, "code", "02_model")
            import sys
            sys.path.append(sys_path)
            from data_utils import load_pickle
            pk = glob.glob(os.path.join(ROOT, "附件2", "aligned_50.pkl"))
            if pk and "id" in book[list(book.keys())[0]].columns:
                d = load_pickle(pk[0])
                ids = set()
                for s in ("train", "valid", "test"):
                    ids |= set(np.asarray(d[s]["id"]).tolist())
                xid = set(book[list(book.keys())[0]]["id"].astype(str).tolist())
                print("\n  [ID 对照] pkl id 数=%d, xlsx id 数=%d, 交集=%d"
                      % (len(ids), len(xid), len(ids & xid)))
                inter = list(ids & xid)[:3]
                print("  交集示例: %s" % inter)
        except Exception as e:
            print("  [ID 对照失败] %r" % e)


if __name__ == "__main__":
    main()
