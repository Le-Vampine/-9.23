"""附件3 是否真的没有文本缺失？——用 token id 判据，而不是"全零帧"判据。

背景：附件3/4 只给 `text_bert`（= (input_ids, attention_mask, token_type_ids)，每样本 3x50）
+ `raw_text`，768 维文本特征是我们用 bert-base-uncased 现算的。BERT 的输出受位置编码
影响**永不为全零**，所以"全零帧"判据对文本无效。

真正的判据（三条，逐条检查）：
  P1 有效长度 vlen：若某样本 vlen≈2（只有 [CLS][SEP]），说明该样本文本为空 → 类似缺失
  P2 有效区间内的 [PAD](id=0)：出现内部 0 说明文本被打洞
  P3 raw_text 是否为空：空字符串 = 转写失败

用法：python att3_text_check.py --root <工程根>
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

sys.stdout.reconfigure(errors="replace")


def norm_tb(tb):
    """把 text_bert 规范成 (N,3,L)。"""
    a = np.asarray(tb)
    if a.ndim == 3:
        return a if a.shape[1] == 3 else np.transpose(a, (0, 2, 1))
    if a.ndim == 2:
        if a.shape[0] == 3:
            return a[None]
        if a.shape[1] == 3:
            return np.transpose(a, (1, 0))[None]
        return a[None]
    return a.reshape(1, 1, -1)


def check_tb(tb, tag):
    """返回 (vlen 列表, 内部零 token 总数, 零注意力样本数)。"""
    A = norm_tb(tb)
    vlens, interior_zeros, zero_mask = [], 0, 0
    for n in range(A.shape[0]):
        ids = np.asarray(A[n, 0]).astype(np.int64)
        att = np.asarray(A[n, 1]).astype(np.int64) if A.shape[1] > 1 else (ids != 0).astype(np.int64)
        v = int(att.sum()) if att.sum() > 0 else int((ids != 0).sum())
        vlens.append(v)
        if v <= 0:
            zero_mask += 1
            continue
        seg = ids[:v]
        interior_zeros += int((seg == 0).sum())
    return vlens, interior_zeros, zero_mask


def dump(name, vlens, iz, zm, raw=None):
    v = np.asarray(vlens)
    print("\n  [%s] 样本数=%d" % (name, len(v)))
    print("    vlen: min=%d p25=%.0f 中位=%.0f p75=%.0f max=%d 均值=%.1f"
          % (v.min(), np.percentile(v, 25), np.median(v), np.percentile(v, 75),
             v.max(), v.mean()))
    print("    vlen<=2 的样本数（= 文本几乎为空）: %d" % int((v <= 2).sum()))
    print("    vlen<=5 的样本数: %d" % int((v <= 5).sum()))
    print("    有效区间内出现的 [PAD](id=0) 总数: %d  → %s"
          % (iz, "无内部打洞" if iz == 0 else "**存在内部打洞**"))
    print("    注意力全零（无有效 token）样本数: %d" % zm)
    if raw is not None:
        lens = [len(str(x).strip()) for x in raw]
        print("    raw_text 字符长度: min=%d 中位=%.0f max=%d；空串数=%d / %d"
              % (min(lens), np.median(lens), max(lens),
                 sum(1 for x in lens if x == 0), len(lens)))
        ne = [i for i, x in enumerate(lens) if x == 0]
        if ne:
            print("    **空 raw_text 的样本下标（前 10 个）**: %s" % ne[:10])
    print("    逐样本 vlen（前 30）:", v[:30].tolist())


def peek(d, depth=0, maxd=2):
    """递归打印 dict 的键与数组形状。"""
    pad = "    " + "  " * depth
    if isinstance(d, dict):
        for k in d:
            v = d[k]
            if isinstance(v, dict):
                print("%s%s: dict" % (pad, k))
                if depth < maxd:
                    peek(v, depth + 1, maxd)
            elif isinstance(v, np.ndarray):
                print("%s%s: ndarray%s %s" % (pad, k, v.shape, v.dtype))
            elif isinstance(v, (list, tuple)):
                print("%s%s: %s len=%d 首元素=%r" % (pad, k, type(v).__name__, len(v),
                                                 str(v[0])[:40] if v else None))
            else:
                print("%s%s: %s %r" % (pad, k, type(v).__name__, str(v)[:60]))
    else:
        print("%s%s" % (pad, type(d).__name__))


def descend(d):
    """附件3 原始 pkl 顶层可能是 {'test': {...}}，下钻到含 text_bert 的那一层。"""
    for _ in range(3):
        if isinstance(d, dict) and "text_bert" not in d and len(d) == 1:
            k = list(d.keys())[0]
            if isinstance(d[k], dict):
                d = d[k]
                continue
        break
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--peek", action="store_true", help="打印原始 pkl 结构")
    args = ap.parse_args()

    print("=" * 76)
    print("判据：vlen（有效 token 数）+ 内部 [PAD] + raw_text 是否为空")
    print("=" * 76)

    # ---- 1) 附件3 原始 30 个 pkl（对齐版本）----
    files = sorted(glob.glob(os.path.join(args.root, "附件3*", "**", "*.pkl"),
                             recursive=True))
    files = [f for f in files if "未对齐" not in f]
    if args.peek and files:
        print("\n[0] 结构探查 %s" % os.path.basename(files[0]))
        with open(files[0], "rb") as fh:
            peek(pickle.load(fh))
    print("\n[1] 附件3 原始 pkl（对齐版本）：找到 %d 个文件" % len(files))
    v_all, iz_all, zm_all, raw_all, ok = [], 0, 0, [], 0
    for f in files:
        try:
            with open(f, "rb") as fh:
                d = pickle.load(fh)
        except Exception as e:
            print("    !! 读取失败 %s: %s" % (os.path.basename(f), e))
            continue
        d = descend(d)
        if "text_bert" not in d:
            print("    !! %s 无 text_bert，键=%s" % (os.path.basename(f), list(d.keys())))
            continue
        ok += 1
        v, iz, zm = check_tb(d["text_bert"], os.path.basename(f))
        v_all += v
        iz_all += iz
        zm_all += zm
        if "raw_text" in d:
            rt = d["raw_text"]
            raw_all += list(rt) if isinstance(rt, (list, tuple, np.ndarray)) else [rt]
    if v_all:
        dump("附件3 原文 token", v_all, iz_all, zm_all, raw_all or None)
        print("    成功解析文件数=%d / %d" % (ok, len(files)))


    # ---- 2) 我们生成的 att3_aligned.pkl ----
    for fn, tag in (("att3_aligned.pkl", "附件3(我们编码后)"),
                    ("att4_aligned.pkl", "附件4(我们编码后)")):
        p = os.path.join(args.root, "data_att", fn)
        if not os.path.isfile(p):
            continue
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        n = len(d["id"])
        print("\n[2] %s  n=%d  keys=%s" % (tag, n, sorted(d.keys())))
        raw = d.get("raw_text", None)
        X = np.asarray(d["text"], dtype=np.float32)
        # 文本特征的行范数（BERT 输出不会全零，但可看是否"退化"）
        nrm = np.linalg.norm(X, axis=-1)
        print("    文本特征行范数: 均值=%.2f 最小=%.2f（全零行数=%d）"
              % (nrm.mean(), nrm.min(), int((nrm == 0).sum())))
        if raw is not None:
            lens = [len(str(x).strip()) for x in raw]
            print("    raw_text 字符长度: min=%d 中位=%.0f max=%d；空串数=%d / %d"
                  % (min(lens), np.median(lens), max(lens),
                     sum(1 for x in lens if x == 0), len(lens)))
            print("    空串下标: %s" % [i for i, x in enumerate(lens) if x == 0][:10])
            print("    示例 raw_text[0]: %r" % str(raw[0])[:80])

    # ---- 3) 对照：附件2 训练集的 text_bert vlen ----
    p2 = os.path.join(args.root, "附件2", "aligned_50.pkl")
    if os.path.isfile(p2):
        with open(p2, "rb") as fh:
            d2 = pickle.load(fh)
        tr = d2["train"]
        if "text_bert" in tr:
            v, iz, zm = check_tb(tr["text_bert"], "train")
            dump("对照 附件2 train text_bert", v, iz, zm,
                 tr.get("raw_text", None))


if __name__ == "__main__":
    main()
