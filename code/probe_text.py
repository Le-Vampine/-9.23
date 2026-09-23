# -*- coding: utf-8 -*-
r"""
附件3 文本模态缺失表示方式探查 + HuggingFace 连通性（绕过系统代理）。

背景：附件3 只提供 text_bert（词元编号/掩码/分段），不提供 text(768)。
需要确认"文本缺失"在 text_bert 中如何体现（是否某些位置三通道全零）。
"""
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "02_model"))
import att_paths                                      # noqa: E402
from data_utils import load_pickle                    # noqa: E402


def norm_tb(tb):
    """把 text_bert 规范成 (N,3,50)。"""
    a = np.asarray(tb)
    if a.ndim == 2 and a.shape[0] == 3:
        return a[None]
    if a.ndim == 2 and a.shape[1] == 3:
        return np.transpose(a[:, None], (1, 2, 0)) if False else a.T[None]
    if a.ndim == 3 and a.shape[1] == 3:
        return a
    if a.ndim == 3 and a.shape[2] == 3:
        return np.transpose(a, (0, 2, 1))
    if a.ndim == 1:
        return a.reshape(1, 1, -1)
    return a


def analyze_text(kind, version, max_n=40):
    files = att_paths.discover(kind, version)
    print("\n" + "=" * 78)
    print("[文本模态缺失探查] %s / %s  (%d 文件)" % (kind, version, len(files)))
    if not files:
        return None
    stat = dict(n=0, with_tb=0, interior_zero_cols=0, tail_zero_only=0,
                interior_runs=[], token_id_zero_inside=0)
    samples = []
    for f in files[:max_n]:
        d = load_pickle(f)
        raw = d
        if isinstance(d, dict):
            for k in ("test", "data", "train", "valid"):
                if k in d and isinstance(d[k], dict):
                    raw = d[k]
                    break
        if "text_bert" not in raw:
            stat["n"] += 1
            continue
        tb = norm_tb(raw["text_bert"])           # (N,3,L)
        stat["with_tb"] += 1
        for i in range(tb.shape[0]):
            stat["n"] += 1
            x = tb[i]                                   # (3,L)
            am = x[1].astype(np.int64)                  # attention mask
            ids = x[0]
            nz = np.nonzero(am)[0]
            vlen = int(nz[-1] + 1) if len(nz) else 0
            allzero = (np.abs(x).sum(axis=0) < 1e-8)    # 每列三通道是否全零
            inner = allzero[:vlen]
            runs, j = [], 0
            while j < len(inner):
                if inner[j]:
                    k = j
                    while k < len(inner) and inner[k]:
                        k += 1
                    if k - j >= 2:
                        runs.append((j, k))
                    j = k
                else:
                    j += 1
            if runs:
                stat["interior_zero_cols"] += 1
                stat["interior_runs"] += [e - s for s, e in runs]
            if np.any(ids[:vlen] == 0) and len(np.nonzero(ids[:vlen] == 0)[0]):
                stat["token_id_zero_inside"] += 1
            samples.append(dict(file=os.path.basename(f), shape=list(np.asarray(raw["text_bert"]).shape),
                                vlen=vlen, interior_runs=runs,
                                mask_head=[int(v) for v in am[:12]],
                                ids_head=[int(v) for v in ids[:12]]))
    print("[样本] 有 text_bert 的文件 %d/%d" % (stat["with_tb"], stat["n"]))
    print("[文本有效长度] %s" % [s["vlen"] for s in samples[:15]])
    print("[attention mask 前12] %s" % [s["mask_head"] for s in samples[:3]])
    print("[token ids 前12]      %s" % [s["ids_head"] for s in samples[:3]])
    print("[有效区间内出现'三通道全零列'的样本数] %d" % stat["interior_zero_cols"])
    if stat["interior_runs"]:
        r = np.asarray(stat["interior_runs"])
        print("   这些零段长度 min/mean/max = %d/%.1f/%d" % (r.min(), r.mean(), r.max()))
    else:
        print("   → 附件3 的文本缺失**不在 text_bert 中体现**（词元编号/掩码无中断）")
    stat["examples"] = samples[:5]
    return stat


def check_hf():
    print("\n" + "=" * 78)
    print("[HuggingFace 连通性]")
    try:
        urllib.request.getproxies_registry = lambda: {}
    except Exception:
        pass
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    urls = ["https://huggingface.co", "https://hf-mirror.com",
            "https://modelscope.cn", "https://pypi.tuna.tsinghua.edu.cn"]
    res = {}
    for u in urls:
        try:
            r = opener.open(urllib.request.Request(u, method="HEAD"), timeout=10)
            res[u] = "OK %d" % r.status
        except Exception as e:
            res[u] = "FAIL %s" % str(e)[:60]
        print("  %-40s %s" % (u, res[u]))
    return res


if __name__ == "__main__":
    out = {}
    for kind, ver in (("att3", "aligned"), ("att4", "aligned")):
        out["%s_%s" % (kind, ver)] = analyze_text(kind, ver)
    out["hf"] = check_hf()
    p = os.path.join(HERE, "..", "runs", "text_probe.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("[SAVE] %s" % os.path.normpath(p))
