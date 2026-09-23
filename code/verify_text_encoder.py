# -*- coding: utf-8 -*-
r"""
关键验证：附件3/4 只提供 text_bert（词元编号），而模型是在附件2 的 text(768) 上训练的。
本脚本检验"用 bert-base-uncased 编码 text_bert 能否复现附件2 的 text"。

若能复现（余弦相似度高）→ 可直接用同一 BERT 为附件3/4 生成 text 特征，训练好的模型依然有效。
若不能 → 必须自建文本编码器并重训（保持训练/测试同源）。

用法：python verify_text_encoder.py [--n 8] [--version aligned]
"""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"E:\hf_cache")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_model"))
from data_utils import load_pickle                      # noqa: E402
import att_paths                                        # noqa: E402

import torch                                            # noqa: E402
from transformers import AutoModel, AutoTokenizer       # noqa: E402

MODEL_NAME = "bert-base-uncased"


def cos(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--model", default=MODEL_NAME)
    a = ap.parse_args()

    tk = AutoTokenizer.from_pretrained(a.model)
    bert = AutoModel.from_pretrained(a.model, output_hidden_states=True)
    bert.eval()

    pkl = os.path.join(att_paths.ROOT, "附件2", "aligned_50.pkl")
    if not os.path.isfile(pkl):
        cands = att_paths.discover_att3("aligned")
        raise SystemExit("找不到附件2: %s" % pkl)
    print("[DATA] %s" % pkl)
    d = load_pickle(pkl)
    tr = d["train"]
    tb = np.asarray(tr["text_bert"])[:a.n].astype(np.int64)      # (n,3,50)
    text = np.asarray(tr["text"])[:a.n].astype(np.float32)       # (n,50,768)
    print("[SHAPE] text_bert=%s  text=%s" % (tb.shape, text.shape))

    ids = torch.tensor(tb[:, 0, :], dtype=torch.long)
    am = torch.tensor(tb[:, 1, :], dtype=torch.long)
    seg = torch.tensor(tb[:, 2, :], dtype=torch.long)
    with torch.no_grad():
        out = bert(input_ids=ids, attention_mask=am, token_type_ids=seg)

    cands = {
        "last_hidden": out.last_hidden_state.numpy(),
        "layer_-2": out.hidden_states[-2].numpy(),
        "layer_-3": out.hidden_states[-3].numpy(),
        "last4_mean": np.mean([h.numpy() for h in out.hidden_states[-4:]], axis=0),
    }

    print("\n" + "=" * 78)
    print("[验证] 各编码方案与附件2 text 的一致性（逐样本平均余弦）")
    best = None
    for name, H in cands.items():
        per = []
        for i in range(a.n):
            L = int(am[i].sum())
            c = cos(H[i, :L], text[i, :L])
            per.append(c)
        m = float(np.mean(per))
        print("  %-12s mean_cos=%.4f  min=%.4f  max=%.4f  |  首位置 cos=%.4f"
              % (name, m, min(per), max(per),
                 float(np.mean([cos(H[i, 0], text[i, 0]) for i in range(a.n)]))))
        if best is None or m > best[1]:
            best = (name, m, H)

    print("\n[最佳方案] %s (mean_cos=%.4f)" % (best[0], best[1]))
    H = best[2]
    # 维度级检查：是否只差一个线性/尺度变换
    X = np.concatenate([H[i, :int(am[i].sum())] for i in range(a.n)], axis=0).astype(np.float64)
    Y = np.concatenate([text[i, :int(am[i].sum())] for i in range(a.n)], axis=0).astype(np.float64)
    print("[统计] 编码输出 std=%.4f 附件2 text std=%.4f" % (X.std(), Y.std()))
    print("[统计] 编码输出 mean=%.4f 附件2 text mean=%.4f" % (X.mean(), Y.mean()))
    # 逐维相关（看是否存在维度置换/线性关系）
    xs = X - X.mean(0)
    ys = Y - Y.mean(0)
    num = (xs * ys).sum(0)
    den = np.sqrt((xs ** 2).sum(0) * (ys ** 2).sum(0)) + 1e-12
    per_dim = num / den
    print("[逐维相关] mean=%.4f  p50=%.4f  p90=%.4f  >0.9 的维度占比=%.2f"
          % (per_dim.mean(), np.percentile(per_dim, 50), np.percentile(per_dim, 90),
             float((per_dim > 0.9).mean())))

    verdict = ("可直接共用同一文本特征空间（用 bert-base-uncased 为附件3/4 生成 text）"
               if best[1] > 0.9 else
               "**不能直接复用** —— 需自建文本编码器并重训，保证训练/测试同源")
    print("\n[结论] %s" % verdict)


if __name__ == "__main__":
    main()
