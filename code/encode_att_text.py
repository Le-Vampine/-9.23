# -*- coding: utf-8 -*-
r"""
为附件3/附件4 生成文本模态特征（它们只提供 text_bert / raw_text，没有预计算的 text(768)）。

已通过 verify_text_encoder.py 验证：bert-base-uncased 的 last_hidden_state 与附件2 的
text 余弦相似度 = 1.0000，即同一特征空间。因此用同一 BERT 编码附件3/4 的 text_bert，
训练好的模型可直接推理，无需重训。

产出：一个与附件2 同构的单 split pkl（字段：id, text, audio, vision [, *_lengths]）
      保存到 data_att/，供 calibrate_missing.py / infer_att3.py 使用。

用法：
  python encode_att_text.py --kind att3 --version aligned
  python encode_att_text.py --kind att4 --version aligned
  python encode_att_text.py --kind att3 --version unaligned   # 走 raw_text 分词
"""
import argparse
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"E:\hf_cache")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "02_model"))
import att_paths                                    # noqa: E402
from data_utils import load_pickle                  # noqa: E402

import torch                                        # noqa: E402
from transformers import AutoModel, AutoTokenizer   # noqa: E402

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 50


def norm_tb(tb, L=MAX_LEN):
    """把 text_bert 规范成 (N,3,L)：兼容 (N,3,L) / (3,L) / (L,) 等存储形态。"""
    a = np.asarray(tb)
    if a.ndim == 3:
        return a if a.shape[1] == 3 else np.transpose(a, (0, 2, 1))
    if a.ndim == 2:
        if a.shape[0] == 3:
            return a[None]
        if a.shape[1] == 3:
            return np.transpose(a, (1, 0))[None]
        return a[None]
    if a.ndim == 1:
        return a.reshape(1, 1, -1)
    raise ValueError("无法识别的 text_bert 形状 %s" % (a.shape,))


def as3d(a):
    """把任意存储形态规范成 (N,L,d)。附件4 的数组是二维存储（每文件 1 个样本）。"""
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(1, -1, 1)
    if a.ndim == 2:
        return a[None]
    return a


def raw_of(d):
    raw = d
    if isinstance(d, dict):
        for k in ("test", "data", "train", "valid"):
            if k in d and isinstance(d[k], dict):
                raw = d[k]
                break
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="att3", choices=["att3", "att4"])
    ap.add_argument("--version", default="aligned", choices=["aligned", "unaligned"])
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()

    files = att_paths.discover(a.kind, a.version)
    if not files:
        raise SystemExit("未找到 %s/%s 的 pkl" % (a.kind, a.version))
    print("[FILES] %d 个（%s / %s）" % (len(files), a.kind, a.version))

    tk = AutoTokenizer.from_pretrained(a.model)
    bert = AutoModel.from_pretrained(a.model).eval()

    ids_all, text_all, aud_all, vis_all = [], [], [], []
    alen_all, vlen_all = [], []
    for f in files:
        raw = raw_of(load_pickle(f))
        stem = os.path.splitext(os.path.basename(f))[0]
        A = as3d(raw["audio"]) if "audio" in raw else None
        V = as3d(raw["vision"]) if "vision" in raw else None
        n = max([x.shape[0] for x in (A, V) if x is not None] + [1])

        # ---------- 文本 ----------
        if "text" in raw:
            T = np.asarray(raw["text"], dtype=np.float32)
        elif "text_bert" in raw:
            tb = norm_tb(raw["text_bert"])
            ids = torch.tensor(tb[:, 0, :].astype(np.int64), dtype=torch.long)
            am = torch.tensor(tb[:, 1, :].astype(np.int64), dtype=torch.long)
            seg = torch.tensor(tb[:, 2, :].astype(np.int64), dtype=torch.long)
            with torch.no_grad():
                T = bert(input_ids=ids, attention_mask=am, token_type_ids=seg).last_hidden_state
            T = T.numpy().astype(np.float32)
        elif "raw_text" in raw:
            rt = raw["raw_text"]
            rt = [rt] if isinstance(rt, str) else list(np.asarray(rt).reshape(-1))
            enc = tk(list(rt), max_length=MAX_LEN, padding="max_length",
                     truncation=True, return_tensors="pt")
            with torch.no_grad():
                T = bert(**enc).last_hidden_state.numpy().astype(np.float32)
        else:
            raise SystemExit("%s 既无 text/text_bert/raw_text" % f)
        if T.ndim == 2:
            T = T[None]
        if T.shape[0] < n:                       # 文本行数不足时复制补齐
            T = np.repeat(T[:1], n, axis=0)

        A = as3d(raw["audio"]).astype(np.float32)
        V = as3d(raw["vision"]).astype(np.float32)

        L = T.shape[1]
        ids_all.append([stem] if n == 1 else ["%s_%d" % (stem, i) for i in range(n)])
        text_all.append(T[:, :L])
        aud_all.append(A)
        vis_all.append(V)
        if "audio_lengths" in raw:
            alen_all.append(np.asarray(raw["audio_lengths"], dtype=np.int64))
        if "vision_lengths" in raw:
            vlen_all.append(np.asarray(raw["vision_lengths"], dtype=np.int64))
        print("  %-28s n=%d  text=%s audio=%s vision=%s" % (stem, n, T.shape, A.shape, V.shape))

    out = dict(
        id=np.concatenate([np.asarray(x, dtype=object) for x in ids_all]),
        text=np.concatenate(text_all, axis=0),
        audio=np.concatenate(aud_all, axis=0),
        vision=np.concatenate(vis_all, axis=0),
    )
    if alen_all and len(alen_all) == len(files):
        out["audio_lengths"] = np.concatenate(alen_all)
    if vlen_all and len(vlen_all) == len(files):
        out["vision_lengths"] = np.concatenate(vlen_all)
    out["raw_text"] = np.array([""] * len(out["id"]), dtype=object)

    out_dir = a.out_dir or os.path.join(HERE, "..", "data_att")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "%s_%s.pkl" % (a.kind, a.version))
    with open(p, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print("\n[SAVE] %s" % os.path.normpath(p))
    print("[SHAPE] N=%d text=%s audio=%s vision=%s"
          % (len(out["id"]), out["text"].shape, out["audio"].shape, out["vision"].shape))
    print("[IDS] %s ... %s" % (out["id"][0], out["id"][-1]))
    print("[提示] 该文件与附件2 同构，可直接用于 calibrate_missing.py 与 infer_att3.py")


if __name__ == "__main__":
    main()
