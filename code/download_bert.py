# -*- coding: utf-8 -*-
r"""下载并本地缓存 BERT（走 hf-mirror.com 镜像，本机 huggingface.co 直连被拒）。"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"E:\hf_cache")

from transformers import AutoModel, AutoTokenizer     # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
print("HF_ENDPOINT =", os.environ["HF_ENDPOINT"])
print("HF_HOME     =", os.environ["HF_HOME"])
tk = AutoTokenizer.from_pretrained(name)
mo = AutoModel.from_pretrained(name)
n = sum(p.numel() for p in mo.parameters()) / 1e6
print("OK  model=%s  params=%.1fM  vocab=%d  hidden=%d"
      % (name, n, tk.vocab_size, mo.config.hidden_size))
print("cache dir   =", mo.name_or_path)
