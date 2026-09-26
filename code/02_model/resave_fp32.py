# -*- coding: utf-8 -*-
r"""
把 float16 保存的 ckpt 重存为 float32（用于验证/规避 fp16 量化带来的精度损失）。

用法：
  python 02_model\resave_fp32.py --src ..\runs\exp_c1\student_s42.pt --dst_dir ..\runs\exp_c1_fp32
"""
import argparse
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.dst_dir, exist_ok=True)
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    n_half = sum(1 for v in ck["state"].values() if torch.is_tensor(v) and v.dtype == torch.float16)
    ck["state"] = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                   for k, v in ck["state"].items()}
    ck["half"] = False
    dst = os.path.join(a.dst_dir, os.path.basename(a.src))
    torch.save(ck, dst)
    print("[RESAVE] %s -> %s（fp16 张量 %d 个已转 fp32，%.2f MB）"
          % (a.src, dst, n_half, os.path.getsize(dst) / 1e6))


if __name__ == "__main__":
    main()
