# -*- coding: utf-8 -*-
"""修正 att_meta.json 中的视频时长：改用 ffmpeg 读取容器时长（此前 mvhd 手工解析有误）。

背景：2026-09-25 用 imageio-ffmpeg 自带的 ffmpeg 逐个读取容器 Duration，
发现 16/20 条附件四样本与 att_meta 记录不一致（最大 2.2 倍），导致：
  * 语音证据秒区间越出视频时长；
  * 关键帧按错误的时刻抽取（定位偏移）。
本脚本把 duration_s 改为 ffmpeg 读到的容器时长，旧值保留在 duration_s_mvhd_legacy 以便审计。

用法：
  python fix_durations.py            # 打印对照表，不改文件
  python fix_durations.py --apply    # 写回 data_att/att_meta.json（自动备份）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
META = os.path.join(ROOT, "data_att", "att_meta.json")


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_duration(path, ff):
    out = subprocess.run([ff, "-i", path], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    ff = ffmpeg_exe()
    meta = json.load(open(META, encoding="utf-8"))
    print("%-6s %-10s %10s %10s %8s" % ("集合", "id", "旧(mvhd)", "新(ffmpeg)", "比值"))
    n_fix = 0
    for key in ("att3", "att4"):
        for e in meta.get(key, []):
            v = e.get("video")
            old = e.get("duration_s")
            if not v or not os.path.exists(v):
                continue
            new = probe_duration(v, ff)
            if new is None:
                print("%-6s %-10s %10s %10s  探测失败" % (key, e.get("id"), old, "-"))
                continue
            flag = abs(new - (old or 0)) > 0.01
            if flag:
                n_fix += 1
            if key == "att4" or flag:
                print("%-6s %-10s %10.3f %10.3f %8s%s" % (
                    key, e.get("id"), old or 0, new, "%.3f" % (new / old) if old else "-",
                    "  <-- 修正" if flag else ""))
            e["duration_s_mvhd_legacy"] = old
            e["duration_s"] = round(new, 3)
    print("\n需修正条数:", n_fix)

    if args.apply:
        bak = META + ".bak_dur%d" % int(time.time())
        shutil.copy2(META, bak)
        json.dump(meta, open(META, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[APPLY] 已写回", META, "\n[备份]", bak)
        print("[提示] 需重跑：evidence.py → explain_card.py → infer_explain_att4.py → verify_q3.py")
    else:
        print("（未写入，加 --apply 执行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
