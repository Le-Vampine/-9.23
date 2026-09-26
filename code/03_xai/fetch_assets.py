# -*- coding: utf-8 -*-
r"""获取问题3 需要的外部静态资源（只下载词表，不引入任何模型或情感数据集）。

用途：把 `text_bert` 的词元编号解码为可读词/词片段，用于文本证据定位。
说明：`bert-base-uncased` 的 `vocab.txt` **行号即 token id**，因此无需安装 transformers
      即可完成解码，避免为一次解释任务引入整套深度学习框架。

来源：https://hf-mirror.com/bert-base-uncased/resolve/main/vocab.txt
      该文件是**分词器静态词表**（通用文本预处理资源），不含任何情感标注信息，
      不属于赛题禁止引入的"其他公开或私有情感数据集"。

用法：
  python fetch_assets.py                     # 下载到 <仓库根>/code/assets/
  python fetch_assets.py --check             # 只校验已存在的文件
"""
import argparse
import hashlib
import json
import os
import sys

URL = "https://hf-mirror.com/bert-base-uncased/resolve/main/vocab.txt"
EXPECT_LINES = 30522
EXPECT_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "code", "assets")
OUT_PKL = os.path.join(OUT_DIR, "bert-base-uncased_vocab.txt")
MANIFEST = os.path.join(OUT_DIR, "assets_manifest.json")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest):
    import requests
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("[GET] %s" % URL)
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)
    print("[OK ] %s (%d bytes)" % (dest, len(r.content)))


def verify(path):
    if not os.path.isfile(path):
        return False, "文件不存在"
    lines = open(path, encoding="utf-8").read().splitlines()
    digest = sha256_of(path)
    ok = (len(lines) == EXPECT_LINES)
    msg = "行数=%d（期望 %d）  sha256=%s%s" % (
        len(lines), EXPECT_LINES, digest,
        "" if digest == EXPECT_SHA256 else "  ⚠ 与参考值不同（hf 镜像可能重打包，行数正确即可）")
    return ok, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不下载")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    a = ap.parse_args()

    if not a.check and (a.force or not os.path.isfile(OUT_PKL)):
        download(OUT_PKL)
    ok, msg = verify(OUT_PKL)
    print("[CHK] %s" % msg)
    if not ok:
        raise SystemExit("词表校验失败")
    man = {
        "file": os.path.basename(OUT_PKL),
        "source_url": URL,
        "lines": EXPECT_LINES,
        "sha256": sha256_of(OUT_PKL),
        "purpose": "text_bert token id -> wordpiece 解码（证据定位）",
        "note": "静态分词词表，非情感数据集；行号即 token id",
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print("[SAVE] %s" % MANIFEST)
    print("ASSETS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
