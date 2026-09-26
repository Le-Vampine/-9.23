# -*- coding: utf-8 -*-
r"""只读工具：按 check_layout.py 的口径统计 main.tex 中每条 \caption 的字数，
并对超长者给出"按句截断"的建议文本（不写回文件，供人工确认后用编辑工具应用）。

口径与 skill/scripts/check_layout.py 一致：
  字数 = 汉字数 + 拉丁/数字串数（每串计 1）
用法：python tools_audit_captions.py --tex main.tex
"""
import argparse
import re
import sys

CJK = re.compile(r"[\u4e00-\u9fff]")
TOK = re.compile(r"[A-Za-z0-9]+(?:[.,][0-9]+)*")
CAP = re.compile(r"^(\s*)\\caption\{(.*)\}\s*$")


def count_zi(s):
    return len(CJK.findall(s)) + len(TOK.findall(s))


def trim(s, lo=100, hi=150):
    """按句号（中文。）从尾部逐步删除，直到落进区间。"""
    parts = [p for p in re.split(r"(?<=。)", s) if p]
    while parts and count_zi("".join(parts)) > hi:
        parts.pop()
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="main.tex")
    ap.add_argument("--lo", type=int, default=100)
    ap.add_argument("--hi", type=int, default=150)
    a = ap.parse_args()

    lines = open(a.tex, encoding="utf-8").read().split("\n")
    bad = []
    for i, ln in enumerate(lines):
        m = CAP.match(ln)
        if not m:
            continue
        body = m.group(2)
        zi = count_zi(body)
        if zi < a.lo:
            bad.append(("过短", i + 1, zi, body, None))
        elif zi > a.hi:
            bad.append(("过长", i + 1, zi, body, trim(body, a.lo, a.hi)))

    print("共发现 %d 条说明文字不在 [%d, %d] 区间：" % (len(bad), a.lo, a.hi))
    for kind, ln, zi, body, sug in bad:
        print("\n--- L%d  %s  %d 字 ---" % (ln, kind, zi))
        if sug is not None:
            print("建议（%d 字，可直接替换该行）：" % count_zi(sug))
            print(sug)
        else:
            print("现状：%s" % body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
