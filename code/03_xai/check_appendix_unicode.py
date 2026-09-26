# -*- coding: utf-8 -*-
"""检查附录代码中的非 ASCII、非中文符号，并给出 ASCII 等价替换建议。

背景：论文用 SimSun + Times New Roman + Latin Modern Mono 排版，代码注释里的 ∈、≤、→、π
等符号在等宽/西文字体中缺字形，会渲染为空白（缺字）。本脚本把它们替换为 ASCII 等价形式，
仅影响注释、docstring 与输出文案，可执行语义不变；替换时避免与相邻字母粘连。

用法：
    python code/03_xai/check_appendix_unicode.py            # 只报告
    python code/03_xai/check_appendix_unicode.py --fix      # 就地替换 05_paper/code_appendix/
    python code/03_xai/check_appendix_unicode.py --fix --diff   # 并逐行列出差异供核对
"""
import argparse
import collections
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEST = os.path.join(ROOT, "05_paper", "code_appendix")

# Unicode 数学/箭头符号 → ASCII 等价（仅出现在注释与 docstring 中，替换不影响可执行语义）
MAP = {
    "↔": "<->", "∈": "in", "∉": "not in", "≠": "!=", "≤": "<=", "≥": ">=",
    "⊆": "subset", "⊂": "subset", "∪": "|", "∩": "&", "∖": "-", "∅": "{}",
    "⊙": "(*)", "∇": "grad", "×": "x", "→": "->", "←": "<-", "⇒": "=>",
    "±": "+/-", "·": ".", "…": "...", "—": "--", "–": "-", "′": "'", "″": '"',
    "²": "^2", "³": "^3", "≈": "~=", "≡": "==", "∀": "all", "∃": "exists",
    "∑": "sum", "∏": "prod", "√": "sqrt", "∞": "inf", "α": "alpha", "β": "beta",
    "γ": "gamma", "δ": "delta", "ε": "eps", "θ": "theta", "λ": "lambda",
    "μ": "mu", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "φ": "phi",
    "ω": "omega", "Δ": "Delta", "Σ": "Sigma", "Γ": "Gamma", "Ω": "Omega",
    "Φ": "Phi", "Θ": "Theta",
    "−": "-", "ŷ": "y_hat", "§": "section", "▲": "*", "\u3000": " ",
    "∂": "d", "̃": "~", "̄": "-", "‖": "||", "＝": "=", "✱": "*",
}


def sub_token(text, key, val):
    """把 key 换成 val；若相邻字符为字母则补空格，避免粘连成新词（如 semantic∈ → semantic in）。"""
    out, i, n = [], 0, len(text)
    while True:
        j = text.find(key, i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        prev = text[j - 1] if j > 0 else ""
        nxt = text[j + len(key)] if j + len(key) < n else ""
        pre = " " if (prev and prev.isalpha() and val[:1].isalpha()) else ""
        post = " " if (nxt and nxt.isalpha() and val[-1:].isalpha()) else ""
        out.append(pre + val + post)
        i = j + len(key)


def fix_all():
    """对 code_appendix/ 下所有 .py 就地替换，返回处理文件数。"""
    n = 0
    for f in sorted(os.listdir(DEST)):
        if not f.endswith(".py"):
            continue
        p = os.path.join(DEST, f)
        t = open(p, encoding="utf-8").read()
        for k, v in MAP.items():
            if k in t:
                t = sub_token(t, k, v)
        open(p, "w", encoding="utf-8").write(t)
        n += 1
    return n


def bad_chars(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = collections.Counter()
    for ch in txt:
        code = ord(ch)
        if code < 128 or 0x4E00 <= code <= 0x9FFF or ch in "，。；：、（）【】《》“”‘’！？—…":
            continue
        out[ch] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--diff", action="store_true", help="逐行列出替换后的差异（核对是否全在注释内）")
    a = ap.parse_args()
    allc = collections.Counter()
    files = 0
    for f in sorted(os.listdir(DEST)):
        if not f.endswith(".py"):
            continue
        files += 1
        c = bad_chars(os.path.join(DEST, f))
        if c:
            print("%-32s %s" % (f, dict(c)))
        allc.update(c)
    print("--- 共 %d 个文件，非中文非 ASCII 字符 %d 种 ---" % (files, len(allc)))
    unknown = {k: v for k, v in allc.items() if k not in MAP}
    if unknown:
        print("!! 未映射字符：", dict(unknown))
    if not a.fix:
        return 0
    m = fix_all()
    print("[FIX] 已按 MAP 替换，处理 %d 个文件（共 %d 类符号）" % (m, len(MAP)))
    if a.diff:
        import difflib
        for f in sorted(os.listdir(DEST)):
            if not f.endswith(".py"):
                continue
            rel = f.replace("__", "/")
            orig = os.path.join(ROOT, "code", rel.replace("/", os.sep))
            if not os.path.isfile(orig):
                print("  [缺原文件] %s" % rel)
                continue
            d = [l for l in difflib.unified_diff(
                open(orig, encoding="utf-8").read().split("\n"),
                open(os.path.join(DEST, f), encoding="utf-8").read().split("\n"),
                lineterm="", n=0)
                if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
            for l in d:
                print("  %-30s %s" % (rel, l[:110]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
