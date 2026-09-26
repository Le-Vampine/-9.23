# -*- coding: utf-8 -*-
r"""问题3 统一绘图风格：宋体 + Times、四色学术配色、去 top/right 边框。

与问题2 的图表风格保持一致（评审要求"全文一个字体语言、配色不混用"）。
用法：from style import apply_style, COLORS; apply_style()
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 与 handover_problem2 一致的四个自然低饱和色
COLORS = {
    "text": "#3B5F8A",      # 深蓝
    "audio": "#A23B3B",     # 砖红
    "vision": "#4A7C59",    # 墨绿
    "neutral": "#B8B8B8",   # 灰
    "accent": "#C8843C",    # 橙（强调/标注）
    "grid": "#D9D9D9",
}
MOD_COLORS = [COLORS["text"], COLORS["audio"], COLORS["vision"]]
LABEL_COLORS = [COLORS["accent"], COLORS["neutral"], COLORS["text"]]   # 负/中/正

_RC = {
    "font.family": ["SimSun", "Times New Roman", "DejaVu Serif"],
    "font.serif": ["SimSun", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.6,
    "grid.color": COLORS["grid"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.8,
    "legend.frameon": False,
}


def apply_style():
    plt.rcParams.update(_RC)
    return plt


def annotate_bars(ax, bars, fmt="%.3f", dy=0.01, fontsize=8.5, color="#333333"):
    """在柱顶直接标数值（评审要求"数值直接标注"）。"""
    ymax = max([b.get_height() for b in bars] + [1e-9])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy * ymax,
                fmt % b.get_height(), ha="center", va="bottom",
                fontsize=fontsize, color=color)


def caption_hint(text):
    """论文图表说明文字的字数自查（要求 100–150 字：汉字 + 拉丁/数字串）。"""
    import re
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    tok = len(re.findall(r"[A-Za-z0-9]+(?:[.,][0-9]+)*", text))
    return cjk + tok
