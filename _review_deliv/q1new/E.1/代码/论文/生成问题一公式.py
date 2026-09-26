"""Render the four equation previews used by the Problem 1 paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parents[2] / "论文" / "图表精修" / "论文插图"
EQUATIONS = {
    1: r"t_{i}^{\mathrm{source}}=t_{i}^{\mathrm{WAV}}+\Delta_i^{\mathrm{audio}}",
    2: r"x_{ij}^{t}=\frac{1}{|P_{ij}|}\sum_{r\in P_{ij}}h_{ir}^{(L)},\qquad x_{ij}^{t}\in\mathbb{R}^{768}",
    3: r"\mathcal{K}_{ij}^{a}=\{k:t_{ik}^{a}\in[s_{ij},e_{ij})\},\qquad\mathcal{K}_{ij}^{v}=\{\ell:t_{i\ell}^{v}\in[s_{ij},e_{ij}),\ q_{i\ell}=1\}",
    4: r"x_{ij}^{a}=\frac{\sum_{k\in\mathcal{K}_{ij}^{a}}a_{ik}}{|\mathcal{K}_{ij}^{a}|},\qquad x_{ij}^{v}=\frac{\sum_{\ell\in\mathcal{K}_{ij}^{v}}v_{i\ell}}{|\mathcal{K}_{ij}^{v}|}",
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for number, latex in EQUATIONS.items():
        fig, ax = plt.subplots(figsize=(14, 0.9), dpi=220)
        ax.axis("off")
        ax.text(0.5, 0.5, "$" + latex + "$", ha="center", va="center", fontsize=22)
        fig.savefig(OUT / f"eq{number:02d}.png", transparent=True, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
