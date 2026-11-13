"""Colour scheme and matplotlib style shared by every figure (values in eddies/config.yaml ``plots``).

NAVY ink / text, ARROW (also the Aero hand), GRID, BLUE (Franka / Eddies / tube), ORANGE (G1 / envelope),
VIOLET (LEAP hand / certified bound), GREY, PANEL, STOP (certified stop), MOVE.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eddies.common.config import load  # noqa: E402

_P = load().plots
COLORS = {k: str(v) for k, v in _P.colors.items()}
NAVY, ARROW, GRID = COLORS["navy"], COLORS["arrow"], COLORS["grid"]
BLUE, ORANGE, VIOLET, GREY = COLORS["blue"], COLORS["orange"], COLORS["violet"], COLORS["grey"]
PANEL, STOP, MOVE = COLORS["panel"], COLORS["stop"], COLORS["move"]
FONT, FONT_SIZE = str(_P.font.family), float(_P.font.size)


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": FONT_SIZE,
                         "axes.titlesize": FONT_SIZE + 0.5, "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE - 0.5,
                         "ytick.labelsize": FONT_SIZE - 0.5,
                         "text.color": NAVY, "axes.labelcolor": NAVY, "axes.edgecolor": GREY, "xtick.color": NAVY,
                         "ytick.color": NAVY, "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "lines.linewidth": 1.8,
                         "lines.markersize": 4})


def grid(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, out_dir, name, dpi=160):
    """<out_dir>/<name>.pdf and .png, tight."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    plt.close(fig)
    return out / f"{name}.pdf"
