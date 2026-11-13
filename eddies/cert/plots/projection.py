#!/usr/bin/env python3
"""Figure: the Streamer's projection of a proposed command onto the trust ball and the slew ball.

    python -m eddies.cert.plots.projection        # writes assets/results/generated/projection.{svg,pdf}

Geometry (metric M = I for the drawing): anchor of the running skill, trust radius delta, previous
command e_k, slew radius rho, and a command proposed by the primitive controller outside both balls.
The command sent is the closest point of the intersection; projecting onto the slew ball alone would
leave the trust ball.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from eddies.common.paths import FIGURES as OUT
# colour scheme of the paper's figures (navy text, blue / orange / violet families, light panels)
NAVY, ARROW = "#221b45", "#2e4b6c"
BLUE, BLUE_L, PANEL, PANEL_EDGE = "#2e90ef", "#e6f3fc", "#f3f9fe", "#9cc6ea"
ORANGE, PEACH = "#f39a2b", "#fee8cb"
VIOLET, LAVENDER = "#6a1fd6", "#ddcdf7"
GREY = "#a4adbb"

A = np.array([300., 260.])      # anchor of the running skill
DELTA = 150.                    # trust radius
P = np.array([380., 200.])      # previous command e_k
RHO = 70.                       # slew radius
X = np.array([620., 150.])      # proposed command


def project(X):
    """Closest point of the intersection of the two disks to X (dense boundary search)."""
    best = None
    for th in np.linspace(0, 2 * np.pi, 40000):
        for C, R, O, Ro in ((A, DELTA, P, RHO), (P, RHO, A, DELTA)):
            q = C + R * np.array([np.cos(th), np.sin(th)])
            if np.linalg.norm(q - O) <= Ro + 1e-9:
                d = np.linalg.norm(q - X)
                if best is None or d < best[0]:
                    best = (d, q)
    return best[1]


def main():
    E = project(X)
    S = P + RHO * (X - P) / np.linalg.norm(X - P)          # slew ball alone
    trail = [np.array(v) for v in ((305., 255.), (328., 238.), (352., 220.))] + [P]

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
                         "font.size": 8.5, "text.color": NAVY, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.set_xlim(80, 1030), ax.set_ylim(462, 22)            # y down, as in the coordinates above
    ax.set_aspect("equal"), ax.axis("off")

    def badge(x, y, n):
        ax.add_patch(Circle((x, y), 10, fc=NAVY, ec="none", zorder=6))
        ax.text(x, y + 0.5, str(n), color="white", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=7)

    # panel
    ax.add_patch(FancyBboxPatch((90, 35), 610, 415, boxstyle="round,pad=0,rounding_size=14", fc=PANEL, ec=PANEL_EDGE, lw=1.0))
    ax.text(104, 46, "One control step of the Streamer", fontsize=9, fontweight="bold", va="top")

    # sets
    ax.add_patch(Circle(A, DELTA, fc=BLUE_L, ec="none"))
    ax.add_patch(Circle(P, RHO, fc=PEACH, ec="none"))
    lens = Circle(P, RHO, fc=LAVENDER, ec="none")          # slew ball clipped to the trust ball
    ax.add_patch(lens)
    lens.set_clip_path(Circle(A, DELTA, transform=ax.transData))
    ax.add_patch(Circle(A, DELTA, fill=False, ec=BLUE, lw=1.6))
    ax.add_patch(Circle(P, RHO, fill=False, ec=ORANGE, lw=1.6))

    # radii
    tb = A + DELTA * np.array([np.cos(np.radians(215)), -np.sin(np.radians(215))])
    ax.plot(*zip(A, tb), color=BLUE, lw=1.0)
    ax.text(*(A + tb) / 2 + np.array([-2, -8]), r"$\delta_i$", color=BLUE, ha="right", va="bottom", fontsize=10)
    sb = P + RHO * np.array([np.cos(np.radians(-55)), -np.sin(np.radians(-55))])
    ax.plot(*zip(P, sb), color=ORANGE, lw=1.0)
    ax.text(*(P + sb) / 2 + np.array([7, 0]), r"$\rho_i$", color=ORANGE, va="center", fontsize=10)

    # set labels
    ax.text(300, 338, "Trust ball", fontweight="bold", color=BLUE, ha="center", va="top")
    ax.text(300, 356, "stays near the anchor", ha="center", va="top", fontsize=7.5)
    ax.text(462, 84, "Slew ball", fontweight="bold", color=ORANGE, ha="left", va="bottom")
    ax.text(462, 102, r"moves $\leq\rho_i$ per step", ha="left", va="bottom", fontsize=7.5)
    ax.annotate("both limits hold", (350, 182), xytext=(104, 92), color=VIOLET, fontsize=7.5, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="-", color=VIOLET, lw=0.7, shrinkB=0))

    # previous commands and anchor
    T = np.array(trail)
    ax.plot(T[:, 0], T[:, 1], color=GREY, lw=1.0)
    ax.scatter(T[:-1, 0], T[:-1, 1], s=10, color=GREY, zorder=3)
    ax.scatter(*P, s=26, color=NAVY, zorder=4)
    ax.text(P[0] - 6, P[1] - 7, r"$e_k$", ha="right", va="bottom", fontsize=10)
    ax.scatter(*A, s=26, color=BLUE, edgecolor="white", lw=0.8, zorder=4)
    ax.text(A[0] + 8, A[1] + 8, r"anchor $\bar e_i$", ha="left", va="top", color=BLUE)

    # proposed -> sent
    ax.scatter(*X, marker="x", s=50, color=NAVY, lw=1.6, zorder=4)
    ax.text(X[0], X[1] + 14, "proposed", ha="center", va="top", fontweight="bold")
    badge(X[0] + 30, X[1] - 16, 1)
    ax.add_patch(FancyArrowPatch(X, E, arrowstyle="-|>", mutation_scale=10, color=ARROW, lw=1.1, ls=(0, (4, 3)),
                                 shrinkA=6, shrinkB=6, zorder=3))
    badge(*((X + E) / 2 + np.array([2, -18])), 2)
    ax.scatter(*S, s=30, facecolor="white", edgecolor=ORANGE, lw=1.4, zorder=4)
    ax.annotate("slew limit alone would\nleave the trust ball", S, xytext=(482, 232), fontsize=7.2, color=ORANGE,
                va="top", arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7, shrinkB=4))
    ax.scatter(*E, s=44, color=VIOLET, edgecolor="white", lw=0.9, zorder=5)
    ax.annotate(r"sent $e_{k+1}$", E, xytext=(482, 292), color=VIOLET, fontweight="bold", va="top",
                arrowprops=dict(arrowstyle="-", color=VIOLET, lw=0.7, shrinkB=4))
    badge(628, 300, 3)

    # steps
    steps = [(1, "The skill's controller proposes\nthe next command."),
             (2, "The Streamer projects it onto the\ncommands that stay in the trust\nball and move at most $\\rho_i$\nfrom the last command."),
             (3, "The closest such command is\nsent to the robot, after the gate\nchecks it (Section 4.2).")]
    y = 60
    for n, txt in steps:
        badge(735, y + 8, n)
        ax.text(755, y, txt, va="top", fontsize=7.8, linespacing=1.3)
        y += 19 * (txt.count("\n") + 1) + 26

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT / f"projection.{ext}", bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT / "projection.svg", OUT / "projection.pdf", "| sent", E.round(1), "slew-only", S.round(1))


if __name__ == "__main__":
    main()
