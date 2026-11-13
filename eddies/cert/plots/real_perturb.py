#!/usr/bin/env python3
"""Real Franka: cube stacking under perturbation (side-view video Cube-Stack-Multi.MOV).

    python -m eddies.cert.plots.real_perturb VIDEO
Writes assets/results/generated/real_perturb.{pdf,png}: six frames (top) and a timeline (bottom) of when the arm moves,
when the cube is pushed with a stick and when the gate holds the arm (certified stop).

Times are video seconds, read off the video: the arm starts moving at 19.25 s (time 0 of the timeline);
the arm-region image is static from 30.75 s to 46.25 s (frame differences, 4 fps); the stick is on the
table during PUSHES (dark pixels on the table surface, 4 fps).
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from eddies.common.paths import FIGURES as OUT
NAVY = "#221b45"
STOP = "#d6452b"
MOVE = "#c9d6e3"
T0 = 19.25                                   # arm starts moving (video time)
STOPPED = (30.75, 46.25)                     # arm held by the gate
PUSHES = [(20.75, 24.5), (26.75, 29.5), (36.75, 42.5), (43.75, 47.5), (48.75, 51.5)]
END = 74.0                                   # arm back above the stack
CROP = (200, 120, 1700, 1080)                # x0, y0, x1, y1 in the 1920x1080 frame
FRAMES = [(22.5, "cube pushed:\narm re-targets"), (27.5, "pushed again:\narm follows"),
          (38.5, "out of the safe set:\ncertified stop"), (48.0, "back in the set:\narm resumes"),
          (60.5, "grasp"), (73.0, "stacked")]


def grab(video, t):
    with tempfile.TemporaryDirectory() as d:
        png = Path(d) / "f.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t}", "-i", video, "-frames:v", "1", str(png)],
                       check=True)
        return Image.open(png).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "text.color": NAVY,
                         "pdf.fonttype": 42, "axes.edgecolor": NAVY, "xtick.color": NAVY})
    n = len(FRAMES)
    cw, ch = CROP[2] - CROP[0], CROP[3] - CROP[1]
    W = 6.9
    img_h = W / n * ch / cw
    fig = plt.figure(figsize=(W, img_h + 0.42))
    gs = fig.add_gridspec(2, n, height_ratios=[img_h, 0.42], wspace=0.025, hspace=0.08,
                          left=0, right=1, top=1, bottom=0)
    for i, (t, label) in enumerate(FRAMES):
        ax = fig.add_subplot(gs[0, i])
        im = grab(a.video, t).crop(CROP).resize((600, round(600 * ch / cw)), Image.LANCZOS)
        ax.imshow(np.asarray(im))
        ax.set_xticks([]), ax.set_yticks([])
        stop = STOPPED[0] <= t <= STOPPED[1]
        for sp in ax.spines.values():
            sp.set_edgecolor(STOP if stop else "#c9d6e3"), sp.set_linewidth(1.4 if stop else 0.6)
        ax.text(0.03, 0.96, f"({chr(97 + i)}) {label}", transform=ax.transAxes, ha="left", va="top", fontsize=4.6,
                color="white", fontweight="bold" if stop else "normal", linespacing=1.1,
                bbox=dict(boxstyle="round,pad=0.25", fc=STOP if stop else NAVY, ec="none", alpha=0.85))
        ax.text(0.97, 0.04, f"{t - T0:.0f} s", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.4,
                color="white", bbox=dict(boxstyle="round,pad=0.2", fc=NAVY, ec="none", alpha=0.7))

    # timeline
    ax = fig.add_subplot(gs[1, :])
    span = END - T0
    ax.set_xlim(-0.6, span + 0.6), ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0.38), span, 0.24, fc=MOVE, ec="none"))
    s0, s1 = STOPPED[0] - T0, STOPPED[1] - T0
    ax.add_patch(Rectangle((s0, 0.38), s1 - s0, 0.24, fc=STOP, ec="none"))
    ax.text((s0 + s1) / 2, 0.50, "certified stop (arm held)", ha="center", va="center", fontsize=4.6, color="white",
            fontweight="bold")
    ax.text(3.0 + (s0 - 3.0) / 2 + 1.2, 0.50, "moving", ha="center", va="center", fontsize=4.4, color=NAVY)
    for p0, p1 in PUSHES:
        ax.add_patch(Rectangle((p0 - T0, 0.68), p1 - p0, 0.12, fc=NAVY, ec="none"))
    ax.text(PUSHES[0][0] - T0, 0.90, "cube pushed", ha="left", va="center", fontsize=4.3, color=NAVY)
    for i, (t, _) in enumerate(FRAMES):
        x = t - T0
        ax.plot([x, x], [0.30, 0.38], color=NAVY, lw=0.6)
        ax.text(x, 0.17, f"({chr(97 + i)})", ha="center", va="center", fontsize=4.4, color=NAVY)
    ax.set_yticks([])
    ax.set_xticks(range(0, int(span) + 1, 10))
    ax.tick_params(axis="x", labelsize=4.4, length=1.5, width=0.4, pad=1)
    ax.set_xlabel("time since the arm starts moving [s]", fontsize=4.6, labelpad=1)
    for k in ("left", "right", "top"):
        ax.spines[k].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "real_perturb.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    fig.savefig(OUT / "real_perturb.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
    print("wrote", OUT / "real_perturb.pdf")


if __name__ == "__main__":
    main()
