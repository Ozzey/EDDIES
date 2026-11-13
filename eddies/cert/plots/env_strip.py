#!/usr/bin/env python3
"""Collage of the ten simulated tasks: one row per simulator (MuJoCo; Isaac Sim when its stills exist).

    MUJOCO_GL=egl python -m eddies.envs.scripts.stills --task <t> --seed 2000 --steps <n> --out <dir>
    python -m eddies.isaac.stills --tasks <t>:<n>,... --seed 2000 --out <dir_isaac>      # Isaac Sim, GPU
    python -m eddies.cert.plots.env_strip --stills <dir> [--stills-isaac <dir_isaac>]
Writes assets/results/generated/envs_strip.{pdf,png}; task labels are drawn on the images of the top row.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eddies.common.paths import FIGURES as OUT
NAVY = "#221b45"
PANEL = "#f3f9fe"
ROBOT_COLOR = {"Franka": "#2e90ef", "G1": "#f39a2b", "LEAP hand": "#6a1fd6", "Aero hand": "#2e4b6c"}
# (still file stem, robot, task, square crop box in the 960x720 still)
TILES = [
    ("franka_fmb_s2000_main_step250", "Franka", "staged insertion", (210, 110, 650, 550)),
    ("franka_peg_insertion_s2000_main_step150", "Franka", "peg insertion", (245, 80, 685, 520)),
    ("franka_cube_stack_s2000_main_step150", "Franka", "cube stack", (230, 100, 670, 540)),
    ("franka_drawer_s2000_main_step200", "Franka", "drawer + box", (110, 20, 730, 640)),
    ("g1_peg_insertion_sb_s2000_main_step150", "G1", "peg insertion", (260, 140, 780, 660)),
    ("g1_drawer_sb_s2000_main_step200", "G1", "drawer + caddy", (160, 60, 780, 680)),
    ("leap_wheel_sb_s2000_main_step150", "LEAP hand", "wheel", (200, 10, 820, 630)),
    ("leap_peg_tighten_sb_s2000_main_step150", "LEAP hand", "peg tightening", (270, 80, 770, 580)),
    ("aero_wheel_sb_s2000_main_step150", "Aero hand", "wheel", (210, 0, 830, 620)),
    ("aero_peg_tighten_sb_s2000_main_step150", "Aero hand", "peg tightening", (270, 80, 770, 580)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", required=True)
    ap.add_argument("--stills-isaac", default=None)
    ap.add_argument("--cols", type=int, default=5, help="tasks per block (10 = one block)")
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "text.color": NAVY,
                         "pdf.fonttype": 42})
    sims = [("MuJoCo", Path(a.stills))]
    if a.stills_isaac and all((Path(a.stills_isaac) / f"{t[0]}.png").exists() for t in TILES):
        sims.append(("Isaac Sim", Path(a.stills_isaac)))
    blocks = [TILES[i:i + a.cols] for i in range(0, len(TILES), a.cols)]
    # per block: first simulator row, label band, remaining simulator rows; a thin spacer between blocks
    layout, heights = [], []
    for bi, block in enumerate(blocks):
        if bi:
            layout.append(("gap", None)), heights.append(0.10)
        layout.append(("img", (bi, sims[0]))), heights.append(1.0)
        layout.append(("lab", bi)), heights.append(0.26)
        for sim in sims[1:]:
            layout.append(("img", (bi, sim))), heights.append(1.0)
    w = 6.9
    tile = w / a.cols
    fig, axes = plt.subplots(len(layout), a.cols, figsize=(w, tile * sum(heights) * 0.93), squeeze=False,
                             gridspec_kw={"wspace": 0.03 if a.cols <= 5 else 0.07, "hspace": 0.03, "height_ratios": heights})
    for r, (kind, val) in enumerate(layout):
        for c in range(a.cols):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            if kind == "gap":
                ax.axis("off")
                continue
            bi = val if kind == "lab" else val[0]
            if c >= len(blocks[bi]):
                ax.axis("off")
                continue
            stem, robot, task, box = blocks[bi][c]
            if kind == "lab":
                ax.axis("off")
                ax.text(0.5, 0.74, robot, transform=ax.transAxes, ha="center", va="center", fontsize=7 if a.cols <= 5 else 5.4,
                        fontweight="bold", color=ROBOT_COLOR[robot])
                ax.text(0.5, 0.24, task, transform=ax.transAxes, ha="center", va="center", fontsize=6.6 if a.cols <= 5 else 5.0, color=NAVY)
                continue
            sim, d = val[1]
            im = Image.open(d / f"{stem}.png").convert("RGB").crop(box).resize((500, 500), Image.LANCZOS)
            ax.imshow(np.asarray(im))
            for sp in ax.spines.values():
                sp.set_edgecolor("#c9d6e3"), sp.set_linewidth(0.6)
            if c == 0:
                ax.set_ylabel(sim, fontsize=7, fontweight="bold", color=NAVY, labelpad=3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "envs_strip.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    fig.savefig(OUT / "envs_strip.png", bbox_inches="tight", pad_inches=0.01, dpi=200)
    print("wrote", OUT / "envs_strip.pdf", "sims:", [x[0] for x in sims], "blocks:", len(blocks))


if __name__ == "__main__":
    main()
