#!/usr/bin/env python3
"""Real Franka figure: setup, the robot camera's RGB and depth, and stacking under perturbation.

    E=external/Franka-Stack-Camera/data
    python -m eddies.cert.plots.real_setup --multi Cube-Stack-Multi.MOV --rgb-video $E/runs/Perturbation-1.mp4 \
        --capture $E/captures/static_20261111_180710 --calib $E/calib/T_base_cam.yaml [--photo figs/stack_real.jpg]

Top row: (a) the Franka grasping the grey cube (side photo with a close-up inset);
(b) a frame of the robot camera's RGB feed during a run (the recorded overlay: 3D cube boxes and the active
Eddie's target frame), cropped to the table;
(c) a static capture of the same RealSense (depth aligned to colour) shown as height above the table, from the
depth, K and the hand-eye calibration; the two cubes are outlined, coloured by height and labelled with their
position in the robot base frame.
Bottom: six frames of the side-view video of a perturbed run, captioned with the phase (times from
real_perturb). Writes assets/results/generated/real_setup.{pdf,png}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

from eddies.cert.plots import real_perturb as RP

from eddies.common.paths import FIGURES as OUT
INK, STOP, MOVE = "#111111", "#d62728", "#bdbdbd"
CUBE = {"grey": "#1f77b4", "black": "#ff7f0e"}       # outline / label colours of the two cubes
AX = ("#d62728", "#2ca02c", "#1f77b4")             # x, y, z of the base frame
FEED_CROP = (170, 180, 880, 700)                   # 1280x720 RGB feed: table, cubes, gripper; no status text
# readable labels for the feed's own (tiny) ones, in FEED_CROP pixels: (x, y, text, colour key)
C_CROP = (380, 300, 880, 666)                      # height panel: the two cubes and the base frame only
FEED_LABELS = [(268, 392, "grey cube", "grey"), (608, 382, "black cube", "black"), (92, 222, "Eddie target", None)]
TEXTWIDTH = 6.0                                    # inches: drawn at print size, so font sizes are as printed
FS, FS_S = 7.0, 6.0                                # captions / annotations [pt], as printed
SHORT = ["reach: cube pushed", "reach: re-targets", "certified stop", "reach: resumes", "grasp", "place: stacked"]
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]


def cubes_from_capture(d: Path, calib: Path):
    """Height above the table from a static RealSense capture (rgb.png, depth.npy in m aligned to colour, K.json)
    and the hand-eye calibration (T_base_cam.yaml); cubes = connected regions 3-9 cm above the table."""
    import yaml
    rgb = np.asarray(Image.open(d / "rgb.png").convert("RGB"))
    z = np.load(d / "depth.npy").astype(float)
    K = np.array(json.loads((d / "K.json").read_text())["K"])
    T = np.array(yaml.safe_load(calib.read_text())["T_base_cam"])
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z, np.ones_like(z)], -1) @ T.T
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    ok = (z > 0.2) & (z < 3.0)
    table = ok & (X > 0.35) & (X < 0.85) & (np.abs(Y) < 0.35)
    H = np.where(ok, Z - np.median(Z[table]), np.nan)       # height above the (locally measured) table level
    near = ok & (X > 0.25) & (X < 0.85) & (np.abs(Y) < 0.40)
    lab, n = ndi.label(ndi.binary_opening(near & (H > 0.03) & (H < 0.09), iterations=4))
    cubes = {}
    for i in range(1, n + 1):
        mk = lab == i
        if mk.sum() < 1500:
            continue
        col = rgb[mk].mean(0)
        name = "grey" if col.mean() > 70 else "black"
        # keep the pixels with the cube's own colour (depth bleeds past the edges), largest piece, holes filled
        c8 = rgb.astype(int)
        own = (c8[..., 2] - c8[..., 0] > 8) if name == "grey" else (c8.mean(-1) < 90)
        lab2, n2 = ndi.label(ndi.binary_opening(mk & own, iterations=2))
        if n2:
            mk = lab2 == 1 + int(np.argmax(ndi.sum(mk, lab2, range(1, n2 + 1))))
        cubes[name] = dict(mask=ndi.binary_fill_holes(ndi.binary_closing(mk, iterations=3)), x=float(np.median(X[mk])), y=float(np.median(Y[mk])),
                           top=float(np.percentile(H[mk], 95)), u=float(u[mk].mean()), v=float(v[mk].min()))
    Tcb = np.linalg.inv(T)

    def proj(p):
        q = Tcb @ np.r_[p, 1.0]
        return K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2]
    axes = [proj(np.zeros(3))] + [proj(0.1 * e) for e in np.eye(3)]
    return rgb, H, cubes, axes


def outline(ax, mask, color):
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.0)


def note(ax, x, y, text, color=INK, ha="center", va="bottom", size=FS_S):
    """On-image annotation: coloured text with a thin white halo, no box."""
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color, linespacing=1.05, path_effects=HALO)


def bare(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4), sp.set_edgecolor(INK)


def caption(fig, ax, text, color=INK):
    b = ax.get_position()
    fig.text((b.x0 + b.x1) / 2, b.y0 - 0.008, text, ha="center", va="top", fontsize=FS, color=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi", required=True, help="side-view video of the perturbed run")
    ap.add_argument("--capture", required=True, help="static capture folder: rgb.png, depth.npy, K.json")
    ap.add_argument("--calib", required=True, help="T_base_cam.yaml of the same camera")
    ap.add_argument("--photo", default=str(OUT / "stack_real.jpg"))
    ap.add_argument("--rgb-video", required=True, help="the robot camera's RGB feed with the pipeline's labels")
    ap.add_argument("--rgb-t", type=float, default=41.0)
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                         "text.color": INK, "pdf.fonttype": 42, "axes.edgecolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.linewidth": 0.5})
    photo = Image.open(a.photo).convert("RGB")
    rgb, hgt, cubes, axes = cubes_from_capture(Path(a.capture), Path(a.calib))
    x0, y0, x1, y1 = C_CROP
    feed = RP.grab(a.rgb_video, a.rgb_t).crop(FEED_CROP)
    r_ph, r_cam, r_feed = photo.width / photo.height, (x1 - x0) / (y1 - y0), feed.width / feed.height

    W = TEXTWIDTH
    n = len(RP.FRAMES)
    cw, ch = RP.CROP[2] - RP.CROP[0], RP.CROP[3] - RP.CROP[1]
    gap, sgap, cap = 0.05, 0.03, 0.17                  # inches: between panels, between frames, caption line
    h_top = (W - 2 * gap) / (r_ph + r_feed + r_cam)
    wf = (W - (n - 1) * sgap) / n
    h_strip = wf * ch / cw
    h_tl = 0.0                                         # no timeline: the captions name the phases
    H = h_top + cap + 0.06 + h_strip + cap
    fig = plt.figure(figsize=(W, H))

    def axes_at(x, y, w, h):                            # inches from the bottom-left corner
        return fig.add_axes([x / W, y / H, w / W, h / H])

    ytop = H - h_top
    # (a) setup photo
    ax = axes_at(0, ytop, h_top * r_ph, h_top)
    ax.imshow(np.asarray(photo)), bare(ax)
    caption(fig, ax, "(a) Franka Panda (inset: grasp)")
    # (b) RGB feed with the pipeline's overlay
    xb = h_top * r_ph + gap
    ax = axes_at(xb, ytop, h_top * r_feed, h_top)
    ax.imshow(np.asarray(feed)), bare(ax)
    for x, y, text, key in FEED_LABELS:
        note(ax, x, y, text, CUBE.get(key, INK), va="top")
    caption(fig, ax, "(b) Robot camera, RGB with detections")
    # (c) height above the table, cubes only, over the dimmed camera image
    xc = xb + h_top * r_feed + gap
    ax = axes_at(xc, ytop, h_top * r_cam, h_top)
    ax.imshow(0.30 + 0.45 * rgb.mean(-1, keepdims=True).repeat(3, -1) / 255.0)
    cubes_px = np.any([c["mask"] for c in cubes.values()], axis=0)
    hc = np.where(cubes_px, np.clip(hgt * 100, 0, 9), np.nan)          # [cm]
    im = ax.imshow(hc, cmap="viridis", vmin=0, vmax=9, interpolation="nearest")
    bare(ax)
    for name, c in cubes.items():
        outline(ax, c["mask"], CUBE[name])
        note(ax, c["u"], c["v"] - 8, f"{name}: ({c['x']:.2f}, {c['y']:+.2f}) m\ntop {100 * c['top']:.1f} cm",
             CUBE[name])
    ax.set_xlim(x0, x1), ax.set_ylim(y1, y0)
    caption(fig, ax, "(c) Height above table from depth; base-frame position")
    cax = axes_at(xc + h_top * r_cam - 0.07, ytop + 0.06, 0.035, h_top * 0.42)
    cb = fig.colorbar(im, cax=cax, ticks=[0, 3, 6, 9])
    cb.ax.tick_params(labelsize=FS_S - 0.5, length=1.5, width=0.4, pad=1, left=True, right=False,
                      labelleft=True, labelright=False)
    for t in cb.ax.get_yticklabels():
        t.set_path_effects(HALO)
    cb.outline.set_linewidth(0.4)
    cb.ax.set_title("cm", fontsize=FS_S - 0.5, pad=1.5, path_effects=HALO)

    # perturbation strip
    ys = cap
    for i, (t, _) in enumerate(RP.FRAMES):
        ax = axes_at(i * (wf + sgap), ys, wf, h_strip)
        img = RP.grab(a.multi, t).crop(RP.CROP).resize((1000, round(1000 * ch / cw)), Image.LANCZOS)
        ax.imshow(np.asarray(img)), bare(ax)
        stop = RP.STOPPED[0] <= t <= RP.STOPPED[1]
        if stop:
            for sp in ax.spines.values():
                sp.set_edgecolor(STOP), sp.set_linewidth(1.2)
        caption(fig, ax, f"({chr(100 + i)}) {SHORT[i]}", STOP if stop else INK)

    fig.savefig(OUT / "real_setup.pdf", bbox_inches="tight", pad_inches=0.01, dpi=a.dpi)
    fig.savefig(OUT / "real_setup.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
    print("wrote", OUT / "real_setup.pdf", {k: (round(c["x"], 3), round(c["y"], 3), round(c["top"], 3))
                                             for k, c in cubes.items()})


if __name__ == "__main__":
    main()
