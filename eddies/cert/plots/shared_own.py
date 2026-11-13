"""Tube of the same Eddie on different robots, each Eddie with its own task's calibration.

Reads results/embodiments/embodiments.json, groups.<robot>.tasks.<task>.plan_tube_cm (the ultimate tube of
every Eddie in the task's plan, with the task's own conformal level q).  An Eddie that runs in several tasks of
one robot gets the median (and range) over those tasks.  Writes assets/results/generated/app_shared_own.pdf/.png and prints
the LaTeX rows of the supplement's per-Eddie table (tab:shared).

    python -m eddies.cert.plots.shared_own
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eddies.cert.plots.results import ARROW, BLUE, GREY, GRID, ORANGE, VIOLET, style

from eddies.common.paths import FIGURES as OUT
from eddies.common.paths import ROOT as R
ROB = ["franka", "g1", "leap", "aero"]


PAPER_NAMES = {  # wording of the supplement's Table D.6 (tab:shared) where it differs from words()
    "transport:hole_above": "transport (above hole)", "transport:goal_above": "transport (above goal)",
    "insert:hole": "insert (hole)", "place:goal": "place (goal)", "turn:unwind": "turn peg (unwind)",
}


def words(name):
    """'reach:peg_above' -> 'reach peg (above)' (Table D.6 wording for the frames in PAPER_NAMES)."""
    if name in PAPER_NAMES:
        return PAPER_NAMES[name]
    prim, _, frame = name.partition(":")
    obj, _, part = frame.partition("_")
    return f"{prim} {obj}" + (f" ({part})" if part else "")


def own_tubes():
    emb = json.loads((R / "embodiments" / "embodiments.json").read_text())
    by = defaultdict(lambda: defaultdict(list))
    for g in ROB:
        for t, T in emb["groups"][g]["tasks"].items():
            for e, v in T["plan_tube_cm"].items():
                by[e][g].append(v)
    return by


def main():
    by = own_tubes()
    shared = {e: v for e, v in by.items() if len(v) > 1}
    style()
    fig, ax = plt.subplots(figsize=(6.9, 2.0))
    pairs = [("franka", "g1", "Franka vs G1", ORANGE), ("leap", "aero", "LEAP vs Aero", VIOLET),
             ("franka", "leap", "Franka vs LEAP", BLUE), ("g1", "aero", "G1 vs Aero", ARROW)]
    mx = 0
    for a, b, lab, col in pairs:
        X = np.array([(np.median(v[a]), np.median(v[b])) for v in shared.values() if a in v and b in v])
        if len(X):
            ax.scatter(X[:, 0], X[:, 1], s=16, color=col, label=f"{lab} ({len(X)} Eddies)", zorder=3,
                       edgecolors="white", linewidths=0.3)
            mx = max(mx, X.max())
    ax.plot([0, mx * 1.05], [0, mx * 1.05], color=GREY, lw=0.8, ls=(0, (4, 2)))
    ax.set_xlabel("tube on the first robot [cm]"), ax.set_ylabel("tube on the second [cm]")
    ax.set_title("The same Eddie gets a different tube on a different robot (own-task calibration)",
                 loc="left", fontweight="bold")
    ax.grid(True, color=GRID, lw=0.6), ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 160})):
        fig.savefig(OUT / f"app_shared_own.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    print("wrote app_shared_own")

    def cell(v):
        if not v:
            return "--"
        lo, hi = min(v), max(v)
        return f"{lo:.1f}" if round(lo, 1) == round(hi, 1) else f"{lo:.1f}--{hi:.1f}"
    order = sorted(shared, key=lambda e: (-len(shared[e]), e))
    for e in order:
        print(f"{words(e)} & " + " & ".join(cell(shared[e].get(g, [])) for g in ROB) + r"\\")


if __name__ == "__main__":
    main()
