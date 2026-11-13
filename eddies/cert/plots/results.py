#!/usr/bin/env python3
"""Result figures of the paper (Fig. 5 and Fig. 7), one message per figure.

    python -m eddies.cert.plots.results            # reads results/, writes assets/results/generated/res_*.pdf

res_gate.pdf     finished runs vs tolerance per robot (mean over its tasks, band = min..max over tasks):
                 self-calibrating gate vs fixed calibration
res_heldout.pdf  held-out Franka tasks: finished vs tolerance (mean, band over tasks) from no calibration vs
                 the task's own calibration; running envelope-miss rate along the episode stream
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eddies.common.paths import FIGURES as OUT
from eddies.common.paths import ROOT as R
from eddies.cert.plots.style import ARROW, BLUE, COLORS, GREY, GRID, NAVY, ORANGE, VIOLET, grid, style  # noqa: F401
from eddies.common.config import load, robots

CFG = load()
ROBOTS = [(r.short, COLORS[r.color], r.name, list(r.tasks)) for r in robots(CFG).values()]
HELD = list(CFG.franka_tasks)


def curve(rows, mus):
    """finished % at each tolerance (cm) from a run's rows {mu_m(str): summary}."""
    by = {round(100 * float(k), 2): v["summary"] for k, v in rows.items()}
    return np.array([100 * by[m]["completed"] / by[m]["n"] for m in mus])


# --------------------------------------------------------------------------- Fig. 5 -> res_gate
def fig_gate():
    emb = json.loads((R / "embodiments" / "embodiments.json").read_text())
    mus = [round(100 * m, 2) for m in emb["mu_m"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 1.75), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, title, fixed in ((axes[0], "Self-calibrating gate", False), (axes[1], "Fixed calibration", True)):
        for robot, col, g, tasks in ROBOTS:
            if fixed:
                Y = np.array([[100 * x["completed"] / x["n"] for x in emb["groups"][g]["tasks"][t]["gate"]] for t in tasks])
            else:
                Y = np.array([curve(json.loads((R / "adaptive" / f"adaptive_{t}_known_{g}.json").read_text())["rows"], mus)
                              for t in tasks])
            ax.plot(mus, Y.mean(0), color=col, marker="o", label=robot)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks([2, 3, 5, 7.5, 10, 15], ["2", "3", "5", "7.5", "10", "15"])
        ax.set_xlabel(r"tolerance $\mu$ [cm]")
        ax.set_ylim(-3, 103)
        ax.axvline(5, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=0)
        grid(ax)
    axes[0].set_ylabel("finished runs [%]")
    axes[1].legend(loc="lower right", handlelength=1.8)
    save(fig, "res_gate")


# --------------------------------------------------------------------------- Fig. 7 -> res_heldout
def fig_heldout():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 1.75), gridspec_kw={"wspace": 0.28})
    mus = [1, 2, 3, 5, 7.5, 10, 15]
    for known, col, lab in ((True, BLUE, "own calibration as prior"), (False, VIOLET, "no calibration of the task")):
        Y = np.array([curve(json.loads((R / "adaptive" / f"adaptive_{t}{'_known' if known else ''}.json").read_text())["rows"],
                            mus) for t in HELD])
        a.fill_between(mus, Y.min(0), Y.max(0), color=col, alpha=0.13, lw=0)
        a.plot(mus, Y.mean(0), color=col, marker="o", label=lab)
    a.set_xticks([1, 2, 3, 5, 7.5, 10, 15], ["1", "2", "3", "5", "7.5", "10", "15"])
    a.set_xlabel(r"tolerance $\mu$ [cm]")
    a.set_ylabel("finished runs [%]")
    a.set_ylim(-3, 103)
    a.set_title("Finished runs on a new task", loc="left", fontweight="bold")
    grid(a)
    a.legend(loc="lower right", handlelength=1.8)
    E = []
    for t in HELD:
        err = np.asarray(json.loads((R / "adaptive" / f"adaptive_{t}.json").read_text())["rows"]["0.05"]["trace"]["err"])
        E.append(100 * np.cumsum(err) / np.arange(1, len(err) + 1))
    E = np.array(E)
    x = np.arange(1, E.shape[1] + 1)
    b.fill_between(x, E.min(0), E.max(0), color=VIOLET, alpha=0.13, lw=0)
    b.plot(x, E.mean(0), color=VIOLET, label="no calibration of the task")
    b.axhline(10, color=ARROW, lw=1.0, ls=(0, (4, 2)))
    b.text(99, 12, r"target $\alpha$ = 10%", ha="right", va="bottom", fontsize=7, color=ARROW)
    b.set_xlabel("episodes of the new task")
    b.set_ylabel("episodes outside the\nenvelope so far [%]")
    b.set_xlim(1, 100), b.set_ylim(0, 50)
    b.set_title("The envelope calibrates itself", loc="left", fontweight="bold")
    grid(b)
    save(fig, "res_heldout")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=220)
    print("wrote", OUT / f"{name}.pdf")


if __name__ == "__main__":
    style()
    fig_gate()
    fig_heldout()
