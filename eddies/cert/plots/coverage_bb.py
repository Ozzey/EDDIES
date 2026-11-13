"""Test coverage that exchangeability allows for split conformal calibration with N calibration and M test
episodes, against the coverage measured on the four Franka tasks (results/pool/final_indist_eddy.json).

Under exchangeability the coverage conditional on the calibration set is Beta(l, N+1-l), l = ceil((N+1)(1-alpha)),
so the number of covered test episodes is BetaBinomial(M, l, N+1-l).  Writes app_coverage_bb.pdf/.png to assets/results/generated/ (paths.figures).

    python -m eddies.cert.plots.coverage_bb
"""
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import betabinom

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eddies.cert.plots.results import BLUE, GREY, GRID, NAVY, ORANGE, VIOLET, style

from eddies.common.paths import FIGURES as OUT
from eddies.common.paths import ROOT as R
NAMES = {"franka_fmb": "staged insertion", "franka_peg_insertion": "peg insertion",
         "franka_cube_stack": "cube stack", "franka_drawer": "drawer + box"}


def measured():
    d = json.loads((R / "pool" / "final_indist_eddy.json").read_text())
    rows = d.get("tasks", d)
    out = {}
    for t, name in NAMES.items():
        v = rows[t]
        out[name] = (v["coverage"], v.get("n_calib", 200), v.get("n_test", 200))
    return out


def main(alpha=0.1):
    meas = measured()
    N = M = 200
    l = math.ceil((N + 1) * (1 - alpha))
    dist = betabinom(M, l, N + 1 - l)
    k = np.arange(150, M + 1)
    style()
    fig, ax = plt.subplots(figsize=(6.9, 1.8))
    ax.bar(k / M, dist.pmf(k), width=0.8 / M, color=BLUE, alpha=0.6, label="allowed by exchangeability")
    ax.axvline(1 - alpha, color=GREY, lw=0.8, ls=(0, (3, 2)))
    cols = [VIOLET, NAVY, ORANGE, ORANGE]
    ymax = dist.pmf(k).max()
    for (name, (c, n_cal, n_test)), col, dy in zip(meas.items(), cols, (0.97, 0.72, 0.45, 0.97)):
        ax.axvline(c, color=col, lw=1.4)
        p = dist.cdf(round(c * M)) if c < 1 - alpha else dist.sf(round(c * M) - 1)
        side = "P(≤)" if c < 1 - alpha else "P(≥)"
        ax.text(c + 0.002, ymax * dy, f"{name} {c:.3f}\n{side} = {p:.3f}", color=col, fontsize=6.5, va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6))
    ax.set_xlim(0.80, 0.985)
    ax.set_xlabel("coverage on 200 test episodes")
    ax.set_ylabel("probability")
    ax.set_title(f"Coverage with {N} calibration and {M} test episodes, " + r"$\alpha=0.1$",
                 loc="left", fontweight="bold")
    ax.grid(True, axis="y", color=GRID, lw=0.6), ax.set_axisbelow(True)
    ax.legend(loc="center left", fontsize=6.5)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 160})):
        fig.savefig(OUT / f"app_coverage_bb.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    lo = dist.cdf(167)
    print("wrote app_coverage_bb; sd %.4f; P(K<=167) %.3f; P(min of 4 <= 167) %.3f" %
          (dist.std() / M, lo, 1 - (1 - lo) ** 4), meas)


if __name__ == "__main__":
    main()
