#!/usr/bin/env python3
"""Fig. 4 (res_cert): conformal calibration on staged insertion (a) and the chained bound along one test
episode (b).  Reads results/figs_data/trace_fmb.json written by eddies.cert.experiments.trace_figs.

    python -m eddies.cert.plots.cert_pair [--out assets/results/generated]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from eddies.cert.plots.style import GREY, NAVY, ORANGE, VIOLET, grid, save, style
from eddies.common.config import load
from eddies.common.paths import FIGURES, ROOT

W1 = 6.9           # full-width panel size [in]


def fig_cert_pair(tr, out_dir=FIGURES, dt=None):
    import matplotlib.pyplot as plt
    dt = float(load().sim.control_dt) if dt is None else dt
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 1.75), gridspec_kw={"wspace": 0.22, "width_ratios": [1, 2.1]})
    sc, st, q = np.asarray(tr["scores_cal"]), np.asarray(tr["scores_test"]), tr["q"]
    hi = 1.75
    bins = np.linspace(0.85, hi, 30)
    a.hist(np.minimum(sc, hi), bins=bins, color=ORANGE, alpha=0.75, label="calibration")
    a.hist(np.minimum(st, hi), bins=bins, histtype="step", color=NAVY, lw=1.1, label="test")
    a.axvline(q, color=VIOLET, lw=1.5)
    a.text(q + 0.02, a.get_ylim()[1] * 0.92, f"q: {100 * np.mean(st <= q):.0f}% of test\nwithin (target 90%)",
           color=VIOLET, fontsize=6.6, va="top")
    a.set_xlabel("episode score $s$"), a.set_ylabel("episodes")
    a.set_title("(a) Calibration", loc="left", fontweight="bold")
    a.legend(loc="center right", fontsize=6.4)
    grid(a)
    seed = sorted(tr["traces"], key=int)[0]
    t = tr["traces"][seed]
    k = np.asarray(t["k"]) * dt
    b.fill_between(k, 0, 100 * np.asarray(t["bound"]), color=VIOLET, alpha=0.13, lw=0)
    b.plot(k, 100 * np.asarray(t["bound"]), color=VIOLET, lw=1.2, label="chained bound")
    b.plot(k, 100 * np.asarray(t["err"]), color=NAVY, lw=0.9, label=r"measured $\|S\eta_k\|$")
    for h in t["handover"]:
        b.axvline(h * dt, color=GREY, lw=0.5, ls=(0, (2, 2)), zorder=0)
    b.set_xlabel("time [s]"), b.set_ylabel("hand error [cm]")
    b.set_title("(b) Chained bound, one test episode (dotted: hand-overs)", loc="left", fontweight="bold")
    b.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), fontsize=6.6, frameon=True, facecolor="white",
             edgecolor="none", framealpha=0.9)
    grid(b)
    return save(fig, out_dir, "res_cert")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(ROOT / "figs_data" / "trace_fmb.json"))
    ap.add_argument("--out", default=str(FIGURES))
    a = ap.parse_args(argv)
    style()
    print("wrote", fig_cert_pair(json.loads(open(a.trace).read()), a.out))


if __name__ == "__main__":
    main()
