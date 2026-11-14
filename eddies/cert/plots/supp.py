#!/usr/bin/env python3
"""The four figures of the compact supplement, drawn at the L4DC text width (6.0 in) with >= 7 pt text.

    python -m eddies.cert.plots.supp          # writes assets/results/generated/supp_*.pdf/.png

supp_gains       one-step gain a_i vs spectral radius, and the factor one run passes to the next Eddie
supp_gate_tasks  finished runs vs tolerance for every task, self-calibrating vs fixed gate
supp_qtrace      conformal quantile q_t of the self-calibrating gate on each held-out Franka task
supp_dynmap      PCA of the dynamics descriptors and descriptor distances (same Eddie / other Eddie / PhysX)

Light: reads result JSONs only.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eddies.cert.plots.results import ARROW, BLUE, GREY, GRID, NAVY, ORANGE, ROBOTS, VIOLET, curve, grid, style
from eddies.cert.plots.dynamics_map import STYLE as DM_STYLE, features, identifiable

from eddies.common.paths import FIGURES as OUT
from eddies.common.paths import ROOT
from eddies.common.paths import ROOT as R
GAINS = R / "facts" / "gain_all.json"          # written by scripts/facts/gain_check.py
TW = 6.0                      # L4DC text width [in] (433.62 pt)
FS = 7.0                      # smallest font used anywhere [pt]
GAIN_ROBOTS = [("franka", "Franka", BLUE, "o"), ("g1", "G1", ORANGE, "s"),
               ("leap", "LEAP hand", VIOLET, "^"), ("aero", "Aero hand", ARROW, "D")]


def setup():
    style()
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                         "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=160)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------------------------------ gains
def fig_gains():
    d = json.loads(GAINS.read_text())
    data = {k: [(v["a_i"], v["rho_A"], v["segment_factor_a_times_cA_Lminus1_med_max"])
                for v in d[k]["eddies"].values() if v.get("a_i") is not None] for k, *_ in GAIN_ROBOTS if k in d}
    fig, axes = plt.subplots(1, 2, figsize=(TW, 1.8), gridspec_kw={"wspace": 0.3, "width_ratios": [1.35, 1]})
    ax = axes[0]
    lo = 0.55
    ax.plot([lo, 1.02], [lo, 1.02], color=GREY, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.axhline(1.0, color=NAVY, lw=0.7, zorder=1)
    ax.text(0.935, 1.008, r"$a_i=1$", color=NAVY, fontsize=FS, va="bottom")
    ax.text(0.72, 0.68, r"$a_i=\max|\lambda(A_i)|$", color=GREY, fontsize=FS, ha="left", va="top")
    for key, name, col, mk in GAIN_ROBOTS:
        X = np.array([(r[1], r[0]) for r in data[key]])
        ax.scatter(X[:, 0], X[:, 1], s=15, color=col, marker=mk, label=f"{name} ({len(X)})", zorder=3,
                   edgecolors="white", linewidths=0.3)
    amax = max(r[0] for rows in data.values() for r in rows)
    ax.set_xlim(lo, 1.02), ax.set_ylim(lo, max(1.06, amax + 0.05))
    ax.set_xlabel(r"spectral radius $\max|\lambda(A_i)|$"), ax.set_ylabel(r"one-step gain $a_i$")
    ax.set_title("(a) one-step gain of every certified Eddie", loc="left")
    ax.grid(True, color=GRID, lw=0.6), ax.set_axisbelow(True)
    ax.legend(loc="upper left", handletextpad=0.2, borderaxespad=0.3)

    ax = axes[1]
    floor = 1e-10
    ticks, labels = [], []
    for pos, (key, name, col, mk) in enumerate(GAIN_ROBOTS):
        F = np.clip(np.array([r[2] for r in data[key]], float), floor, None)
        jit = np.random.default_rng(0).uniform(-0.18, 0.18, len(F))
        ax.scatter(pos + jit, F[:, 0], s=11, color=col, marker=mk, zorder=3, edgecolors="white", linewidths=0.3,
                   label="median run" if pos == 0 else None)
        ax.scatter(pos + jit, F[:, 1], s=11, facecolors="none", edgecolors=col, marker=mk, linewidths=0.6, zorder=3,
                   label="largest run" if pos == 0 else None)
        ticks.append(pos), labels.append(name.replace(" hand", ""))
    ax.set_yscale("log"), ax.set_ylim(floor / 3, 3e2)
    ax.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
    ax.axhline(1.0, color=NAVY, lw=0.7)
    ax.set_xticks(ticks, labels), ax.set_xlim(-0.6, len(ticks) - 0.4)
    ax.set_ylabel(r"$a_i\,\|W^{1/2}A_i^{L-1}W^{-1/2}\|$")
    ax.set_title("(b) carried over by one run", loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.6), ax.set_axisbelow(True)
    ax.legend(loc="upper right", handletextpad=0.2, ncol=2, columnspacing=0.8)
    save(fig, "supp_gains")


# ------------------------------------------------------------------------------------- gate tasks
def fig_gate_tasks():
    emb = json.loads((R / "embodiments" / "embodiments.json").read_text())
    mus = [round(100 * m, 2) for m in emb["mu_m"]]
    names = {"franka_fmb": "staged", "franka_peg_insertion": "peg", "franka_cube_stack": "stack", "franka_drawer": "drawer",
             "g1_peg_insertion_sb": "peg", "g1_drawer_sb": "drawer", "leap_wheel_sb": "wheel",
             "leap_peg_tighten_sb": "peg tighten", "aero_wheel_sb": "wheel", "aero_peg_tighten_sb": "peg tighten"}
    fig, axes = plt.subplots(1, 4, figsize=(TW, 1.6), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (robot, col, g, tasks) in zip(axes, ROBOTS):
        for t, mk in zip(tasks, "osD^"):
            fixed = [100 * x["completed"] / x["n"] for x in emb["groups"][g]["tasks"][t]["gate"]]
            ad = curve(json.loads((R / "adaptive" / f"adaptive_{t}_known_{g}.json").read_text())["rows"], mus)
            ax.plot(mus, ad, color=col, marker=mk, lw=1.2, label=names[t])
            ax.plot(mus, fixed, color=col, marker=mk, mfc="white", lw=1.0, ls=(0, (3, 2)), alpha=0.8)
        ax.set_title(robot, color=col, fontweight="bold")
        ax.set_xticks([2, 5, 10, 15], ["2", "5", "10", "15"]), ax.set_xlabel(r"$\mu$ [cm]")
        grid(ax)
        ax.legend(loc="center right", bbox_to_anchor=(1.02, 0.47), handlelength=1.3, borderaxespad=0.1, labelspacing=0.25, fontsize=FS)
    axes[0].set_ylabel("finished [%]")
    save(fig, "supp_gate_tasks")


# ----------------------------------------------------------------------------------------- qtrace
def fig_qtrace():
    fig, ax = plt.subplots(figsize=(TW, 1.45))
    for (task, name), col in zip([("franka_fmb", "staged insertion"), ("franka_peg_insertion", "peg insertion"),
                                  ("franka_cube_stack", "cube stack"), ("franka_drawer", "drawer + box")],
                                 (BLUE, ORANGE, VIOLET, ARROW)):
        r = json.loads((R / "adaptive" / f"adaptive_{task}.json").read_text())
        q = [r["q0"]] + r["rows"]["0.05"]["trace"]["q"]
        ax.plot(np.arange(len(q)), q, color=col, lw=1.4, label=name)
    ax.set_xlabel("episode of the new task"), ax.set_ylabel("conformal quantile $q_t$")
    grid(ax)
    ax.legend(loc="upper right", ncol=4, handlelength=1.6, columnspacing=1.2)
    save(fig, "supp_qtrace")


# ----------------------------------------------------------------------------------------- dynmap
def fig_dynmap():
    rows = json.loads((ROOT / "dynamics_map" / "dynamics_map.json").read_text())["eddies"]
    X = np.array([features(r, "own") for r in rows], float)
    keep = np.all(np.isfinite(X), 1) & np.array([identifiable(r, "own") for r in rows])
    rows, X = [r for r, k in zip(rows, keep) if k], X[keep]
    iqr = np.subtract(*np.quantile(X, [.75, .25], axis=0))
    Z = (X - np.median(X, 0)) / np.where(iqr > 1e-9, iqr, X.std(0) + 1e-9)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    P, var = Z @ Vt[:2].T, S ** 2 / (S ** 2).sum()
    emb = np.array([f'{r["embodiment"]}/{r["sim"]}' for r in rows])
    name = np.array([r["eddie"] for r in rows])
    mj = np.array([r["sim"] == "mujoco" for r in rows])
    same_eddie, same_body, sim2sim = [], [], []
    for i, j in combinations(range(len(rows)), 2):
        d = float(np.linalg.norm(Z[i] - Z[j]))
        if mj[i] and mj[j]:
            if name[i] == name[j] and emb[i] != emb[j]:
                same_eddie.append(d)
            elif emb[i] == emb[j] and name[i] != name[j]:
                same_body.append(d)
        elif name[i] == name[j] and rows[i]["embodiment"] == rows[j]["embodiment"] == "franka":
            sim2sim.append(d)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TW, 1.95), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.3})
    for i, j in combinations(range(len(rows)), 2):
        if name[i] == name[j] and {rows[i]["sim"], rows[j]["sim"]} == {"mujoco", "physx"} \
                and rows[i]["embodiment"] == rows[j]["embodiment"] == "franka":
            ax.plot(P[[i, j], 0], P[[i, j], 1], color=GREY, lw=0.6, alpha=0.6, zorder=1)
    for key, (c, m, lab) in DM_STYLE.items():
        sel = np.array([(r["embodiment"], r["sim"]) == key for r in rows])
        if not sel.any():
            continue
        physx = key[1] == "physx"
        ax.scatter(P[sel, 0], P[sel, 1], s=20, marker=m, facecolor="white" if physx else c,
                   edgecolor=c if physx else "white", linewidth=1.0 if physx else 0.5, label=f"{lab} ({sel.sum()})",
                   zorder=3)
    ax.set_xlabel(f"PC 1 ({100 * var[0]:.0f}% of variance)"), ax.set_ylabel(f"PC 2 ({100 * var[1]:.0f}%)")
    # symmetric-log axes: a few G1 outliers (|PC| up to 50) would otherwise flatten the other 70 points
    for setter, ticks in ((ax.set_xscale, [-50, -10, -3, 0, 3, 10, 50]), (ax.set_yscale, [-30, -10, -3, 0, 3, 10, 30])):
        setter("symlog", linthresh=3.0, linscale=0.8)
    ax.set_xticks([-50, -10, -3, 0, 3, 10, 50], ["-50", "-10", "-3", "0", "3", "10", "50"])
    ax.set_yticks([-30, -10, -3, 0, 3, 10, 30], ["-30", "-10", "-3", "0", "3", "10", "30"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()), ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(color=GRID, lw=0.5), ax.set_axisbelow(True)
    ax.set_title("(a) identified Eddie dynamics, PCA", loc="left")

    groups = [("same Eddie,\nother robot", same_eddie), ("other Eddie,\nsame robot", same_body),
              ("same Eddie,\nMuJoCo/PhysX", sim2sim)]
    rng = np.random.default_rng(0)
    for k, (g, v) in enumerate(groups):
        x = k + rng.uniform(-0.18, 0.18, len(v))
        bx.scatter(x, v, s=6, color=BLUE, alpha=0.35, linewidth=0, zorder=2)
        q1, md, q3 = np.quantile(v, [.25, .5, .75])
        bx.plot([k - 0.28, k + 0.28], [md, md], color=NAVY, lw=1.6, solid_capstyle="round", zorder=3)
        bx.plot([k, k], [q1, q3], color=NAVY, lw=0.8, zorder=3)
        bx.text(k + 0.3, md, f"{md:.1f}", va="center", fontsize=FS, color=NAVY)
    bx.set_xticks(range(len(groups)), [g for g, _ in groups], fontsize=FS, linespacing=1.1)
    bx.set_xlim(-0.5, len(groups) - 0.5)
    bx.set_ylabel("descriptor distance")
    grid(bx)
    bx.set_title("(b) distance between fits", loc="left")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, handletextpad=0.3, columnspacing=0.8,
              frameon=False, fontsize=FS)
    save(fig, "supp_dynmap")


if __name__ == "__main__":
    setup()
    fig_gains(), fig_gate_tasks(), fig_qtrace(), fig_dynmap()
