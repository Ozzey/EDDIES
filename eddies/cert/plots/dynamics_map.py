#!/usr/bin/env python3
"""Dynamics-similarity map of the Eddies (light; reads the JSON of eddies.cert.experiments.dynamics_map).

    python -m eddies.cert.plots.dynamics_map [--json path] [--model own|cert]

(a) PCA of the standardised dynamics descriptors (one point per Eddie and embodiment; Franka FMB /
    peg Eddies also in PhysX, joined to their MuJoCo point).
(b) Descriptor distance between: the same Eddie on two embodiments; two different Eddies on one
    embodiment; the same Eddie in MuJoCo and PhysX.
Also prints k-means agreement (adjusted Rand index) with the embodiment and with the primitive, and
nearest-neighbour label agreement, each against a label-permutation baseline.
Writes dynamics_map.pdf / .png and dynamics_map_stats.json next to the JSON.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.special import comb

from eddies.common.paths import FIGURES, ROOT

# reference categorical palette, fixed order (dataviz skill, references/palette.md), plus a shape each
STYLE = {("franka", "mujoco"): ("#2a78d6", "o", "Franka (MuJoCo)"),
         ("g1", "mujoco"): ("#eb6834", "s", "G1 + Dex3"),
         ("leap", "mujoco"): ("#1baf7a", "^", "UR5e + LEAP"),
         ("aero", "mujoco"): ("#eda100", "D", "UR5e + Aero"),
         ("franka", "physx"): ("#e87ba4", "o", "Franka (PhysX)")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


SR_MAX, N_MIN, K_EIG = 0.99, 500, 5     # identifiable fits only; top-5 modes (hands certify 5 coordinates)


def identifiable(r, model):
    d = r[model] if model in r else r["own"]
    return max(d["eig"]) < SR_MAX and r["n_train"] >= N_MIN


def features(r, model):
    """Scale-free, bounded descriptors: time constants of the 5 slowest modes [steps], position DC
    gains and step-rise fractions clipped to a physical range, log(1 + coupling)."""
    d = r[model] if model in r else r["own"]
    lam = np.clip(np.sort(d["eig"])[::-1][:K_EIG], 1e-3, SR_MAX)
    tau = -1.0 / np.log(lam)
    g = np.clip(np.r_[d["gain"], [np.nan] * (3 - len(d["gain"]))], 0.0, 2.0)
    rise = np.clip(d["rise"], 0.0, 1.5)
    return np.r_[np.log(tau), g, rise, np.log1p(d["coupling"])]


def kmeans(X, k, seeds=20, iters=100):
    best = None
    for s in range(seeds):
        rng = np.random.default_rng(s)
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
            C2 = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(C2, C):
                break
            C = C2
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return best[1]


def ari(a, b):
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    M = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(M, (a, b), 1)
    s = comb(M, 2).sum()
    sa, sb, n = comb(M.sum(1), 2).sum(), comb(M.sum(0), 2).sum(), comb(len(a), 2)
    exp = sa * sb / n
    return float((s - exp) / (0.5 * (sa + sb) - exp))


def nn_agree(X, lab):
    D = ((X[:, None] - X[None]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    return float(np.mean(lab[np.argmin(D, 1)] == lab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "dynamics_map" / "dynamics_map.json"))
    ap.add_argument("--model", choices=("own", "cert"), default="own")
    a = ap.parse_args()
    src = Path(a.json)
    rows = json.loads(src.read_text())["eddies"]
    X = np.array([features(r, a.model) for r in rows], float)
    keep = np.all(np.isfinite(X), 1) & np.array([identifiable(r, a.model) for r in rows])
    dropped = [f'{r["embodiment"]}/{r["sim"]}/{r["eddie"]}' for r, k in zip(rows, keep) if not k]
    rows, X = [r for r, k in zip(rows, keep) if k], X[keep]
    iqr = np.subtract(*np.quantile(X, [.75, .25], axis=0))
    Z = (X - np.median(X, 0)) / np.where(iqr > 1e-9, iqr, X.std(0) + 1e-9)     # robust scaling
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    P = Z @ Vt[:2].T
    var = S ** 2 / (S ** 2).sum()

    emb = np.array([f'{r["embodiment"]}/{r["sim"]}' for r in rows])
    prim = np.array([r["primitive"] for r in rows])
    name = np.array([r["eddie"] for r in rows])
    mj = np.array([r["sim"] == "mujoco" for r in rows])

    # clustering statistics on the MuJoCo embodiments (PhysX is a second simulator, not a body)
    Zm, em, pm = Z[mj], emb[mj], prim[mj]
    rng = np.random.default_rng(0)
    perm = lambda f, lab: float(np.mean([f(rng.permutation(lab)) for _ in range(200)]))
    lab_k = kmeans(Zm, len(np.unique(em)))
    stats = {"model": a.model, "n": int(len(rows)), "dropped": dropped, "pca_var": var[:2].tolist(),
             "ari_kmeans_embodiment": ari(lab_k, em), "ari_kmeans_primitive": ari(lab_k, pm),
             "nn_same_embodiment": nn_agree(Zm, em), "nn_same_embodiment_chance": perm(lambda l: nn_agree(Zm, l), em),
             "nn_same_primitive": nn_agree(Zm, pm), "nn_same_primitive_chance": perm(lambda l: nn_agree(Zm, l), pm)}
    lab_p = kmeans(Zm, len(np.unique(pm)))
    stats["ari_kmeans_k=primitives_vs_primitive"] = ari(lab_p, pm)
    stats["ari_kmeans_k=primitives_vs_embodiment"] = ari(lab_p, em)

    # distances
    dist = lambda i, j: float(np.linalg.norm(Z[i] - Z[j]))
    same_eddie, same_body, sim2sim = [], [], []
    for i, j in combinations(range(len(rows)), 2):
        if mj[i] and mj[j]:
            if name[i] == name[j] and emb[i] != emb[j]:
                same_eddie.append(dist(i, j))
            elif emb[i] == emb[j] and name[i] != name[j]:
                same_body.append(dist(i, j))
        elif name[i] == name[j] and rows[i]["embodiment"] == rows[j]["embodiment"] == "franka":
            sim2sim.append(dist(i, j))
    stats.update({k: {"n": len(v), "median": float(np.median(v)) if v else None,
                      "q25": float(np.quantile(v, .25)) if v else None, "q75": float(np.quantile(v, .75)) if v else None}
                  for k, v in (("d_same_eddie_other_embodiment", same_eddie), ("d_other_eddie_same_embodiment", same_body),
                               ("d_same_eddie_mujoco_physx", sim2sim))})
    # contraction rates, own vs certified model (the shrinkage check)
    stats["spectral_radius"] = {}
    for e in sorted(set(emb)):
        sel = [r for r in rows if f'{r["embodiment"]}/{r["sim"]}' == e]
        so = [r["sr_own"] for r in sel]
        sc = [r["sr_cert"] for r in sel if r.get("certified")]
        stats["spectral_radius"][e] = {"own_median": float(np.median(so)), "own_iqr": float(np.subtract(*np.quantile(so, [.75, .25]))),
                                        "cert_median": float(np.median(sc)) if sc else None,
                                        "cert_iqr": float(np.subtract(*np.quantile(sc, [.75, .25]))) if sc else None,
                                        "shrunk": sum(r.get("shrunk", False) for r in sel), "n": len(sel)}
    print(json.dumps(stats, indent=1))
    (src.parent / f"dynamics_map_stats_{a.model}.json").write_text(json.dumps(stats, indent=1))

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "serif", "pdf.fonttype": 42})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw={"width_ratios": [1.55, 1]})
    for i, j in combinations(range(len(rows)), 2):           # MuJoCo -> PhysX for the same Franka Eddie
        if name[i] == name[j] and {rows[i]["sim"], rows[j]["sim"]} == {"mujoco", "physx"} \
                and rows[i]["embodiment"] == rows[j]["embodiment"] == "franka":
            ax.plot(P[[i, j], 0], P[[i, j], 1], color=MUTED, lw=0.6, alpha=0.6, zorder=1)
    for key, (c, m, lab) in STYLE.items():
        sel = np.array([(r["embodiment"], r["sim"]) == key for r in rows])
        if not sel.any():
            continue
        face = "white" if key[1] == "physx" else c
        ax.scatter(P[sel, 0], P[sel, 1], s=22, marker=m, facecolor=face, edgecolor=c if key[1] == "physx" else "white",
                   linewidth=1.0 if key[1] == "physx" else 0.6, label=f"{lab} ({sel.sum()})", zorder=3)
    ax.set_xlabel(f"PC 1 ({100 * var[0]:.0f} % of variance)")
    ax.set_ylabel(f"PC 2 ({100 * var[1]:.0f} %)")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="best", handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(a) Eddie dynamics, PCA", fontsize=8, loc="left", color=INK)

    groups = [("same Eddie,\nother body", same_eddie), ("other Eddie,\nsame body", same_body),
              ("same Eddie,\nMuJoCo vs PhysX", sim2sim)]
    groups = [(g, v) for g, v in groups if v]
    for k, (g, v) in enumerate(groups):
        x = k + rng.uniform(-0.18, 0.18, len(v))
        bx.scatter(x, v, s=6, color="#2a78d6", alpha=0.35, linewidth=0, zorder=2)
        q1, md, q3 = np.quantile(v, [.25, .5, .75])
        bx.plot([k - 0.28, k + 0.28], [md, md], color=INK, lw=1.6, solid_capstyle="round", zorder=3)
        bx.plot([k, k], [q1, q3], color=INK, lw=0.8, zorder=3)
        bx.text(k + 0.3, md, f"{md:.1f}", va="center", fontsize=6.5, color=INK)
    bx.set_xticks(range(len(groups)), [g for g, _ in groups], fontsize=6.5)
    bx.set_ylabel("descriptor distance (std. units)")
    bx.grid(axis="y", color=GRID, lw=0.5)
    bx.set_axisbelow(True)
    bx.set_title("(b) what sets an Eddie's dynamics", fontsize=8, loc="left", color=INK)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(src.parent / f"dynamics_map_{a.model}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", src.parent / f"dynamics_map_{a.model}.pdf")


if __name__ == "__main__":
    main()
