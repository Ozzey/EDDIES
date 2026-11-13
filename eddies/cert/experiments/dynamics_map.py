#!/usr/bin/env python3
"""Dynamics descriptors of every Eddie, for the dynamics-similarity map (PCA / clustering).

    python -m eddies.cert.experiments.dynamics_map            # heavy (8 cores, ~48 GB): fits every Eddie of every embodiment
    python -m eddies.cert.plots.dynamics_map           # local: PCA, clustering, figure

Per embodiment (embodiments.GROUPS, MuJoCo) and for Franka FMB + peg in Isaac (PhysX), each
Eddie's OWN least-squares model z+ = A z + B e + d is fitted on the training split (no shrinkage:
the certificate shrinks weak Eddies toward a per-embodiment reference model, which would make
Eddies of one embodiment look alike by construction).  The certified (selected) model is kept
for comparison.  Descriptors are invariant to each embodiment's coordinate scaling:
  eig      |eigenvalues| of A, sorted, padded to 6 (contraction modes)
  gain     DC gain of each position axis, G = (I - A)^-1 B, G[k, J[k]] (1 = exact tracking)
  rise     fraction of that gain reached 2, 5, 10, 20 steps after a unit command step
  coupling off-diagonal DC gain of the position block, relative to its diagonal
Writes $EDDIES_RESULTS/dynamics_map/dynamics_map.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

import eddies.cert.data as bdata
from eddies.common.paths import EPISODES_ISAAC, ROOT
from eddies.common.config import load as load_config
from eddies.common.config import blocks, mu_list, mu_str, parse_mu, robots, seeds, span
from eddies.cert.experiments.table1_eddies import embedding
from eddies.cert.experiments.embodiments import GROUPS, TRAIN
from eddies.cert.experiments.gate_fixed import build_certificates
from eddies.cert.tube import make_tube

CFG = load_config("experiments/dynamics_map")
OUT = ROOT / "dynamics_map"
ISAAC = EPISODES_ISAAC
PHYSX_TASKS = tuple(CFG.physx_tasks)
RISE = tuple(CFG.rise)


def descriptors(t, pos):
    A, B, J = np.asarray(t.A), np.asarray(t.B), np.asarray(t.J)
    eig = sorted(np.abs(np.linalg.eigvals(A)), reverse=True)
    eig = list(eig) + [0.0] * (6 - len(eig))
    rows = [k for k, j in enumerate(J) if j in pos]
    try:
        G = np.linalg.solve(np.eye(len(J)) - A, B)
    except np.linalg.LinAlgError:
        G = np.full_like(B, np.nan)
    gain = [float(G[k, J[k]]) for k in rows]
    Gp = np.array([[G[k, J[c]] for c in rows] for k in rows])
    off = Gp - np.diag(np.diag(Gp))
    coupling = float(np.linalg.norm(off) / max(np.linalg.norm(np.diag(Gp)), 1e-9))
    rise, y, Aj = [], np.zeros_like(B), np.eye(len(J))
    for n in range(1, max(RISE) + 1):
        y = y + Aj @ B
        Aj = Aj @ A
        if n in RISE:
            rise.append(float(np.mean([y[k, J[k]] / G[k, J[k]] for k in rows])))
    return {"eig": [float(x) for x in eig], "gain": gain, "rise": rise, "coupling": coupling}


def trim(eps):
    if not eps:
        return eps
    w = min(ep["z"].shape[1] for ep in eps)
    for ep in eps:
        ep["z"], ep["e"] = ep["z"][:, :w], ep["e"][:, :w]
    return eps


def fit_group(name, sim, train, tasks):
    J, W, scale, pos, rho = embedding("eddie", train)
    cert, _, ref = build_certificates(train, J, W, scale, pos, rho)
    rows = []
    for s in [x for x in dict.fromkeys(x for ep in train for x in ep["skill"]) if x]:
        own = make_tube(train, s, J, W, scale, pos)
        if own is None:
            continue
        r = {"embodiment": name, "sim": sim, "eddie": s, "primitive": s.split(":")[0], "n_train": int(own.n_train),
             "tasks": sorted({t for t, eps in tasks.items() if any(s in ep["skill"] for ep in eps)}),
             "own": descriptors(own, pos), "sr_own": float(max(abs(np.linalg.eigvals(own.A))))}
        if s in cert:
            c = cert[s]
            r.update(certified=True, cert=descriptors(c, pos), sr_cert=float(c.sr),
                     shrunk=bool(np.abs(np.asarray(c.A) - np.asarray(own.A)).max() > 1e-9))
        else:
            r["certified"] = False
        rows.append(r)
    print(name, sim, "Eddies", len(rows), "certified", sum(r["certified"] for r in rows),
          "shrunk", sum(r.get("shrunk", False) for r in rows), "reference", ref, flush=True)
    return rows, ref


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="eddies.cert.experiments.dynamics_map", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT / "dynamics_map.json"), help="output JSON (default results/dynamics_map/dynamics_map.json)")
    a = ap.parse_args(argv)
    rows, refs = [], {}
    for g, tasks in GROUPS.items():
        data = {t: bdata.load_split("sb", "nominal", t, TRAIN) for t in tasks}
        train = trim([ep for t in tasks for ep in data[t]])
        if not train:
            sys.exit(f"dynamics_map: no episodes for {g} ({', '.join(tasks)}) under {bdata.EPISODES}/sb "
                     f"(set $EDDIES_EPISODES or fetch them, see datasets/README.md)")
        r, refs[g] = fit_group(g, "mujoco", train, data)
        rows += r
    old = bdata.EPISODES
    bdata.EPISODES = ISAAC
    try:
        data = {t: bdata.load_split("sb", "nominal", t, seeds(CFG.splits.isaac.train)) for t in PHYSX_TASKS}
    finally:
        bdata.EPISODES = old
    if all(data.values()):
        r, refs["franka_physx"] = fit_group("franka", "physx", [ep for t in PHYSX_TASKS for ep in data[t]], data)
        rows += r
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rise_steps": RISE, "reference": refs, "eddies": rows}, indent=1))


if __name__ == "__main__":
    main()
