#!/usr/bin/env python3
"""The same Eddies under different physics: MuJoCo -> Isaac Sim (PhysX), Franka FMB and peg insertion.

    python -m eddies.cert.experiments.physics [--mu 5,7.5,10,15] [--alpha 0.1]

The simulated counterpart of deploying on the real robot: identical tasks, Eddies and controller,
only the physics engine (contact, friction, solver) changes.  Data: MuJoCo skillbook episodes (tag sb,
train 1000-1199, calibration 1200-1399) and PhysX skillbook episodes ($EDDIES_RESULTS_ISAAC, tag sb: train
1000-1049, calibration 1200-1249, test 2000-2049).  Certificates on the PhysX test episodes:
  mujoco      MuJoCo models and MuJoCo calibration, unchanged
  recal<n>    MuJoCo models, joint quantile on n PhysX calibration episodes (no refit)
  refit<k>    k PhysX training episodes per task added to the MuJoCo fit (shrinkage keeps the MuJoCo
              model as the prior where PhysX data is scarce), calibrated on 50 PhysX episodes
  physx       PhysX training episodes only, calibrated on 50 PhysX episodes
Per Eddie: contraction rate and envelope in each engine (the token carries the dynamics it is run on).
Writes $EDDIES_RESULTS/physics/physics.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import eddies.cert.data as bdata
from eddies.common.paths import EPISODES, EPISODES_ISAAC, ROOT
from eddies.common.config import load as load_config
from eddies.common.config import blocks, mu_list, mu_str, parse_mu, robots, seeds, span
from eddies.cert.gate import ultimate_tube
from eddies.cert.experiments.table1_eddies import embedding, gate_rows, radius_stats
from eddies.cert.experiments.certify_task import evaluate, summ
from eddies.cert.experiments.gate_fixed import TASKS, build_certificates
from eddies.cert.tube import conformal_quantile, joint_scores

ISAAC = EPISODES_ISAAC
CFG = load_config("experiments/physics")
OUT = ROOT / "physics"
PHYSX_TASKS = list(CFG.tasks)
SP, SI = CFG.splits, CFG.splits.isaac


def load(root, tag, task, seeds):
    old = bdata.EPISODES
    bdata.EPISODES = root
    try:
        return bdata.load_split(tag, "nominal", task, seeds)
    finally:
        bdata.EPISODES = old


def nominal_plan(task):
    from eddies.envs.tasks import make_task
    return list(dict.fromkeys(make_task(task, 2000).nominal_plan()))


def certify(train):
    J, W, scale, pos, rho = embedding("eddie", train)
    tubes, kappa, _ = build_certificates(train, J, W, scale, pos, rho)
    return tubes, kappa, rho


def evaluate_on(tubes, kappa, rho, cal, test, plan, mus, alpha):
    q = conformal_quantile(joint_scores(tubes, kappa, cal), alpha)
    eps = {s: q * kappa[s] for s in tubes}
    return {"q": q, "n_cal": len(cal), "n_test": len(test),
            "violation": summ(evaluate(tubes, eps, test, rho), "D_viol"),
            "radius": radius_stats(tubes, eps, rho, test), "gate": gate_rows(test, tubes, eps, rho, mus, plan)}, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", default=mu_str(CFG.mu_cm))
    ap.add_argument("--alpha", type=float, default=CFG.conformal.alpha)
    ap.add_argument("--refit", default=",".join(str(int(k)) for k in CFG.refit))
    a = ap.parse_args()
    mus = [float(x) / 100 for x in a.mu.split(",")]
    mj = {t: {"train": load(EPISODES, "sb", t, seeds(SP.train)), "calib": load(EPISODES, "sb", t, seeds(SP.calib))}
          for t in TASKS}
    px = {t: {"train": load(ISAAC, "sb", t, seeds(SI.train)), "calib": load(ISAAC, "sb", t, seeds(SI.calib)),
              "test": load(ISAAC, "sb", t, seeds(SI.test))} for t in PHYSX_TASKS}
    for t, d in px.items():
        print("physx", t, {k: len(v) for k, v in d.items()}, flush=True)
    res = {"alpha": a.alpha, "mu_m": mus, "tasks": {t: {} for t in PHYSX_TASKS}, "eddies": {}}
    mj_train = [ep for t in TASKS for ep in mj[t]["train"]]
    T, K, rho = certify(mj_train)
    eps_by = {}

    def record(mode, t, r, eps):
        res["tasks"][t][mode] = r
        eps_by[(mode, t)] = eps
        print(t, mode, "q", round(r["q"], 3), "viol", r["violation"]["violations"], "/", r["n_test"],
              "r90", r["radius"]["p90_cm"], "gate", [x["completed"] for x in r["gate"]],
              "beyond", [x["completed_exceed_mu"] for x in r["gate"]], flush=True)
    for t in PHYSX_TASKS:
        plan, test = nominal_plan(t), px[t]["test"]
        record("mujoco", t, *evaluate_on(T, K, rho, mj[t]["calib"], test, plan, mus, a.alpha))
        for n in (10, 20, 50):
            record(f"recal{n}", t, *evaluate_on(T, K, rho, px[t]["calib"][:n], test, plan, mus, a.alpha))
    fits = {"physx": certify([ep for t in PHYSX_TASKS for ep in px[t]["train"]])}
    for k in [int(x) for x in a.refit.split(",") if x]:
        fits[f"refit{k}"] = certify(mj_train + [ep for t in PHYSX_TASKS for ep in px[t]["train"][:k]])
    for mode, (T2, K2, rho2) in fits.items():
        for t in PHYSX_TASKS:
            record(mode, t, *evaluate_on(T2, K2, rho2, px[t]["calib"], px[t]["test"], nominal_plan(t), mus, a.alpha))
    # per Eddie: the certificate under each engine
    T_px, _, rho_px = fits["physx"]
    for t in PHYSX_TASKS:
        for s in nominal_plan(t):
            row = {}
            for mode, tubes, r in (("mujoco", T, rho), ("recal50", T, rho), ("physx", T_px, rho_px)):
                e = eps_by.get((mode, t), {}).get(s)
                if s in tubes and e is not None:
                    row[mode] = {"spectral_radius": tubes[s].sr, "eps": e, "tube_cm": 100 * ultimate_tube(tubes[s], r, e)}
            res["eddies"][f"{t}/{s}"] = row
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "physics.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
