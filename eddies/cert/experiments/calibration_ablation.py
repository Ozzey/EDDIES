#!/usr/bin/env python3
"""Table 1, row "Eddies, per-step bound": same Eddie models and tubes, a per-step envelope calibration.

    python -m eddies.cert.experiments.calibration_ablation [--mu 3,5,7.5,10,15] [--alpha 0.1] [--robot g1|leap|aero]

On the robot's tasks (models from gate_fixed's build_certificates, per-task calibration on seeds
1200-1399, test on 2000-2099 for the Franka, 2000-2199 for the other robots), only the residual
bound eps_i changes:
  eddie   ours: one episode-level score s = max_k ||W^1/2 w_k|| / kappa_i(k), eps_i = q kappa_i
  b1      per-step conformal with a union bound over the horizon (Hirano & Tsukamoto, ICRA 2026):
          eps_i = quantile at level alpha / K of Eddie i's per-step residuals ||W^1/2 w_k|| on the
          calibration episodes, K = the longest calibration episode [steps]; the union bound over
          the K steps of an episode gives episode coverage >= 1 - alpha (steps treated as
          exchangeable, as in the original)
The W-weighted residual is used for both (it is what the tube needs); this is tighter than the
unweighted residual of the original per-step bound, so the comparison does not disadvantage it.
Reports per task: eps per Eddie, violations, tube radius, gate completions vs mu.
Writes $EDDIES_RESULTS/table1/calibration.json (--out: another directory under $EDDIES_RESULTS).
--episodes adds, for b1, each calibration episode's length and, per plan Eddie, its step count and
largest per-step residuals (table1_ci.b1_bootstrap); --swap exchanges the training and calibration
seeds.  --robot g1|leap|aero: another robot's own tasks, splits and block (table1_eddies.ROBOTS).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from eddies.common.paths import ROOT
from eddies.common.config import load as load_config
from eddies.common.config import mu_str
from eddies.cert.ident import _transitions
from eddies.cert.experiments import table1_eddies as t1
from eddies.cert.experiments.table1_eddies import a_priori_plan, embedding, gate_rows, load, radius_stats
from eddies.cert.experiments.certify_task import evaluate, summ
from eddies.cert.experiments.gate_fixed import build_certificates
from eddies.cert.tube import conformal_quantile, joint_scores

CFG = load_config("experiments/calibration_ablation")
OUT = ROOT / "table1"


def step_residuals(t, eps_list):
    """||W^1/2 w_k|| for every step of Eddie t in the episodes, and the per-episode maximum."""
    data = _transitions(eps_list, t.skill, t.J, 1, None)
    if data is None:
        return np.array([]), np.full(len(eps_list), -np.inf)
    X, E, Y, ep_id = data
    r = np.linalg.norm((Y - (X @ t.A.T + E @ t.B.T + t.d)) * t.Wz, axis=1)
    per_ep = np.full(len(eps_list), -np.inf)
    np.maximum.at(per_ep, ep_id, r)
    return r, per_ep


def b1_cal_record(t, eps_list, m=30):
    """Per calibration episode: number of steps of Eddie t and its m largest per-step residuals
    (an extreme per-step quantile of a resample is determined by the pooled largest values)."""
    data = _transitions(eps_list, t.skill, t.J, 1, None)
    n, top = [0] * len(eps_list), [[] for _ in eps_list]
    if data is None:
        return n, top
    X, E, Y, ep_id = data
    r = np.linalg.norm((Y - (X @ t.A.T + E @ t.B.T + t.d)) * t.Wz, axis=1)
    for i in np.unique(ep_id):
        ri = r[ep_id == i]
        n[i], top[i] = int(len(ri)), np.sort(ri)[::-1][:m].tolist()
    return n, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", default=mu_str(CFG.mu_cm))
    ap.add_argument("--alpha", type=float, default=CFG.conformal.alpha)
    ap.add_argument("--episodes", action="store_true", help="also save per-episode records (bootstrap intervals)")
    ap.add_argument("--out", default="table1", help="output directory under $EDDIES_RESULTS")
    ap.add_argument("--swap", action="store_true", help="exchange training and calibration seeds (second split)")
    ap.add_argument("--robot", choices=tuple(t1.ROBOTS), default="franka")
    a = ap.parse_args()
    tasks, t1.TEST, t1.BLOCK = t1.ROBOTS[a.robot]
    if a.swap:
        t1.TRAIN, t1.CALIB = t1.CALIB, t1.TRAIN
    mus = [float(x) / 100 for x in a.mu.split(",")]
    data = load(tasks)
    train = [ep for t in tasks for ep in data[t]["train"]]
    J, W, scale, pos, rho = embedding("eddie", train)
    tubes, kappa, _ = build_certificates(train, J, W, scale, pos, rho)
    res = {"alpha": a.alpha, "mu_m": mus, "tasks": {}}
    if a.robot != "franka":
        res["robot"] = a.robot
    for task in tasks:
        cal, test, plan = data[task]["calib"], data[task]["test"], a_priori_plan(task)
        K = max(len(ep["z"]) for ep in cal)
        steps = {s: step_residuals(t, cal)[0] for s, t in tubes.items()}
        q = conformal_quantile(joint_scores(tubes, kappa, cal), a.alpha)
        variants = {
            "eddie": {s: q * kappa[s] for s in tubes},
            "b1": {s: conformal_quantile(steps[s], a.alpha / K) if len(steps[s]) else np.inf for s in tubes},
        }
        res["tasks"][task] = {"K": K}
        for v, eps in variants.items():
            cert = {s: tubes[s] for s in eps if np.isfinite(eps[s])}
            e = {s: eps[s] for s in cert}
            row = {"certified_in_plan": sum(s in cert for s in plan), "plan": len(plan),
                   "violation": summ(evaluate(cert, e, test, rho), "D_viol"),
                   "radius": radius_stats(cert, e, rho, test),
                   "gate": gate_rows(test, cert, e, rho, mus, plan),
                   "eps_plan": {s: (float(eps[s]) if np.isfinite(eps[s]) else None) for s in plan}}
            if a.episodes and v == "b1":
                row["episodes"] = {"len": [len(ep["z"]) for ep in cal], "n": {}, "top": {}}
                for s in plan:
                    if s in tubes:
                        n_s, top = b1_cal_record(tubes[s], cal)
                        row["episodes"]["n"][s], row["episodes"]["top"][s] = n_s, top
            res["tasks"][task][v] = row
            print(task, v, "K", K, "certified", row["certified_in_plan"], "/", len(plan),
                  "viol", row["violation"]["violations"], "r50/r90", row["radius"]["median_cm"], row["radius"]["p90_cm"],
                  "gate", [g["completed"] for g in row["gate"]], "beyond", [g["completed_exceed_mu"] for g in row["gate"]], flush=True)
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
