#!/usr/bin/env python3
"""Certify-or-stop experiment on the Eddie tasks.

    python -m eddies.cert.experiments.gate_fixed [--pool eddy|primitive] [--mu 3,4,5,7.5,10,15] [--alpha 0.1]

Certificates: per-Eddie (or per-primitive) models fitted on tag ``sb`` training seeds (all four
tasks, nominal speed), model chosen on the training split (smallest estimated tube), eps
calibrated per task (task-conditional, episode-level joint conformal) on the task's calibration
seeds.  Test: seeds 2000-2099 at nominal speed (tag sb) and with every Eddie commanded faster,
slew bound x {1.5, 2, 3, 4} (tag gate, labels bistl_rho<s>).  For every episode and tolerance mu
the gate (eddies.cert.gate) is replayed on the logged trajectory; the counterfactual outcome of
the full, ungated execution is known from the same log.
Writes $EDDIES_RESULTS/gate/<name>.json.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from eddies.common.paths import ROOT
from eddies.common.config import load as load_config
from eddies.common.config import blocks, mu_list, mu_str, parse_mu, robots, seeds, span
from eddies.cert.data import load_split
from eddies.cert.gate import evaluate_episode
from eddies.cert.ident import fit
from eddies.cert.experiments.certify_task import BLOCKS, CALIB, MIN_STD, TRAIN, family, wilson_upper
from eddies.cert.tube import conformal_quantile, joint_scores, make_tube
from eddies.envs.skillbook import primitive_of

CFG = load_config("experiments/gate_fixed")
TASKS = list(CFG.franka_tasks)
TEST = seeds(CFG.splits.gate_test)
SCALES = [float(s) for s in CFG.gate.speed_scales]
IDENT = CFG.identification
KAPPA_Q = float(CFG.conformal.kappa_quantile)


def relabel(eps, pool):
    for ep in eps:
        ep["token"] = ep["skill"]
        if pool == "primitive":
            ep["skill"] = np.array([primitive_of(s) for s in ep["skill"]], dtype=object)
    return eps


def build_certificates(train, J, W, scale, pos, rho0):
    """Per label: own fit or shrunk toward the best-conditioned reference, chosen on TRAIN."""
    labels = [s for s in dict.fromkeys(x for ep in train for x in ep["skill"]) if s]

    def est(t):
        return t.slew_radius(rho0, float(np.nanquantile(t.residual_scores(train), KAPPA_Q)))
    own, n = {}, {}
    for s in labels:
        t = make_tube(train, s, J, W, scale, pos)
        if t is not None and t.cg is not None:
            n[s] = (t.n_train, est(t))
    own = {s: r for s, (k, r) in n.items() if k >= IDENT.reference.min_transitions}
    if not own:     # few training episodes (another engine, the robot): the best-populated half of the labels
        top = max(k for k, _ in n.values())
        own = {s: r for s, (k, r) in n.items() if k >= top * IDENT.reference.fallback_fraction}
        print(f"build_certificates: no label has {IDENT.reference.min_transitions} transitions (most {top}); reference among {sorted(own)}", flush=True)
    ref = min(own, key=own.get)
    prior = fit(train, ref, np.asarray(J), int(IDENT.order), W).info["Theta"]
    tubes, kappa = {}, {}
    for s in labels:
        best = None
        for lam in IDENT.shrink_grid:
            t = make_tube(train, s, J, W, scale, pos, prior=prior, shrink=lam)
            if t is None or t.cg is None:
                continue
            t.shrink = lam                  # recorded for reporting only
            r = est(t)
            if best is None or r < best[0]:
                best = (r, t)
        if best:
            tubes[s] = best[1]
            kappa[s] = float(np.nanquantile(best[1].residual_scores(train), KAPPA_Q))
    return tubes, kappa, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=("eddy", "primitive"), default="eddy")
    ap.add_argument("--mu", default=mu_str(CFG.mu_cm), help="tolerances [cm]")
    ap.add_argument("--alpha", type=float, default=CFG.conformal.alpha)
    ap.add_argument("--recal", action="store_true",
                    help="calibrate eps at each commanded speed on the test seeds below splits.recal_split (models unchanged), test on the rest")
    ap.add_argument("--name", default=None)
    ap.add_argument("--tasks", default=",".join(TASKS),
                    help="comma list; certificates are fitted on these tasks only (default: the four Franka tasks)")
    a = ap.parse_args()
    mus = [float(x) / 100 for x in a.mu.split(",")]
    from eddies.envs.tasks import make_task
    tasks = a.tasks.split(",")
    train = [ep for t in tasks for ep in relabel(load_split("sb", "nominal", t, TRAIN), a.pool)]
    emb = train[0]["manifest"]["embedding"]
    W = np.asarray(emb["metric_diag"], float)
    scale = (np.asarray(emb["phys_upper"]) - np.asarray(emb["phys_lower"])) / \
            (np.asarray(emb["norm_upper"]) - np.asarray(emb["norm_lower"]))
    J0, pos = BLOCKS["franka"]
    Z = np.concatenate([ep["z"] for ep in train])
    J = [j for j in J0 if Z[:, j].std() > MIN_STD]
    rho0 = float(train[0]["manifest"]["method_info"]["rho_step"])
    tubes, kappa, ref = build_certificates(train, J, W, scale, pos, rho0)
    print(f"certified {len(tubes)} labels (reference model {ref})")

    out = {"pool": a.pool, "alpha": a.alpha, "mu_m": mus, "scales": SCALES, "tasks": {}}
    for task in tasks:
        cal = relabel(load_split("sb", "nominal", task, CALIB), a.pool)
        q = conformal_quantile(joint_scores(tubes, kappa, cal), a.alpha)
        eps = {s: q * kappa[s] for s in tubes}
        plan = make_task(task, 2000).nominal_plan()
        if a.pool == "primitive":
            plan = [primitive_of(x) for x in plan]
        res = {"n_calib": len(cal), "q": q, "plan_ultimate_tube_m": {}, "rows": [], "q_by_scale": {}}
        for sc in SCALES:
            test = (load_split("sb", "nominal", task, TEST) if sc == 1.0 else
                    load_split("gate", "nominal", task, TEST, method=f"bistl_rho{sc:g}"))
            test = relabel(test, a.pool)
            if not test:
                continue
            if a.recal:                       # calibrate at the operating speed; test on the other seeds
                ctest = [ep for ep in test if ep["seed"] < CFG.splits.recal_split]
                test = [ep for ep in test if ep["seed"] >= CFG.splits.recal_split]
                if sc != 1.0:
                    qs = conformal_quantile(joint_scores(tubes, kappa, ctest), a.alpha)
                    eps = {s: qs * kappa[s] for s in tubes}
                else:
                    eps = {s: q * kappa[s] for s in tubes}
                res["q_by_scale"][f"{sc:g}"] = float(eps[next(iter(tubes))] / kappa[next(iter(tubes))])
            rho = float(test[0]["manifest"]["method_info"]["rho_step"])
            res["plan_ultimate_tube_m"][f"{sc:g}"] = {s: tubes[s].slew_radius(rho, eps[s]) for s in dict.fromkeys(plan) if s in tubes}
            fail = np.array([not ep["manifest"]["success"] or ep["manifest"]["bad_steps"] > 0 for ep in test])
            bad = np.array([ep["manifest"]["bad_steps"] > 0 for ep in test])
            for mu in mus:
                D, T = zip(*[evaluate_episode(ep, plan, tubes, eps, mu, rho) for ep in test])
                acc = np.array([d.accepted for d in D])
                comp = np.array([d.completed for d in D])
                exceed = np.array([t["exceeds_mu"] for t in T])
                causes = defaultdict(int)
                for d in D:
                    causes[d.cause] += 1
                n = len(test)
                stopped = ~comp
                res["rows"].append({
                    "scale": sc, "rho": rho, "mu_m": mu, "n": n,
                    "ungated_success": float(1 - fail.mean()), "ungated_bad": int(bad.sum()),
                    "ungated_exceed_mu": int(exceed.sum()),
                    "accepted": int(acc.sum()), "completed": int(comp.sum()), "causes": dict(causes),
                    "completed_exceed_mu": int(sum(d.exceeded_executed for d in D if d.completed)),
                    "completed_failed": int((comp & fail).sum()), "completed_bad": int((comp & bad).sum()),
                    "stopped_would_exceed": int((stopped & exceed).sum()),
                    "stopped_would_fail": int((stopped & fail).sum()),
                    "exceed_caught": int((exceed & stopped).sum()), "exceed_total": int(exceed.sum()),
                    "fail_caught": int((fail & stopped).sum()), "fail_total": int(fail.sum()),
                })
        out["tasks"][task] = res
    name = a.name or f"gate_{a.pool}"
    path = ROOT / "gate" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))

    print(f"\n{'task':22s} {'x':>4s} {'mu':>5s} {'ok%':>5s} {'acc':>4s} {'comp':>4s} {'cmpExc':>6s} {'cmpFail':>7s} "
          f"{'exc caught':>11s} {'fail caught':>11s}  stop causes")
    for task, r in out["tasks"].items():
        for row in r["rows"]:
            print(f"{task:22s} {row['scale']:4g} {100 * row['mu_m']:5.1f} {100 * row['ungated_success']:5.0f} "
                  f"{row['accepted']:4d} {row['completed']:4d} {row['completed_exceed_mu']:6d} {row['completed_failed']:7d} "
                  f"{row['exceed_caught']:5d}/{row['exceed_total']:<5d} {row['fail_caught']:5d}/{row['fail_total']:<5d}  "
                  f"{ {k: v for k, v in row['causes'].items() if k != 'completed'} }")


if __name__ == "__main__":
    main()
