#!/usr/bin/env python3
"""Eddies on several embodiments (MuJoCo, nominal): the ten manipulation tasks of the paper.

    python -m eddies.cert.experiments.embodiments [--mu 2,3,5,7.5,10,15] [--alpha 0.1]

Embodiments and tasks; every task is composed only from its embodiment's primitive set (skillbook
tasks, tag sb, Eddie names logged directly):
  franka   skillbook/primitives.py: staged insertion (FMB), peg insertion, cube stack, drawer
  g1       skillbook/g1.py (G1 + Dex3): peg insertion, drawer
  leap     skillbook/hand.py (UR5e + LEAP hand): wheel, peg tightening
  aero     skillbook/hand.py (UR5e + Aero hand): the same two tasks
(configs/robots/<robot>.yaml lists the tasks of each embodiment)
Per embodiment, certificates are fitted on the training episodes of all its tasks (build_certificates,
as gate_fixed), calibrated per task (joint episode-level quantile) and tested on held-out seeds:
violations, ultimate tube radius, and the certify-or-stop gate with the a-priori (nominal) plan.

The token carries the embodiment's dynamics:
  shared   every Eddie certified on more than one embodiment (the peg Eddies on Franka and G1; every
           LEAP / Aero Eddie): contraction rate, n_train and ultimate tube on each embodiment
  cross    LEAP <-> Aero (same arm, tasks, Eddies and embedding; only the hand differs): the other
           hand's certificate applied unchanged (violations, gate), and its models re-calibrated on
           this hand (models transfer, envelopes need this hand's data)
Writes $EDDIES_RESULTS/embodiments/embodiments.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from eddies.common.paths import ROOT
from eddies.common.config import load as load_config
from eddies.common.config import blocks, mu_list, mu_str, parse_mu, robots, seeds, span
from eddies.cert.data import load_split
from eddies.cert.gate import ultimate_tube
from eddies.cert.experiments.table1_eddies import embedding, gate_rows, radius_stats
from eddies.cert.experiments.certify_task import evaluate, summ
from eddies.cert.experiments.gate_fixed import TASKS as SB_TASKS
from eddies.cert.experiments.gate_fixed import build_certificates
from eddies.cert.tube import conformal_quantile, joint_scores

OUT = ROOT / "embodiments"
CFG = load_config("experiments/embodiments")
TRAIN, CALIB = seeds(CFG.splits.train), seeds(CFG.splits.calib)
TEST = seeds(CFG.splits.test)
GROUPS = {r: list(c.tasks) for r, c in robots(CFG).items()}     # every task is composed from its robot's primitive set
assert GROUPS["franka"] == list(SB_TASKS)


def load_task(task):
    return {split: load_split("sb", "nominal", task, seeds) for split, seeds in
            (("train", TRAIN), ("calib", CALIB), ("test", TEST))}


def nominal_plan(task):
    from eddies.envs.tasks import make_task
    return list(dict.fromkeys(make_task(task, 2000).nominal_plan()))


def certify(train):
    J, W, scale, pos, rho = embedding("eddie", train)
    tubes, kappa, ref = build_certificates(train, J, W, scale, pos, rho)
    return {"tubes": tubes, "kappa": kappa, "ref": ref, "rho": rho, "J": [int(j) for j in J]}


def task_eval(cert, task, d, mus, alpha, calib=None):
    tubes, kappa, rho = cert["tubes"], cert["kappa"], cert["rho"]
    q = conformal_quantile(joint_scores(tubes, kappa, calib if calib is not None else d["calib"]), alpha)
    eps = {s: q * kappa[s] for s in tubes}
    plan = nominal_plan(task)
    test = d["test"]
    tube_cm = {s: (100 * ultimate_tube(tubes[s], rho, eps[s]) if s in tubes else None) for s in plan}
    return {"q": q, "n_test": len(test), "eddies": len(plan), "plan_certified": sum(s in tubes for s in plan),
            "plan_tube_cm": tube_cm,
            "violation": summ(evaluate(tubes, eps, test, rho), "D_viol"),
            "radius": radius_stats(tubes, eps, rho, test), "gate": gate_rows(test, tubes, eps, rho, mus, plan)}, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", default=mu_str(CFG.mu_cm))
    ap.add_argument("--alpha", type=float, default=CFG.conformal.alpha)
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--name", default="embodiments", help="output file name under results/embodiments/")
    a = ap.parse_args()
    mus = [float(x) / 100 for x in a.mu.split(",")]
    groups = a.groups.split(",")
    data = {t: load_task(t) for g in groups for t in GROUPS[g]}
    for t, d in data.items():
        print(t, {k: len(v) for k, v in d.items()}, flush=True)
    # one command layout per embodiment: a task may append task-specific extra dims after the shared
    # ones (the G1 microwave's hook/fist 'mode', constant within every Eddie but the posture change);
    # certificates use only the pose block (BLOCKS), so the extra dims are dropped, not certified
    for g in groups:
        eps_g = [ep for t in GROUPS[g] for sp in data[t].values() for ep in sp]
        w = min((ep["z"].shape[1] for ep in eps_g), default=0)
        if any(ep["z"].shape[1] != w for ep in eps_g):
            print(g, "command width trimmed to", w, flush=True)
            for ep in eps_g:
                ep["z"], ep["e"] = ep["z"][:, :w], ep["e"][:, :w]
    res = {"alpha": a.alpha, "mu_m": mus, "groups": {}, "shared": {}, "cross": {}}
    certs, eps_of = {}, {}
    for g in groups:
        train = [ep for t in GROUPS[g] for ep in data[t]["train"]]
        if not train:
            continue
        certs[g] = c = certify(train)
        res["groups"][g] = {"tasks": {}, "certified": len(c["tubes"]), "rho": c["rho"], "J": c["J"], "reference": c["ref"]}
        for t in GROUPS[g]:
            if not data[t]["test"]:
                continue
            r, eps = task_eval(c, t, data[t], mus, a.alpha)
            res["groups"][g]["tasks"][t] = r
            eps_of[(g, t)] = eps
            print(g, t, "viol", r["violation"]["violations"], "/", r["n_test"], "r50/r90",
                  r["radius"]["median_cm"], r["radius"]["p90_cm"], "gate", [x["completed"] for x in r["gate"]],
                  "beyond mu", [x["completed_exceed_mu"] for x in r["gate"]], "stops@max mu", r["gate"][-1]["causes"],
                  "widest", sorted(((v, k) for k, v in r["plan_tube_cm"].items() if v is not None), reverse=True)[:3],
                  "uncertified", [k for k, v in r["plan_tube_cm"].items() if v is None], flush=True)
    # the same Eddie on several embodiments
    names = {}
    for g, c in certs.items():
        for s in c["tubes"]:
            names.setdefault(s, []).append(g)
    for s, gs in sorted(names.items()):
        if len(gs) < 2:
            continue
        row = {}
        for g in gs:
            t = certs[g]["tubes"][s]
            ev = [eps_of[(g, task)][s] for task in GROUPS[g] if (g, task) in eps_of and s in eps_of[(g, task)]]
            e = float(np.median(ev)) if ev else np.nan
            row[g] = {"spectral_radius": t.sr, "n_train": t.n_train, "eps": e,
                      "tube_cm": 100 * ultimate_tube(t, certs[g]["rho"], e) if np.isfinite(e) else None}
        res["shared"][s] = row
        print("shared", s, {g: (round(v["spectral_radius"], 3), v["tube_cm"] and round(v["tube_cm"], 1)) for g, v in row.items()})
    # LEAP <-> Aero: certificate of the other hand, unchanged and re-calibrated on this hand
    for src, dst in (("leap", "aero"), ("aero", "leap")):
        if src not in certs or dst not in certs:
            continue
        for t_dst in GROUPS[dst]:
            t_src = t_dst.replace(dst, src, 1)
            if (src, t_src) not in eps_of:
                continue
            c = certs[src]
            eps = eps_of[(src, t_src)]
            test, plan = data[t_dst]["test"], nominal_plan(t_dst)
            unchanged = {"violation": summ(evaluate(c["tubes"], eps, test, c["rho"]), "D_viol"),
                         "gate": gate_rows(test, c["tubes"], eps, c["rho"], mus, plan)}
            recal, _ = task_eval(c, t_dst, data[t_dst], mus, a.alpha)
            res["cross"][f"{src}->{t_dst}"] = {"unchanged": unchanged, "recalibrated": recal}
            print("cross", src, "->", t_dst, "unchanged viol", unchanged["violation"]["violations"],
                  "gate", [x["completed"] for x in unchanged["gate"]], "| recal viol", recal["violation"]["violations"],
                  "gate", [x["completed"] for x in recal["gate"]], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.name}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
