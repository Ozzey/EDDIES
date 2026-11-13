#!/usr/bin/env python3
"""Composition from LTLf specifications: execution summary + certify-or-stop replay.

    python -m eddies.cert.experiments.gate_ltl [--mu 3,5,7.5,10,15] [--alpha 0.1]

Episodes: tag ``ltl`` (scripts/reproduce.sh collect-ltl), seeds 2000-2099, nominal and perturbed:
the four hand-written token tasks, the same goals composed by the Streamer from a formula
(``*_ltl``, label bistl), and new goals in the same scenes (label = spec name).
Certificates: the per-Eddie models and per-scene envelopes of gate_fixed (tag sb, nominal
training / calibration seeds of the HAND-WRITTEN tasks; nothing is refitted for composed or
new specs).  The gate's plan check uses the plan the Streamer synthesised for the spec.
Writes $EDDIES_RESULTS/gate/ltl.json.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from eddies.common.paths import ROOT
from eddies.common.config import load as load_config
from eddies.common.config import blocks, mu_list, mu_str, parse_mu, robots, seeds, span
from eddies.cert.data import load_split
from eddies.cert.gate import evaluate_episode
from eddies.cert.experiments.certify_task import BLOCKS, CALIB, MIN_STD, TRAIN
from eddies.cert.experiments.gate_fixed import TASKS, TEST, build_certificates
from eddies.cert.tube import conformal_quantile, joint_scores

# (scene, task directory, label, spec or None for the hand-written task): configs/experiments/ltl.yaml
CFG = load_config("experiments/ltl")
RUNS = [tuple(r) for r in CFG.runs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", default=mu_str(CFG.mu_cm))
    ap.add_argument("--alpha", type=float, default=CFG.conformal.alpha)
    ap.add_argument("--tag", default=CFG.tag)
    a = ap.parse_args()
    mus = [float(x) / 100 for x in a.mu.split(",")]
    from eddies.envs.tasks import make_task
    train = [ep for t in TASKS for ep in load_split("sb", "nominal", t, TRAIN)]
    for ep in train:
        ep["token"] = ep["skill"]
    emb = train[0]["manifest"]["embedding"]
    W = np.asarray(emb["metric_diag"], float)
    scale = (np.asarray(emb["phys_upper"]) - np.asarray(emb["phys_lower"])) / \
            (np.asarray(emb["norm_upper"]) - np.asarray(emb["norm_lower"]))
    J0, pos = BLOCKS["franka"]
    Z = np.concatenate([ep["z"] for ep in train])
    J = [j for j in J0 if Z[:, j].std() > MIN_STD]
    rho = float(train[0]["manifest"]["method_info"]["rho_step"])
    tubes, kappa, _ = build_certificates(train, J, W, scale, pos, rho)
    eps_by_scene = {}
    for scene in TASKS:
        q = conformal_quantile(joint_scores(tubes, kappa, load_split("sb", "nominal", scene, CALIB)), a.alpha)
        eps_by_scene[scene] = {s: q * kappa[s] for s in tubes}

    out = {"alpha": a.alpha, "mu_m": mus, "runs": []}
    hdr = f"{'run':44s} {'cond':9s} {'n':>4s} {'succ':>5s} {'cov.v':>5s} {'bad':>4s}  plan"
    print(hdr)
    for scene, tdir, label, spec in RUNS:
        t = make_task(tdir, 2000, **({"spec": spec} if spec else {}))
        plan = t.nominal_plan()
        for cond in ("nominal", "perturbed"):
            eps = load_split(a.tag, cond, tdir, TEST, method=label)
            if not eps:
                continue
            man = [ep["manifest"] for ep in eps]
            sup = [m.get("method_info", {}).get("supervisor", {}) for m in man]
            succ = np.array([m["success"] for m in man])
            cv = np.array([s.get("cover_violations", 0) or 0 for s in sup])
            bad = np.array([m.get("bad_steps", 0) > 0 for m in man])
            seqs = Counter(" ".join(d["skill"] for d in s.get("decisions", [])) == " ".join(plan) for s in sup)
            row = {"scene": scene, "task": tdir, "label": label, "spec": spec, "condition": cond, "plan": plan,
                   "n": len(eps), "success": float(succ.mean()), "episodes_with_cover_violation": int((cv > 0).sum()),
                   "cover_violations": int(cv.sum()), "bad": int(bad.sum()),
                   "executed_nominal_plan": int(seqs.get(True, 0)),
                   "extensions": int(sum(s.get("graph_extensions", 0) or 0 for s in sup)),
                   "stop_reasons": dict(Counter(m.get("stop_reason") for m in man)), "gate": []}
            if cond == "nominal":
                eps_s = eps_by_scene[scene]
                fail = ~succ | bad
                for mu in mus:
                    D, T = zip(*[evaluate_episode(ep, plan, tubes, eps_s, mu, rho) for ep in eps])
                    comp = np.array([d.completed for d in D])
                    exceed = np.array([x["exceeds_mu"] for x in T])
                    row["gate"].append({"mu_m": mu, "accepted": int(sum(d.accepted for d in D)),
                                        "completed": int(comp.sum()),
                                        "completed_exceed_mu": int(sum(d.exceeded_executed for d in D if d.completed)),
                                        "exceed_caught": int((exceed & ~comp).sum()), "exceed_total": int(exceed.sum()),
                                        "completed_failed": int((comp & fail).sum()),
                                        "causes": dict(Counter(d.cause for d in D))})
            out["runs"].append(row)
            name = (label if label != "bistl" else ("composed" if spec else "hand-written")) + f" [{scene}]"
            print(f"{name:44s} {cond:9s} {row['n']:4d} {100 * row['success']:5.1f} {row['cover_violations']:5d} "
                  f"{row['bad']:4d}  {len(plan)} Eddies; executed exactly the plan in {row['executed_nominal_plan']}")
            for g in row["gate"]:
                print(f"{'':44s}   mu {100 * g['mu_m']:4.1f} cm: accepted {g['accepted']:3d} completed {g['completed']:3d} "
                      f"completed>mu {g['completed_exceed_mu']}  >mu caught {g['exceed_caught']}/{g['exceed_total']}  "
                      f"{ {k: v for k, v in g['causes'].items() if k != 'completed'} }")
    path = ROOT / "gate" / "ltl.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
