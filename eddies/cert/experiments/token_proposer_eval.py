#!/usr/bin/env python3
"""Closed-loop evaluation of the learned Eddie proposer against the symbolic supervisor, same seeds.

    python -m eddies.cert.experiments.token_proposer_eval --task franka_cube_stack [--tag tokprop] [--gate]

Episodes: $EDDIES_RESULTS/episodes/<tag>/<condition>/<task>/<method>/ (eddies.cert.collect), methods
bistl (symbolic supervisor), bistl_learned (+ _raw, _state, _state_raw; eddies/envs/methods/learned_tokens.py).
Per method x condition: LTLf success, stop reasons, proposals, contract rejections (proposal not a rank-decreasing
contract of the measured product state), agreement with the symbolic choice, supervisor cover / budget violations.
--gate also replays both gates of the paper on the logged episodes, with the task's EXISTING certificate (fitted on
the tag-sb nominal training episodes of the four Franka tasks, calibrated on the task's sb calibration episodes;
nothing refitted on the new logs), exactly as stack_compare's mujoco row: certificate violations (episodes
leaving a tube of the starting certificate), fixed-calibration gate and self-calibrating gate per mu.
Writes $EDDIES_RESULTS/tokprop/eval_<task>.json.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

import eddies.cert.data as bdata
from eddies.common.paths import ROOT, parse_seeds
from eddies.common.config import load as load_config
from eddies.common.config import blocks, mu_list, mu_str, parse_mu, robots, seeds, span

CFG = load_config("experiments/token_proposer")
METHODS = ",".join(CFG.methods)


def closed_loop(eps):
    man = [ep["manifest"] for ep in eps]
    sup = [m.get("method_info", {}).get("supervisor", {}) for m in man]
    prop = [s.get("proposer") for s in sup]
    r = {"n": len(man), "success": int(sum(bool(m["success"]) for m in man)),
         "task_success": int(sum(bool(m["task_success"]) for m in man)),
         "property_success": int(sum(bool(m["property_success"]) for m in man)),
         "bad_episodes": int(sum(m.get("bad_steps", 0) > 0 for m in man)),
         "stop_reasons": dict(Counter(m.get("stop_reason") for m in man)),
         "cover_violations": int(sum(s.get("cover_violations", 0) or 0 for s in sup)),
         "budget_assumption_violations": int(sum(s.get("assumption_violations", 0) or 0 for s in sup)),
         "eddies_executed_mean": float(np.mean([len(m["method_info"].get("skills", [])) for m in man])),
         "completion_time_mean_s": (float(np.mean([m["completion_time_s"] for m in man if m.get("completion_time_s")]))
                                    if any(m.get("completion_time_s") for m in man) else None)}
    if any(p is not None for p in prop):
        keys = ("proposals", "rejected", "agree_symbolic", "masked_choice_not_top", "unvetted_executed",
                "done_proposed", "done_premature", "guard_stops")
        r["proposer"] = {k: int(sum((p or {}).get(k, 0) for p in prop)) for k in keys}
        r["proposer"]["episodes_with_rejection"] = int(sum(((p or {}).get("rejected", 0) > 0) for p in prop))
        rej = Counter()
        for p in prop:
            for e in (p or {}).get("log", []):
                if not e["passes"] and e["rank"] is not None:
                    ok = "<done>" if e["rank"] == 0 else ",".join(e["allowed"])
                    rej[f"{e['cell']}: {e['raw']} (allowed {ok})"] += 1
        r["proposer"]["rejections_by_state"] = dict(rej.most_common(12))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=CFG.tasks[0])
    ap.add_argument("--tag", default=CFG.tag)
    ap.add_argument("--methods", default=METHODS)
    ap.add_argument("--conditions", default="nominal,perturbed")
    ap.add_argument("--seeds", default=span(CFG.splits.gate_test))
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--mu", default=mu_str(CFG.mu_cm))
    ap.add_argument("--alpha", type=float, default=CFG.conformal.alpha)
    ap.add_argument("--eta", type=float, default=CFG.adaptive_gate.eta)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    mus = [float(x) / 100 for x in a.mu.split(",")]
    out = {"task": a.task, "tag": a.tag, "seeds": a.seeds, "mu_m": mus, "alpha": a.alpha, "rows": {}}
    eps_by = {}
    for cond in a.conditions.split(","):
        for m in a.methods.split(","):
            eps = bdata.load_split(a.tag, cond, a.task, seeds, method=m)
            if not eps:
                continue
            eps_by[(m, cond)] = eps
            out["rows"][f"{m}|{cond}"] = {"method": m, "condition": cond, "closed_loop": closed_loop(eps)}
            print(m, cond, json.dumps(out["rows"][f"{m}|{cond}"]["closed_loop"]), flush=True)
    if a.gate:
        import eddies.cert.experiments.stack_compare as RSC
        from eddies.cert.experiments.gate_fixed import TASKS
        RSC.TASK = a.task
        sb = {t: {"train": bdata.load_split("sb", "nominal", t, seeds(CFG.splits.train)),
                  "calib": bdata.load_split("sb", "nominal", t, seeds(CFG.splits.calib)),
                  "test": bdata.load_split("sb", "nominal", t, seeds(CFG.splits.gate_test))} for t in TASKS}
        train = [ep for t in TASKS for ep in sb[t]["train"]]
        E, eps_c, q, rho = RSC.starting_certificate(train, sb[a.task]["calib"], a.alpha)
        REF = RSC.reference(train)
        ga = argparse.Namespace(alpha=a.alpha, eta=a.eta)
        for (m, cond), eps in eps_by.items():
            st = RSC.static_metrics(E, eps_c, q, rho, eps)
            fixed = RSC.fixed_gate(E, eps_c, eps, REF, mus)
            nom = RSC.stream(sb, eps, None, mus, ga)
            row = out["rows"][f"{m}|{cond}"]
            row["certificate"] = st
            row["fixed_gate"] = fixed
            row["adaptive_gate"] = {str(mu): {k: nom["rows"][str(mu)]["summary"][k] for k in
                                              ("completed", "stopped", "would_exceed", "caught_before", "false_stops",
                                               "completed_beyond_mu", "miscoverage", "causes")} for mu in mus}
            print(m, cond, "viol", st["violations"], "/", st["n"],
                  "fixed", {k: (v["completed"], v["stopped"], v["completed_beyond_mu"]) for k, v in fixed.items()},
                  "adaptive", {k: (v["completed"], v["stopped"], v["completed_beyond_mu"]) for k, v in row["adaptive_gate"].items()},
                  flush=True)
    path = ROOT / "tokprop" / f"eval_{a.task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", path)


if __name__ == "__main__":
    main()
