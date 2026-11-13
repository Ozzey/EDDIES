#!/usr/bin/env python3
"""Cube stacking in MuJoCo, in Isaac Sim (PhysX) and on the real Franka: the rows of the real-robot table.

    python -m eddies.cert.experiments.stack_compare [--mu 2,3,4,5,7.5,10] [--real-train 0-49 --real-calib 50-99
                                           --real-test 100-149]

Every row replays both gates on a stream of logged test episodes, started from a prior certificate, as they
would run on the robot: the fixed-calibration gate (gate_fixed: plan check, hand-over and residual checks with the
starting certificate unchanged) and the self-calibrating gate (gate_adaptive: one-step check, models and envelope
level updated online).  Rows:
  mujoco       MuJoCo test episodes (seeds 2000-2099), started from the MuJoCo certificate (own calibration)
  physx_sim    PhysX test episodes (2000-2049), started from the MuJoCo certificate unchanged (simulation prior)
  physx_own    PhysX test episodes, started from a certificate re-identified on PhysX: the MuJoCo training
               episodes plus the PhysX training episodes (1000-1049), calibrated on PhysX (1200-1249)
  real_sim     real-robot test episodes, started from the MuJoCo certificate     (when $EDDIES_RESULTS_REAL has them)
  real_own     real-robot test episodes, started from a certificate re-identified on the robot
The truth (whether an execution would leave mu, and when) uses reference certificates fitted on the test
system's own episodes, for both gates: MuJoCo, the training episodes of the four Franka tasks (as gate_adaptive);
PhysX, the training episodes of the three tasks run on PhysX (staged insertion, peg insertion, stacking); the
robot, its stacking training and calibration episodes.  Per row: episodes, success, viol. (test episodes leaving a
tube of the starting certificate), tube p90 (90th percentile of the tube radius over the executed test steps, cm,
as in Table 1 and physics; also over the planned Eddies), finished / stopped per mu, and
caught: runs stopped or refused before leaving mu, of those that would, on the faster-speed runs (tag gate,
2-4x slew bound; seeds as the test stream).
Episode roots: $EDDIES_EPISODES (MuJoCo), $EDDIES_RESULTS_ISAAC (PhysX), $EDDIES_RESULTS_REAL
(real robot, same layout: episodes/<tag>/nominal/franka_cube_stack/bistl/seed_XXXX/).
--rows picks rows (default all; e.g. one row per job); each row is written to $EDDIES_RESULTS/compare/stack_<row>.json,
and the table printed at the end merges every row file there.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import eddies.cert.data as bdata
import eddies.cert.experiments.gate_adaptive as RA
from eddies.common.paths import EPISODES, EPISODES_ISAAC, EPISODES_REAL, ROOT
from eddies.common.config import load as load_config
from eddies.common.config import blocks, mu_list, mu_str, parse_mu, robots, seeds, span
from eddies.cert.gate import evaluate_episode, ultimate_tube
from eddies.cert.experiments.table1_eddies import embedding, radius_stats
from eddies.cert.experiments.certify_task import evaluate, summ
from eddies.cert.experiments.gate_fixed import TASKS, build_certificates
from eddies.cert.experiments.transfer import nominal_plan
from eddies.cert.tube import conformal_quantile, joint_scores, ordered_segments

CFG = load_config("experiments/stack_compare")
TASK = str(CFG.task)
SPEEDS = tuple(CFG.speeds)        # default faster-speed runs for "caught" (--speeds)
SP, SI, SR = CFG.splits, CFG.splits.isaac, CFG.splits.real
ISAAC = EPISODES_ISAAC
REAL = EPISODES_REAL if EPISODES_REAL.is_dir() else None


def seeds(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1)) if part else []
    return out


def load(root, tag, task, ss, method="bistl"):
    old = bdata.EPISODES
    bdata.EPISODES = root
    try:
        return bdata.load_split(tag, "nominal", task, ss, method=method)
    finally:
        bdata.EPISODES = old


# gate_adaptive refits its prior and reference certificates on every call; they repeat here, so cache them by
# episode set.
def _key(eps):
    return tuple((ep["manifest"].get("task"), ep["seed"], ep["manifest"].get("simulator")) for ep in eps)


_cert_cache, _ref_cache = {}, {}
_orig_certificates = RA.certificates


def _cached_certificates(train):
    k = _key(train)
    if k not in _cert_cache:
        _cert_cache[k] = _orig_certificates(train)
    return _cert_cache[k]


def _cached_build(train, J, W, scale, pos, rho):
    k = (_key(train), tuple(int(j) for j in J), float(rho))
    if k not in _ref_cache:
        _ref_cache[k] = build_certificates(train, J, W, scale, pos, rho)
    return _ref_cache[k]


RA.certificates = _cached_certificates
RA.build_certificates = _cached_build


def reference(ref_train):
    """Truth models: certificates fitted on the test system's own training episodes (as gate_adaptive builds them)."""
    J, W, scale, pos, rho = embedding("eddie", ref_train)
    REF, _, ref = _cached_build(ref_train, J, W, scale, pos, rho)
    print(f"reference: {len(ref_train)} episodes, J={list(map(int, J))}, {len(REF)} labels, best-conditioned {ref}", flush=True)
    return REF


def ref_error(REF, ep):
    """Tracking error [m] at every step under the reference models (0 where the Eddie has no reference)."""
    z, e = ep["z"], ep["e"]
    err = np.zeros(len(z))
    for s, k0, k1 in ordered_segments(ep["skill"]):
        r = REF.get(s) if s else None
        if r is None:
            continue
        Zs = np.linalg.solve(np.eye(len(r.J)) - r.A, (e[k0:k1 + 1] @ r.B.T + r.d).T).T
        err[k0:k1 + 1] = np.linalg.norm((z[k0:k1 + 1][:, r.J] - Zs) @ r.S.T, axis=1)
    return err


def fixed_gate(E, eps, test, REF, mus):
    """The fixed-calibration gate on logged episodes, judged against the reference truth: caught = refused at the
    plan, or stopped before the reference error first exceeds mu."""
    if not test:
        return {}
    plan = list(dict.fromkeys(nominal_plan(TASK)))
    rho = float(test[0]["manifest"]["method_info"]["rho_step"])
    errs = [ref_error(REF, ep) for ep in test]
    out = {}
    for mu in mus:
        r = {"n": len(test), "accepted": 0, "completed": 0, "stopped": 0, "would_exceed": 0, "caught_before": 0,
             "completed_beyond_mu": 0, "causes": {}}
        for ep, err in zip(test, errs):
            d, _ = evaluate_episode(ep, plan, E, eps, mu, rho)
            would = bool(err.max() > mu)
            r["accepted"] += d.accepted
            r["completed"] += d.completed
            r["stopped"] += not d.completed
            r["would_exceed"] += would
            r["causes"][d.cause] = r["causes"].get(d.cause, 0) + 1
            if d.completed:
                r["completed_beyond_mu"] += would
            elif would and (not d.accepted or err[: d.stop_step + 1].max() <= mu):
                r["caught_before"] += 1
        out[str(mu)] = r
    return out


def starting_certificate(train, calib, alpha):
    """The certificate both gates start from: fixed models, joint quantile on calib."""
    E, kE, _, _, rho = _cached_certificates(train)
    q = conformal_quantile(joint_scores(E, kE, calib), alpha)
    return E, {s: q * kE[s] for s in E}, q, rho


def static_metrics(E, eps, q, rho, test):
    """viol. and tube p90 of the starting certificate."""
    viol = summ(evaluate(E, eps, test, rho), "D_viol")
    plan = [s for s in dict.fromkeys(nominal_plan(TASK)) if s in E]
    tubes = np.array([ultimate_tube(E[s], rho, eps[s]) for s in plan]) * 100
    return {"q": q, "violations": viol["violations"], "n": viol["n"],
            "tube_p90_cm": float(np.quantile(tubes, 0.9)), "tube_max_cm": float(tubes.max()),
            "tube_p90_steps_cm": radius_stats(E, eps, rho, test)["p90_cm"],
            "planned_certified": len(plan), "planned": len(dict.fromkeys(nominal_plan(TASK)))}


def stream(data, test, ref_train, mus, a):
    return RA.run(TASK, data, mus, a.alpha, a.eta, len(test), 0, True, TASKS, ref_train=ref_train, test=test)


def row(name, data, test, speed_tests, ref_train, mus, a):
    E, eps, q, rho = starting_certificate([ep for t in TASKS for ep in data[t]["train"]], data[TASK]["calib"], a.alpha)
    st = static_metrics(E, eps, q, rho, test)
    REF = reference(ref_train if ref_train is not None else [ep for t in TASKS for ep in data[t]["train"]])
    out = {"episodes": len(test), "success": int(sum(bool(ep["manifest"]["success"]) for ep in test)),
           "bad_steps_episodes": int(sum(ep["manifest"].get("bad_steps", 0) > 0 for ep in test)),
           **st, "fixed": fixed_gate(E, eps, test, REF, mus),
           "fixed_speed": {str(sp): fixed_gate(E, eps, tst, REF, mus) for sp, tst in speed_tests.items() if tst},
           "gate": {}, "speed": {}}
    print(name, "fixed", json.dumps({k: (v["completed"], v["stopped"], v["completed_beyond_mu"])
                                     for k, v in out["fixed"].items()}), flush=True)
    nom = stream(data, test, ref_train, mus, a)
    for mu in mus:
        s = nom["rows"][str(mu)]["summary"]
        out["gate"][str(mu)] = {k: s[k] for k in ("completed", "stopped", "would_exceed", "caught_before",
                                                  "false_stops", "completed_beyond_mu", "miscoverage")}
    for sp, tst in speed_tests.items():
        if not tst:
            continue
        r = stream(data, tst, ref_train, mus, a)
        out["speed"][str(sp)] = {str(mu): {k: r["rows"][str(mu)]["summary"][k] for k in
                                           ("n", "completed", "would_exceed", "caught_before", "completed_beyond_mu")}
                                 for mu in mus}
    print(name, json.dumps({k: v for k, v in out.items() if k not in ("gate", "speed", "fixed", "fixed_speed")}),
          flush=True)
    path = ROOT / "compare" / f"stack_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task": TASK, "alpha": a.alpha, "eta": a.eta, "mu_m": mus, "speeds": list(SPEEDS),
                                "row": out}, indent=1, default=float))
    return out


def caught(r, key, mu):
    """Summed over the faster speeds this row has (2-4x in simulation, possibly 1.5-2x on the robot)."""
    c = sum(r[key][s].get(str(mu), {}).get("caught_before", 0) for s in r[key])
    w = sum(r[key][s].get(str(mu), {}).get("would_exceed", 0) for s in r[key])
    return f"{c}/{w}"


def table(mus):
    rows = {}
    for name in ("mujoco", "physx_sim", "physx_own", "real_sim", "real_own"):
        p = ROOT / "compare" / f"stack_{name}.json"
        if p.exists():
            rows[name] = json.loads(p.read_text())["row"]
    for gate, nom, sp in (("fixed-calibration gate", "fixed", "fixed_speed"), ("self-calibrating gate", "gate", "speed")):
        print(f"\n{gate}\n{'row':10s} {'eps':>4s} {'succ':>4s} {'viol':>4s} {'p90':>5s} " +
              " ".join(f"{100 * m:>5g}cm fin/stop/beyond" for m in mus) + "   caught@speeds per mu")
        for name, r in rows.items():
            cells = []
            for m in mus:
                g = r[nom].get(str(m))
                cells.append(f"{g['completed']:>9d}/{g['stopped']:<3d}/{g['completed_beyond_mu']:<5d}" if g else " " * 19)
            print(f"{name:10s} {r['episodes']:4d} {r['success']:4d} {r['violations']:4d} {r['tube_p90_steps_cm']:5.1f} " +
                  " ".join(cells) + "   " + " ".join(caught(r, sp, m) for m in mus))


def main():
    global SPEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", default=mu_str(CFG.mu_cm))
    ap.add_argument("--alpha", type=float, default=CFG.conformal.alpha)
    ap.add_argument("--eta", type=float, default=CFG.adaptive_gate.eta)
    ap.add_argument("--speeds", default=",".join(f"{s:g}" for s in SPEEDS), help="slew-bound multiples of the faster runs used for caught")
    ap.add_argument("--real-train", default=span(SR.train))
    ap.add_argument("--real-calib", default=span(SR.calib))
    ap.add_argument("--real-test", default=span(SR.test))
    ap.add_argument("--rows", default=",".join(CFG.rows))
    ap.add_argument("--table", action="store_true", help="only print the table of the row files already written")
    a = ap.parse_args()
    mus = [float(x) / 100 for x in a.mu.split(",")]
    SPEEDS = tuple(float(x) if "." in x else int(x) for x in a.speeds.split(","))
    rows = set(a.rows.split(","))
    if a.table:
        return table(mus)

    mj = {t: {"train": load(EPISODES, "sb", t, seeds(SP.train)), "calib": load(EPISODES, "sb", t, seeds(SP.calib)),
              "test": load(EPISODES, "sb", t, seeds(SP.gate_test))} for t in TASKS}
    mj_train = [ep for t in TASKS for ep in mj[t]["train"]]
    if "mujoco" in rows:
        mj_speed = {s: load(EPISODES, "gate", TASK, seeds(SP.gate_test), f"bistl_rho{s}") for s in SPEEDS}
        row("mujoco", mj, mj[TASK]["test"], mj_speed, None, mus, a)

    def other_system(prefix, root, tr, ca, te, ref_tasks=()):
        if not rows & {f"{prefix}_sim", f"{prefix}_own"}:
            return
        train, calib, test = load(root, "sb", TASK, tr), load(root, "sb", TASK, ca), load(root, "sb", TASK, te)
        print(prefix, "episodes", len(train), len(calib), len(test), flush=True)
        if not (train and calib and test):
            return
        speed = {s: load(root, "gate", TASK, te, f"bistl_rho{s}") for s in SPEEDS}
        # truth: the system's own dynamics, from as many of its episodes as the gates never replay
        ref = train + [ep for t in ref_tasks for ep in load(root, "sb", t, tr)] if ref_tasks else train + calib
        print(prefix, "reference episodes", len(ref), flush=True)
        if f"{prefix}_sim" in rows:
            row(f"{prefix}_sim", mj, test, speed, ref, mus, a)
        if f"{prefix}_own" in rows:
            own = {t: dict(d) for t, d in mj.items()}
            own[TASK] = {"train": mj[TASK]["train"] + train, "calib": calib, "test": test}
            row(f"{prefix}_own", own, test, speed, ref, mus, a)

    other_system("physx", ISAAC, seeds(SI.train), seeds(SI.calib), seeds(SI.test), ref_tasks=tuple(CFG.physx_ref_tasks))
    if REAL is not None and a.real_test:
        other_system("real", REAL, seeds(a.real_train), seeds(a.real_calib), seeds(a.real_test))
    table(mus)


if __name__ == "__main__":
    main()
