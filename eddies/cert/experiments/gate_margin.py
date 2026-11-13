#!/usr/bin/env python3
"""Gate vs. threshold monitor at intermediate tolerances (offline replay, tag sb, nominal, 10 tasks).

    python -m eddies.cert.experiments.gate_margin [--tasks a,b] [--mu-a 2,2.5,3,3.5,4,4.5,5] [--workers 10] [--smoke]

Per task, on its 200 logged test episodes, with the certificates of its robot (all its tasks' training
episodes) and the task's own calibration (as embodiments / gate_adaptive --known):
  (a) self-calibrating gate from the task's own calibration: gate_adaptive.run(known=True), unchanged,
      one run per mu in --mu-a (its state depends on mu)
  (b) fixed-calibration gate (gate.evaluate_episode: plan check, hand-over check, residual monitor)
  (c) the self-calibrating gate's one-step check with the fixed calibration: models, kappa and q fixed
      at the conformal quantile (no refits, no ACI); variant 'w' keeps the within-episode raise of the
      envelope to the observed residual score (as gate_adaptive), variant 'f' keeps eps fixed
  plus per step the measured error ||S eta_k|| and ||S (eta_k - eta_{k-1})|| (same model) for the
  threshold monitors, and the truth error of gate_adaptive (reference certificates, all tasks' training).
(b) and (c) depend on mu only through a threshold, so they are stored as per-step stop values
v_k (stop at the first k with v_k > mu; a step's value uses only data up to step k) and evaluated for
any mu offline (eddies.cert.tables.gate_margin, whose --out writes the committed summary.json).
Writes $EDDIES_RESULTS/gate_margin/<task>.npz and <task>_a.json (also in the dataset, derived/gate_margin/).
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np

from eddies.common.paths import ROOT
from eddies.common.config import load as load_config
from eddies.common.config import blocks, mu_list, mu_str, parse_mu, robots, seeds, span
from eddies.cert.gate import equilibrium, evaluate_episode, handover_bound, ultimate_tube
from eddies.cert.experiments.gate_adaptive import eta_of, run, safe_next
from eddies.cert.experiments.table1_eddies import embedding
from eddies.cert.experiments.gate_fixed import build_certificates
from eddies.cert.experiments.transfer import certificates
from eddies.cert.tube import conformal_quantile, joint_scores, make_tube, ordered_segments
from eddies.envs.skillbook import primitive_of

OUT = ROOT / "gate_margin"
CFG = load_config("experiments/gate_margin")
GROUP_OF = {t: r for r, c in robots(CFG).items() for t in c.tasks}
KAPPA_Q = float(CFG.conformal.kappa_quantile)


def load_group(g, smoke):
    from eddies.cert.experiments.embodiments import GROUPS, load_task
    tasks = GROUPS[g]
    data = {t: load_task(t) for t in tasks}
    w = min(ep["z"].shape[1] for d in data.values() for sp in d.values() for ep in sp)
    for d in data.values():
        for sp in d.values():
            for ep in sp:
                ep["z"], ep["e"] = ep["z"][:, :w], ep["e"][:, :w]
    if smoke:
        data = {t: {k: v[:CFG.smoke.per_split] for k, v in d.items()} for t, d in data.items()}
    return tasks, data


def arrays(H, data, tasks, alpha):
    """Per-step stop values of (b), (c), truth and monitor inputs on the test episodes of H."""
    train = [ep for t in tasks for ep in data[t]["train"]]
    E, kE, P, kP, rho = certificates(train)                     # = gate_adaptive known=True priors
    J, W, scale, pos, _ = embedding("eddie", train)
    glob = make_tube(train, None, J, W, scale, pos)
    kG = float(np.nanquantile(glob.residual_scores(train), KAPPA_Q))
    q0 = conformal_quantile(joint_scores(E, kE, data[H]["calib"]), alpha)
    Jr, Wr, sr_, pr, _ = embedding("eddie", train)               # truth, exactly as gate_adaptive
    REF, _, _ = build_certificates(train, Jr, Wr, sr_, pr, rho)
    eps = {s: q0 * kE[s] for s in E}                              # fixed gate (embodiments.task_eval)
    from eddies.cert.experiments.embodiments import nominal_plan
    plan = nominal_plan(H)
    U = max((ultimate_tube(E[s], rho, eps[s]) if s in E else np.inf) for s in plan)

    def model(s):                                                 # gate_adaptive's initial prior choice
        if s in E:
            return E[s], kE[s]
        p = primitive_of(s)
        return (P[p], kP[p]) if p in P else (glob, kG)

    test = data[H]["test"]
    keys = ("err", "meas", "vel", "b", "cw", "cf")
    cols = {k: [] for k in keys}
    lens, seeds = [], []
    for ep in test:
        z, e = ep["z"], ep["e"]
        n = len(z)
        A = {k: np.zeros(n) for k in keys}
        A["b"][:] = -np.inf
        A["cw"][:] = -np.inf
        A["cf"][:] = -np.inf
        # truth error (gate_adaptive: reference certificate of the running Eddie)
        for s, k0, k1 in ordered_segments(ep["skill"]):
            ref = REF.get(s) if s else None
            if ref is not None:
                for k in range(k0, k1 + 1):
                    A["err"][k] = float(np.linalg.norm(ref.S @ eta_of(ref, z[k], e[k])))
        # (b) fixed-calibration gate, as gate.evaluate_episode (plan check: U, stored once)
        stopped_b = False
        for s, k0, k1 in ordered_segments(ep["skill"]):
            if s is None or stopped_b:
                continue
            t = E.get(s)
            if t is None:
                A["b"][k0] = np.inf
                stopped_b = True
                continue
            A["b"][k0] = handover_bound(t, rho, eps[s], z[k0, t.J] - equilibrium(t, e[k0]))
            for k in range(k0, k1):
                w_ = z[k + 1, t.J] - (t.A @ z[k, t.J] + t.B @ e[k] + t.d)
                if float(np.linalg.norm(t.Wz * w_)) > eps[s] * (1 + 1e-9):
                    A["b"][k + 1] = np.inf
                    stopped_b = True
                    break
        # (c) one-step check of gate_adaptive with fixed models / kappa / q; monitors' measured error
        segs = [(s, k0, k1) for s, k0, k1 in ordered_segments(ep["skill"]) if s is not None]
        q_ep = max(q0, 0.0)
        for n_seg, (s, k0, k1) in enumerate(segs):
            t, kap = model(s)
            nxt = segs[n_seg + 1] if n_seg + 1 < len(segs) and segs[n_seg + 1][1] == k1 + 1 else None
            t_hand = model(nxt[0])[0] if nxt else t
            for k in range(k0, k1 + 1):
                eta = eta_of(t, z[k], e[k])
                m = float(np.linalg.norm(t.S @ eta))
                A["meas"][k] = m
                if k > 0:
                    A["vel"][k] = float(np.linalg.norm(t.S @ (eta - eta_of(t, z[k - 1], e[k - 1]))))
                has_next = k < k1 or nxt is not None
                pw = pf = -np.inf
                if has_next:
                    tn = t if k < k1 else t_hand
                    pw = safe_next(t, tn, z[k], e[k], e[k + 1], q_ep * kap, 0)
                    pf = safe_next(t, tn, z[k], e[k], e[k + 1], q0 * kap, 0)
                    w_ = z[k + 1, t.J] - (t.A @ z[k, t.J] + t.B @ e[k] + t.d)
                    q_ep = max(q_ep, float(np.linalg.norm(t.Wz * w_)) / kap)
                A["cw"][k], A["cf"][k] = max(m, pw), max(m, pf)
        for k_ in keys:
            cols[k_].append(A[k_])
        lens.append(n)
        seeds.append(int(ep["seed"]))
    # cross-check (b) against gate.evaluate_episode at a few tolerances
    check = {}
    for mu in (0.03, 0.05, 0.075, 0.1, 0.15):
        comp = sum(evaluate_episode(ep, plan, E, eps, mu, rho)[0].completed for ep in test)
        comp_arr = 0 if U > mu else sum(bool(np.all(b <= mu)) for b in cols["b"])
        check[str(mu)] = [int(comp), int(comp_arr)]
    print(H, "q0", q0, "U_plan", U, "fixed-gate check [evaluate_episode, arrays]", check, flush=True)
    return {"q0": q0, "U_plan": U, "rho": rho, "check_b": check, "lens": np.array(lens), "seeds": np.array(seeds),
            **{k: np.concatenate(v) for k, v in cols.items()}}


def one(args):
    H, mus_a, alpha, smoke, do_a = args
    g = GROUP_OF[H]
    tasks, data = load_group(g, smoke)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if smoke else ""
    r = arrays(H, data, tasks, alpha)
    meta = {k: r.pop(k) for k in ("q0", "U_plan", "rho", "check_b")}
    np.savez_compressed(OUT / f"{H}{tag}.npz", **r, meta=json.dumps(meta, default=float))
    if do_a:
        res = run(H, data, mus_a, alpha, CFG.adaptive_gate.eta, CFG.smoke.n_test if smoke else CFG.n_test, CFG.adaptive_gate.lookahead, True, tasks)
        (OUT / f"{H}{tag}_a.json").write_text(json.dumps(res, indent=0, default=float))
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(GROUP_OF))
    ap.add_argument("--mu-a", default=mu_str(CFG.mu_cm), help="tolerances [cm] for the self-calibrating gate")
    ap.add_argument("--alpha", type=float, default=CFG.conformal.alpha)
    ap.add_argument("--workers", type=int, default=CFG.workers)
    ap.add_argument("--no-a", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="smoke.per_split episodes per split")
    a = ap.parse_args()
    mus = [float(x) / 100 for x in a.mu_a.split(",")]
    jobs = [(t, mus, a.alpha, a.smoke, not a.no_a) for t in a.tasks.split(",")]
    with Pool(min(a.workers, len(jobs))) as pool:
        for h in pool.imap_unordered(one, jobs):
            print("done", h, flush=True)


if __name__ == "__main__":
    main()
