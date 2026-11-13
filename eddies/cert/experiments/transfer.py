#!/usr/bin/env python3
"""Generalisation to an unseen task through composition (leave-one-task-out, MuJoCo skillbook data).

    python -m eddies.cert.experiments.transfer [--mu 5,7.5,10,15] [--alpha 0.1] [--fewshot 5,10,20] [--smoke]

The held-out task H is a NEW COMPOSITION of tokens: some of its Eddies were seen in the other tasks
(same primitive and binding), some are new bindings of known primitives.  Certificates come from the
other three tasks only, and the gate is replayed on H's test episodes with its a-priori (nominal) plan.

Certificates (labelling of H):
  eddie       per-Eddie certificate; a new binding has none (the current zero-shot result)
  hier        per-Eddie certificate where the Eddie was seen, else the certificate of its PRIMITIVE
              (pooled over every binding of that primitive in the training tasks)
Calibration:
  joint0      one joint quantile from the training tasks' calibration episodes (task-level; not
              exchangeable with H -- the zero-shot reference)
  label       per-token calibration composed by a union bound over H's plan: eps_s is the conformal
              quantile at level alpha/m (m = tokens in H's plan) of token s's episode score over every
              training-task calibration episode that runs s.  Needs NO data of H; valid if a token's
              residuals do not depend on the task it is composed into (checked here, not assumed).
  label_max   as label, but the largest per-task quantile over the training tasks running s (robust to
              which task the token was calibrated in)
  recal<n>    hier certificates unchanged, joint quantile on n calibration episodes of H (no refit)
  few<k>      k training episodes of H added to the fit (new bindings identified, shrunk toward the
              reference), joint quantile on 50 calibration episodes of H
In-distribution reference: label calibration on all four tasks (what composing the calibration costs
against the task-level joint quantile).
Writes $EDDIES_RESULTS/transfer/transfer.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from eddies.common.paths import ROOT
from eddies.common.config import load as load_config
from eddies.common.config import blocks, mu_list, mu_str, parse_mu, robots, seeds, span
from eddies.cert.experiments.table1_eddies import _relabel, embedding, gate_rows, load, radius_stats
from eddies.cert.experiments.certify_task import evaluate, summ
from eddies.cert.experiments.gate_fixed import TASKS, build_certificates
from eddies.cert.tube import conformal_quantile, joint_scores, ordered_segments
from eddies.envs.skillbook import primitive_of

CFG = load_config("experiments/transfer")
OUT = ROOT / "transfer"
CAL_H = int(CFG.cal_h)             # calibration episodes of the held-out task


def certificates(train):
    """Per-Eddie and per-primitive certificates from the same training episodes."""
    J, W, scale, pos, rho = embedding("eddie", train)
    E, kE, _ = build_certificates(train, J, W, scale, pos, rho)
    P, kP, _ = build_certificates([prim(ep) for ep in train], J, W, scale, pos, rho)
    return E, kE, P, kP, rho


def prim(ep):
    return _relabel(ep, [primitive_of(s) for s in ep["skill"]])


def labeller(E, P, hier):
    def lab(s):
        if s is None or s in E:
            return s
        p = primitive_of(s)
        return p if hier and p in P else s
    return lambda ep: _relabel(ep, [lab(s) for s in ep["skill"]])


def nominal_plan(task):
    """The nominal Eddie sequence in execution order (repeats kept: the hand-over context of a
    re-used Eddie is its predecessor in this sequence)."""
    from eddies.envs.tasks import make_task
    return list(make_task(task, 2000).nominal_plan())


# ------------------------------------------------------------ hand-over context (ctx mode)
LEVELS = tuple(CFG.levels)     # 1 Eddie@predecessor primitive, 2 primitive@predecessor primitive, 3 primitive


def ctx_labels(ep):
    """Per step: (Eddie, primitive of the previous executed token, or 'start')."""
    out, pred = [None] * len(ep["skill"]), "start"
    for s, k0, k1 in ordered_segments(ep["skill"]):
        if s is None:
            continue
        for k in range(k0, k1 + 1):
            out[k] = (s, pred)
        pred = primitive_of(s)
    return out


def ctx_name(s, pred, level):
    return f"{s}@{pred}" if level == 1 else f"{primitive_of(s)}@{pred}" if level == 2 else primitive_of(s)


def ctx_level(label):
    return 1 if "@" in label and ":" in label else 2 if "@" in label else 3


def ctx_relabel(ep, level):
    return _relabel(ep, [None if c is None else ctx_name(*c, level) for c in ctx_labels(ep)])


def ctx_certificates(train):
    J, W, scale, pos, rho = embedding("eddie", train)
    C, K = {}, {}
    for level in LEVELS:
        T, Kl, _ = build_certificates([ctx_relabel(ep, level) for ep in train], J, W, scale, pos, rho)
        C.update(T)
        K.update(Kl)
    return C, K, rho


def ctx_labeller(C):
    """The most specific certified label: Eddie in its hand-over context, else its primitive in
    that context, else its primitive."""
    def lab(ep):
        out = []
        for c in ctx_labels(ep):
            if c is None:
                out.append(None)
                continue
            names = [ctx_name(*c, lv) for lv in LEVELS]
            out.append(next((n for n in names if n in C), names[0]))
        return _relabel(ep, out)
    return lab


def label_eps(tubes, cal_by_task, plan, alpha, robust):
    """Per-token quantile at alpha/m, from every training-task calibration episode that runs the token.
    cal_by_task: task -> {"E": Eddie-labelled, "P": primitive-labelled episodes}; a primitive
    certificate (label without ':') is scored on the primitive labelling."""
    m = max(len(plan), 1)
    eps = {}
    for s in plan:
        t = tubes.get(s)
        if t is None:
            continue
        per = []
        for c in cal_by_task.values():
            eps_t = c[ctx_level(s)] if "@" in s or 3 in c else c["E"] if ":" in s else c["P"]
            sc = t.residual_scores(eps_t)
            sc = sc[np.isfinite(sc)]
            if len(sc):
                per.append(sc)
        if not per:
            continue
        eps[s] = max(conformal_quantile(x, alpha / m) for x in per) if robust else \
            conformal_quantile(np.concatenate(per), alpha / m)
    return eps


def evaluate_mode(tubes, eps, test, plan, rho, mus):
    cert = {s: tubes[s] for s in eps if np.isfinite(eps[s])}
    eps = {s: eps[s] for s in cert}
    steps = [s for ep in test for s in ep["skill"] if s]
    fallback = float(np.mean([":" not in s for s in steps])) if steps else 0.0     # primitive-level steps
    return {"n_test": len(test), "plan": plan, "plan_certified": [s in cert for s in plan],
            "fallback_steps": fallback, "violation": summ(evaluate(cert, eps, test, rho), "D_viol"),
            "radius": radius_stats(cert, eps, rho, test), "gate": gate_rows(test, cert, eps, rho, mus, plan),
            "eps": {s: float(v) for s, v in eps.items()}}


def held_out(H, data, mus, alpha, fewshot, only_ctx=False):
    rest = [t for t in TASKS if t != H]
    train = [ep for t in rest for ep in data[t]["train"]]
    E, kE, P, kP, rho = certificates(train)
    nominal = nominal_plan(H)
    out = {}
    # hand-over context: Eddie@predecessor -> primitive@predecessor -> primitive
    C, K, rhoC = ctx_certificates(train)
    lab = ctx_labeller(C)
    test = [lab(ep) for ep in data[H]["test"]]
    plan = list(dict.fromkeys(lab({"skill": nominal})["skill"]))
    cal_by_task = {t: {lv: [ctx_relabel(ep, lv) for ep in data[t]["calib"]] for lv in LEVELS} for t in rest}
    modes = {"label": label_eps(C, cal_by_task, plan, alpha, False),
             "label_max": label_eps(C, cal_by_task, plan, alpha, True)}
    q = conformal_quantile(joint_scores(C, K, [lab(ep) for ep in data[H]["calib"][:20]]), alpha)
    modes["recal20"] = {s: q * K[s] for s in C}
    for mode, eps in modes.items():
        g = out[f"ctx/{mode}"] = evaluate_mode(C, eps, test, plan, rhoC, mus)
        g["levels"] = {str(lv): float(np.mean([ctx_level(s) == lv for ep in test for s in ep["skill"] if s])) for lv in LEVELS}
        print(H, f"ctx/{mode}", "viol", g["violation"]["violations"], "levels", g["levels"],
              "gate", [r["completed"] for r in g["gate"]], [r["completed_exceed_mu"] for r in g["gate"]], flush=True)
    if only_ctx:
        return out
    for name, hier in (("eddie", False), ("hier", True)):
        lab = labeller(E, P, hier)
        tubes = {**P, **E} if hier else dict(E)
        kappa = {**kP, **kE} if hier else dict(kE)
        test = [lab(ep) for ep in data[H]["test"]]
        plan = list(dict.fromkeys(lab({"skill": nominal})["skill"]))
        cal_by_task = {t: {"E": data[t]["calib"], "P": [prim(ep) for ep in data[t]["calib"]]} for t in rest}
        calE = [ep for t in rest for ep in data[t]["calib"]]
        q0 = conformal_quantile(joint_scores(E, kE, calE), alpha)
        if hier:   # primitive certificates scored on the primitive labelling of the same episodes
            q0 = max(q0, conformal_quantile(joint_scores(P, kP, [prim(ep) for ep in calE]), alpha))
        modes = {"joint0": {s: q0 * kappa[s] for s in tubes},
                 "label": label_eps(tubes, cal_by_task, plan, alpha, False),
                 "label_max": label_eps(tubes, cal_by_task, plan, alpha, True)}
        if hier:
            for n in (20, CAL_H):
                cal = [lab(ep) for ep in data[H]["calib"][:n]]
                q = conformal_quantile(joint_scores(tubes, kappa, cal), alpha)
                modes[f"recal{n}"] = {s: q * kappa[s] for s in tubes}
        for mode, eps in modes.items():
            out[f"{name}/{mode}"] = evaluate_mode(tubes, eps, test, plan, rho, mus)
            g = out[f"{name}/{mode}"]
            print(H, f"{name}/{mode}", "viol", g["violation"]["violations"], "fallback", round(g["fallback_steps"], 2),
                  "gate", [r["completed"] for r in g["gate"]], [r["completed_exceed_mu"] for r in g["gate"]], flush=True)
    for k in fewshot:
        E2, kE2, P2, kP2, rho2 = certificates(train + data[H]["train"][:k])
        lab = labeller(E2, P2, True)
        tubes, kappa = {**P2, **E2}, {**kP2, **kE2}
        test = [lab(ep) for ep in data[H]["test"]]
        plan = list(dict.fromkeys(lab({"skill": nominal})["skill"]))
        q = conformal_quantile(joint_scores(tubes, kappa, [lab(ep) for ep in data[H]["calib"][:CAL_H]]), alpha)
        g = out[f"hier/few{k}"] = evaluate_mode(tubes, {s: q * kappa[s] for s in tubes}, test, plan, rho2, mus)
        print(H, f"hier/few{k}", "viol", g["violation"]["violations"], "fallback", round(g["fallback_steps"], 2),
              "gate", [r["completed"] for r in g["gate"]], flush=True)
    return out


def in_distribution(data, mus, alpha):
    train = [ep for t in TASKS for ep in data[t]["train"]]
    E, kE, _, _, rho = certificates(train)
    out = {}
    C, K, rhoC = ctx_certificates(train)
    lab = ctx_labeller(C)
    for H in TASKS:
        plan = list(dict.fromkeys(nominal_plan(H)))
        test = data[H]["test"]
        eps = label_eps(E, {t: {"E": data[t]["calib"], "P": []} for t in TASKS}, plan, alpha, False)
        q = conformal_quantile(joint_scores(E, kE, data[H]["calib"]), alpha)
        out[H] = {"label": evaluate_mode(E, eps, test, plan, rho, mus),
                  "task_joint": evaluate_mode(E, {s: q * kE[s] for s in E}, test, plan, rho, mus)}
        ctest = [lab(ep) for ep in test]
        cplan = list(dict.fromkeys(lab({"skill": nominal_plan(H)})["skill"]))
        cal = {t: {lv: [ctx_relabel(ep, lv) for ep in data[t]["calib"]] for lv in LEVELS} for t in TASKS}
        out[H]["ctx_label"] = evaluate_mode(C, label_eps(C, cal, cplan, alpha, False), ctest, cplan, rhoC, mus)
        print("in-dist", H, {m: ([r["completed"] for r in x["gate"]], x["violation"]["violations"]) for m, x in out[H].items()},
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", default=mu_str(CFG.mu_cm))
    ap.add_argument("--alpha", type=float, default=CFG.conformal.alpha)
    ap.add_argument("--fewshot", default=",".join(str(int(k)) for k in CFG.fewshot))
    ap.add_argument("--task", default=None, help="only this held-out task")
    ap.add_argument("--no-indist", action="store_true")
    ap.add_argument("--only-indist", action="store_true")
    ap.add_argument("--only-ctx", action="store_true", help="held-out tasks: only the hand-over-context modes")
    a = ap.parse_args()
    mus = [float(x) / 100 for x in a.mu.split(",")]
    fewshot = [int(x) for x in a.fewshot.split(",") if x]
    data = load(TASKS + ([a.task] if a.task and a.task not in TASKS else []))
    res = {"alpha": a.alpha, "mu_m": mus, "loto": {}}
    for H in ([] if a.only_indist else [a.task] if a.task else TASKS):
        res["loto"][H] = held_out(H, data, mus, a.alpha, fewshot, a.only_ctx)
    if not a.no_indist:
        res["indist"] = in_distribution(data, mus, a.alpha)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "transfer_indist.json" if a.only_indist else \
        f"transfer{'_ctx' if a.only_ctx else ''}{'_' + a.task if a.task else ''}.json"
    (OUT / name).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
