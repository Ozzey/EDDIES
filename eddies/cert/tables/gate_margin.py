#!/usr/bin/env python3
"""Tables for eddies.cert.experiments.gate_margin: gate (a, b, c) vs. threshold monitors at intermediate tolerances.

    python -m eddies.cert.tables.gate_margin [--dir $EDDIES_RESULTS/gate_margin] [--smoke] [--out summary.json]

Inputs: the per-step logs <task>.npz and <task>_a.json in --dir (written by eddies.cert.experiments.gate_margin;
also in the dataset under derived/gate_margin/).  Without them the tables are printed from --dir/summary.json,
the counts of every task / method / tolerance computed from those logs (`--out results/gate_margin/summary.json`
writes it); the printed tables are the same either way.

Per episode and method, with the truth error err_k of gate_adaptive (reference certificates):
  would    max_k err_k > mu (the ungated execution leaves the tolerance)
  stop     first step k with stop value v_k > mu (fixed gate: before motion if its plan check fails)
  caught   would and stopped at a step before err first exceeds mu
  missed   would and not caught (never stopped, or stopped at / after the first exceedance)
  false    stopped although the execution stays within mu
Monitors (measured error ||S eta_k|| of the running Eddie's model = the truth model on certified Eddies):
  margin m    v_k = ||S eta_k|| + m                      (stop once the error exceeds mu - m)
  velocity c  v_k = ||S eta_k|| + c ||S (eta_k - eta_{k-1})||
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eddies.common.paths import ROOT
from eddies.common.config import load as load_config
from eddies.common.config import mu_list, robots

CFG = load_config("experiments/gate_margin")
GROUP_OF = {t: r for r, c in robots(CFG).items() for t in c.tasks}     # as eddies.cert.experiments.gate_margin
MUS = mu_list(CFG.mu_cm)
MARGINS = [float(m) for m in CFG.margins_m]
VEL = [float(v) for v in CFG.velocities]
FIELDS = ("n", "would", "caught", "missed", "false", "finished")
METHODS = {"a self-cal gate": "a", "b fixed gate": "b", "c one-step, fixed cal (env. raised)": "cw",
           "c' one-step, fixed cal (eps fixed)": "cf"}
METHODS.update({f"monitor margin {100 * m:g} cm": ("m", m) for m in MARGINS})
METHODS.update({f"monitor velocity c={c:g}": ("v", c) for c in VEL})
SUMMARY = "summary.json"


def counts(err_eps, stop_k, mu):
    out = dict.fromkeys(FIELDS, 0)
    for err, s in zip(err_eps, stop_k):
        exc = np.flatnonzero(err > mu)
        first = exc[0] if len(exc) else np.inf
        would, stopped = len(exc) > 0, s is not None
        caught = would and stopped and s < first
        out["n"] += 1
        out["would"] += would
        out["caught"] += caught
        out["missed"] += would and not caught
        out["false"] += stopped and not would
        out["finished"] += not stopped
    return {k: int(v) for k, v in out.items()}


def first_above(v, mu):
    i = np.flatnonzero(v > mu)
    return int(i[0]) if len(i) else None


def load(d, smoke):
    tasks = {}
    for t in GROUP_OF:
        f = d / f"{t}{'_smoke' if smoke else ''}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        cut = np.cumsum(z["lens"])[:-1]
        ep = {k: np.split(z[k], cut) for k in ("err", "meas", "vel", "b", "cw", "cf")}
        a = d / f"{t}{'_smoke' if smoke else ''}_a.json"
        tasks[t] = {"meta": meta, "ep": ep, "a": json.loads(a.read_text()) if a.exists() else None}
    return tasks


def method_stops(T, method, mu):
    ep = T["ep"]
    if method == "b":
        if T["meta"]["U_plan"] > mu:
            return [-1] * len(ep["err"])
        return [first_above(v, mu) for v in ep["b"]]
    if method in ("cw", "cf"):
        return [first_above(v, mu) for v in ep[method]]
    kind, x = method
    if kind == "m":
        return [first_above(v + x, mu) for v in ep["meas"]]
    return [first_above(m + x * v, mu) for m, v in zip(ep["meas"], ep["vel"])]


def a_counts(T, mu):
    if T["a"] is None or str(mu) not in T["a"]["rows"]:
        return None
    s = T["a"]["rows"][str(mu)]["summary"]
    return {"n": s["n"], "would": s["would_exceed"], "caught": s["caught_before"],
            "missed": s["would_exceed"] - s["caught_before"], "false": s["false_stops"], "finished": s["completed"]}


def add(acc, c):
    for k in FIELDS:
        acc[k] = acc.get(k, 0) + c[k]
    return acc


def rates(c):
    w, n = c["would"], c["n"]
    return (c["caught"] / w if w else np.nan), (c["false"] / (n - w) if n > w else np.nan)


def fmt(c):
    cr, fr = rates(c)
    return " ".join(f"{c[k]:5d}" for k in FIELDS) + f"   catch {100 * cr:5.1f}%  false {100 * fr:5.1f}%"


def groups_of(tasks):
    return {"pooled": list(tasks)} | {g: [t for t in tasks if GROUP_OF[t] == g] for g in CFG.robots}


def pool(per_task, ts):
    """Sum of the task counts of ts; None when one of them is missing."""
    acc = {}
    for t in ts:
        if per_task[t] is None:
            return None
        add(acc, per_task[t])
    return acc


def summarize(T):
    """Everything the tables print, from the per-step logs: per tolerance, method and task the counts, and the
    margin monitor at the gates' false-stop / catch rates (needs the per-step errors, so stored as numbers)."""
    groups = groups_of(T)
    per_task = {}
    for mu in MUS:
        per_task[str(mu)] = {name: {t: a_counts(T[t], mu) if m == "a" else counts(T[t]["ep"]["err"], method_stops(T[t], m, mu), mu)
                                    for t in T} for name, m in METHODS.items()}
    # catch rate of the margin monitor at the gate's false-stop rate (fine margin grid, pooled / per robot)
    fine = np.round(np.arange(0, 0.0301, 0.0005), 6)
    equal = []
    for mu in MUS:
        for g, ts in groups.items():
            curve = []
            for m in fine:
                acc = {}
                for t in ts:
                    add(acc, counts(T[t]["ep"]["err"], method_stops(T[t], ("m", m), mu), mu))
                curve.append((m,) + rates(acc))
            M, CR, FR = (np.array(x) for x in zip(*curve))
            for gate in ("a", "cw", "cf", "b"):
                name = [k for k, v in METHODS.items() if v == gate][0]
                c = pool(per_task[str(mu)][name], ts)
                if not c or not c["would"]:
                    continue
                cr, fr = rates(c)
                o = np.argsort(FR, kind="stable")
                # best monitor catch rate with false-stop rate <= the gate's, and interpolated at it
                ok = FR <= fr + 1e-12
                best = float(CR[ok].max()) if ok.any() else np.nan
                interp = float(np.interp(fr, FR[o], np.maximum.accumulate(CR[o])))
                reach = CR >= cr - 1e-12                       # lowest monitor false-stop rate reaching the gate's catch rate
                fr_at = float(FR[reach].min()) if reach.any() else np.nan
                m_at = float(np.interp(fr, FR[o], M[o]))
                equal.append(dict(mu=mu, group=g, gate=gate, gate_catch=float(cr), gate_false=float(fr), mon_catch_best=best,
                                  mon_catch_interp=interp, mon_false_at_gate_catch=fr_at, m_at=m_at))
    meta = {t: {k: x["meta"][k] for k in ("q0", "U_plan", "check_b")} for t, x in T.items()}
    return {"about": "counts of eddies.cert.tables.gate_margin from the per-step logs <task>.npz / <task>_a.json "
                     "(dataset derived/gate_margin/); per_task[mu][method][task] = {n, would, caught, missed, false, finished}",
            "mu": MUS, "tasks": meta, "per_task": per_task, "equal_rate": equal}


def summary(d=None, smoke=False):
    """The summary of --dir: computed from the per-step logs when they are there, else read from summary.json;
    None when neither is present."""
    d = Path(d or ROOT / "gate_margin")
    if any(d.glob(f"*{'_smoke' if smoke else ''}.npz")):
        return summarize(load(d, smoke))
    if not smoke and (d / SUMMARY).exists():
        return json.loads((d / SUMMARY).read_text())
    return None


def report(S):
    tasks = list(S["tasks"])
    print("tasks:", tasks)
    for t, x in S["tasks"].items():
        print(f"  {t:22s} q0 {x['q0']:.3f} U_plan {100 * x['U_plan']:6.2f} cm  fixed-gate check {x['check_b']}")
    groups = groups_of(tasks)
    for mu in S["mu"]:
        P = S["per_task"][str(mu)]
        print(f"\n=== mu = {100 * mu:g} cm                     " + " ".join(f"{k:>5s}" for k in FIELDS))
        for g, ts in groups.items():
            print(f" [{g}]")
            for name in METHODS:
                c = pool(P[name], ts)
                if c:
                    print(f"  {name:38s} {fmt(c)}")
    print("\n=== catch rate at equal false-stop rate (margin monitor, m on a 0.05 cm grid, linear interpolation)")
    for e in S["equal_rate"]:
        g, gate = e["group"], e["gate"]
        if g == "pooled" or gate == "a":
            print(f" mu {100 * e['mu']:4.1f} {g:7s} {gate:3s} gate catch {100 * e['gate_catch']:5.1f}% false {100 * e['gate_false']:5.1f}% | "
                  f"monitor @ same false: catch {100 * e['mon_catch_best']:5.1f}% (best m with FSR<=) {100 * e['mon_catch_interp']:5.1f}% (interp, m~{100 * e['m_at']:.2f} cm)"
                  f" | monitor false @ same catch {100 * e['mon_false_at_gate_catch']:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "gate_margin"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None, help="write the summary (all counts) as json, e.g. results/gate_margin/summary.json")
    a = ap.parse_args()
    S = summary(a.dir, a.smoke)
    if S is None:
        raise SystemExit(f"no per-step logs (<task>.npz) or {SUMMARY} in {a.dir}: datasets/download.sh --subset gate_margin")
    report(S)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(S, indent=1) + "\n")


if __name__ == "__main__":
    main()
