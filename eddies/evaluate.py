#!/usr/bin/env python3
"""Regenerate the tables and figures of the paper from the result files in results/.

    python -m eddies.evaluate all                      # every table (markdown) and figure
    python -m eddies.evaluate table1 table3 fig5       # a subset
    python -m eddies.evaluate --list

Tables are printed as markdown (no LaTeX); figures go to assets/results/generated/ (paths.figures).
Inputs: $EDDIES_RESULTS (default results/); nothing here touches episode logs or the simulator.

  table1   Table 1   Eddies rows: Eddies and the two ablations  results/table1, table1_robots/<robot>
  table2   Table 2   certificates moved between systems              embodiments/embodiments.json, physics/physics.json
  table3   Table 3   cube stacking: MuJoCo / Isaac Sim / robot       compare/stack_*.json
  tableD1  Table D.1 per-task certificates, both gates               embodiments.json, adaptive/adaptive_<t>_known_<r>.json
  tableD2  Table D.2 Franka tasks held out                           adaptive/adaptive_<t>.json, adaptive_<t>_known.json
  tableD3  Table D.3 per-Eddie vs per-primitive (+ App. D mixture)   pool/final_indist_{eddy,primitive}.json, pool/mixture_*.json
  tableD4  Table D.4 funnel containment vs recursion                 gate/containment.json
  tableD5  Table D.5 certify-or-stop under faster commands           gate/gate_eddy.json, gate_eddy_recal.json
  tableD6  Table D.6 shared Eddies' tubes                            embodiments.json
  pooled   App. D    pooled gate comparison ("1,968")                gate_margin/summary.json (or the per-step *.npz)
  fig4     Fig. 4    calibration + chained bound                     figs_data/trace_fmb.json
  fig5     Fig. 5    finished runs vs tolerance, both gates          embodiments.json, adaptive_*_known_*.json
  fig7     Fig. 7    held-out Franka tasks                           adaptive_<t>.json
  supp     Figs. B.1, D.1, E.1 and the per-task gate figure          facts/gain_all.json, dynamics_map/, adaptive/

Table 1 prints the Eddies rows of the paper's table per robot: Eddies, "Eddies, one reference model" and
"Eddies, per-step bound" (the other token rows are not recomputed by this release; docs/PAPER_RESULTS.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from eddies.common.config import load, mu_list, robots, tasks as task_cfgs
from eddies.common.paths import FIGURES, ROOT

CFG = load("experiments/evaluate")
R = robots(CFG)
NAME = {t: c.display for t, c in task_cfgs(CFG).items()}
ROBOT_OF = {t: r for r, c in R.items() for t in c.tasks}


# ------------------------------------------------------------------------------- helpers
def md(header, rows, title=None):
    w = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    line = lambda r: "| " + " | ".join(str(x).ljust(k) for x, k in zip(r, w)) + " |"
    out = ([f"**{title}**", ""] if title else []) + [line(header), "|" + "|".join("-" * (k + 2) for k in w) + "|"]
    out += [line(r) for r in rows]
    print("\n".join(out) + "\n")


def jload(rel):
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(f"{p} (results/ incomplete; see datasets/README.md)")
    return json.loads(p.read_text())


def gate_at(rows, mu_cm, key="completed"):
    for r in rows:
        if abs(100 * r["mu_m"] - mu_cm) < 1e-6:
            return r[key]
    raise KeyError(f"no gate row at {mu_cm} cm")


def have(*rels):
    return all((ROOT / r).exists() for r in rels)


# ------------------------------------------------------------------------------- Table 1
def table1(B=None):
    from eddies.cert.tables import table1_ci as tc
    T1 = CFG.table1
    B = int(T1.B if B is None else B)
    for r, c in R.items():
        d = ROOT / T1.dirs[r]
        if not (d / "eddie.json").exists():
            print(f"table1: {d} missing, skipped\n")
            continue
        K = json.loads((d / "eddie.json").read_text())["indist"]["labels"]
        rows = [("Eddies", "eddie"), ("Eddies, one reference model", "eddie_refonly")]
        fin_cm = [float(m) for m in T1.finished_cm[r]]
        out, D, seeds = [], None, None
        for name, v in rows:
            f = d / f"{v}.json"
            if not f.exists():
                continue
            ind = json.loads(f.read_text())["indist"]
            tasks_ = ind["tasks"]
            p90 = [x["radius"]["p90_cm"] for x in tasks_.values() if x["radius"]["p90_cm"] is not None]
            fin = {m: sum(gate_at(x["gate"], m) for x in tasks_.values()) for m in fin_cm}
            ci = ""
            if B > 0:
                tk, mus, alpha, _ = tc.load_variant(f)
                T = {t: tc.Task(x, mus, alpha) for t, x in sorted(tk.items())}
                if D is None:
                    D, seeds = tc.draws(T, B, 0), {t: x.seeds for t, x in T.items()}
                if {t: x.seeds for t, x in T.items()} == seeds:
                    _, fb = tc.bootstrap(T, D, False)
                    j = next((k for k, m in enumerate(tc.MUS) if abs(100 * m - fin_cm[0]) < 1e-6), 0)   # CI at the first reported tolerance
                    lo, hi = tc.ci(fb[:, j])
                    ci = f"[{lo:.0f}, {hi:.0f}]"
            n = sum(x["n_test"] for x in tasks_.values())
            out.append([name, ind["labels"], ind.get("certified", ""), f"{max(p90):.1f}" if p90 else "--",
                        " / ".join(str(fin[m]) for m in fin_cm) + (f"  {ci}" if ci else ""), n])
        cal = d / "calibration.json"
        if cal.exists():
            cj = json.loads(cal.read_text())
            mu = cj["mu_m"]
            for name, key in (("Eddies, per-step bound", "b1"),):
                p90 = [x[key]["radius"]["p90_cm"] for x in cj["tasks"].values() if x[key]["radius"]["p90_cm"] is not None]
                fin = [sum(x[key]["gate"][mu.index(m / 100)]["completed"] for x in cj["tasks"].values()) for m in fin_cm]
                # paper convention (Table 1): a tube is printed only where the bound lets some run finish at some
                # tolerance of the calibration grid; a bound that finishes nothing anywhere gets "--"
                any_fin = any(g["completed"] for x in cj["tasks"].values() for g in x[key]["gate"])
                out.append([name, K, "", f"{max(p90):.1f}" if p90 and any_fin else "--", " / ".join(map(str, fin)), sum(x["n_test"] for x in cj["tasks"].values()) if "n_test" in next(iter(cj["tasks"].values())) else ""])
        md(["Tokens", "#", "certified", "tube p90 [cm]", "finished at " + " / ".join(f"{m:g}" for m in fin_cm) + " cm" + (" (95% CI at the first)" if B else ""), "n"],
           out, f"Table 1, Eddies rows ({c.short}; {d.relative_to(ROOT)}; test-episode bootstrap B={B})")


# ------------------------------------------------------------------------------- Table 2
def table2():
    T2 = CFG.table2
    mu = float(T2.mu_cm)
    emb = jload("embodiments/embodiments.json")
    rows = []
    for src, dst in T2.hand_pairs:
        for i, suf in enumerate(T2.hand_tasks):
            t = f"{dst}_{suf}"
            x = emb["cross"][f"{src}->{t}"]["unchanged"]
            own = emb["groups"][dst]["tasks"][t]
            rows.append([f"{R[src].short} -> {R[dst].short}" if i == 0 else "", NAME[t], x["violation"]["n"],
                         x["violation"]["violations"], gate_at(x["gate"], mu), own["violation"]["violations"], gate_at(own["gate"], mu)])
    if have("physics/physics.json"):
        ph = jload("physics/physics.json")
        for i, t in enumerate(T2.physx_tasks):
            b, re_ = ph["tasks"][t]["mujoco"], ph["tasks"][t][T2.physx_refit]
            rows.append(["MuJoCo -> Isaac Sim" if i == 0 else "", NAME[t], b["violation"]["n"], b["violation"]["violations"],
                         gate_at(b["gate"], mu), re_["violation"]["violations"], gate_at(re_["gate"], mu)])
    md(["Source -> target", "Task", "episodes", "borrowed viol.", "borrowed finished", "re-identified viol.", "re-identified finished"],
       rows, f"Table 2 (finished at mu = {mu:g} cm; Isaac re-identified = {T2.physx_refit})")


# ------------------------------------------------------------------------------- Table 3
def table3():
    from eddies.cert.experiments import stack_compare
    print("**Table 3** (eddies.cert.experiments.stack_compare --table; mu columns in cm)\n")
    stack_compare.table(mu_list(CFG.table3.mu_cm))
    print()
    if have("real/table3_robot_rows.json"):                 # robot rows: values reported in the paper
        rr = jload("real/table3_robot_rows.json")
        rows = [[r["system"], r["certificate"], r["runs"], r["success"], r["viol"], f"{r['tube_p90_cm']:.1f}",
                 f"{r['fixed_gate_finished']} / {r['fixed_gate_stopped']}", f"{r['fixed_gate_caught']} / {r['fixed_gate_would_leave']}",
                 f"{r['selfcal_gate_finished']} / {r['selfcal_gate_stopped']}", f"{r['selfcal_gate_caught']} / {r['selfcal_gate_would_leave']}"]
                for r in rr["rows"]]
        md(["system", "certificate", "runs", "success", "viol.", "tube p90 [cm]", "fixed: fin. / stop.", "caught", "self-calibrating: fin. / stop.", "caught"],
           rows, f"Table 3, robot rows at mu = {rr['mu_cm']:g} cm (values reported in the paper, real/table3_robot_rows.json)")


# ------------------------------------------------------------------------------- Table D.1
def tableD1():
    D1 = CFG.tableD1
    fin, cat = float(D1.finished_cm), float(D1.caught_cm)
    emb = jload("embodiments/embodiments.json")
    rows, tot = [], dict(eddies=0, viol=0, fixed=0, self=0, caught=0, would=0, cert=0)
    # "Eddies" is the paper's count (tasks/<t>.yaml n_eddies: the steps of the nominal plan, except that the
    # tightening loop of the two peg-tightening tasks counts once, i.e. its distinct Eddies); "distinct" is the
    # number of distinct Eddies the task's plan uses (embodiments.json), see docs/PAPER_RESULTS.md
    plan_len = {t: int(c.n_eddies) for t, c in task_cfgs(CFG).items()}
    one = one_step_finished(fin)                                              # the paper's "one-step" column
    tot["plan"] = tot["one"] = 0
    for r, c in R.items():
        tot["cert"] += emb["groups"][r]["certified"]                           # distinct Eddies certified on the robot
        for i, t in enumerate(c.tasks):
            x = emb["groups"][r]["tasks"][t]
            ad = ROOT / "adaptive" / f"adaptive_{t}_known_{r}.json"
            if ad.exists():
                a = json.loads(ad.read_text())["rows"]
                s_fin = a[f"{fin / 100:g}"]["summary"]["completed"]
                s2 = a[f"{cat / 100:g}"]["summary"]
                caught = f"{s2['caught_before']} / {s2['would_exceed']}"
                tot["self"] += s_fin; tot["caught"] += s2["caught_before"]; tot["would"] += s2["would_exceed"]
            else:
                s_fin, caught = "--", "--"
            tot["plan"] += plan_len[t]
            tot["one"] += one.get(t, 0)
            rows.append([c.short if i == 0 else "", NAME[t], plan_len[t], x["eddies"], x["violation"]["violations"],
                         f"{x['radius']['median_cm']:.1f} / {x['radius']['p90_cm']:.1f}", gate_at(x["gate"], fin), one.get(t, "--"), s_fin, caught])
            tot["eddies"] += x["eddies"]; tot["viol"] += x["violation"]["violations"]; tot["fixed"] += gate_at(x["gate"], fin)
    rows.append(["All", f"({tot['cert']} distinct Eddies certified)", tot["plan"], tot["eddies"], tot["viol"], "", tot["fixed"],
                 tot["one"] if one else "--", tot["self"], f"{tot['caught']} / {tot['would']}"])
    md(["Robot", "Task", "Eddies", "distinct", "viol.", "tube median / p90 [cm]", f"finished at {fin:g} cm, fixed", "one-step", "self-calibrating", f"caught at {cat:g} cm"],
       rows, "Table D.1 (200 test episodes per task; Eddies = the paper's count of the nominal plan, distinct = distinct Eddies in it"
             + ("" if one else "; one-step column needs results/gate_margin/summary.json") + ")")


def one_step_finished(mu_cm):
    """Table D.1 'one-step' column, {task: finished at mu}: the gate that holds the calibration fixed and checks only
    the next command (method "c one-step, fixed cal (env. raised)" of eddies.cert.tables.gate_margin), counted from
    the per-step logs results/gate_margin/<task>.npz when present, else from results/gate_margin/summary.json.
    {} when neither is there."""
    from eddies.cert.tables import gate_margin as gm
    S = gm.summary(ROOT / "gate_margin")
    if S is None:
        return {}
    P = S["per_task"][str(mu_cm / 100)]["c one-step, fixed cal (env. raised)"]
    return {t: c["finished"] for t, c in P.items()}


# ------------------------------------------------------------------------------- Table D.2
def tableD2():
    rows = []
    for t in CFG.franka_tasks:
        for j, mu in enumerate(CFG.tableD2.mu_cm):
            cells = [NAME[t] if j == 0 else "", f"{mu:g}"]
            for suffix in ("", "_known"):
                p = ROOT / "adaptive" / f"adaptive_{t}{suffix}.json"
                if not p.exists():
                    cells += ["--", "--", "--"]
                    continue
                s = json.loads(p.read_text())["rows"][f"{mu / 100:g}"]["summary"]
                cells += [f"{s['caught_before']} / {s['would_exceed']}" if s["would_exceed"] else "--", s["false_stops"], s["completed"]]
            rows.append(cells)
    md(["Held-out task", "mu [cm]", "new task: caught", "false stops", "finished", "own calibration as prior: caught", "false stops", "finished"],
       rows, "Table D.2 (first 100 test episodes; self-calibrating gate)")


# ------------------------------------------------------------------------------- Table D.3
def tableD3():
    e, p = jload("pool/final_indist_eddy.json"), jload("pool/final_indist_primitive.json")
    rows = []
    for t in CFG.franka_tasks:
        a, b = e["tasks"][t], p["tasks"][t]
        rows.append([NAME[t], f"{a['eddies_per_episode']:g}", f"{a['D_violations']}/{a['compose_violations']}", f"{a['coverage']:.3f}",
                     f"{100 * a['compose_bound_median_m']:.1f}", f"{a['compose_tightness_median']:.2f}",
                     b["compose_violations"], f"{b['coverage']:.3f}", f"{100 * b['compose_bound_median_m']:.1f}"])
    md(["Task", "Eddies", "per Eddie: viol. (tube/chain)", "coverage", "bound [cm]", "tightness", "per primitive: viol. (chain)", "coverage", "bound [cm]"],
       rows, "Table D.3 (Franka, 200 held-out episodes per task)")
    # App. D text after Table D.3: one quantile calibrated over the mixture of the four tasks (no per-task Mondrian
    # calibration), same models and selection; the per-task coverage is the number quoted for stacking.
    mix = {k: jload(f"pool/mixture_{k}.json") for k in ("eddy", "primitive") if (ROOT / "pool" / f"mixture_{k}.json").exists()}
    if mix:
        rows = [[NAME[t]] + [f"{mix[k]['tasks'][t]['coverage']:.3f}" if k in mix else "--" for k in ("eddy", "primitive")]
                for t in CFG.franka_tasks]
        md(["Task", "mixture quantile, per Eddie: coverage", "per primitive: coverage"], rows,
           "App. D text: one quantile over the mixture of tasks (not valid per task)")


# ------------------------------------------------------------------------------- Table D.4
def tableD4():
    c = jload("gate/containment.json")
    rows = []
    for mu in CFG.tableD4.mu_cm:
        cells = [f"{mu:g}"]
        for t in CFG.franka_tasks:
            r = next(x for x in c["tasks"][t] if abs(100 * x["mu_m"] - mu) < 1e-6)
            cells.append(f"{100 * r['containment'] / r['handovers']:.0f} / {100 * r['recursion'] / r['handovers']:.0f}")
            cells.append(f"{100 * r['episodes_containment'] / r['episodes']:.0f} / {100 * r['episodes_recursion'] / r['episodes']:.0f}")
        rows.append(cells)
    hdr = ["mu [cm]"] + [f"{NAME[t]}: {k}" for t in CFG.franka_tasks for k in ("hand-overs", "episodes")]
    md(hdr, rows, "Table D.4 (% certified: funnel containment / recursion; Franka, 100 test episodes per task)")


# ------------------------------------------------------------------------------- Table D.5
def tableD5():
    files = [f for f in CFG.tableD5.files if (ROOT / "gate" / f"{f}.json").exists()]
    data = {f: jload(f"gate/{f}.json") for f in files}
    rows = []
    for sc in data[files[0]]["scales"]:
        for j, mu in enumerate(CFG.tableD5.mu_cm):
            cells = [f"{sc:g}x" if j == 0 else "", f"{mu:g}"]
            for k, f in enumerate(files):
                rs = [r for t in data[f]["tasks"].values() for r in t["rows"] if r["scale"] == sc and abs(100 * r["mu_m"] - mu) < 1e-6]
                n = sum(r["n"] for r in rs)
                if not n:
                    cells += (["--"] * 4) if k == 0 else ["--"] * 3
                    continue
                ref = sum(r["n"] - r["accepted"] for r in rs)
                comp = sum(r["completed"] for r in rs)
                if k == 0:
                    cells.append(f"{100 * sum(r['ungated_exceed_mu'] for r in rs) / n:.0f}")
                cells += [f"{100 * ref / n:.0f}", f"{100 * (n - ref - comp) / n:.0f}", f"{100 * comp / n:.0f}"]
            rows.append(cells)
    hdr = ["speed", "mu [cm]", "leave mu %"] + [f"{f}: {k}" for f in files for k in ("refused %", "stopped %", "completed %")]
    md(hdr, rows, "Table D.5 (Franka, four tasks; gate_eddy = calibrated at nominal speed, gate_eddy_recal = at the commanded speed)")


# ------------------------------------------------------------------------------- Table D.6
def tableD6():
    from eddies.cert.plots import shared_own
    print("**Table D.6** (eddies.cert.plots.shared_own; LaTeX rows: Eddie & Franka & G1 & LEAP & Aero)\n")
    shared_own.main()
    print()


# ------------------------------------------------------------------------------- pooled gate comparison
def pooled():
    from eddies.cert.tables import gate_margin
    S = gate_margin.summary(ROOT / "gate_margin")
    if S is None:
        print("pooled: results/gate_margin/summary.json missing, skipped\n")
        return
    print("**Appendix D, pooled gates** (eddies.cert.tables.gate_margin)\n")
    gate_margin.report(S)
    print()


# ------------------------------------------------------------------------------- figures
def fig4():
    from eddies.cert.plots import cert_pair
    from eddies.cert.plots.style import style
    style()
    print("fig4 ->", cert_pair.fig_cert_pair(jload("figs_data/trace_fmb.json"), FIGURES))


def fig5():
    from eddies.cert.plots import results as pr
    pr.style(); pr.fig_gate(); print("fig5 -> res_gate")


def fig7():
    from eddies.cert.plots import results as pr
    pr.style(); pr.fig_heldout(); print("fig7 -> res_heldout")


def supp():
    from eddies.cert.plots import supp as ps
    ps.setup()
    for name, fn, need in (("B.1", ps.fig_gains, "facts/gain_all.json"), ("gate_tasks", ps.fig_gate_tasks, "embodiments/embodiments.json"),
                           ("E.1", ps.fig_qtrace, "adaptive/adaptive_franka_fmb.json"), ("D.1", ps.fig_dynmap, "dynamics_map/dynamics_map.json")):
        if have(need):
            fn(); print(f"supp {name} -> done")
        else:
            print(f"supp {name}: {need} missing, skipped")


STEPS = {"table1": table1, "table2": table2, "table3": table3, "tableD1": tableD1, "tableD2": tableD2, "tableD3": tableD3,
         "tableD4": tableD4, "tableD5": tableD5, "tableD6": tableD6, "pooled": pooled,
         "fig4": fig4, "fig5": fig5, "fig7": fig7, "supp": supp}
NEEDS = {"table1": ["table1/eddie.json"], "table2": ["embodiments/embodiments.json"], "table3": ["compare/stack_mujoco.json"],
         "tableD1": ["embodiments/embodiments.json"], "tableD2": ["adaptive/adaptive_franka_fmb.json"],
         "tableD3": ["pool/final_indist_eddy.json", "pool/final_indist_primitive.json"], "tableD4": ["gate/containment.json"],
         "tableD5": ["gate/gate_eddy.json"], "tableD6": ["embodiments/embodiments.json"], "pooled": ["gate_margin/summary.json"],
         "fig4": ["figs_data/trace_fmb.json"], "fig5": ["embodiments/embodiments.json"], "fig7": ["adaptive/adaptive_franka_fmb.json"],
         "supp": ["adaptive/adaptive_franka_fmb.json"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", help="table1 ... supp, or all")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--B", type=int, default=None, help="Table 1 bootstrap draws (0 = none)")
    ap.add_argument("--strict", action="store_true", help="fail on a missing input instead of skipping the step")
    a = ap.parse_args(argv)
    if a.list or not a.steps:
        for k in STEPS:
            print(f"{k:9s} needs results/{NEEDS[k][0]}" + ("" if have(*NEEDS[k]) else "   (missing)"))
        return 0
    steps = list(STEPS) if a.steps == ["all"] else a.steps
    print(f"results: {ROOT}\nfigures: {FIGURES}\n")
    rc = 0
    for s in steps:
        if s not in STEPS:
            ap.error(f"unknown step {s}; one of {', '.join(STEPS)}")
        if not have(*NEEDS[s]) and not a.strict:
            print(f"{s}: results/{NEEDS[s][0]} missing, skipped\n")
            continue
        try:
            STEPS[s](a.B) if s == "table1" else STEPS[s]()
        except Exception as ex:          # one failing step must not hide the others
            rc = 1
            print(f"{s}: FAILED {ex!r}\n")
            if a.strict:
                raise
    return rc


if __name__ == "__main__":
    sys.exit(main())
