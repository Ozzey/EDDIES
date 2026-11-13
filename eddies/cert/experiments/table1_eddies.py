#!/usr/bin/env python3
"""Table 1: Eddies and the one-reference-model ablation under the same certificate, per robot.

    python -m eddies.cert.experiments.table1_eddies --variant eddie [--robot g1|leap|aero] [--episodes] [--swap]
    python -m eddies.cert.experiments.table1_eddies --table            # collect the per-variant JSONs

Both variants certify the SAME logged trajectories (tag sb, nominal; Franka: the four tasks, train
1000-1199, calibration 1200-1399, test 2000-2099) with the pipeline of gate_fixed: per-Eddie affine
model with shrinkage (build_certificates), episode-level joint conformal eps per task, propagated
tubes (Theorem D), and the certify-or-stop gate replayed on the test log with the task's nominal plan
(known before the robot moves).

  eddie           Eddies (ours): every Eddie has its own shrunk fit
  eddie_refonly   "Eddies, one reference model": every Eddie uses the reference model Theta_ref (the
                  label build_certificates selects as `ref`, its own unshrunk fit) instead of its own
                  shrunk fit; the per-Eddie normalisers kappa_i (0.9 quantile of the Eddie's
                  episode-max residual under that model on TRAIN) and the per-task conformal quantile
                  are kept
The third row of Table 1, "Eddies, per-step bound", is eddies.cert.experiments.calibration_ablation.

Metrics per task: Eddies certified; in-distribution coverage (episodes with any Theorem-D violation,
target <= alpha); ultimate tube radius over the executed steps; gate (plan check on the nominal plan)
completed / completed beyond mu / exceedances caught.
Writes $EDDIES_RESULTS/table1/<variant>.json (--out: another directory under $EDDIES_RESULTS).
--episodes adds per-episode records (episode_records) for the bootstrap intervals of
eddies.cert.tables.table1_ci; --swap exchanges the training and calibration seeds (a second split).
--robot g1|leap|aero runs the same comparison on another robot's own tasks (default franka): the
tasks, splits and coordinate block of its per-robot certificate (embodiments: tag sb, nominal,
train 1000-1199, calibration 1200-1399, test 2000-2199).

The helpers embedding, gate_rows, radius_stats, episode_records, load and _relabel are shared by the
other experiments (embodiments, physics, transfer, gate_adaptive, gate_margin, stack_compare, ...).
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter

import numpy as np

from eddies.common.paths import ROOT
from eddies.common.config import load as load_config
from eddies.common.config import mu_str, robots, seeds
from eddies.cert.data import load_split
from eddies.cert.gate import evaluate_episode, ultimate_tube
from eddies.cert.experiments.certify_task import BLOCKS, CALIB, MIN_STD, TRAIN, evaluate, summ
from eddies.cert.experiments.gate_fixed import TEST, build_certificates
from eddies.cert.tube import conformal_quantile, joint_scores, make_tube, ordered_segments

CFG = load_config("experiments/table1_eddies")
VARIANTS = tuple(str(v) for v in CFG.variants)
KAPPA_Q = float(CFG.conformal.kappa_quantile)
# other robots: the task groups, test seeds and coordinate block of embodiments (GROUPS, TEST);
# the Franka rows use the first 100 test episodes (splits.gate_test), the other robots all 200 (splits.test)
ROBOTS = {r: (list(c.tasks), TEST if r == "franka" else seeds(CFG.splits.test), c.family)
          for r, c in robots(CFG).items()}
BLOCK = "franka"          # BLOCKS key of the robot (set by --robot)
OUT = ROOT / "table1"


# ----------------------------------------------------------------------------- labelling
def _relabel(ep, lab):
    """A copy of the episode with another label per step (used by transfer for the held-out task)."""
    out = dict(ep)
    out["skill"] = np.array(lab, dtype=object)
    return out


# ----------------------------------------------------------------------------- embedding
def embedding(variant, train):
    """Certified block J, metric W, physical scale, position rows and slew bound rho of the paper's
    embedding (the robot's block, restricted to the coordinates that move on the training data).
    ``variant`` is an Eddie labelling (eddie or eddie_refonly): the embedding is the same for both."""
    if not train:
        raise FileNotFoundError("no training episodes found under datasets/episodes (run datasets/download.sh, "
                                "see datasets/README.md)")
    emb = train[0]["manifest"]["embedding"]
    W = np.asarray(emb["metric_diag"], float)
    scale = (np.asarray(emb["phys_upper"]) - np.asarray(emb["phys_lower"])) / \
            (np.asarray(emb["norm_upper"]) - np.asarray(emb["norm_lower"]))
    J0, pos = BLOCKS[BLOCK]
    Z = np.concatenate([ep["z"] for ep in train])
    moving = [j for j in range(Z.shape[1]) if Z[:, j].std() > MIN_STD]
    J = [j for j in J0 if j in moving]
    rho = float(train[0]["manifest"]["method_info"]["rho_step"])
    return J, W, scale, pos, rho


# ----------------------------------------------------------------------------- evaluation
def plan_of(ep):
    return list(dict.fromkeys(s for s, _, _ in ordered_segments(ep["skill"]) if s))


def radius_stats(tubes, eps, rho, test):
    """Ultimate tube radius [m], weighted by executed steps (uncertified steps excluded)."""
    r, w, unc, tot = [], [], 0, 0
    for ep in test:
        for s, k0, k1 in ordered_segments(ep["skill"]):
            if s is None:
                continue
            tot += k1 - k0 + 1
            if s in tubes:
                r.append(ultimate_tube(tubes[s], rho, eps[s]))
                w.append(k1 - k0 + 1)
            else:
                unc += k1 - k0 + 1
    if not r:
        return {"median_cm": None, "p90_cm": None, "uncertified_steps": unc / max(tot, 1)}
    r, w = np.asarray(r), np.asarray(w, float)
    o = np.argsort(r)
    cw = np.cumsum(w[o]) / w.sum()
    q = lambda p: float(100 * r[o][np.searchsorted(cw, p)])  # noqa: E731
    return {"median_cm": q(0.5), "p90_cm": q(0.9), "uncertified_steps": unc / max(tot, 1)}


def token_stats(tubes, test):
    """Segment lengths, hand-overs per episode, and the one-step position prediction error [mm] of
    each step's Eddie model (how much embodiment dynamics the Eddie carries)."""
    seg, ho, err, unc = [], [], [], 0
    for ep in test:
        segs = [(s, k0, k1) for s, k0, k1 in ordered_segments(ep["skill"]) if s]
        seg += [k1 - k0 + 1 for _, k0, k1 in segs]
        ho.append(max(len(segs) - 1, 0))
        z, e, sk = ep["z"], ep["e"], ep["skill"]
        for s, k0, k1 in segs:
            t = tubes.get(s)
            if t is None:
                unc += k1 - k0
                continue
            Z = z[k0:k1 + 1][:, t.J]
            w = Z[1:] - (Z[:-1] @ t.A.T + e[k0:k1] @ t.B.T + t.d)
            err.append(1000 * np.linalg.norm(w @ t.S.T, axis=1))
    err = np.concatenate(err) if err else np.array([np.nan])
    return {"segment_len_median": float(np.median(seg)) if seg else None, "handovers_per_episode": float(np.mean(ho)),
            "pred_err_mm_median": float(np.nanmedian(err)), "pred_err_mm_p99": float(np.nanquantile(err, 0.99))}


def a_priori_plan(task):
    """The Eddie set the gate's plan check sees BEFORE motion: the task's nominal Eddie plan
    (synthesised from the task for the LTL tasks), never the test episode's own sequence."""
    from eddies.envs.tasks import make_task
    return list(dict.fromkeys(make_task(task, 2000).nominal_plan()))


def gate_rows(test, tubes, eps, rho, mus, plan=None, detail=None):
    rows = []
    fail = np.array([not ep["manifest"]["success"] or ep["manifest"]["bad_steps"] > 0 for ep in test])
    for mu in mus:
        D, T = zip(*[evaluate_episode(ep, plan if plan is not None else plan_of(ep), tubes, eps, mu, rho)
                    for ep in test])
        comp = np.array([d.completed for d in D])
        exceed = np.array([t["exceeds_mu"] for t in T])
        if detail is not None:                  # per-episode completion (episode_records checks against it)
            detail.append(comp.tolist())
        rows.append({"mu_m": mu, "n": len(test), "accepted": int(sum(d.accepted for d in D)),
                     "completed": int(comp.sum()),
                     "completed_exceed_mu": int(sum(d.exceeded_executed for d in D if d.completed)),
                     "exceed_total": int(exceed.sum()), "exceed_caught": int((exceed & ~comp).sum()),
                     "completed_failed": int((comp & fail).sum()),
                     "causes": dict(Counter(d.cause for d in D))})
    return rows


def build_ref_only(train, J, W, scale, pos, rho):
    """Ablation eddie_refonly: the labels build_certificates certifies, each with the reference model
    (the label it selects as `ref`, fitted alone without shrinkage, i.e. its prior Theta) instead of
    its own shrunk fit; kappa_i = 0.9 quantile of label i's episode-max residual under that model."""
    own, _, ref = build_certificates(train, J, W, scale, pos, rho)
    t_ref = make_tube(train, ref, J, W, scale, pos)
    tubes, kappa = {}, {}
    for s, t0 in own.items():
        t = copy.copy(t_ref)
        t.skill, t.n_train, t.shrink = s, t0.n_train, None
        tubes[s] = t
        kappa[s] = float(np.nanquantile(t.residual_scores(train), KAPPA_Q))
    return tubes, kappa, ref


def episode_records(test, tubes, kappa, q, cal_scores, rho, mus, plan, completed):
    """Per-episode data for bootstrap intervals.  With eps_i = q' kappa_i the replayed gate completes
    an episode iff no Eddie is uncertified and q_res <= q' <= min(q_plan(mu), q_ho(mu)) (residual
    monitor, plan check and hand-over check are each monotone in q'), so test AND calibration
    episodes can be resampled offline.  The reconstruction is checked against the replayed gate
    (``completed``: per mu, per episode) at the calibrated q."""
    coef = {s: (rho * float(t.cg.sum()), kappa[s] * float(t.cw.sum())) for s, t in tubes.items()}  # r_s = a + q b [m]
    q_plan = [(-np.inf if any(s not in tubes for s in plan) else
               min([(mu - coef[s][0]) / coef[s][1] for s in plan], default=np.inf)) for mu in mus]
    eps_rec = []
    for i, ep in enumerate(test):
        z, e = ep["z"], ep["e"]
        steps, q_res, q_ho, uncert = Counter(), -np.inf, [np.inf] * len(mus), False
        for s, k0, k1 in ordered_segments(ep["skill"]):
            if s is None:
                continue
            steps[s] += k1 - k0 + 1
            t = tubes.get(s)
            if t is None:
                uncert = True
                continue
            if k1 > k0:                          # residual monitor: steps k0 .. k1-1
                Z = z[k0:k1 + 1][:, t.J]
                w = Z[1:] - (Z[:-1] @ t.A.T + e[k0:k1] @ t.B.T + t.d)
                q_res = max(q_res, float(np.linalg.norm(w * t.Wz, axis=1).max()) / (kappa[s] * (1 + 1e-9)))
            eta0 = z[k0, t.J] - np.linalg.solve(np.eye(len(t.J)) - t.A, t.B @ e[k0] + t.d)
            tr = np.linalg.norm(np.einsum("jab,b->ja", t.powers_S, eta0), axis=1)
            cg = np.concatenate([[0], np.cumsum(t.cg)])[: len(tr)]
            cw = np.concatenate([[0], np.cumsum(t.cw)])[: len(tr)]
            for m, mu in enumerate(mus):         # max_j tr_j + rho cg_j + q kappa cw_j <= mu
                num = mu - tr - rho * cg
                thr = -np.inf if num[0] < 0 else float(np.min(num[1:] / (kappa[s] * cw[1:])))
                q_ho[m] = min(q_ho[m], thr)
        rec = [(not uncert) and q_res <= q <= min(q_plan[m], q_ho[m]) for m in range(len(mus))]
        bad = [m for m in range(len(mus)) if rec[m] != completed[m][i]]
        if bad:
            print(f"episode_records: seed {ep['seed']} reconstruction differs at mu {[mus[m] for m in bad]}", flush=True)
        eps_rec.append({"seed": int(ep["seed"]), "steps": dict(steps), "uncertified": uncert, "q_res": q_res,
                        "q_ho": q_ho, "completed": [bool(completed[m][i]) for m in range(len(mus))],
                        "recon_ok": not bad})
    return {"q": q, "cal_scores": [float(x) for x in cal_scores], "coef": coef, "q_plan": q_plan,
            "plan": list(plan), "test": eps_rec}


def run(variant, data, mus, alpha, tasks, episodes=False):
    if variant not in VARIANTS:
        raise KeyError(variant)
    train = [ep for t in tasks for ep in data[t]["train"]]
    J, W, scale, pos, rho = embedding(variant, train)
    build = build_ref_only if variant == "eddie_refonly" else build_certificates
    tubes, kappa, ref = build(train, J, W, scale, pos, rho)
    labels = sorted({s for ep in train for s in ep["skill"] if s})
    res = {"labels": len(labels), "certified": len(tubes), "J": [int(j) for j in J], "rho": rho,
           "W": W.tolist(), "reference": ref, "tasks": {},
           "shrink": dict(Counter(str(getattr(t, "shrink", None)) for t in tubes.values()))}
    for task in tasks:
        cal = data[task]["calib"]
        cal_scores = joint_scores(tubes, kappa, cal)
        q = conformal_quantile(cal_scores, alpha)
        eps = {s: q * kappa[s] for s in tubes}
        test = data[task]["test"]
        viol = summ(evaluate(tubes, eps, test, rho), "D_viol")
        plan = a_priori_plan(task)
        detail = []
        res["tasks"][task] = {"q": q, "n_test": len(test), "violation": viol, "a_priori_plan": plan,
                              "radius": radius_stats(tubes, eps, rho, test), "tokens": token_stats(tubes, test),
                              "gate": gate_rows(test, tubes, eps, rho, mus, plan, detail)}
        if episodes:
            res["tasks"][task]["episodes"] = episode_records(test, tubes, kappa, q, cal_scores, rho, mus, plan, detail)
    return res


def load(tasks):
    data = {t: {"train": load_split("sb", "nominal", t, TRAIN), "calib": load_split("sb", "nominal", t, CALIB),
                "test": load_split("sb", "nominal", t, TEST)} for t in tasks}
    # one command layout per robot, as embodiments: task-specific extra dims after the shared ones
    # are dropped (no-op when every task has the same width, e.g. the Franka)
    eps = [ep for d in data.values() for sp in d.values() for ep in sp]
    w = min((ep["z"].shape[1] for ep in eps), default=0)
    if any(ep["z"].shape[1] != w for ep in eps):
        print("command width trimmed to", w, flush=True)
        for ep in eps:
            ep["z"], ep["e"] = ep["z"][:, :w], ep["e"][:, :w]
    return data


# ----------------------------------------------------------------------------- table
def table():
    rows = []
    for v in VARIANTS:
        f = OUT / f"{v}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        ind = r["indist"]
        n = sum(x["n_test"] for x in ind["tasks"].values())
        viol = sum(x["violation"]["violations"] for x in ind["tasks"].values())
        med = np.median([x["radius"]["median_cm"] for x in ind["tasks"].values() if x["radius"]["median_cm"]])
        p90 = max(x["radius"]["p90_cm"] or 0 for x in ind["tasks"].values())
        g = {f"{100 * row['mu_m']:g}": [0, 0, 0] for row in next(iter(ind["tasks"].values()))["gate"]}
        for x in ind["tasks"].values():
            for row in x["gate"]:
                k = f"{100 * row['mu_m']:g}"
                g[k][0] += row["completed"]
                g[k][1] += row["completed_exceed_mu"]
                g[k][2] += row["completed_failed"]
        rows.append((v, ind["labels"], ind["certified"], viol, n, med, p90, g))
    mus = list(rows[0][7]) if rows else []
    print(f"{'variant':15s} {'tok':>4s} {'cert':>4s} {'viol':>8s} {'r50':>5s} {'r90':>5s} " +
          " ".join(f"{'gate@' + m:>11s}" for m in mus))
    for v, nl, nc, viol, n, med, p90, g in rows:
        print(f"{v:15s} {nl:4d} {nc:4d} {viol:3d}/{n:<4d} {med:5.1f} {p90:5.1f} " +
              " ".join(f"{c:4d} ({e},{f})".rjust(11) for c, e, f in g.values()))
    print("viol: test episodes with a tube violation; r50/r90: step-weighted ultimate tube radius [cm];"
          " gate@mu: completed (completed beyond mu, completed but failed).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, help="eddie (Table 1 'Eddies') or eddie_refonly ('Eddies, one reference model')")
    ap.add_argument("--robot", choices=tuple(ROBOTS), default="franka",
                    help="robot whose own tasks are certified (default franka: Table 1)")
    ap.add_argument("--mu", default=mu_str(CFG.mu_cm))
    ap.add_argument("--alpha", type=float, default=CFG.conformal.alpha)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="full training split, smoke.n_calib calibration / smoke.n_test_tok test seeds (pipeline check)")
    ap.add_argument("--episodes", action="store_true", help="also save per-episode records (bootstrap intervals)")
    ap.add_argument("--out", default="table1", help="output directory under $EDDIES_RESULTS")
    ap.add_argument("--swap", action="store_true", help="exchange training and calibration seeds (second split)")
    a = ap.parse_args()
    if a.table:
        return table()
    if a.variant is None:
        ap.error("--variant is required (or --table)")
    global TRAIN, CALIB, TEST, BLOCK
    tasks, TEST, BLOCK = ROBOTS[a.robot]
    if a.swap:
        TRAIN, CALIB = CALIB, TRAIN
    if a.smoke:
        CALIB, TEST = CALIB[:CFG.smoke.n_calib], TEST[:CFG.smoke.n_test_tok]
    mus = [float(x) / 100 for x in a.mu.split(",")]
    data = load(tasks)
    out = {"variant": a.variant, "alpha": a.alpha, "mu_m": mus, "train": [TRAIN[0], TRAIN[-1]], "calib": [CALIB[0], CALIB[-1]]}
    if a.robot != "franka":
        out.update(robot=a.robot, tasks=tasks, test=[TEST[0], TEST[-1]])
    out["indist"] = run(a.variant, data, mus, a.alpha, tasks, a.episodes)
    print(a.variant, "in-distribution:", {t: (x["violation"]["violations"], x["radius"]["median_cm"],
                                                [g["completed"] for g in x["gate"]]) for t, x in out["indist"]["tasks"].items()},
          flush=True)
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{a.variant}{'_smoke' if a.smoke else ''}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
