#!/usr/bin/env python3
"""Eddie certificates: per-primitive pooled tubes, held-out tasks and vocabulary growth.

    python -m eddies.cert.experiments.pool --train franka_peg_insertion,franka_cube_stack,franka_drawer \
        --test franka_fmb --pool primitive [--calib joint|per_label] [--tag sb] [--name heldout_fmb]

Labels: ``--pool eddy`` certifies every Eddie (primitive:binding) separately; ``--pool
primitive`` pools every Eddie of a primitive over all bindings and training tasks (one model
and one tube per primitive).  Consecutive tokens of the same primitive then form one segment
(same model, zero mode jump at that hand-over).  (``token`` is accepted for ``eddy``.)

Calibration: ``joint`` -- one conformal quantile of the episode score max_l r_l / kappa_l at
level alpha (as certify_task); ``per_label`` -- each label is calibrated on its own at alpha times
the primitive's registered budget share (1/12 each in v1, 1/12 reserved; union bound), so adding
a primitive leaves every other certificate unchanged (vocabulary growth is a monotone refinement).

``--extra a:p1+p2`` fits/calibrates labels p1, p2 additionally from task a's training and
calibration episodes only (vocabulary growth: a new primitive certified from the one task that
introduced it).  Models: train seeds 1000-1199; calibration 1200-1399 of the TRAINING tasks;
test 2000-2199 of every test task (for a held-out task, test episodes are not exchangeable
with the calibration episodes: transfer is the empirical question).
Writes $EDDIES_RESULTS/pool/<name>.json.
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
from eddies.cert.ident import fit
from eddies.cert.experiments.certify_task import BLOCKS, CALIB, MIN_STD, TRAIN, family, wilson_upper
from eddies.cert.tube import check_episode, compose_episode, conformal_quantile, joint_scores, make_tube, ordered_segments
from eddies.envs.skillbook import BOOK, primitive_of

CFG = load_config("experiments/pool")
TEST = seeds(CFG.splits.test)


def relabel(eps, pool, task):
    for ep in eps:
        ep["task"] = task
        ep["token"] = ep["skill"]
        if pool == "primitive":
            ep["skill"] = np.array([primitive_of(s) for s in ep["skill"]], dtype=object)
    return eps


def only_labels(eps, keep):
    """Copies of episodes whose label column keeps only ``keep`` (others -> None)."""
    out = []
    for ep in eps:
        e = dict(ep)
        e["skill"] = np.array([s if s in keep else None for s in ep["skill"]], dtype=object)
        out.append(e)
    return out


def rho_of_ep(ep):
    return float(ep["manifest"]["method_info"]["rho_step"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--pool", choices=("eddy", "token", "primitive"), default="primitive")
    ap.add_argument("--calib", choices=("joint", "per_label"), default="joint")
    ap.add_argument("--extra", default=None, help="task:label1+label2 (vocabulary growth)")
    ap.add_argument("--exclude", default="", help="labels left uncertified (comma separated)")
    ap.add_argument("--alpha", type=float, default=CFG.conformal.alpha)
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="shrink every label's model toward the robot's pooled tracking model (lam = shrink * n)")
    ap.add_argument("--mondrian", action="store_true",
                    help="calibrate eps per test task on that task's own calibration episodes (models stay pooled)")
    ap.add_argument("--transfer-max", action="store_true",
                    help="no data from the test task: eps = label-wise max of the per-task eps of the training tasks")
    ap.add_argument("--renorm", action="store_true",
                    help="with --mondrian (joint): re-estimate the normalisers kappa on half of the task's calibration episodes")
    ap.add_argument("--n-calib", type=int, default=None, help="with --mondrian: use only the first N calibration episodes")
    ap.add_argument("--select", action="store_true",
                    help="per label, choose on the TRAINING split the model with the smallest estimated tube among its own "
                         "fit and fits shrunk toward the best-conditioned reference model (calibration is untouched)")
    ap.add_argument("--tag", default=CFG.tag)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    if a.pool == "token":
        a.pool = "eddy"
    train_tasks, test_tasks = a.train.split(","), a.test.split(",")
    J0, pos = BLOCKS[family(train_tasks[0])]

    def load(task, seeds):
        return relabel(load_split(a.tag, "nominal", task, seeds), a.pool, task)
    train = [ep for t in train_tasks for ep in load(t, TRAIN)]
    calib = [ep for t in train_tasks for ep in load(t, CALIB)]
    extra_labels, extra_train, extra_calib = set(), [], []
    if a.extra:
        et, labs = a.extra.split(":")
        extra_labels = set(labs.split("+"))
        extra_train = only_labels(load(et, TRAIN), extra_labels)
        extra_calib = only_labels(load(et, CALIB), extra_labels)
    test = {t: load(t, TEST) for t in test_tasks}

    if not train:
        raise FileNotFoundError(f"no training episodes for {train_tasks} under datasets/episodes (tag {a.tag}; "
                                "run datasets/download.sh, see datasets/README.md)")
    man = train[0]["manifest"]
    emb = man["embedding"]
    W = np.asarray(emb["metric_diag"], float)
    scale = (np.asarray(emb["phys_upper"]) - np.asarray(emb["phys_lower"])) / \
            (np.asarray(emb["norm_upper"]) - np.asarray(emb["norm_lower"]))
    for ep in train + extra_train:                       # one shared embedding chart
        assert np.allclose(ep["manifest"]["embedding"]["metric_diag"], W), ep["task"]
    Z = np.concatenate([ep["z"] for ep in train])
    J = [j for j in J0 if Z[:, j].std() > MIN_STD]
    excl = {x for x in a.exclude.split(",") if x}
    labels = [s for s in dict.fromkeys(x for ep in train for x in ep["skill"]) if s and s not in excl]
    labels += [s for s in sorted(extra_labels) if s not in labels]

    prior = None
    if a.shrink > 0:                        # the robot's pooled tracking model (all labels, all training tasks)
        pm = fit(train, None, np.asarray(J), 1, W)
        prior = pm.info["Theta"]
    rho0 = rho_of_ep(train[0])

    def train_tube(t, tr):
        """Estimated slew tube on the training split (eps = 0.9 quantile of the episode residual)."""
        e = float(np.nanquantile(t.residual_scores(tr), 0.9))
        return t.slew_radius(rho0, e)
    choice = {}
    if a.select:
        # reference: the best-conditioned label fit (smallest estimated tube, >= 5000 transitions)
        own = {}
        for s in labels:
            tr = extra_train if s in extra_labels else train
            t = make_tube(tr, s, J, W, scale, pos)
            if t is not None and t.cg is not None and t.n_train >= 5000:
                own[s] = train_tube(t, tr)
        ref = min(own, key=own.get)
        prior = fit(extra_train if ref in extra_labels else train, ref, np.asarray(J), 1, W).info["Theta"]
        print(f"  reference model: {ref} (estimated tube {100 * own[ref]:.1f} cm)")
    tubes, kappa, uncert, sources = {}, {}, {}, {}
    for s in labels:
        tr = extra_train if s in extra_labels else train
        if a.select:
            best = None
            for lam in (0.0, 0.03, 0.3, 3.0):
                t = make_tube(tr, s, J, W, scale, pos, prior=prior, shrink=lam)
                if t is None or t.cg is None:
                    continue
                r = train_tube(t, tr)
                if best is None or r < best[0]:
                    best = (r, lam, t)
            t = best[2] if best else make_tube(tr, s, J, W, scale, pos)
            choice[s] = best[1] if best else None
        else:
            t = make_tube(tr, s, J, W, scale, pos, prior=prior, shrink=a.shrink)
        if t is None:
            continue
        if t.cg is None:
            uncert[s] = t.sr
            continue
        tubes[s] = t
        kappa[s] = float(np.nanquantile(t.residual_scores(tr), 0.9))
        sources[s] = sorted({ep["task"] for ep in tr if s in set(ep["skill"])})

    # ------------------------------------------------------------ calibration
    if a.calib == "joint":
        alpha_l = {s: a.alpha for s in tubes}
    else:
        # fixed per-primitive budget (registered share); an Eddie label splits its primitive's share
        n_of = defaultdict(int)
        for s in tubes:
            n_of[primitive_of(s)] += 1
        alpha_l = {s: a.alpha * BOOK.primitives[primitive_of(s)].alpha_share / n_of[primitive_of(s)] for s in tubes}

    def calibrate_eps(cal, cal_extra, renorm=False):
        if a.calib == "joint":
            kap = kappa
            if renorm and len(cal) >= 4:
                # few-shot: normalisers re-estimated on half of the new task's episodes, the
                # quantile on the other half (the split keeps the conformal guarantee)
                half = len(cal) // 2
                kap = {}
                for s, t in tubes.items():
                    sc = t.residual_scores(cal[:half])
                    sc = sc[np.isfinite(sc)]
                    kap[s] = float(np.quantile(sc, 0.9)) if len(sc) else kappa[s]
                cal = cal[half:]
            q = conformal_quantile(joint_scores(tubes, kap, cal + cal_extra), a.alpha)
            return {s: q * kap[s] for s in tubes}
        e = {}
        for s, t in tubes.items():
            sc = t.residual_scores(cal_extra if s in extra_labels else cal)
            e[s] = conformal_quantile(sc[np.isfinite(sc)], alpha_l[s])
        return e
    eps_pooled = calibrate_eps(calib, extra_calib)
    if a.transfer_max:                      # worst per-task eps over the training tasks
        per = [calibrate_eps([ep for ep in calib if ep["task"] == t], []) for t in train_tasks]
        eps_pooled = {s: max(e.get(s, 0.0) for e in per if np.isfinite(e.get(s, np.nan))) for s in tubes}
    eps_task, n_cal_task = {}, {}
    for task in test_tasks:
        if a.mondrian:          # task-conditional: the test task's own calibration episodes (models unchanged)
            ct = load(task, CALIB)
            if a.n_calib:
                ct = ct[: a.n_calib]
            eps_task[task] = calibrate_eps(ct, ct if extra_labels else [], renorm=a.renorm)
            n_cal_task[task] = len(ct)
        else:
            eps_task[task] = eps_pooled
            n_cal_task[task] = len(calib)
    eps = eps_pooled

    # ------------------------------------------------------------ test
    out = {"train": train_tasks, "test": test_tasks, "pool": a.pool, "calib": a.calib, "alpha": a.alpha,
           "extra": a.extra, "exclude": sorted(excl), "shrink": a.shrink, "select": a.select, "mondrian": a.mondrian, "transfer_max": a.transfer_max, "renorm": a.renorm, "n_calib": a.n_calib, "block": J, "uncertified": uncert, "tasks": {}, "labels": {}}
    for task, eps_list in test.items():
        eps = eps_task[task]
        rows, comp = [], []
        covered = np.ones(len(eps_list), bool)
        for s, t in tubes.items():
            sc = t.residual_scores(eps_list)
            covered &= ~(np.isfinite(sc) & (sc > eps[s]))
        for ep in eps_list:
            rho = rho_of_ep(ep)
            r = {"D_viol": False, "D_ratio": 0.0}
            for s, t in tubes.items():
                c = check_episode(t, eps[s], rho, ep)
                r["D_viol"] |= c["D_viol"]
                r["D_ratio"] = max(r["D_ratio"], c["D_ratio"])
            rows.append(r)
            comp.append(compose_episode(tubes, eps, lambda s, rho=rho: rho, ep))
        n = len(rows)
        kD = int(sum(r["D_viol"] for r in rows))
        kC = int(sum(c["violated"] for c in comp))
        seq = [len([x for x in ordered_segments(ep["token"]) if x[0]]) for ep in eps_list]
        out["tasks"][task] = {
            "n_test": n, "n_calib": n_cal_task[task], "eddies_per_episode": float(np.median(seq)) if seq else np.nan,
            "coverage": float(covered.mean()) if n else np.nan,
            "D_violations": kD, "D_wilson_upper": wilson_upper(kD, n),
            "D_tightness_median": float(np.median([r["D_ratio"] for r in rows])) if n else np.nan,
            "compose_violations": kC, "compose_wilson_upper": wilson_upper(kC, n),
            "compose_tightness_median": float(np.median([c["max_ratio"] for c in comp])) if n else np.nan,
            "compose_bound_median_m": float(np.median([c["max_bound_m"] for c in comp])) if n else np.nan,
            "chained_median": float(np.median([c["n_tokens"] for c in comp])) if n else np.nan,
            "restarts_median": float(np.median([c["restarts"] for c in comp])) if n else np.nan,
            "uncertified_labels": sorted({s for ep in eps_list for s in ep["skill"] if s and s not in tubes}),
        }
    eps = eps_pooled
    for s, t in tubes.items():
        out["labels"][s] = {"n_train": t.n_train, "shrink_selected": choice.get(s), "tasks": sources[s], "spectral_radius": t.sr, "eps": eps[s],
                            "alpha": alpha_l[s], "slew_tube_m": t.slew_radius(rho0, eps[s]),
                            "noise_tube_m": t.slew_radius(0.0, eps[s]), "rho_star_5cm": t.rho_star(0.05, eps[s])}
    name = a.name or f"{a.pool}_{a.calib}_{'+'.join(train_tasks)}__{'+'.join(test_tasks)}"
    path = ROOT / "pool" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))

    print(f"\n[{name}] pool={a.pool} calib={a.calib} alpha={a.alpha} shrink={a.shrink} block={J}  "
          f"certified labels {len(tubes)}" + (f" (unstable: {', '.join(uncert)})" if uncert else ""))
    print(f"  {'label':26s} {'n':>6s} {'sr':>6s} {'eps':>7s} {'slew tube':>9s} {'lam':>5s}  tasks")
    for s, e in out["labels"].items():
        lam = e["shrink_selected"]
        print(f"  {s:26s} {e['n_train']:6d} {e['spectral_radius']:6.3f} {e['eps']:7.4f} {100 * e['slew_tube_m']:8.1f}cm "
              f"{'-' if lam is None else f'{lam:5.2f}':>5s}  "
              f"{','.join(x.replace('franka_', '') for x in e['tasks'])}")
    print(f"  {'test task':22s} {'ncal':>4s} {'n':>4s} {'eddy':>4s} {'cover':>6s} {'D viol':>7s} {'D tight':>7s} {'C viol':>7s} "
          f"{'C tight':>7s} {'C bound':>8s} {'chain':>6s} {'restart':>7s}  uncertified")
    for t, r in out["tasks"].items():
        print(f"  {t:22s} {r['n_calib']:4d} {r['n_test']:4d} {r['eddies_per_episode']:4.0f} {r['coverage']:6.3f} {r['D_violations']:7d} "
              f"{r['D_tightness_median']:7.2f} {r['compose_violations']:7d} {r['compose_tightness_median']:7.2f} "
              f"{100 * r['compose_bound_median_m']:7.1f}cm {r['chained_median']:6.0f} {r['restarts_median']:7.0f}  "
              f"{','.join(r['uncertified_labels'])}")


if __name__ == "__main__":
    main()
