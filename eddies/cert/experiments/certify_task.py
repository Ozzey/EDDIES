#!/usr/bin/env python3
"""Identify, calibrate and test the tube certificates (Theorems A and D) for one task.

    python -m eddies.cert.experiments.certify_task --task franka_drawer [--alpha 0.1]

Splits (seeds): train 1000-1199 (models, normalisers kappa), calibration 1200-1399
(conformal quantile), test 2000-2199 (+2200-2599 if present).  Models are fitted on
nominal + perturbed training episodes; eps is calibrated per condition; a shift test
calibrates on nominal and tests on perturbed.  Single process, light memory.
Writes $EDDIES_RESULTS/cert/<task>.json.  Config: eddies/configs/experiments/certify_task.yaml.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from eddies.common.paths import ROOT
from eddies.common.config import load as load_config
from eddies.common.config import blocks, mu_list, mu_str, parse_mu, robots, seeds, span
from eddies.cert.data import anchors, load_split
from eddies.cert.tube import check_episode, conformal_quantile, joint_scores, make_tube

CFG = load_config("experiments/certify_task")
TRAIN, CALIB = seeds(CFG.splits.train), seeds(CFG.splits.calib)
TEST = seeds(CFG.splits.test) + seeds(CFG.splits.test_extra)
BLOCKS = blocks(CFG)                       # family -> (certified pose block, position rows), configs/robots
MIN_STD = float(CFG.identification.min_std)   # coordinates that never move in the task are not identifiable
KAPPA_Q = float(CFG.conformal.kappa_quantile)


def family(task):
    for k, pre in (("franka", "franka"), ("g1", "g1"), ("hand", ("leap", "aero"))):
        if task.startswith(pre):
            return k
    raise KeyError(task)


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return np.nan
    p = k / n
    den = 1 + z * z / n
    return float((p + z * z / (2 * n) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den)


def calibrate(tubes, kappa, calib, alpha):
    q = conformal_quantile(joint_scores(tubes, kappa, calib), alpha)
    return {s: q * kappa[s] for s in tubes}, q


def evaluate(tubes, eps_hat, test, rho, anc=None):
    rows = []
    for ep in test:
        r = {"seed": ep["seed"], "D_viol": False, "A_viol": False, "slew_viol": False, "outside_ball": False,
             "D_ratio": 0.0, "A_ratio": 0.0}
        for s, t in tubes.items():
            a = anc.get(ep["seed"], {}).get(s) if anc else None
            c = check_episode(t, eps_hat[s], rho, ep, *(a if a else (None, None)))
            for k in ("D_viol", "A_viol", "slew_viol", "outside_ball"):
                r[k] |= c[k]
            r["D_ratio"] = max(r["D_ratio"], c["D_ratio"])
            r["A_ratio"] = max(r["A_ratio"], c["A_ratio"])
        rows.append(r)
    return rows


def summ(rows, key):
    n, k = len(rows), int(sum(r[key] for r in rows))
    return {"n": n, "violations": k, "rate": k / n if n else np.nan, "wilson_upper": wilson_upper(k, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--alpha", type=float, default=CFG.conformal.alpha)
    ap.add_argument("--tag", default=CFG.tag)
    ap.add_argument("--no-anchors", action="store_true")
    ap.add_argument("--conditions", default=",".join(CFG.conditions),
                    help="comma list; a task collected nominal-only passes --conditions nominal")
    a = ap.parse_args()
    task, alpha = a.task, a.alpha
    conds = tuple(a.conditions.split(","))
    J0, pos = BLOCKS[family(task)]
    data = {(c, n): load_split(a.tag, c, task, seeds) for c in conds
            for n, seeds in (("train", TRAIN), ("calib", CALIB), ("test", TEST))}
    train = [ep for c in conds for ep in data[c, "train"]]
    man = train[0]["manifest"]
    emb = man["embedding"]
    W = np.asarray(emb["metric_diag"], float)
    scale = (np.asarray(emb["phys_upper"]) - np.asarray(emb["phys_lower"])) / \
            (np.asarray(emb["norm_upper"]) - np.asarray(emb["norm_lower"]))
    rho = float(man["method_info"]["rho_step"])
    Z = np.concatenate([ep["z"] for ep in train])
    J = [j for j in J0 if Z[:, j].std() > MIN_STD]
    skills = [s for s in dict.fromkeys(x for ep in train for x in ep["skill"]) if s]
    tubes, kappa, uncert = {}, {}, {}
    for s in skills:
        t = make_tube(train, s, J, W, scale, pos)
        if t is None:
            continue
        if t.cg is None:
            uncert[s] = t.sr
            continue
        tubes[s] = t
        kappa[s] = float(np.nanquantile(t.residual_scores(train), KAPPA_Q))
    anc = {}
    if not a.no_anchors:
        for c in conds:
            anc[c], _ = anchors(task, [ep["seed"] for ep in data[c, "test"]], c == "perturbed", workers=1)
    out = {"task": task, "alpha": alpha, "rho_step": rho, "block": J, "uncertified": uncert,
           "conditions": {}, "skills": {}}
    for c in conds:
        eps_hat, q = calibrate(tubes, kappa, data[c, "calib"], alpha)
        rows = evaluate(tubes, eps_hat, data[c, "test"], rho, anc.get(c))
        jt = joint_scores(tubes, kappa, data[c, "test"])
        out["conditions"][c] = {"n_calib": len(data[c, "calib"]), "n_test": len(rows), "q": q,
                                "residual_coverage": float(np.mean(jt <= q)),
                                "theorem_D": summ(rows, "D_viol"), "theorem_A": summ(rows, "A_viol"),
                                "outside_ball_episodes": int(sum(r["outside_ball"] for r in rows)),
                                "slew_violation_episodes": int(sum(r["slew_viol"] for r in rows)),
                                "D_tightness_median": float(np.median([r["D_ratio"] for r in rows])),
                                "eps": eps_hat}
    if "perturbed" in conds:
        eps_nom, q_nom = calibrate(tubes, kappa, data["nominal", "calib"], alpha)
        rows = evaluate(tubes, eps_nom, data["perturbed", "test"], rho)
        jt = joint_scores(tubes, kappa, data["perturbed", "test"])
        out["shift_nominal_to_perturbed"] = {"residual_coverage": float(np.mean(jt <= q_nom)),
                                             "theorem_D": summ(rows, "D_viol")}
    for s, t in tubes.items():
        deltas = [v[s][1] for v in anc.get("nominal", {}).values() if s in v]
        delta = float(np.median(deltas)) if deltas else np.nan
        ent = {"n_train": t.n_train, "spectral_radius": t.sr, "delta_hand": delta}
        for tag_ in conds:
            e = out["conditions"][tag_]["eps"][s]
            ent[tag_] = {"eps": e, "slew_tube_m": t.slew_radius(rho, e),
                         "noise_tube_m": t.slew_radius(0.0, e),
                         "trust_tube_m": t.trust_radius(delta, e) if np.isfinite(delta) else np.nan,
                         "rho_star": {f"{int(mu * 100)}cm": t.rho_star(mu, e) for mu in CFG.rho_star_mu_m},
                         "delta_star_10cm": t.delta_star(0.10, e)}
        out["skills"][s] = ent
    path = ROOT / "cert" / f"{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\n{task}: rho={rho} alpha={alpha} block={J}  certified {len(tubes)}/{len(tubes) + len(uncert)} skills"
          + (f"  (unstable: {', '.join(uncert)})" if uncert else ""))
    for c, r in out["conditions"].items():
        print(f"  {c:9s} calib={r['n_calib']} test={r['n_test']} coverage={r['residual_coverage']:.3f}"
              f"  D viol={r['theorem_D']['violations']} A viol={r['theorem_A']['violations']}"
              f"  outside-ball={r['outside_ball_episodes']}  tightness={r['D_tightness_median']:.2f}")
    if "shift_nominal_to_perturbed" in out:
        sh = out["shift_nominal_to_perturbed"]
        print(f"  shift nom->pert: coverage={sh['residual_coverage']:.3f}  D viol={sh['theorem_D']['violations']}/{sh['theorem_D']['n']}")
    last = conds[-1]
    print(f"  {'skill':18s} {'sr':>5s} {'slew tube':>9s} {'noise':>6s} {'trust tube':>10s} {'delta':>6s} {'rho*5cm':>8s} {'d*10cm':>7s}  ({last} eps, metres->cm)")
    for s, e in out["skills"].items():
        p = e[last]
        print(f"  {s:18s} {e['spectral_radius']:5.3f} {100 * p['slew_tube_m']:9.1f} {100 * p['noise_tube_m']:6.1f}"
              f" {100 * p['trust_tube_m']:10.1f} {e['delta_hand']:6.3f} {p['rho_star']['5cm']:8.4f} {p['delta_star_10cm']:7.4f}")


if __name__ == "__main__":
    main()
