#!/usr/bin/env python3
"""Hand-over certification: funnel-style containment vs the Eddie composition recursion.

    python -m eddies.cert.experiments.containment [--mu 3,4,5,7.5,10] [--alpha 0.1]

For every hand-over i -> i' of the test episodes (nominal speed, seeds 2000-2099) and a tolerance
mu, the next Eddie i' admits an entry set {||W^1/2 eta|| <= r_i'} with
    r_i' = min_j (mu - rho cumG_i'(j) - eps_i' cumW_i'(j)) / cw_i'(j)       (its tube stays within mu).
* Containment (funnel libraries, Majumdar & Tedrake 2017 Def. 2): the terminal funnel of i, a ball of
  the ultimate radius R_i = rho sum cgW_i + eps_i sum cA_i around z*_i(e_end), must lie inside the
  entry set around z*_i'(e_new):  J + R_i <= r_i'.
* Eddie recursion (ours): b'_0 = ||W^1/2 A_i W^-1/2|| b_end + eps_i + J <= r_i', with b_end the bound
  propagated through the chain from the episode-start measurement.
Reports the fraction of hand-overs and of whole episodes (every hand-over) each test certifies, and
checks that the recursion's bound is never violated.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from eddies.common.paths import ROOT
from eddies.common.config import load as load_config
from eddies.common.config import blocks, mu_list, mu_str, parse_mu, robots, seeds, span
CFG = load_config("experiments/containment")
from eddies.cert.data import load_split
from eddies.cert.experiments.certify_task import BLOCKS, CALIB, MIN_STD, TRAIN
from eddies.cert.experiments.gate_fixed import TASKS, TEST, build_certificates
from eddies.cert.tube import conformal_quantile, joint_scores, ordered_segments


def entry_radius(t, rho, eps, mu):
    cg = np.concatenate([[0], np.cumsum(t.cg)])[: len(t.cw)]
    cw = np.concatenate([[0], np.cumsum(t.cw)])[: len(t.cw)]
    r = (mu - rho * cg - eps * cw) / np.maximum(t.cw, 1e-12)
    return max(0.0, float(r.min()))


def zs(t, e):
    return np.linalg.solve(np.eye(len(t.J)) - t.A, t.B @ e + t.d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", default=mu_str(CFG.mu_cm))
    ap.add_argument("--alpha", type=float, default=CFG.conformal.alpha)
    a = ap.parse_args()
    mus = [float(x) / 100 for x in a.mu.split(",")]
    train = [ep for t in TASKS for ep in load_split("sb", "nominal", t, TRAIN)]
    emb = train[0]["manifest"]["embedding"]
    W = np.asarray(emb["metric_diag"], float)
    scale = (np.asarray(emb["phys_upper"]) - np.asarray(emb["phys_lower"])) / \
            (np.asarray(emb["norm_upper"]) - np.asarray(emb["norm_lower"]))
    J0, pos = BLOCKS["franka"]
    Z = np.concatenate([ep["z"] for ep in train])
    J = [j for j in J0 if Z[:, j].std() > MIN_STD]
    rho = float(train[0]["manifest"]["method_info"]["rho_step"])
    tubes, kappa, _ = build_certificates(train, J, W, scale, pos, rho)
    out = {"alpha": a.alpha, "mu_m": mus, "tasks": {}}
    print(f"{'task':22s} {'mu':>5s} {'hand-overs':>10s} {'contain':>8s} {'ours':>6s} {'ep contain':>10s} {'ep ours':>8s} {'ours viol':>9s}")
    for task in TASKS:
        cal = load_split("sb", "nominal", task, CALIB)
        q = conformal_quantile(joint_scores(tubes, kappa, cal), a.alpha)
        eps = {s: q * kappa[s] for s in tubes}
        test = load_split("sb", "nominal", task, TEST)
        rows = []
        for mu in mus:
            n_h = n_c = n_o = 0
            ep_c = ep_o = viol = 0
            for ep in test:
                z, e = ep["z"], ep["e"]
                segs = [x for x in ordered_segments(ep["skill"]) if x[0]]
                prev, all_c, all_o = None, True, True
                for s, k0, k1 in segs:
                    t = tubes.get(s)
                    if t is None:
                        all_c = all_o = False
                        prev = None
                        continue
                    L = k1 - k0 + 1
                    if prev is None:
                        b0 = float(np.linalg.norm(t.Wz * (z[k0, t.J] - zs(t, e[k0]))))
                    else:
                        tp, b_end, eps_p, e_end = prev
                        Jm = float(np.linalg.norm(t.Wz * (zs(tp, e_end) - zs(t, e[k0]))))
                        r_next = entry_radius(t, rho, eps[s], mu)
                        R_prev = rho * tp.cgW.sum() + eps_p * tp.cA.sum()
                        b0 = float(tp.cA[1] * b_end + eps_p + Jm)
                        n_h += 1
                        c_ok, o_ok = Jm + R_prev <= r_next, b0 <= r_next
                        n_c += c_ok
                        n_o += o_ok
                        all_c &= c_ok
                        all_o &= o_ok
                    cumgW = np.concatenate([[0], np.cumsum(t.cgW)])[:L]
                    cumA = np.concatenate([[0], np.cumsum(t.cA)])[:L]
                    b_traj = t.cA[:L] * b0 + rho * cumgW + eps[s] * cumA
                    Zs = np.linalg.solve(np.eye(len(t.J)) - t.A, (e[k0:k1 + 1] @ t.B.T + t.d).T).T
                    actual = np.linalg.norm((z[k0:k1 + 1][:, t.J] - Zs) * t.Wz, axis=1)
                    viol += bool(np.any(actual > b_traj * (1 + 1e-9) + 1e-12))
                    prev = (t, float(b_traj[-1]), eps[s], e[k1])
                ep_c += all_c
                ep_o += all_o
            rows.append({"mu_m": mu, "handovers": n_h, "containment": n_c, "recursion": n_o,
                         "episodes": len(test), "episodes_containment": ep_c, "episodes_recursion": ep_o,
                         "recursion_bound_violations": viol})
            print(f"{task:22s} {100 * mu:5.1f} {n_h:10d} {100 * n_c / max(n_h, 1):7.1f}% {100 * n_o / max(n_h, 1):5.1f}% "
                  f"{100 * ep_c / max(len(test), 1):9.1f}% {100 * ep_o / max(len(test), 1):7.1f}% {viol:9d}")
        out["tasks"][task] = rows
    path = ROOT / "gate" / "containment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
