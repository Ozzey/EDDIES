#!/usr/bin/env python3
"""Per-step data for the composition and calibration figures (Franka, MuJoCo, tag sb).

    python -m eddies.cert.experiments.trace_figs [--out DIR]   # writes $EDDIES_RESULTS/figs_data/trace_fmb.json

Certificates as in gate_fixed (all four Franka tasks, train 1000-1199), envelope calibrated on the FMB
calibration episodes (1200-1399).  Saves:
  scores      FMB calibration and test episode scores, and the conformal quantile q
  traces      for FMB test seeds 2000-2004: per step, the measured position error ||S eta_k|| under the
              running Eddie's model, the composed position bound of Theorem 1 (chain started from the
              measured error at the first step only), the running Eddie, and the hand-over steps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eddies.common.paths import ROOT
from eddies.common.config import load as load_config
from eddies.common.config import blocks, mu_list, mu_str, parse_mu, robots, seeds, span
CFG = load_config("experiments/trace_figs")
from eddies.cert.experiments.table1_eddies import embedding, load
from eddies.cert.experiments.gate_fixed import TASKS, build_certificates
from eddies.cert.tube import conformal_quantile, joint_scores, ordered_segments

OUT = ROOT / "figs_data"


def trace(tubes, eps, rho, ep):
    z, e = ep["z"], ep["e"]
    out = {"k": [], "err": [], "bound": [], "eddie": [], "handover": [], "restart": []}
    prev = None
    zs = lambda tt, ee: np.linalg.solve(np.eye(len(tt.J)) - tt.A, tt.B @ ee + tt.d)
    for s, k0, k1 in ordered_segments(ep["skill"]):
        t = tubes.get(s) if s is not None else None
        if t is None or k1 - k0 + 1 < 2 or k1 - k0 + 1 > t.horizon:
            if s is not None:
                out["restart"].append(int(k0))
            prev = None
            continue
        L = k1 - k0 + 1
        Z, E = z[k0:k1 + 1][:, t.J], e[k0:k1 + 1]
        if prev is None:
            b0 = float(np.linalg.norm(t.Wz * (Z[0] - zs(t, E[0]))))
        else:
            tp, b_end, eps_p, e_end = prev
            J = float(np.linalg.norm(t.Wz * (zs(tp, e_end) - zs(t, E[0]))))
            b0 = float(tp.cA[1] * b_end + eps_p + J)
            out["handover"].append(int(k0))
        cumg = np.concatenate([[0], np.cumsum(t.cg)])[:L]
        cumw = np.concatenate([[0], np.cumsum(t.cw)])[:L]
        cumgW = np.concatenate([[0], np.cumsum(t.cgW)])[:L]
        cumA = np.concatenate([[0], np.cumsum(t.cA)])[:L]
        pos = t.cw[:L] * b0 + rho * cumg + eps[s] * cumw
        btraj = t.cA[:L] * b0 + rho * cumgW + eps[s] * cumA
        Zs = np.linalg.solve(np.eye(len(t.J)) - t.A, (E @ t.B.T + t.d).T).T
        err = np.linalg.norm((Z - Zs) @ t.S.T, axis=1)
        out["k"] += list(range(k0, k1 + 1))
        out["err"] += [float(x) for x in err]
        out["bound"] += [float(x) for x in pos]
        out["eddie"] += [s] * L
        prev = (t, float(btraj[-1]), eps[s], E[-1])
    return out


def main():
    ap = argparse.ArgumentParser(description="per-step traces for Fig. 4 (Franka, MuJoCo, tag sb)")
    ap.add_argument("--out", default=str(OUT), help="output directory (default $EDDIES_RESULTS/figs_data)")
    a = ap.parse_args()
    out = Path(a.out)
    data = load(TASKS)                                   # needs the sb episodes (datasets/README.md)
    train = [ep for t in TASKS for ep in data[t]["train"]]
    J, W, scale, pos, rho = embedding("eddie", train)
    tubes, kappa, _ = build_certificates(train, J, W, scale, pos, rho)
    cal, test = data["franka_fmb"]["calib"], data["franka_fmb"]["test"]
    s_cal = joint_scores(tubes, kappa, cal)
    s_test = joint_scores(tubes, kappa, test)
    q = conformal_quantile(s_cal, 0.1)
    eps = {s: q * kappa[s] for s in tubes}
    res = {"q": q, "alpha": 0.1, "rho": rho, "scores_cal": [float(x) for x in s_cal], "scores_test": [float(x) for x in s_test],
           "traces": {int(ep["seed"]): trace(tubes, eps, rho, ep) for ep in test[:5]}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace_fmb.json").write_text(json.dumps(res))
    print("q", q, "coverage test", float(np.mean(np.asarray(s_test) <= q)), "wrote", out / "trace_fmb.json")


if __name__ == "__main__":
    main()
