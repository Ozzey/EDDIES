#!/usr/bin/env python3
"""Light probe: replicate build_certificates on Franka sb training data and log choices.

    python scripts/facts/shrink_probe.py [--out results/facts/shrink_probe.json]

Needs the sb episodes (datasets/README.md).
"""
import argparse
import json
import time
from collections import Counter

import numpy as np

from eddies.cert.experiments.table1_eddies import embedding
from eddies.cert.experiments.gate_fixed import TASKS
from eddies.cert.experiments.certify_task import TRAIN
from eddies.cert.data import load_split
from eddies.cert.ident import fit, _transitions
from eddies.cert.tube import make_tube


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/facts/shrink_probe.json")
    a = ap.parse_args()

    t0 = time.time()
    train = [ep for t in TASKS for ep in load_split("sb", "nominal", t, TRAIN)]
    print("episodes", len(train), "load s", round(time.time() - t0, 1), flush=True)
    J, W, scale, pos, rho = embedding("eddie", train)           # raises a clear error without the episodes
    Z = np.concatenate([ep["z"] for ep in train]); E = np.concatenate([ep["e"] for ep in train])
    print("z dim", Z.shape[1], "e dim", E.shape[1], "J", J, "rho", rho)
    print("scale", scale.tolist())
    print("Z std", np.round(Z.std(0), 4).tolist())
    print("E std", np.round(E.std(0), 4).tolist())
    print("z[:,7] force range", float(Z[:, 7].min()), float(Z[:, 7].max()), "e[:,7] range", float(E[:, 7].min()), float(E[:, 7].max()))
    labels = [s for s in dict.fromkeys(x for ep in train for x in ep["skill"]) if s]

    def est(t):
        return t.slew_radius(rho, float(np.nanquantile(t.residual_scores(train), 0.9)))

    own, info = {}, {}
    for s in labels:
        d = _transitions(train, s, np.asarray(J), 1, None)
        n = 0 if d is None else len(d[0])
        m = fit(train, s, np.asarray(J), 1, W)
        sr = float(max(abs(np.linalg.eigvals(m.A)))) if m is not None else None
        cond = None
        if d is not None:
            X, Ee, Y, _ = d
            Phi = np.hstack([X, Ee, np.ones((len(X), 1))])
            cond = float(np.linalg.cond(Phi.T @ Phi))
        t = make_tube(train, s, J, W, scale, pos)
        ok = t is not None and t.cg is not None
        info[s] = {"n": n, "sr_own": sr, "cond_gram": cond, "own_certified": ok}
        if ok and t.n_train >= 5000:
            own[s] = est(t)
    ref = min(own, key=own.get)
    print("candidates n>=5000:", len(own), "ref", ref, "own est", own[ref], flush=True)
    prior = fit(train, ref, np.asarray(J), 1, W).info["Theta"]
    for s in labels:
        best = None; per = {}
        for lam in (0.0, 0.03, 0.3, 3.0):
            t = make_tube(train, s, J, W, scale, pos, prior=prior, shrink=lam)
            if t is None or t.cg is None:
                per[lam] = None if t is None else f"unstable sr={t.sr:.4f}"
                continue
            r = est(t); per[lam] = round(r * 100, 3)
            if best is None or r < best[0]:
                best = (r, lam, t.sr)
        info[s].update({"tube_cm_by_lam": per, "chosen_lam": None if best is None else best[1],
                        "sr_chosen": None if best is None else best[2]})
        print(s, info[s], flush=True)
    print("chosen lam counts", Counter(v["chosen_lam"] for v in info.values()))
    print("total s", round(time.time() - t0, 1))
    with open(a.out, "w") as f:
        json.dump({"ref": ref, "J": J, "info": info}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
