#!/usr/bin/env python3
"""Equilibrium offset vs. command.  Read-only analysis; writes one JSON.

For every certified Eddie (build_certificates exactly as gate_fixed / embodiments) on the test
seeds 2000-2199 (tag sb, nominal), at every executed step k of the Eddie:
  off_k   = ||S (z*_i(e_k) - e_k[J])||      equilibrium offset from the commanded pose [m]
  cmd_k   = ||S (z_k - e_k[J])||            tracking error to the command [m]
  eta_k   = ||S (z_k - z*_i(e_k))||         certified (model-relative) error [m]
and the Eddie's ultimate tube U_i = slew_radius(rho, q_task * kappa_i) per task.
    python eq_offset.py --robots franka[,g1,leap,aero] --out file.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from eddies.cert.data import load_split
from eddies.cert.ident import fit
from eddies.cert.experiments.certify_task import BLOCKS, MIN_STD
from eddies.cert.experiments.gate_fixed import build_certificates
from eddies.cert.tube import conformal_quantile, joint_scores, ordered_segments

TRAIN, CALIB, TEST = range(1000, 1200), range(1200, 1400), range(2000, 2200)
GROUPS = {"franka": ["franka_fmb", "franka_peg_insertion", "franka_cube_stack", "franka_drawer"],
          "g1": ["g1_peg_insertion_sb", "g1_drawer_sb"],
          "leap": ["leap_wheel_sb", "leap_peg_tighten_sb"], "aero": ["aero_wheel_sb", "aero_peg_tighten_sb"]}


def embedding(train):
    emb = train[0]["manifest"]["embedding"]
    W = np.asarray(emb["metric_diag"], float)
    scale = (np.asarray(emb["phys_upper"]) - np.asarray(emb["phys_lower"])) / \
            (np.asarray(emb["norm_upper"]) - np.asarray(emb["norm_lower"]))
    J0, pos = BLOCKS["franka"]
    Z = np.concatenate([ep["z"] for ep in train])
    J = [j for j in J0 if Z[:, j].std() > MIN_STD]
    rho = float(train[0]["manifest"]["method_info"]["rho_step"])
    return J, W, scale, pos, rho, emb


def zstar(t, E):
    return np.linalg.solve(np.eye(len(t.J)) - t.A, (E @ t.B.T + t.d).T).T


def q(x, p):
    return float(np.quantile(x, p)) if len(x) else None


def stats(x):
    x = np.asarray(x)
    return {"n": int(len(x)), "med_cm": 100 * q(x, .5), "p95_cm": 100 * q(x, .95), "max_cm": 100 * float(x.max())} if len(x) else None


def run_group(g, tasks):
    data = {t: {sp: load_split("sb", "nominal", t, seeds) for sp, seeds in
                (("train", TRAIN), ("calib", CALIB), ("test", TEST))} for t in tasks}
    eps_all = [ep for t in tasks for sp in data[t].values() for ep in sp]
    w = min(ep["z"].shape[1] for ep in eps_all)
    for ep in eps_all:
        ep["z"], ep["e"] = ep["z"][:, :w], ep["e"][:, :w]
    for t in tasks:
        print(g, t, {k: len(v) for k, v in data[t].items()}, flush=True)
    train = [ep for t in tasks for ep in data[t]["train"]]
    J, W, scale, pos, rho, emb = embedding(train)
    print(g, "J", J, "names", emb.get("names"), "rho", rho, flush=True)
    tubes, kappa, ref = build_certificates(train, J, W, scale, pos, rho)
    # own-fit (lambda = 0) models for comparison of the equilibrium
    own = {s: fit(train, s, np.asarray(J), 1, W, prior=None, shrink=0.0) for s in tubes}
    out = {"J": J, "rho": rho, "ref": ref, "names": emb.get("names"), "eddies": {}, "tasks": {}}
    acc = {s: {"off": [], "cmd": [], "eta": [], "over": [], "U": [], "off_own": []} for s in tubes}
    for t in tasks:
        qq = conformal_quantile(joint_scores(tubes, kappa, data[t]["calib"]), 0.1)
        U = {s: tubes[s].slew_radius(rho, qq * kappa[s]) for s in tubes}
        out["tasks"][t] = {"q": qq}
        for ep in data[t]["test"]:
            z, e = ep["z"], ep["e"]
            for s, k0, k1 in ordered_segments(ep["skill"]):
                if s is None or s not in tubes:
                    continue
                tb = tubes[s]
                Z, E = z[k0:k1 + 1][:, tb.J], e[k0:k1 + 1]
                Zs = zstar(tb, E)
                Ep = E[:, tb.J]
                off = np.linalg.norm((Zs - Ep) @ tb.S.T, axis=1)
                cmd = np.linalg.norm((Z - Ep) @ tb.S.T, axis=1)
                eta = np.linalg.norm((Z - Zs) @ tb.S.T, axis=1)
                a = acc[s]
                a["off"].append(off); a["cmd"].append(cmd); a["eta"].append(eta)
                a["over"].append(cmd > U[s]); a["U"].append(np.full(len(off), U[s]))
                m = own[s]
                if m is not None:
                    Zo = np.linalg.solve(np.eye(len(J)) - m.A, (E @ m.B[:len(J)].T + m.d).T).T
                    a["off_own"].append(np.linalg.norm((Zo - Ep) @ tb.S.T, axis=1))
    # a-priori offset bound: max equilibrium offset over the commands of the training + calibration episodes
    O = {}
    for s_ in tubes:
        tb = tubes[s_]
        Es = [ep["e"][k0:k1 + 1] for t in tasks for sp in ("train", "calib") for ep in data[t][sp]
              for s2, k0, k1 in ordered_segments(ep["skill"]) if s2 == s_]
        if Es:
            E = np.concatenate(Es)
            O[s_] = float(np.linalg.norm((zstar(tb, E) - E[:, tb.J]) @ tb.S.T, axis=1).max())
    for s_, a in acc.items():
        a["over2"] = [c > u + O.get(s_, np.inf) for c, u in zip(a["cmd"], a["U"])]
    pooled = {k: [] for k in ("off", "cmd", "eta", "over", "over2")}
    for s, a in acc.items():
        if not a["off"]:
            continue
        cat = {k: np.concatenate(v) for k, v in a.items() if v}
        for k in pooled:
            pooled[k].append(cat[k])
        tb = tubes[s]
        own_sr = float(max(abs(np.linalg.eigvals(own[s].A)))) if own[s] is not None else None
        out["eddies"][s] = {"lambda": tb.shrink, "n_train": tb.n_train, "sr": tb.sr, "own_sr": own_sr,
                            "U_cm": [100 * float(np.min(cat["U"])), 100 * float(np.max(cat["U"]))],
                            "off": stats(cat["off"]), "cmd": stats(cat["cmd"]), "eta": stats(cat["eta"]),
                            "frac_cmd_over_U": float(cat["over"].mean()),
                            "frac_eta_over_U": float((cat["eta"] > cat["U"]).mean()),
                            "O_traincal_cm": 100 * O.get(s, np.nan), "frac_cmd_over_U_plus_O": float(cat["over2"].mean()),
                            "off_own": stats(cat["off_own"]) if "off_own" in cat else None}
    # eddies certified but never executed on test
    out["not_executed"] = [s for s, a in acc.items() if not a["off"]]
    P = {k: np.concatenate(v) for k, v in pooled.items()}
    out["pooled"] = {"off": stats(P["off"]), "cmd": stats(P["cmd"]), "eta": stats(P["eta"]),
                     "frac_cmd_over_U": float(P["over"].mean()), "frac_cmd_over_U_plus_O": float(P["over2"].mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", default="franka")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for g in a.robots.split(","):
        res[g] = run_group(g, GROUPS[g])
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=float)
        r = res[g]
        print(f"\n== {g}  ref {r['ref']}  pooled off {r['pooled']['off']}  cmd {r['pooled']['cmd']}  eta {r['pooled']['eta']}"
              f"  cmd>U {100 * r['pooled']['frac_cmd_over_U']:.2f}%  cmd>U+O {100 * r['pooled']['frac_cmd_over_U_plus_O']:.3f}%")
        print(f"{'eddie':28s} {'lam':>5s} {'n':>6s} {'U cm':>11s} | {'off med':>7s} {'p95':>6s} {'max':>6s} | "
              f"{'cmd med':>7s} {'p95':>6s} {'max':>6s} | {'eta med':>7s} {'p95':>6s} | {'cmd>U':>6s} {'eta>U':>6s} | own off med/p95")
        for s, x in sorted(r["eddies"].items()):
            oo = x["off_own"]
            print(f"{s:28s} {x['lambda']:5g} {x['n_train']:6d} {x['U_cm'][0]:5.2f}-{x['U_cm'][1]:5.2f} | "
                  f"{x['off']['med_cm']:7.2f} {x['off']['p95_cm']:6.2f} {x['off']['max_cm']:6.2f} | "
                  f"{x['cmd']['med_cm']:7.2f} {x['cmd']['p95_cm']:6.2f} {x['cmd']['max_cm']:6.2f} | "
                  f"{x['eta']['med_cm']:7.2f} {x['eta']['p95_cm']:6.2f} | {100 * x['frac_cmd_over_U']:5.1f}% {100 * x['frac_eta_over_U']:5.1f}% | "
                  + (f"{oo['med_cm']:.2f}/{oo['p95_cm']:.2f}" if oo else "-") + f"  n_test={x['off']['n']}  O={x['O_traincal_cm']:.2f} cmd>U+O {100 * x['frac_cmd_over_U_plus_O']:.2f}%", flush=True)
        print("not executed on test:", r["not_executed"], flush=True)


if __name__ == "__main__":
    main()
