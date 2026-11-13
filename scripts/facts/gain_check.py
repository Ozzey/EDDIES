"""Per-Eddie induced gain a_i=||W^1/2 A W^-1/2|| vs spectral radius, using the paper's code paths."""
import json, sys, time
import numpy as np
from scipy.linalg import solve_discrete_lyapunov, sqrtm
from eddies.cert.experiments.embodiments import GROUPS, TRAIN
from eddies.cert.data import load_split
from eddies.cert.experiments.table1_eddies import embedding
from eddies.cert.experiments.gate_fixed import build_certificates
from eddies.cert.tube import ordered_segments

OUT = sys.argv[1]
res = {}
for g, tasks in GROUPS.items():
    t0 = time.time()
    train = [ep for t in tasks for ep in load_split("sb", "nominal", t, TRAIN)]
    if not train:
        res[g] = {"tasks": tasks, "missing": "no local training episodes"}; print(g, "no data"); continue
    w = min(ep["z"].shape[1] for ep in train)
    for ep in train:
        ep["z"], ep["e"] = ep["z"][:, :w], ep["e"][:, :w]
    J, W, scale, pos, rho = embedding("eddie", train)
    tubes, kappa, ref = build_certificates(train, J, W, scale, pos, rho)
    # segment lengths per Eddie on train episodes
    seglen = {}
    for ep in train:
        for s, k0, k1 in ordered_segments(ep["skill"]):
            if s: seglen.setdefault(s, []).append(k1 - k0 + 1)
    rows = {}
    for s, t in tubes.items():
        A = t.A; Wh = np.diag(t.Wz); iWh = np.diag(1 / t.Wz)
        a = float(np.linalg.norm(Wh @ A @ iWh, 2))
        sr = float(max(abs(np.linalg.eigvals(A))))
        # Lyapunov metric in W-coordinates: Abar = W^1/2 A W^-1/2, Abar^T P Abar - P = -I
        Ab = Wh @ A @ iWh
        P = solve_discrete_lyapunov(Ab.T, np.eye(len(A)))
        Ph = np.real(sqrtm(P)); iPh = np.linalg.inv(Ph)
        a_lyap = float(np.linalg.norm(Ph @ Ab @ iPh, 2))
        condP = float(np.linalg.cond(P))
        cA = t.cA
        jfirst = int(np.argmax(cA[1:] < 1) + 1) if np.any(cA[1:] < 1) else None
        L = np.array(seglen.get(s, [0]))
        Lmed = int(np.median(L))
        # per-segment factor multiplying b0 through one Eddie + its outgoing hand-over: a * cA[L-1]
        fac = [float(a * cA[min(l, len(cA)) - 1]) for l in L if l >= 2]
        rows[s] = {"a_i": a, "a_i_from_cA1": float(cA[1]), "rho_A": sr, "n_dim": len(A),
                   "peak_cA": float(cA.max()), "argmax_cA": int(cA.argmax()), "first_j_cA_lt_1": jfirst,
                   "cA_at_median_len": float(cA[min(Lmed, len(cA)) - 1]) if Lmed >= 1 else None,
                   "seg_len_min_med_max": [int(L.min()), Lmed, int(L.max())],
                   "segment_factor_a_times_cA_Lminus1_med_max": [float(np.median(fac)), float(np.max(fac))] if fac else None,
                   "a_lyap_I": a_lyap, "cond_P_lyap": condP, "n_train": int(t.n_train)}
    res[g] = {"tasks": tasks, "J": [int(j) for j in J], "rho_slew": rho, "reference": ref, "eddies": rows,
              "load_fit_seconds": time.time() - t0}
    print(g, len(rows), "eddies", round(time.time() - t0, 1), "s", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
