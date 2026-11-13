#!/usr/bin/env python3
"""Adaptive, self-calibrating certify-or-stop gate: one mechanism for known and new tasks.
A known task starts from its own calibrated certificate (--known); a new task starts from the other
tasks' certificates; both are then updated online from the closed loop.

    python -m eddies.cert.experiments.gate_adaptive --task franka_fmb [--mu 5,7.5,10,15] [--eta 0.5] [--smoke]

No episode of the new task is used before deployment, and nothing is re-calibrated offline.  The
gate starts from whatever the robot already knows and calibrates itself from the closed loop:

  prior    certificates fitted on the other tasks: the Eddie's own, else its primitive's, else one
           global model of the robot (so every Eddie of the new task has a model from step one)
  model    at every hand-over the Eddie's model is re-fitted by ridge regression toward its prior on
           all of its transitions seen so far in the new task (prior weight LAM0 transitions)
  kappa    each Eddie's normaliser: prior, then the 0.9 quantile of its own episode-max residuals
           once it has KAPPA_MIN episodes in the new task
  q        one envelope level, eps_i = q * kappa_i, updated after every episode by online quantile
           tracking  q <- q + eta * (err - alpha),  err = 1 if the episode's residual score exceeded q
           (adaptive conformal inference, Gibbs & Candes 2021; Angelopoulos et al. 2023): for ANY
           sequence of episodes, |mean(err) - alpha| <= (s_max + eta) / (eta T); no exchangeability
           and no per-task calibration set
Gate, using only data up to the current step (receding, one command ahead):
  before sending e_{k+1}, from the measured error eta_k under the current model,
      max_{j <= L} ||S A_i^j (A_i eta_k - G_i (e_{k+1} - e_k))|| + eps_i omega_i(j+1)  <= mu   else STOP
  L = 0 (default): the next command is sent only if the error after it stays within mu while the
  residual stays within eps_i; L > 0 also requires that holding the command for L more steps would.
  STOP also if the measured error itself exceeds mu.  A residual above eps_i does not stop the robot: the episode's envelope is
  raised to the observed score and the prediction re-checked, so the robot stops only when its own
  dynamics can no longer finish the Eddie within mu.
Ground truth: tracking error under REFERENCE certificates fitted on all four tasks (fixed, never used
by the gate).  Reports per mu: completed, stopped, executions whose true error left mu before the gate
stopped them, completion and q along the episode stream.
Writes $EDDIES_RESULTS/adaptive/adaptive_<task>.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from eddies.common.paths import ROOT
from eddies.common.config import load as load_config
from eddies.common.config import blocks, mu_list, mu_str, parse_mu, robots, seeds, span
from eddies.cert.experiments.table1_eddies import embedding, load
from eddies.cert.experiments.gate_fixed import TASKS, build_certificates
from eddies.cert.experiments.transfer import certificates, nominal_plan, prim
from eddies.cert.tube import Tube, conformal_quantile, joint_scores, ordered_segments
from eddies.envs.skillbook import primitive_of

OUT = ROOT / "adaptive"
CFG = load_config("experiments/gate_adaptive")
AG = CFG.adaptive_gate
LAM0 = {k: float(v) for k, v in AG.lam0.items()}   # prior weight [transitions], by prior level
KAPPA_MIN = int(AG.kappa_min_episodes)   # episodes before an Eddie's own normaliser replaces the prior one
H_MAX = int(AG.h_max_steps)              # longest look-ahead [steps]
KAPPA_Q = float(CFG.conformal.kappa_quantile)


def theta_of(t):
    nz = len(t.J)
    return np.hstack([t.A[:nz, :nz], t.B[:nz], t.d[:nz, None]])


def tube_from(theta, ref, horizon):
    """Tube with the same constants as Tube.prepare (S A^j G We^-1, S A^j Wz^-1), batched."""
    nz = len(ref.J)
    A, B, d = theta[:, :nz], theta[:, nz:-1], theta[:, -1]
    t = Tube(ref.skill, ref.J, A, B, d, ref.Wz, ref.We, ref.S, ref.n_train, horizon=int(horizon))
    t.sr = float(max(abs(np.linalg.eigvals(A))))
    if t.sr >= AG.max_spectral_radius:
        return None
    G = np.linalg.solve(np.eye(nz) - A, B)
    P, X = [], ref.S.copy()
    for _ in range(int(horizon) + 1):
        P.append(X)
        X = X @ A
    P = np.array(P)                                            # S A^j, j = 0..horizon
    t.powers_S = P
    t.cg = np.linalg.norm(P[:-1] @ (G / ref.We), 2, axis=(1, 2))
    t.cw = np.linalg.norm(P[:-1] / ref.Wz, 2, axis=(1, 2))
    return t


class OnlineEddie:
    """Prior model + ridge re-fit toward it on the new task's transitions."""

    def __init__(self, name, prior, kappa, horizon, lam):
        self.name, self.prior, self.kappa0, self.lam = name, prior, kappa, lam
        self.theta_p = theta_of(prior)
        self.G = np.zeros((self.theta_p.shape[1],) * 2)
        self.C = np.zeros_like(self.theta_p.T)
        self.horizon = horizon
        self.tube = tube_from(self.theta_p, prior, min(horizon, H_MAX)) or prior
        self.scores = []                       # episode-max residuals (unnormalised) in the new task

    def add(self, z, e):
        J = self.prior.J
        Phi = np.hstack([z[:-1, J], e[:-1], np.ones((len(z) - 1, 1))])
        self.G += Phi.T @ Phi
        self.C += Phi.T @ z[1:, J]

    def refit(self):
        theta = np.linalg.solve(self.G + self.lam * np.eye(len(self.G)), self.C + self.lam * self.theta_p.T).T
        t = tube_from(theta, self.prior, min(self.horizon, H_MAX))
        if t is not None:
            self.tube = t

    def kappa(self):
        k = float(np.quantile(self.scores, KAPPA_Q)) if len(self.scores) >= KAPPA_MIN else 0.0
        return k if k > 0 else self.kappa0


def eta_of(t, z, e):
    return z[t.J] - np.linalg.solve(np.eye(len(t.J)) - t.A, t.B @ e + t.d)


def safe_next(t, t_next, z, e, e_next, eps, H):
    """Gate bound before sending e_next: predicted next error w.r.t. the Eddie that will run it,
    eta_hat = (A_i z + B_i e + d_i) - z*_{i'}(e_next)   (= A_i eta - G_i (e_next - e) inside an Eddie),
    plus eps ||S W^-1/2||; with H > 0 also the tube of holding e_next for H more steps (model i')."""
    zhat = t.A @ z[t.J] + t.B @ e + t.d
    eta_hat = zhat - np.linalg.solve(np.eye(len(t_next.J)) - t_next.A, t_next.B @ e_next + t_next.d)
    H = int(max(0, min(H, len(t_next.cw) - 1, len(t_next.powers_S) - 1)))
    tr = np.linalg.norm(np.einsum("jab,b->ja", t_next.powers_S[:H + 1], eta_hat), axis=1)
    ew = eps * np.cumsum(t_next.cw[:H + 1]) if H > 0 else np.array([eps * float(np.linalg.norm(t.S / t.Wz, 2))])
    return float(np.max(tr + ew))


def run(H, data, mus, alpha, eta_q, n_test, lookahead=0, known=False, tasks=TASKS, ref_train=None, test=None):
    """known=False: the task is new (prior = the other tasks' certificates, envelope level from their
    calibration).  known=True: the same gate started from the task's own certificate (fitted with its
    training episodes, envelope level calibrated on its calibration episodes).  Both then adapt online.
    ref_train: training episodes of the reference (truth) certificates; default all tasks' training episodes.
    test: the episode stream to run on; default data[H]["test"] (e.g. episodes of another physics engine or
    the real robot, with the prior still taken from data)."""
    rest = [t for t in tasks if t != H] if not known else list(tasks)
    train = [ep for t in rest for ep in data[t]["train"]]
    E, kE, P, kP, rho = certificates(train)
    J, W, scale, pos, _ = embedding("eddie", train)
    from eddies.cert.tube import make_tube
    glob = make_tube(train, None, J, W, scale, pos)
    kG = float(np.nanquantile(glob.residual_scores(train), 0.9))
    calE = [ep for t in rest for ep in data[t]["calib"]] if not known else data[H]["calib"]
    q0 = conformal_quantile(joint_scores(E, kE, calE), alpha)
    # reference (truth): certificates on all four tasks, never seen by the gate
    alltrain = ref_train if ref_train is not None else [ep for t in tasks for ep in data[t]["train"]]
    Jr, Wr, sr_, pr, _ = embedding("eddie", alltrain)
    REF, _, _ = build_certificates(alltrain, Jr, Wr, sr_, pr, rho)
    dur = {}
    for ep in train:
        for s, k0, k1 in ordered_segments(ep["skill"]):
            if s:
                dur[s] = max(dur.get(s, 0), k1 - k0 + 1)
                dur[primitive_of(s)] = max(dur.get(primitive_of(s), 0), k1 - k0 + 1)
    test = (data[H]["test"] if test is None else test)[:n_test]
    out = {"task": H, "alpha": alpha, "eta": eta_q, "q0": q0, "rho": rho, "mu_m": mus, "prior_level": {}, "rows": {}}
    for mu in mus:
        eddies, q = {}, q0
        rows = {"would_exceed": [], "completed": [], "stopped": [], "true_exceed": [], "exceed_uncaught": [], "q": [], "err": [],
                "cause": [], "stop_eddie": []}
        for ep in test:
            z, e = ep["z"], ep["e"]
            would = False                              # counterfactual: the whole execution leaves mu
            for s, k0, k1 in ordered_segments(ep["skill"]):
                ref = REF.get(s) if s else None
                if ref is not None:
                    Zs = np.linalg.solve(np.eye(len(ref.J)) - ref.A, (e[k0:k1 + 1] @ ref.B.T + ref.d).T).T
                    would |= bool(np.linalg.norm((z[k0:k1 + 1][:, ref.J] - Zs) @ ref.S.T, axis=1).max() > mu)
            q_ep, score, stop, cause, stop_eddie, exceeded = max(q, 0.0), 0.0, False, "completed", None, False
            segs = [(s_, k0, k1) for s_, k0, k1 in ordered_segments(ep["skill"]) if s_ is not None]

            def get(s_):
                if s_ not in eddies:
                    p = primitive_of(s_)
                    if s_ in E:
                        prior, kap0, lvl = E[s_], kE[s_], "eddie"
                    elif p in P:
                        prior, kap0, lvl = P[p], kP[p], "primitive"
                    else:
                        prior, kap0, lvl = glob, kG, "global"
                    out["prior_level"][s_] = lvl
                    eddies[s_] = OnlineEddie(s_, prior, kap0, 1.2 * dur.get(s_, dur.get(primitive_of(s_), 200)), LAM0[lvl])
                return eddies[s_]

            for n_seg, (s, k0, k1) in enumerate(segs):
                oe = get(s)
                t, kap = oe.tube, oe.kappa()
                nxt = segs[n_seg + 1] if n_seg + 1 < len(segs) and segs[n_seg + 1][1] == k1 + 1 else None
                t_hand = get(nxt[0]).tube if nxt else t          # model of the Eddie that runs step k1 + 1
                ref = REF.get(s)
                res_max = 0.0
                last = k1 if nxt is None else k1                   # steps k0..k1 of this Eddie
                for k in range(k0, last + 1):
                    if ref is not None:                           # truth: reference tracking error
                        exceeded |= float(np.linalg.norm(ref.S @ eta_of(ref, z[k], e[k]))) > mu
                    eta = eta_of(t, z[k], e[k])
                    has_next = k < k1 or nxt is not None
                    if float(np.linalg.norm(t.S @ eta)) > mu:
                        stop, cause = True, "measured"
                    elif has_next and safe_next(t, t if k < k1 else t_hand, z[k], e[k], e[k + 1], q_ep * kap, lookahead) > mu:
                        stop, cause = True, "predicted"
                    if stop:
                        stop_eddie = s
                        break
                    if has_next:                                   # residual of the transition k -> k+1 (model i)
                        w = z[k + 1, t.J] - (t.A @ z[k, t.J] + t.B @ e[k] + t.d)
                        r = float(np.linalg.norm(t.Wz * w))
                        res_max = max(res_max, r)
                        score = max(score, r / kap)
                        q_ep = max(q_ep, score)                    # widen the envelope for the rest of the episode
                end = k + 1 if stop else k1 + 1
                oe.add(z[k0:end], e[k0:end])
                if res_max > 0:
                    oe.scores.append(res_max)
                oe.horizon = max(oe.horizon, 1.2 * (k1 - k0 + 1))
                oe.refit()
                if stop:
                    break
            err = float(score > q)
            q = q + eta_q * (err - alpha)          # no clipping: |mean(err) - alpha| <= (s_max + eta) / (eta T)
            rows["would_exceed"].append(would)
            rows["completed"].append(not stop)
            rows["stopped"].append(stop)
            rows["true_exceed"].append(bool(exceeded))
            rows["exceed_uncaught"].append(bool(exceeded and not stop))
            rows["q"].append(q)
            rows["err"].append(err)
            rows["cause"].append(cause)
            rows["stop_eddie"].append(stop_eddie)
        n = len(test)
        W_, S_, X_ = (np.array(rows[k]) for k in ("would_exceed", "stopped", "true_exceed"))
        summary = {"n": n, "would_exceed": int(W_.sum()), "caught_before": int((W_ & S_ & ~X_).sum()),
                   "false_stops": int((~W_ & S_).sum()),
                   "completed": int(sum(rows["completed"])), "stopped": int(sum(rows["stopped"])),
                   "true_exceed": int(sum(rows["true_exceed"])), "completed_beyond_mu": int(sum(rows["exceed_uncaught"])),
                   "miscoverage": float(np.mean(rows["err"])),
                   "completed_first_half": int(sum(rows["completed"][: n // 2])),
                   "completed_second_half": int(sum(rows["completed"][n // 2:])),
                   "causes": {c: rows["cause"].count(c) for c in set(rows["cause"])}}
        out["rows"][str(mu)] = {"summary": summary, "trace": rows}
        print(H, "mu", mu, summary, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--mu", default=None, help="tolerances [cm]; default mu_cm (held-out Franka) or mu_cm_group (--group)")
    ap.add_argument("--alpha", type=float, default=CFG.conformal.alpha)
    ap.add_argument("--eta", type=float, default=AG.eta)
    ap.add_argument("--group", default=None, help="robot group of eddies.cert.experiments.embodiments (franka, g1, leap, aero)")
    ap.add_argument("--known", action="store_true", help="start from the task's own calibrated certificate")
    ap.add_argument("--lookahead", type=int, default=AG.lookahead, help="hold horizon after the next command [steps]; 0 = one step")
    ap.add_argument("--smoke", action="store_true", help="smoke.per_split train / calib episodes per task, smoke.n_test test")
    a = ap.parse_args()
    mus = parse_mu(a.mu) if a.mu else mu_list(CFG.mu_cm_group if a.group else CFG.mu_cm)
    sl = int(CFG.smoke.per_split) if a.smoke else None
    tasks = TASKS
    if a.group:                                   # any robot of embodiments (tag sb, 200 test episodes)
        from eddies.cert.experiments.embodiments import GROUPS, load_task
        tasks = GROUPS[a.group]
        data = {t: load_task(t) for t in tasks}
        w = min(ep["z"].shape[1] for d in data.values() for sp in d.values() for ep in sp)
        for d in data.values():
            for sp in d.values():
                for ep in sp:
                    ep["z"], ep["e"] = ep["z"][:, :w], ep["e"][:, :w]
    else:
        data = load(TASKS)
    if sl:
        data = {t: {k: v[:sl] for k, v in d.items()} for t, d in data.items()}
    n_test = int(CFG.smoke.n_test if a.smoke else (CFG.n_test_group if a.group else CFG.n_test))
    res = run(a.task, data, mus, a.alpha, a.eta, n_test, a.lookahead, a.known, tasks)
    res["lookahead"] = a.lookahead
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"adaptive_{a.task}{'_known' if a.known else ''}{'_' + a.group if a.group else ''}{'_smoke' if a.smoke else ''}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
