"""Propagated (convolution) tube certificates for an identified skill model.

Model (per skill, first order, certified block J of the measured embedding):
    z_{k+1} = A z_k + B e_k + d + w_k,     ||W_z^{1/2} w_k|| <= eps   (conformal, episode level)

Slew certificate (Theorem D, propagated form).  With eta_k = z_k - z*(e_k),
z*(e) = (I - A)^{-1}(B e + d), G = (I - A)^{-1} B and ||W_e^{1/2}(e_{k+1} - e_k)|| <= rho:
    eta_k = A^k eta_0 + sum_{j<k} A^j (-G de_{k-j} + w_{k-1-j})
    ||S eta_k|| <= ||S A^k eta_0|| + rho * sum_{j<k} ||S A^j G W_e^{-1/2}|| + eps * sum_{j<k} ||S A^j W_z^{-1/2}||.
Trust-radius certificate (Theorem A, propagated form).  With c = z*(ebar), ||W_e^{1/2}(e_k - ebar)|| <= delta:
    ||S (z_k - c)|| <= ||S A^k (z_0 - c)|| + delta * sum_{j<k} ||S A^j B W_e^{-1/2}|| + eps * sum_{j<k} ||S A^j W_z^{-1/2}||.
S maps the block to metres (position coordinates).  Both follow from the triangle inequality
applied to the exact solution; the one-step bound (g rho + eps)/(1 - a) of the paper's
Theorem 5 is a weaker upper bound of the same quantity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from eddies.cert.ident import _transitions, fit
from eddies.common.config import load

IDENT = load().identification



@dataclass
class Tube:
    skill: str
    J: np.ndarray
    A: np.ndarray
    B: np.ndarray
    d: np.ndarray
    Wz: np.ndarray            # sqrt weights of the block
    We: np.ndarray            # sqrt weights of the command
    S: np.ndarray             # block -> metres (position rows)
    n_train: int
    horizon: int = int(IDENT.horizon)
    sr: float = np.nan
    cg: np.ndarray = field(default=None, repr=False)   # ||S A^j G We^-1||, j = 0..horizon-1
    cb: np.ndarray = field(default=None, repr=False)   # ||S A^j B We^-1||
    cw: np.ndarray = field(default=None, repr=False)   # ||S A^j Wz^-1||
    powers_S: np.ndarray = field(default=None, repr=False)  # S A^j

    def prepare(self) -> bool:
        n = len(self.J)
        self.sr = float(max(abs(np.linalg.eigvals(self.A))))
        if self.sr >= IDENT.max_spectral_radius:
            return False
        self.G = np.linalg.solve(np.eye(n) - self.A, self.B)
        SAj = self.S.copy()
        WAj = np.diag(self.Wz)                       # W_z^{1/2} A^j (full block, for composition)
        cg, cb, cw, P, cA, cgW = [], [], [], [], [], []
        iWe, iWz = np.diag(1 / self.We), np.diag(1 / self.Wz)
        for _ in range(self.horizon):
            P.append(SAj)
            cg.append(np.linalg.norm(SAj @ self.G @ iWe, 2))
            cb.append(np.linalg.norm(SAj @ self.B @ iWe, 2))
            cw.append(np.linalg.norm(SAj @ iWz, 2))
            cA.append(np.linalg.norm(WAj @ iWz, 2))
            cgW.append(np.linalg.norm(WAj @ self.G @ iWe, 2))
            SAj = SAj @ self.A
            WAj = WAj @ self.A
        self.cg, self.cb, self.cw = np.array(cg), np.array(cb), np.array(cw)
        self.cA, self.cgW = np.array(cA), np.array(cgW)
        self.powers_S = np.array(P)
        return True

    # ultimate (k -> infinity) radii in metres
    def slew_radius(self, rho, eps):
        return float(rho * self.cg.sum() + eps * self.cw.sum())

    def trust_radius(self, delta, eps):
        return float(delta * self.cb.sum() + eps * self.cw.sum())

    def rho_star(self, mu, eps):
        """Largest certified slew bound for a tracking tolerance mu [m]."""
        return float((mu - eps * self.cw.sum()) / self.cg.sum())

    def delta_star(self, R, eps):
        """Largest certified trust radius for an invariant tube of radius R [m]."""
        return float((R - eps * self.cw.sum()) / self.cb.sum())

    def equilibrium(self, e):
        return np.linalg.solve(np.eye(len(self.J)) - self.A, self.B @ e + self.d)

    def residual_scores(self, eps_list) -> np.ndarray:
        out = np.full(len(eps_list), np.nan)
        data = _transitions(eps_list, self.skill, self.J, 1, None)
        if data is None:
            return out
        X, E, Y, ep_id = data
        r = np.linalg.norm((Y - (X @ self.A.T + E @ self.B.T + self.d)) * self.Wz, axis=1)
        for i in np.unique(ep_id):
            out[i] = r[ep_id == i].max()
        return out


def make_tube(train_eps, skill, J, weights, scale, pos_coords, prior=None, shrink=0.0) -> Tube | None:
    """prior/shrink: hierarchical shrinkage of the skill model toward ``prior`` (Theta of the
    robot's pooled tracking model); the tube stays valid for any fixed model (eps is calibrated
    on that model's residuals) -- shrinkage only changes its tightness."""
    m = fit(train_eps, skill, np.asarray(J), int(IDENT.order), weights, prior=prior, shrink=shrink)
    if m is None or m.n_train < IDENT.min_train_transitions:
        return None
    J = np.asarray(J)
    rows = [k for k, j in enumerate(J) if j in pos_coords]
    S = np.zeros((len(rows), len(J)))
    for r, k in enumerate(rows):
        S[r, k] = scale[J[k]]
    t = Tube(skill, J, m.A, m.B[: len(J)], m.d, np.sqrt(np.asarray(weights)[J]), np.sqrt(np.asarray(weights)),
             S, m.n_train)
    return t if t.prepare() else t


def segments(skill_col, name):
    out, k0 = [], None
    for k, s in enumerate(list(skill_col) + [None]):
        if s == name and k0 is None:
            k0 = k
        elif s != name and k0 is not None:
            out.append((k0, k - 1))
            k0 = None
    return out


def check_episode(t: Tube, eps: float, rho: float, ep, anchor=None, delta=None) -> dict:
    """Check both propagated tubes on every execution of t.skill in one episode."""
    res = {"D_viol": False, "A_viol": False, "slew_viol": False, "outside_ball": False,
           "D_ratio": 0.0, "A_ratio": 0.0, "max_track_err_m": 0.0, "n_seg": 0}
    z, e = ep["z"], ep["e"]
    cum_g = np.concatenate([[0], np.cumsum(t.cg)])
    cum_b = np.concatenate([[0], np.cumsum(t.cb)])
    cum_w = np.concatenate([[0], np.cumsum(t.cw)])
    for k0, k1 in segments(ep["skill"], t.skill):
        L = k1 - k0 + 1
        if L < 2 or L > t.horizon:
            continue
        res["n_seg"] += 1
        Z, E = z[k0:k1 + 1][:, t.J], e[k0:k1 + 1]
        if np.max(np.linalg.norm(np.diff(E, axis=0) * t.We, axis=1)) > rho * (1 + 1e-6) + 1e-9:
            res["slew_viol"] = True
        Zs = np.linalg.solve(np.eye(len(t.J)) - t.A, (E @ t.B.T + t.d).T).T
        eta = Z - Zs
        err = np.linalg.norm(eta @ t.S.T, axis=1)
        tr = np.linalg.norm(np.einsum("jab,b->ja", t.powers_S[:L], eta[0]), axis=1)
        bound = tr + rho * cum_g[:L] + eps * cum_w[:L]
        res["D_viol"] |= bool(np.any(err > bound * (1 + 1e-9) + 1e-12))
        res["D_ratio"] = max(res["D_ratio"], float(np.max(err[1:] / np.maximum(bound[1:], 1e-12))))
        res["max_track_err_m"] = max(res["max_track_err_m"], float(err[1:].max()))
        if anchor is not None:
            inball = np.linalg.norm((E - anchor) * t.We, axis=1) <= delta * (1 + 1e-6) + 1e-9
            if not inball.all():
                res["outside_ball"] = True
            outside = np.flatnonzero(~inball)
            s0 = int(outside[-1]) + 1 if len(outside) else 0     # premise holds from s0 on
            if L - s0 < 2:
                continue
            Z, E, L = Z[s0:], E[s0:], L - s0
            c = t.equilibrium(anchor)
            dev = Z - c
            dist = np.linalg.norm(dev @ t.S.T, axis=1)
            trA = np.linalg.norm(np.einsum("jab,b->ja", t.powers_S[:L], dev[0]), axis=1)
            bA = trA + delta * cum_b[:L] + eps * cum_w[:L]
            res["A_viol"] |= bool(np.any(dist > bA * (1 + 1e-9) + 1e-12))
            res["A_ratio"] = max(res["A_ratio"], float(np.max(dist[1:] / np.maximum(bA[1:], 1e-12))))
    return res


def conformal_quantile(scores, alpha: float) -> float:
    s = np.sort(np.asarray([x for x in scores if np.isfinite(x)]))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(s[k - 1]) if 1 <= k <= n else np.inf


def joint_scores(tubes: dict, kappa: dict, eps_list) -> np.ndarray:
    names = [s for s in tubes if kappa.get(s, 0) > 0]
    S = np.full((len(eps_list), len(names)), -np.inf)
    for j, s in enumerate(names):
        sc = tubes[s].residual_scores(eps_list) / kappa[s]
        S[:, j] = np.where(np.isfinite(sc), sc, -np.inf)
    return S.max(axis=1) if names else np.full(len(eps_list), -np.inf)


def ordered_segments(skill_col):
    """All token executions of an episode in time order: [(skill, k0, k1)]."""
    out, cur, k0 = [], None, None
    for k, sname in enumerate(list(skill_col) + [None]):
        if sname != cur:
            if cur is not None:
                out.append((cur, k0, k - 1))
            cur, k0 = sname, k
    return out


def compose_episode(tubes: dict, eps: dict, rho_of, ep) -> dict:
    """Composition certificate: chain the per-token tubes through every hand-over.

    b_n(j) bounds ||W^1/2 eta_k|| (full block) in token n, k = k0_n + j:
        b_n(j)   = cA[j] b_n(0) + rho_n cgW[:j].sum() + eps_n cA[:j].sum()
        p_n(j)   = cw[j] b_n(0) + rho_n cg[:j].sum()  + eps_n cw[:j].sum()     (metres, the tube)
    hand-over n -> n+1 (exact identity eta' = A eta + w + z*_n(e_end) - z*_{n+1}(e_new)):
        b_{n+1}(0) = cA[1] b_n(end) + eps_n + J_n,   J_n = ||W^1/2 (z*_n(e_end) - z*_{n+1}(e_new))||
    The chain starts from the MEASURED error at the episode's first token only; a token without
    a certified model restarts the chain from measurement (counted)."""
    z, e = ep["z"], ep["e"]
    res = {"violated": False, "n_tokens": 0, "restarts": 0, "max_ratio": 0.0, "max_J": 0.0,
           "max_bound_m": 0.0}
    prev = None                      # (tube, b_end, eps, e_end)
    for sname, k0, k1 in ordered_segments(ep["skill"]):
        t = tubes.get(sname) if sname is not None else None
        if t is None or k1 - k0 + 1 < 2 or k1 - k0 + 1 > t.horizon:
            if sname is not None:
                res["restarts"] += 1
            prev = None
            continue
        res["n_tokens"] += 1
        L = k1 - k0 + 1
        Z, E = z[k0:k1 + 1][:, t.J], e[k0:k1 + 1]
        zs = lambda tt, ee: np.linalg.solve(np.eye(len(tt.J)) - tt.A, tt.B @ ee + tt.d)
        eta0 = Z[0] - zs(t, E[0])
        if prev is None:
            b0 = float(np.linalg.norm(t.Wz * eta0))                 # measured (chain start)
        else:
            tp, b_end, eps_p, e_end = prev
            J = float(np.linalg.norm(t.Wz * (zs(tp, e_end) - zs(t, E[0])))) if np.array_equal(tp.J, t.J) else np.inf
            res["max_J"] = max(res["max_J"], J)
            b0 = float(tp.cA[1] * b_end + eps_p + J)
        rho, ep_s = rho_of(sname), eps[sname]
        cumg = np.concatenate([[0], np.cumsum(t.cg)])[:L]
        cumw = np.concatenate([[0], np.cumsum(t.cw)])[:L]
        cumgW = np.concatenate([[0], np.cumsum(t.cgW)])[:L]
        cumA = np.concatenate([[0], np.cumsum(t.cA)])[:L]
        pos_bound = t.cw[:L] * b0 + rho * cumg + ep_s * cumw
        b_traj = t.cA[:L] * b0 + rho * cumgW + ep_s * cumA
        Zs = np.linalg.solve(np.eye(len(t.J)) - t.A, (E @ t.B.T + t.d).T).T
        err = np.linalg.norm((Z - Zs) @ t.S.T, axis=1)
        res["violated"] |= bool(np.any(err > pos_bound * (1 + 1e-9) + 1e-12))
        res["max_ratio"] = max(res["max_ratio"], float(np.max(err / np.maximum(pos_bound, 1e-12))))
        res["max_bound_m"] = max(res["max_bound_m"], float(pos_bound.max()))
        prev = (t, float(b_traj[-1]), ep_s, E[-1])
    return res
