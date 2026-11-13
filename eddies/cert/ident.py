"""Per-skill embedding-affine latent models and their exact constants.

Latent state  zeta_k = [z_k, z_{k-1}, ..., z_{k-p+1}]  (measured embedding, active coordinates)
Model         zeta_{k+1} = A zeta_k + B e_k + d + w_k          (companion form, order p)

Because the model is affine in the command e, the Lipschitz constants of the paper's
Theorem 1 are exact:  L^A = 0  and  L^b = ||B||_{M->P} = ||S B M^{-1/2}||_2  (P = S^T S).
The ellipsoid metric P is chosen on TRAINING data only (gamma-scaled Lyapunov equation),
so the conformal calibration that follows stays valid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eddies.common.config import load

IDENT = load().identification
from scipy.linalg import solve_discrete_lyapunov


@dataclass
class SkillModel:
    skill: str
    J: np.ndarray                 # active z coordinates
    p: int                        # model order
    A: np.ndarray                 # (n, n), n = p * |J|
    B: np.ndarray                 # (n, m)
    d: np.ndarray                 # (n,)
    weights: np.ndarray           # metric M diagonal on e (m,)
    n_train: int
    P: np.ndarray | None = None
    S: np.ndarray | None = None   # P = S^T S
    a: float = np.nan             # ||A||_P  (< 1 required)
    Lb: float = np.nan            # ||B||_{M->P}
    g: float = np.nan             # ||(I-A)^{-1} B||_{M->P}
    gamma: float = np.nan
    spectral_radius: float = np.nan
    sigma_M: float = np.nan       # max ||W^{1/2} z_k||  over the unit P-ball (current block)
    sigma_pos: float = np.nan     # max position error [m] over the unit P-ball
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.A.shape[0]

    def zeta(self, z_hist: np.ndarray) -> np.ndarray:
        """z_hist: (p, dim_z) rows z_k, z_{k-1}, ... (full coordinates)."""
        return np.concatenate([z_hist[i, self.J] for i in range(self.p)])

    def equilibrium(self, e: np.ndarray) -> np.ndarray:
        return np.linalg.solve(np.eye(self.n) - self.A, self.B @ e + self.d)

    def pnorm(self, v: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.atleast_2d(v) @ self.S.T, axis=1)


def _transitions(eps, skill, J, p, dim_z):
    """Rows (zeta_k, e_k, z_{k+1}[J]) for control steps issued by ``skill``."""
    X, E, Y, ep_id = [], [], [], []
    for i, ep in enumerate(eps):
        z, e, sk = ep["z"], ep["e"], ep["skill"]
        T = len(z)
        for k in range(p - 1, T - 1):
            if sk[k] is None or (skill is not None and sk[k] != skill):
                continue
            X.append(np.concatenate([z[k - j, J] for j in range(p)]))
            E.append(e[k])
            Y.append(z[k + 1, J])
            ep_id.append(i)
    if not X:
        return None
    return np.asarray(X), np.asarray(E), np.asarray(Y), np.asarray(ep_id)


def active_coords(eps, tol=1e-6) -> np.ndarray:
    Z = np.concatenate([ep["z"] for ep in eps])
    return np.flatnonzero(Z.std(0) > tol)


def fit(eps, skill, J, p, weights, ridge=None, prior=None, shrink=0.0) -> SkillModel | None:
    """Least squares for one skill (skill=None: pooled over every skill).

    prior/shrink: hierarchical shrinkage  min ||Y - Phi Theta||^2 + lam ||Theta - prior||^2
    with lam = shrink * n_transitions, so coordinates the skill does not excite inherit the
    robot's pooled tracking dynamics instead of an unidentified unit eigenvalue."""
    ridge = float(IDENT.ridge) if ridge is None else ridge
    data = _transitions(eps, skill, J, p, None)
    if data is None:
        return None
    X, E, Y, _ = data
    nz = len(J)
    Phi = np.hstack([X, E, np.ones((len(X), 1))])
    G = Phi.T @ Phi
    if prior is not None and shrink > 0:
        lam = shrink * len(X)
        Theta = np.linalg.solve(G + lam * np.eye(G.shape[0]), Phi.T @ Y + lam * prior.T).T
    else:
        lam = ridge * np.trace(G) / G.shape[0]
        Theta = np.linalg.solve(G + lam * np.eye(G.shape[0]), Phi.T @ Y).T   # (nz, cols)
    n = p * nz
    A = np.zeros((n, n))
    A[:nz, :n] = Theta[:, :n]
    if p > 1:
        A[nz:, :n - nz] = np.eye(n - nz)
    m = E.shape[1]
    B = np.zeros((n, m))
    B[:nz] = Theta[:, n:n + m]
    d = np.zeros(n)
    d[:nz] = Theta[:, -1]
    mdl = SkillModel(skill, np.asarray(J), p, A, B, d, np.asarray(weights, float), len(X))
    mdl.info["Theta"] = Theta
    return mdl


def induced(S, Mat, weights=None):
    """||Mat||_{M->P} (weights given) or ||Mat||_P (weights None)."""
    if weights is None:
        return float(np.linalg.norm(S @ Mat @ np.linalg.inv(S), 2))
    return float(np.linalg.norm(S @ Mat @ np.diag(1 / np.sqrt(weights)), 2))


def set_metric(mdl: SkillModel, gamma: float, pos_idx=None, pos_scale=None) -> bool:
    """P from (A/gamma)^T P (A/gamma) - P = -I; then ||A||_P <= gamma."""
    A = mdl.A
    rho = float(max(abs(np.linalg.eigvals(A))))
    mdl.spectral_radius = rho
    if not (rho < gamma < 1):
        return False
    P = solve_discrete_lyapunov((A / gamma).T, np.eye(mdl.n))
    P = (P + P.T) / 2
    try:
        S = np.linalg.cholesky(P).T
    except np.linalg.LinAlgError:
        return False
    mdl.P, mdl.S, mdl.gamma = P, S, gamma
    mdl.a = induced(S, A)
    if mdl.a >= 1:
        return False
    mdl.Lb = induced(S, mdl.B, mdl.weights)
    G = np.linalg.solve(np.eye(mdl.n) - A, mdl.B)
    mdl.g = induced(S, G, mdl.weights)
    Pinv = np.linalg.inv(P)
    nz = len(mdl.J)
    w = mdl.weights[mdl.J] if len(mdl.weights) > max(mdl.J) else np.ones(nz)
    Wh = np.diag(np.sqrt(w))
    mdl.sigma_M = float(np.sqrt(max(np.linalg.eigvalsh(Wh @ Pinv[:nz, :nz] @ Wh))))
    if pos_idx is not None and len(pos_idx):
        rows = [list(mdl.J).index(j) for j in pos_idx if j in mdl.J]
        if rows:
            D = np.diag(np.asarray(pos_scale)[[mdl.J[r] for r in rows]])
            Sub = Pinv[np.ix_(rows, rows)]
            mdl.sigma_pos = float(np.sqrt(max(np.linalg.eigvalsh(D @ Sub @ D))))
    return True


def residual_scores(mdl: SkillModel, eps) -> np.ndarray:
    """Per-episode score: max_k ||w_k||_P over the transitions of this skill (nan if absent)."""
    out = np.full(len(eps), np.nan)
    data = _transitions(eps, mdl.skill, mdl.J, mdl.p, None)
    if data is None:
        return out
    X, E, Y, ep_id = data
    nz = len(mdl.J)
    pred = X @ mdl.A[:nz].T + E @ mdl.B[:nz].T + mdl.d[:nz]
    W = np.zeros((len(X), mdl.n))
    W[:, :nz] = Y - pred
    r = mdl.pnorm(W)
    for i in np.unique(ep_id):
        out[i] = r[ep_id == i].max()
    return out


def choose_gamma(mdl: SkillModel, train_eps, rho: float, pos_idx=None, pos_scale=None,
                 grid=None) -> float:
    """Pick gamma on TRAINING data to minimise the M-metric slew tube (g rho + eps)/(1-a)."""
    best, best_val = None, np.inf
    rho_A = float(max(abs(np.linalg.eigvals(mdl.A))))
    if rho_A >= 0.999:
        return np.nan
    grid = grid if grid is not None else rho_A + (1 - rho_A) * np.array(
        [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    for gamma in grid:
        if not set_metric(mdl, float(gamma), pos_idx, pos_scale):
            continue
        sc = residual_scores(mdl, train_eps)
        eps_tr = np.nanquantile(sc, 0.9) if np.isfinite(sc).any() else np.inf
        val = (mdl.g * rho + eps_tr) / (1 - mdl.a) * mdl.sigma_M
        if val < best_val:
            best, best_val = float(gamma), val
    if best is not None:
        set_metric(mdl, best, pos_idx, pos_scale)
    return best if best is not None else np.nan
