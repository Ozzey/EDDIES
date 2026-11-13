"""Projection onto the INTERSECTION of a fixed domain, trust ball and slew ball.

``EmbeddingProjector.project`` returns the exact M-metric projection

    argmin_x  1/2 (x - p)^T M (x - p)
    s.t.      lower <= x <= upper,  ||x - c_j||_M <= r_j   (slew ball, optional prototype ball)

computed by Lagrangian duality instead of a general NLP solver.  For fixed ball
multipliers lambda >= 0 the inner problem is the M-projection of
z(lambda) = (p + sum_j lambda_j c_j) / (1 + sum_j lambda_j) onto the box (a clip
for diagonal M, bounded least squares otherwise); the dual derivative is
phi_j = (||x(lambda) - c_j||_M^2 - r_j^2) / 2, non-increasing in lambda_j.  One
ball: bisection on lambda.  Two balls: coordinate ascent of the concave dual
with exact 1-D bisections.  The result is accepted only after a relative KKT
check; a guaranteed-feasible line search toward a known feasible point removes
residual round-off, so a feasible problem is never rejected because of solver
failure, and an empty intersection is still rejected (never silently enlarged).
Sequential clipping is never used.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import lsq_linear
from .certificates import vector


class InfeasibleEmbedding(RuntimeError):
    pass


@dataclass
class EmbeddingProjector:
    lower: np.ndarray
    upper: np.ndarray
    metric: np.ndarray
    tolerance: float = 1e-8
    last_info: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        self.lower = vector(self.lower).copy(); self.upper = vector(self.upper).copy()
        self.metric = np.asarray(self.metric, dtype=float).copy()
        n = len(self.lower)
        if self.upper.shape != (n,) or np.any(self.lower > self.upper):
            raise ValueError("Invalid box")
        if (self.metric.shape != (n, n) or not np.isfinite(self.metric).all()
                or not np.allclose(self.metric, self.metric.T)):
            raise ValueError("Invalid metric")
        self.metric = (self.metric + self.metric.T) / 2
        self._chol = np.linalg.cholesky(self.metric)          # raises if not SPD
        self._diag = np.allclose(self.metric, np.diag(np.diag(self.metric)), rtol=0, atol=0)
        self._w = np.diag(self.metric).copy()

    # ------------------------------------------------------------------ norms
    def norm(self, x):
        x = np.asarray(x, dtype=float)
        return float(np.sqrt(max(0., x @ self.metric @ x)))

    def _box(self, z):
        """M-metric projection of z onto the box."""
        if self._diag:
            return np.clip(z, self.lower, self.upper)
        A = self._chol.T
        res = lsq_linear(A, A @ z, bounds=(self.lower, self.upper), method="bvls", tol=1e-15)
        return np.clip(res.x, self.lower, self.upper)

    def _x(self, p, centers, lam):
        s = 1.0 + lam.sum()
        return self._box((p + lam @ centers) / s)

    # ------------------------------------------------------------- projection
    def project(self, proposed, previous, rho, anchor=None, delta=None):
        proposed, previous = vector(proposed), vector(previous)
        if proposed.shape != self.lower.shape or previous.shape != self.lower.shape:
            raise ValueError("Embedding dimension mismatch")
        if not np.isfinite(rho) or rho < 0:
            raise ValueError("rho must be finite and nonnegative")
        balls = [(previous, float(rho))]
        if anchor is not None:
            anchor = vector(anchor)
            if (anchor.shape != self.lower.shape or delta is None or not np.isfinite(delta)
                    or delta < 0):
                raise ValueError("Invalid prototype ball")
            if self.norm(anchor - previous) > rho + delta + self.tolerance * max(1.0, rho + delta):
                raise InfeasibleEmbedding("Slew and prototype balls are disjoint")
            balls.append((anchor, float(delta)))
        centers = np.array([c for c, _ in balls])
        radii = np.array([r for _, r in balls])
        tol = self.tolerance * np.maximum(1.0, radii)
        if rho == 0:
            candidate = previous.copy()
            info = {"method": "rho=0", "lambda": [0.0] * len(balls)}
        else:
            candidate, info = self._solve(proposed, centers, radii)
        viol = self._violations(candidate, centers, radii)
        if viol.max() > 0 and info.get("feasible_point") is not None:
            candidate = self._repair(candidate, info["feasible_point"], centers, radii)
            viol = self._violations(candidate, centers, radii)
            info["repaired"] = True
        info["max_violation"] = float(max(viol.max(), 0.0))
        self.last_info = info
        if (not np.isfinite(candidate).all() or
                np.any(candidate < self.lower - self.tolerance) or
                np.any(candidate > self.upper + self.tolerance) or
                np.any(viol[2:] > tol)):
            raise InfeasibleEmbedding("Projection failed post-solve verification: "
                                      + info.get("status", ""))
        return np.clip(candidate, self.lower, self.upper)

    def _violations(self, x, centers, radii):
        """[box lower, box upper, ball_1, ball_2...] (positive = violated)."""
        out = [float(np.max(self.lower - x)), float(np.max(x - self.upper))]
        out += [self.norm(x - c) - r for c, r in zip(centers, radii)]
        return np.array(out)

    def _phi(self, x, c, r):
        d = x - c
        return 0.5 * (d @ self.metric @ d - r * r)

    def _feasible_point(self, centers, radii):
        """A point of box ∩ balls if one is easy to certify (else None).

        The slew centre is feasible when it lies in the box and every other ball;
        otherwise, for a box-contained anchor, the segment between the two centres
        meets both balls (they intersect) and stays in the convex box.
        """
        prev = centers[0]
        if self._violations(prev, centers, radii).max() <= 0:
            return prev.copy()
        if len(radii) == 2:
            a = centers[1]
            D = self.norm(a - prev)
            if D > 0:
                t = 0.5 * (max(0.0, D - radii[1]) + min(radii[0], D))
                q = prev + (a - prev) * (t / D)
                if self._violations(q, centers, radii).max() <= 0:
                    return q
        return None

    def _solve(self, p, centers, radii):
        m = len(radii)
        info = {"method": "dual", "status": "", "iterations": 0,
                "feasible_point": self._feasible_point(centers, radii)}
        lam = np.zeros(m)
        x = self._x(p, centers, lam)
        phis = np.array([self._phi(x, c, r) for c, r in zip(centers, radii)])
        if np.all(phis <= 0):
            info.update(status="balls-inactive", **{"lambda": lam.tolist()})
            return x, info
        # one active ball: exact 1-D dual solution, optimal if it satisfies the others
        for j in np.argsort(-phis):
            lam1 = np.zeros(m)
            lam1[j] = self._line(p, centers, radii, lam1, j)
            x1 = self._x(p, centers, lam1)
            if all(self._phi(x1, c, r) <= 0 for k, (c, r) in enumerate(zip(centers, radii)) if k != j):
                info.update(status=f"one-active-ball-{j}", **{"lambda": lam1.tolist()})
                return x1, info
        # both balls active: coordinate ascent warm start, then projected Newton (diagonal M)
        lam = np.zeros(m)
        for outer in range(30):
            lam_old = lam.copy()
            for j in range(m):
                lam[j] = self._line(p, centers, radii, lam, j)
            if np.allclose(lam, lam_old, rtol=1e-12, atol=1e-15):
                break
        info["iterations"] = outer + 1
        lam = self._newton(p, centers, radii, lam, info)
        x = self._x(p, centers, lam)
        phis = np.array([self._phi(x, c, r) for c, r in zip(centers, radii)])
        scale = np.maximum(1.0, radii * radii)
        primal = float(np.max(np.maximum(phis, 0) / scale))
        comp = float(np.max(np.abs(lam * phis) / (scale * (1 + lam))))
        info.update(status="kkt-ok" if primal <= 1e-10 and comp <= 1e-8 else "kkt-approx",
                    kkt_primal=primal, kkt_complementarity=comp, **{"lambda": lam.tolist()})
        return x, info

    def _dual_value(self, p, centers, radii, lam):
        x = self._x(p, centers, lam)
        d = x - p
        return 0.5 * d @ self.metric @ d + sum(l * self._phi(x, c, r) for l, c, r in zip(lam, centers, radii))

    def _newton(self, p, centers, radii, lam, info):
        """Projected Newton ascent on the concave dual (both balls active).

        On the free set F of the box projection x(lambda) the KKT conditions give
        x_F = z_F - M_FF^{-1} M_FB (x_B - z_B), hence the exact dual Hessian
        d phi_k / d lambda_j = (x - c_k)^T M dx/dlambda_j with dz/dlambda_j = (c_j - z)/S.
        """
        M = self.metric
        for it in range(60):
            S = 1.0 + lam.sum()
            z = (p + lam @ centers) / S
            x = self._box(z)
            free = (x > self.lower + 1e-13) & (x < self.upper - 1e-13)
            phi = np.array([self._phi(x, c, r) for c, r in zip(centers, radii)])
            if np.max(np.abs(phi) / np.maximum(1.0, radii * radii)) < 1e-14:
                break
            dz = np.array([(cj - z) / S for cj in centers]).T          # n x m
            dx = np.zeros_like(dz)
            if free.any():
                F, B = np.flatnonzero(free), np.flatnonzero(~free)
                corr = np.zeros((len(F), dz.shape[1]))
                if len(B):
                    corr = np.linalg.solve(M[np.ix_(F, F)], M[np.ix_(F, B)] @ dz[B])
                dx[F] = dz[F] + corr
            H = np.array([[(x - ck) @ M @ dx[:, j] for j in range(len(lam))] for ck in centers])
            try:
                step = -np.linalg.solve(H - 1e-14 * np.eye(len(lam)), phi)
            except np.linalg.LinAlgError:
                break
            g0 = self._dual_value(p, centers, radii, lam)
            t = 1.0
            while t > 1e-6:
                cand = np.maximum(lam + t * step, 0.0)
                if self._dual_value(p, centers, radii, cand) >= g0 - 1e-15 * max(1.0, abs(g0)):
                    break
                t *= 0.5
            if t <= 1e-6 or np.allclose(cand, lam, rtol=1e-15, atol=1e-18):
                break
            lam = cand
        info["newton_iterations"] = it + 1
        return lam

    def _line(self, p, centers, radii, lam, j):
        """argmax_{lambda_j >= 0} of the concave dual with the other multipliers fixed."""
        c, r = centers[j], radii[j]
        trial = lam.copy()
        trial[j] = 0.0
        if self._phi(self._x(p, centers, trial), c, r) <= 0:
            return 0.0
        lo, hi = 0.0, max(1.0, 2 * lam[j])
        while True:
            trial[j] = hi
            if self._phi(self._x(p, centers, trial), c, r) <= 0:
                break
            lo, hi = hi, hi * 4.0
            if hi > 1e18:
                raise InfeasibleEmbedding("Box and ball intersection is empty")
        while hi - lo > 1e-13 * max(1.0, hi):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            trial[j] = mid
            if self._phi(self._x(p, centers, trial), c, r) <= 0:
                hi = mid
            else:
                lo = mid
        return hi                       # feasible side of the root

    def _repair(self, x, feasible, centers, radii):
        """Largest step from a feasible point toward x that stays feasible."""
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            y = feasible + mid * (x - feasible)
            if self._violations(y, centers, radii).max() <= 0:
                lo = mid
            else:
                hi = mid
        return feasible + lo * (x - feasible)


def discrete_neighbors(codes: np.ndarray, previous: np.ndarray, rho: float,
                       metric: np.ndarray | None = None):
    codes = np.asarray(codes, dtype=float); previous = vector(previous)
    if codes.ndim != 2 or codes.shape[1] != len(previous) or rho < 0:
        raise ValueError("Invalid codebook or radius")
    if metric is None: metric = np.eye(len(previous))
    diff = codes-previous
    return np.flatnonzero(np.einsum('bi,ij,bj->b',diff,metric,diff) <= rho*rho+1e-12)
