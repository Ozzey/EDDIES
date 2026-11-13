"""Algebraic certificates in a FIXED latent coordinate system.

These certify their stated affine model, not an uncalibrated robot or encoder.
All numerical checks are floating-point checks, not interval-arithmetic proofs.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_triangular

MAX_CONDITION = 1e10   # beyond this the floating-point operator norm is not trustworthy


def vector(x: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError("Expected a finite one-dimensional vector")
    return a


@dataclass(frozen=True)
class Ellipsoid:
    center: NDArray[np.float64]
    P: NDArray[np.float64]
    radius: float

    def __post_init__(self):
        c, p = vector(self.center).copy(), np.asarray(self.P, dtype=float).copy()
        if p.shape != (len(c), len(c)) or not np.isfinite(p).all():
            raise ValueError("Invalid metric shape or values")
        scale = max(1.0, float(np.abs(p).max())) if p.size else 1.0
        if (len(c)==0 or np.abs(p - p.T).max() > 1e-12 * scale or
            not np.isfinite(self.radius) or self.radius <= 0):
            raise ValueError("P must be symmetric positive definite; radius > 0")
        p = (p + p.T) / 2                     # store and factor the symmetric matrix
        L = np.linalg.cholesky(p)             # raises LinAlgError if not positive definite
        ev = np.linalg.eigvalsh(p)
        if ev[0] <= 0 or ev[-1] / ev[0] > MAX_CONDITION:
            raise ValueError("P is too ill-conditioned for a floating-point certificate")
        c.setflags(write=False); p.setflags(write=False)
        object.__setattr__(self, "center", c); object.__setattr__(self, "P", p)
        object.__setattr__(self, "_S", L.T)

    @property
    def whitening(self):
        return self._S

    def norm(self, v: ArrayLike) -> float:
        v = vector(v)
        if len(v) != len(self.center):
            raise ValueError("Dimension mismatch")
        return float(np.linalg.norm(self.whitening @ v))

    def operator_norm(self, a: ArrayLike) -> float:
        a = np.asarray(a, dtype=float)
        if a.shape != self.P.shape or not np.isfinite(a).all():
            raise ValueError("Invalid operator")
        s = self.whitening
        # S A S^{-1} without an explicit inverse:  (S^{-T} (S A)^T)^T
        sa = s @ a
        return float(np.linalg.norm(solve_triangular(s, sa.T, trans='T', lower=False).T, ord=2))

    def contains(self, z: ArrayLike, tolerance: float = 0.0) -> bool:
        return self.norm(vector(z) - self.center) <= self.radius + tolerance


@dataclass(frozen=True)
class AffinePhase:
    A: NDArray[np.float64]
    b: NDArray[np.float64]
    epsilon: float
    L_A: float = 0.0
    L_b: float = 0.0

    def __post_init__(self):
        a=np.asarray(self.A,float).copy(); b=vector(self.b).copy()
        if a.shape!=(len(b),len(b)) or not np.isfinite(a).all():
            raise ValueError("Invalid phase matrix")
        bounds=[self.epsilon,self.L_A,self.L_b]
        if not np.isfinite(bounds).all() or min(bounds)<0:
            raise ValueError("Bounds must be finite and nonnegative")
        a.setflags(write=False); b.setflags(write=False)
        object.__setattr__(self,'A',a); object.__setattr__(self,'b',b)

    def margin(self, region: Ellipsoid, delta: float = 0.0) -> float:
        slope = region.radius * self.L_A + self.L_b
        if np.isnan(delta) or delta < 0 or (np.isinf(delta) and slope > 0):
            raise ValueError("delta must be nonnegative (and finite unless the slope is zero)")
        base = (region.radius - region.radius * region.operator_norm(self.A)
                - region.norm(self.b) - self.epsilon)
        return float(base if slope == 0 else base - slope * delta)

    def admissible_radius(self, region: Ellipsoid) -> float:
        m = self.margin(region)
        if m < 0:
            raise ValueError("Even the prototype has no invariance certificate")
        slope = region.radius * self.L_A + self.L_b
        if slope == 0:
            return np.inf
        d = m / slope
        while d > 0 and self.margin(region, d) < 0:   # round toward zero: the
            d = float(np.nextafter(d, 0.0))             # returned radius passes margin()
        return d

    def step(self, z: ArrayLike, region: Ellipsoid, w: ArrayLike) -> NDArray:
        w = vector(w)
        if region.norm(w) > self.epsilon * (1 + 1e-12) + 1e-15:
            raise ValueError("Disturbance violates the assumed bound")
        return region.center + np.asarray(self.A) @ (vector(z)-region.center) + self.b + w


def compose(first: tuple[NDArray, NDArray], second: tuple[NDArray, NDArray]):
    """Execution order: first, THEN second. All maps use the same center."""
    a1, b1 = first; a2, b2 = second
    a1, a2 = np.asarray(a1, float), np.asarray(a2, float)
    b1, b2 = vector(b1), vector(b2)
    n = len(b1)
    if a1.shape != (n, n) or a2.shape != (n, n) or b2.shape != (n,):
        raise ValueError("compose expects square A matrices and 1-D offsets of one dimension")
    return a2 @ a1, a2 @ b1 + b2


def composed_error(gains: ArrayLike, errors: ArrayLike) -> float:
    gains, errors = vector(gains), vector(errors)
    if len(gains) != len(errors) or np.any(gains < 0) or np.any(errors < 0):
        raise ValueError("Invalid gain/error sequence")
    result = 0.0
    for a, e in zip(gains, errors):
        result = float(a * result + e)
    return result


def decoder_slew_bound(rho: float, dx: float, dtheta: float,
                       L_e: float, L_x: float, L_theta: float,
                       mode_jump: float = 0.0) -> float:
    values = [rho, dx, dtheta, L_e, L_x, L_theta, mode_jump]
    if min(values) < 0 or not np.isfinite(values).all():
        raise ValueError("All bounds must be finite and nonnegative")
    return L_e*rho + L_x*dx + L_theta*dtheta + mode_jump
