"""Physically typed, normalized command embeddings.

An :class:`EmbeddingSpace` fixes the coordinate chart, units, normalization and
metric M of the command embedding e for one robot/task family.  It is FROZEN
for an assurance epoch: the metric gauge must not change while trust radii and
slew bounds are interpreted in it.

Normalized coordinates live in the fixed box [lower_n, upper_n] (by default
[-1, 1], or [0, 1] for one-sided quantities such as aperture/force).  The
projection onto box ∩ trust ball ∩ slew ball is delegated to
``eddies.core.embedding.EmbeddingProjector`` (exact intersection, never sequential
clipping).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from eddies.core.embedding import EmbeddingProjector, InfeasibleEmbedding  # noqa: F401  (re-export)


@dataclass
class EmbeddingSpace:
    names: list[str]
    phys_lower: np.ndarray            # physical value mapped to norm_lower
    phys_upper: np.ndarray            # physical value mapped to norm_upper
    weights: np.ndarray               # diagonal of the metric M (normalized coords)
    norm_lower: np.ndarray | None = None
    norm_upper: np.ndarray | None = None
    units: list[str] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.names)
        self.phys_lower = np.asarray(self.phys_lower, float).reshape(n)
        self.phys_upper = np.asarray(self.phys_upper, float).reshape(n)
        self.weights = np.asarray(self.weights, float).reshape(n)
        if self.norm_lower is None:
            self.norm_lower = -np.ones(n)
        if self.norm_upper is None:
            self.norm_upper = np.ones(n)
        self.norm_lower = np.asarray(self.norm_lower, float).reshape(n)
        self.norm_upper = np.asarray(self.norm_upper, float).reshape(n)
        if np.any(self.phys_upper <= self.phys_lower) or np.any(self.weights <= 0):
            raise ValueError("invalid embedding space bounds/metric")
        self.metric = np.diag(self.weights)
        self.projector = EmbeddingProjector(self.norm_lower, self.norm_upper, self.metric)
        self._scale = (self.phys_upper - self.phys_lower) / (self.norm_upper - self.norm_lower)

    @property
    def dim(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def normalize(self, phys) -> np.ndarray:
        phys = np.asarray(phys, float)
        return self.norm_lower + (phys - self.phys_lower) / self._scale

    def denormalize(self, e) -> np.ndarray:
        e = np.asarray(e, float)
        return self.phys_lower + (e - self.norm_lower) * self._scale

    def norm(self, v) -> float:
        v = np.asarray(v, float)
        return float(np.sqrt(max(0.0, v @ self.metric @ v)))

    def clip(self, e) -> np.ndarray:
        return np.clip(np.asarray(e, float), self.norm_lower, self.norm_upper)

    def in_box(self, e, tol: float = 1e-6) -> bool:
        e = np.asarray(e, float)
        return bool(np.all(e >= self.norm_lower - tol) and np.all(e <= self.norm_upper + tol))

    def describe(self) -> dict:
        return {"names": list(self.names), "units": list(self.units),
                "phys_lower": self.phys_lower.tolist(), "phys_upper": self.phys_upper.tolist(),
                "norm_lower": self.norm_lower.tolist(), "norm_upper": self.norm_upper.tolist(),
                "metric_diag": self.weights.tolist()}


class RotationChart:
    """Local SO(3) chart: rotation vector of R * R_base^{-1} (world frame).

    Raw quaternion differences are unsuitable (q and -q are the same
    orientation); this chart is sign-invariant and is rejected near its
    boundary |rotvec| -> pi.
    """

    def __init__(self, base_quat_wxyz, max_angle: float = np.pi - 0.1):
        q = np.asarray(base_quat_wxyz, float)
        self.base = Rotation.from_quat(q[[1, 2, 3, 0]])
        self.max_angle = float(max_angle)

    def _check(self, v, strict):
        n = np.linalg.norm(v)
        if n > self.max_angle:
            if strict:
                raise InfeasibleEmbedding("orientation outside the SO(3) chart")
            v = v * (self.max_angle / n)      # MEASURED orientation: clip to the chart boundary
        return v

    def encode(self, quat_wxyz, strict: bool = False) -> np.ndarray:
        """strict=True for commands (reject); default clips measured orientations so an
        observation can never crash an episode."""
        q = np.asarray(quat_wxyz, float)
        v = (Rotation.from_quat(q[[1, 2, 3, 0]]) * self.base.inv()).as_rotvec()
        return self._check(v, strict)

    def encode_matrix(self, mat, strict: bool = False) -> np.ndarray:
        v = (Rotation.from_matrix(np.asarray(mat, float).reshape(3, 3)) * self.base.inv()).as_rotvec()
        return self._check(v, strict)

    def decode(self, rotvec) -> np.ndarray:
        """Return quaternion wxyz."""
        r = Rotation.from_rotvec(np.asarray(rotvec, float)) * self.base
        return r.as_quat()[[3, 0, 1, 2]]

    def decode_matrix(self, rotvec) -> np.ndarray:
        return (Rotation.from_rotvec(np.asarray(rotvec, float)) * self.base).as_matrix()
