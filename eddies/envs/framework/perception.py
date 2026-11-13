"""Noisy object-pose estimates (shared by every method; identical per seed).

A static scene camera gives a biased, noisy estimate; when the end-effector is
close to the object a wrist-camera estimate with much smaller error is used.
The bias is drawn once per episode from the task RNG, so all methods see the
same perception errors on the same seed (paired comparison).
"""
from __future__ import annotations

import numpy as np


class PoseEstimator:
    def __init__(self, rng: np.random.Generator, *, bias_far=0.004, sigma_far=0.0015,
                 bias_near=0.0008, sigma_near=0.0004, yaw_bias=np.radians(3.0),
                 yaw_sigma=np.radians(0.7), near_distance=0.10, z_factor=0.3):
        zf = np.array([1.0, 1.0, z_factor])
        self.rng = rng
        self.b_far = rng.normal(0, bias_far, 3) * zf
        self.b_near = rng.normal(0, bias_near, 3) * zf
        self.s_far, self.s_near = sigma_far * zf, sigma_near * zf
        self.yaw_b = rng.normal(0, yaw_bias)
        self.yaw_s = yaw_sigma
        self.near = near_distance

    def estimate(self, true_pos, true_yaw: float = 0.0, sensor_pos=None):
        true_pos = np.asarray(true_pos, float)
        near = sensor_pos is not None and np.linalg.norm(np.asarray(sensor_pos) - true_pos) < self.near
        if near:
            pos = true_pos + self.b_near + self.rng.normal(0, 1, 3) * self.s_near
            yaw = true_yaw + 0.25 * self.yaw_b + self.rng.normal(0, 0.3 * self.yaw_s)
        else:
            pos = true_pos + self.b_far + self.rng.normal(0, 1, 3) * self.s_far
            yaw = true_yaw + self.yaw_b + self.rng.normal(0, self.yaw_s)
        return pos, float(yaw), bool(near)


def wrap_to(angle: float, period: float) -> float:
    """Wrap an angle to (-period/2, period/2] (e.g. period=pi/2 for a cube)."""
    return float((angle + period / 2) % period - period / 2)


def yaw_of_quat(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def tilt_of_quat(q) -> float:
    """Angle between the body z-axis and world z (radians)."""
    w, x, y, z = q
    zz = 1 - 2 * (x * x + y * y)
    return float(np.arccos(np.clip(zz, -1.0, 1.0)))
