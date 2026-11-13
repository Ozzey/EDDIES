"""Damped-least-squares differential IK on a kinematic shadow of the live model.

The shadow ``MjData`` only receives copied joint positions and is advanced
with ``mj_kinematics``; it never feeds back into the simulated physics.
The reference integrator (start from the previous joint reference, re-synced
to the measured configuration when tracking error grows) avoids feeding
contact-induced tracking errors back into the reference.
"""
from __future__ import annotations

import mujoco
import numpy as np

from eddies.envs.framework.scene import dadr, qadr, sid


def rotation_error_mat(R_current: np.ndarray, R_desired: np.ndarray) -> np.ndarray:
    """World-frame rotation vector of R_d R_c^T (same convention as eddies.core.control.rotation_error)."""
    R = R_desired @ R_current.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    if q[0] < 0:
        q = -q
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


class DLSIK:
    def __init__(self, model: mujoco.MjModel, joint_names: list[str], site: str, *,
                 damping: float = 0.03, posture: np.ndarray | None = None,
                 posture_gain: float = 0.05, rot_weight: float = 0.6,
                 max_step: float = 0.05, iterations: int = 2, limit_margin: float = 0.02):
        self.model = model
        self.shadow = mujoco.MjData(model)
        self.joints = list(joint_names)
        self.qadr = np.array([qadr(model, j) for j in self.joints])
        self.dadr = np.array([dadr(model, j) for j in self.joints])
        self.site = sid(model, site)
        self.damping = damping
        rng = np.array([model.jnt_range[model.joint(j).id] for j in self.joints])
        limited = np.array([bool(model.jnt_limited[model.joint(j).id]) for j in self.joints])
        self.lower = np.where(limited, rng[:, 0] + limit_margin, -np.inf)
        self.upper = np.where(limited, rng[:, 1] - limit_margin, np.inf)
        self.posture = None if posture is None else np.asarray(posture, float)
        self.posture_gain = posture_gain
        self.rot_weight = rot_weight
        self.max_step = max_step
        self.iterations = iterations
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.q_ref: np.ndarray | None = None

    def site_pose(self, data: mujoco.MjData):
        return data.site_xpos[self.site].copy(), data.site_xmat[self.site].reshape(3, 3).copy()

    def reset(self, q: np.ndarray) -> None:
        self.q_ref = np.asarray(q, float).copy()

    def solve(self, live_qpos: np.ndarray, target_pos, target_mat=None, *,
              resync_tol: float = 0.15, measured_q: np.ndarray | None = None) -> np.ndarray:
        """Return a joint reference that moves the site toward the target pose."""
        m, d = self.model, self.shadow
        q_meas = live_qpos[self.qadr] if measured_q is None else measured_q
        if self.q_ref is None or np.max(np.abs(self.q_ref - q_meas)) > resync_tol:
            self.q_ref = q_meas.copy()
        d.qpos[:] = live_qpos
        q = self.q_ref.copy()
        n = len(q)
        for _ in range(self.iterations):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos, R = self.site_pose(d)
            err_p = np.asarray(target_pos, float) - pos
            mujoco.mj_jacSite(m, d, self.jacp, self.jacr, self.site)
            if target_mat is None:
                J = self.jacp[:, self.dadr]
                err = err_p
            else:
                err_r = rotation_error_mat(R, np.asarray(target_mat, float).reshape(3, 3))
                J = np.vstack([self.jacp[:, self.dadr], self.rot_weight * self.jacr[:, self.dadr]])
                err = np.concatenate([err_p, self.rot_weight * err_r])
            JJt = J @ J.T + (self.damping ** 2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, err)
            if self.posture is not None:
                Jpinv = J.T @ np.linalg.solve(JJt, np.eye(J.shape[0]))
                null = np.eye(n) - Jpinv @ J
                dq += null @ (self.posture_gain * (self.posture - q))
            mx = np.max(np.abs(dq))
            if mx > self.max_step:
                dq *= self.max_step / mx
            q = np.clip(q + dq, self.lower, self.upper)
        self.q_ref = q
        return q.copy()
