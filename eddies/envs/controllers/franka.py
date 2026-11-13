"""Franka Emika Panda (menagerie ``franka_emika_panda/panda.xml``) decoder.

Edits applied to the menagerie model before attaching (all physical, recorded
in the manifest):
* a ``tcp`` site between the fingertip pads (0.1034 m from the hand frame);
* realistic Franka Hand grip: the menagerie tendon servo (kp = 100 N/m, about
  2.5 N squeeze) is replaced by kp = 3000 N/m with a +-140 N force range, i.e.
  roughly 30-70 N per finger like the real hand's 70 N continuous / 140 N max;
* rubber fingertip pads (mu = 1.1, condim 4 for torsional friction) with a
  stiff contact time constant (solref 4 ms) so friction grasps do not creep;
* arm gravity compensation routed through the actuators (``actuatorgravcomp``),
  as the real robot's controller does; it counts against the 87/12 Nm limits.
  Grasped objects are NOT compensated — the arm carries their real weight.

Embedding (8-D): TCP position (3, world, m), TCP
rotation-vector chart about a downward hand (3, rad), aperture (m), normal
force target (N).
"""
from __future__ import annotations

import mujoco
import numpy as np

from eddies.envs.controllers.ik import DLSIK
from eddies.envs.framework.embedding import EmbeddingSpace, RotationChart
from eddies.envs.framework.physics import FRICTION
from eddies.envs.framework.scene import aid, bid, qadr, sid, subtree_geoms

from eddies.common.config import load

ROBOT = load("robots/franka")
HOME = np.array(ROBOT.home, float)
TCP_OFFSET = 0.1034
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])        # hand z-axis pointing to world -z
GRIP_KP = 3000.0
GRIP_KV = 40.0
MAX_APERTURE = float(ROBOT.embedding.max_aperture)
PAD_SOLREF = (0.004, 1.0)
PAD_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)


def edit_panda(spec: mujoco.MjSpec) -> None:
    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.006] * 3, rgba=[1, 0.2, 0.2, 0.0])
    act = spec.actuator("actuator8")
    act.gainprm[0] = GRIP_KP * 0.04 / 255.0
    act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -GRIP_KP, -GRIP_KV
    act.forcerange = np.array([-140.0, 140.0])
    for name in ("left_finger", "right_finger"):
        for g in spec.body(name).geoms:
            if g.contype or g.conaffinity:
                g.friction = np.array(FRICTION["rubber"])
                g.condim = 4
                g.priority = 2       # pad friction/softness govern pad contacts
                # Stiff, high-impedance pad contact.  With MuJoCo's default 20 ms
                # contact time constant a 36 N pinch let a 0.1 kg cube creep out of
                # the pads by >2 cm during a 0.4 m/s lift (a soft-constraint artifact,
                # not slip: the friction limit was 40 N).  4 ms keeps creep < 5 mm.
                g.solref = np.array(PAD_SOLREF)
                g.solimp = np.array(PAD_SOLIMP)
    for b in spec.bodies:
        if b.name.startswith("link") or b.name in ("hand", "left_finger", "right_finger"):
            b.gravcomp = 1.0
    for j in spec.joints:
        j.actgravcomp = True


def panda_space(x_range, y_range, z_range, force_max: float | None = None) -> EmbeddingSpace:
    force_max = float(ROBOT.embedding.force_max) if force_max is None else force_max
    return EmbeddingSpace(
        names=list(ROBOT.embedding.names),
        phys_lower=[x_range[0], y_range[0], z_range[0], -np.pi, -np.pi, -np.pi, 0.0, 0.0],
        phys_upper=[x_range[1], y_range[1], z_range[1], np.pi, np.pi, np.pi, MAX_APERTURE, force_max],
        weights=[float(w) for w in ROBOT.embedding.weights],
        norm_lower=[-1, -1, -1, -1, -1, -1, 0, 0],
        norm_upper=[1, 1, 1, 1, 1, 1, 1, 1],
        units=["m", "m", "m", "rad", "rad", "rad", "m", "N"])


class FrankaArm:
    """Runtime handle for an attached Panda (prefix e.g. ``"franka/"``)."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "franka/"):
        self.model, self.data, self.p = model, data, prefix
        self.joints = [f"{prefix}joint{i}" for i in range(1, 8)]
        self.qadr = np.array([qadr(model, j) for j in self.joints])
        self.dadr = np.array([model.jnt_dofadr[model.joint(j).id] for j in self.joints])
        self.finger_qadr = np.array([qadr(model, f"{prefix}finger_joint{i}") for i in (1, 2)])
        self.act_arm = np.array([aid(model, f"{prefix}actuator{i}") for i in range(1, 8)])
        self.act_grip = aid(model, f"{prefix}actuator8")
        self.tcp = sid(model, f"{prefix}tcp")
        self.ik = DLSIK(model, self.joints, f"{prefix}tcp", posture=HOME, posture_gain=0.05,
                        max_step=0.04, iterations=2)
        self.chart = RotationChart(DOWN_QUAT)
        self.robot_geoms = subtree_geoms(model, f"{prefix}link0")
        self.finger_geoms = {s: [g for g in subtree_geoms(model, f"{prefix}{s}_finger")
                                 if model.geom_contype[g] or model.geom_conaffinity[g]]
                             for s in ("left", "right")}
        self.hand_body = bid(model, f"{prefix}hand")

    # ------------------------------------------------------------------ state
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def dq(self) -> np.ndarray:
        return self.data.qvel[self.dadr].copy()

    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp].copy()

    def tcp_mat(self) -> np.ndarray:
        return self.data.site_xmat[self.tcp].reshape(3, 3).copy()

    def aperture(self) -> float:
        return float(self.data.qpos[self.finger_qadr].sum())

    # ------------------------------------------------------------------ reset
    def set_configuration(self, q=HOME, aperture: float = MAX_APERTURE) -> None:
        """PRE-EPISODE ONLY: write the joint configuration and matching controls."""
        self.data.qpos[self.qadr] = q
        self.data.qpos[self.finger_qadr] = aperture / 2
        self.data.qvel[self.dadr] = 0
        self.data.ctrl[self.act_arm] = q
        self.data.ctrl[self.act_grip] = aperture / MAX_APERTURE * 255.0
        mujoco.mj_forward(self.model, self.data)
        self.ik.reset(np.asarray(q, float))

    # ---------------------------------------------------------------- command
    def command(self, pos, mat, aperture: float) -> None:
        q_ref = self.ik.solve(self.data.qpos, pos, mat, measured_q=self.q())
        self.data.ctrl[self.act_arm] = q_ref
        self.data.ctrl[self.act_grip] = float(np.clip(aperture, 0, MAX_APERTURE)) / MAX_APERTURE * 255.0

    def command_embedding(self, space: EmbeddingSpace, e) -> dict:
        phys = space.denormalize(e)
        mat = self.chart.decode_matrix(phys[3:6])
        self.command(phys[:3], mat, phys[6])
        return {"pos": phys[:3], "mat": mat, "aperture": phys[6], "force": phys[7]}

    def encode(self, space: EmbeddingSpace, pos=None, mat=None, aperture=None, force=0.0) -> np.ndarray:
        pos = self.tcp_pos() if pos is None else pos
        mat = self.tcp_mat() if mat is None else mat
        aperture = self.aperture() if aperture is None else aperture
        rv = self.chart.encode_matrix(mat)
        return space.normalize(np.r_[pos, rv, aperture, force])

    def joint_limit_margin(self) -> float:
        rng = self.model.jnt_range[[self.model.joint(j).id for j in self.joints]]
        q = self.q()
        return float(np.min(np.minimum(q - rng[:, 0], rng[:, 1] - q)))
