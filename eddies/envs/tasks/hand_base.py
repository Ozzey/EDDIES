"""Shared UR5e + dexterous-hand tabletop setup (LEAP Hand / Tetheria Aero Hand Open).

Reusable by every dexterous-hand task (push button, steering wheel, peg
tightening).  HOW TO BUILD A TASK ON IT (see ``tasks/hand_button.py``):

    class MyTask(HandArmTaskBase):
        def __init__(self, seed, hand="leap", **variant):
            super().__init__(seed, hand=hand, **variant)    # sets self.name = f"{hand}_{self.task_id}"
            sb = self.make_builder()                        # table + UR5e + hand (+ pedestal)
            ... add objects / cameras to sb (mjSpec) ...
            self.finish_build(sb, base_mat=hand_frame(z_axis, x_hint))   # compile, HandArm, embedding
            self.cm = self.contact_monitor({"wheel": ["wheel_body"]}, pairs={...})
            self.q_ready = self.solve_ready(grasp_pos, grasp_mat)        # pre-episode IK

* ``self.ha`` is the :class:`~eddies.envs.controllers.hand_arm.HandArm` runtime
  handle (state, fingertip sites, synergies, IK decoder, contact groups).
* Embedding (9-D, see ``controllers/hand_arm.py``): grasp-site position (3),
  rotation-vector chart about ``base_mat`` (3), synergies power/pinch/point (3).
  Build commands with ``self.cmd(pos, mat, s)`` where ``mat`` is the desired
  WORLD rotation of the hand frame (columns x_h = palm normal, y_h = thumb side,
  z_h = finger direction; ``hand_frame(z, x_hint)`` builds one) and
  ``s = (power, pinch, point)`` in [0, 1].
* Fingertip end points: ``self.ha.tip_pos(f)`` for f in ``self.ha.hand.fingers``
  (``th, if, mf, rf`` [+ ``pf`` on the Aero]); measured offset in the hand frame
  ``self.ha.tip_in_hand(f)``.  To put fingertip f at world point p with hand
  orientation R: ``grasp_target = p - R @ offset``.
* Contact groups for ``contact_monitor``: ``arm``, ``hand``, ``palm``, ``fingers``,
  ``finger_<f>``, ``dist_<f>``, ``tip_<f>`` plus ``table`` and your object groups.
* Measured finger flexion (sum of flexion-joint encoders of one finger):
  ``self.finger_flexion(f)``.
* Physics: 2 ms step, gravity compensation of arm + hand payload through the
  arm actuators (UR controller payload model).  Objects you grasp are NOT
  compensated.

PITFALLS (found while building the push-button task; please read):

* MuJoCo 3.15's native CCD returns a FLIPPED contact normal when a box corner
  touches the FLAT face of a cylinder (the Aero fingertip pads are boxes): the
  solver then pushes the two bodies INTO each other (100+ N in a few ms).  Model
  hand-contacted round parts as spheres / capsules / boxes / convex meshes
  (e.g. a steering-wheel rim from capsule segments, a domed button cap);
  touching a cylinder's curved side is fine.
* ``power`` = 1 is a TIGHT fist (collision-free without an object), so an
  enclosed object is squeezed by the servos (LEAP <= 0.5 N m per joint, Aero
  <= 75 N per cable).  A looser "power" posture never squeezes and lets the
  object slide out.  The Aero thumb cables are commanded to stop short of their
  force limit on a ~4 cm object: a force-saturated cable has no position
  stiffness and ratchets inward under transport loads, dragging the object.
* The closing fingers pull an object toward the palm; the grasp site is where a
  4 cm bar settles, so for a power grasp put the grasp site on the object axis.
* Verified (``stiff_pad_transport_test``): 0.1-0.2 kg 4x4x14 cm bar, side power
  grasp, 0.25 m transport at 0.3 m/s -> drift LEAP 0.6-0.8 mm, Aero 3.5-4.0 mm
  (< 5 mm); with MuJoCo's default soft contact: LEAP 2.4-3.1 mm, Aero 16.3 mm.
* The rotation chart is valid only within pi - 0.1 rad of ``base_mat``; choose the
  base near the task's working orientations (``RotationChart`` raises otherwise).
* The arm reference is rate-limited by the IK (3 rad/s joints); reset with
  ``solve_ready`` + ``reset()`` (joint writes only before the episode).
"""
from __future__ import annotations

import mujoco
import numpy as np

from eddies.common.config import load

ROBOT = load("robots/leap")   # LEAP and Aero share every constant below

from eddies.envs.controllers.hand_arm import (ARM_JOINTS, ARM_PREFIX, HANDS, HandArm, attach_hand_arm,
                                               hand_arm_space, hand_frame)
from eddies.envs.controllers.ik import DLSIK
from eddies.envs.framework.contacts import ContactMonitor
from eddies.envs.framework.scene import SceneBuilder, body_geoms, gid
from eddies.envs.framework.task import Task

TABLE_H = 0.75
Q_HOME = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])   # menagerie 'home'

# flexion joints per finger (measured encoders) for each hand
FLEX_JOINTS = {
    "leap": {"if": ("if_mcp", "if_pip", "if_dip"), "mf": ("mf_mcp", "mf_pip", "mf_dip"),
             "rf": ("rf_mcp", "rf_pip", "rf_dip"), "th": ("th_mcp", "th_ipl")},
    "aero": {"if": ("right_index_mcp_flex", "right_index_pip", "right_index_dip"),
             "mf": ("right_middle_mcp_flex", "right_middle_pip", "right_middle_dip"),
             "rf": ("right_ring_mcp_flex", "right_ring_pip", "right_ring_dip"),
             "pf": ("right_pinky_mcp_flex", "right_pinky_pip", "right_pinky_dip"),
             "th": ("right_thumb_cmc_flex", "right_thumb_mcp", "right_thumb_ip")},
}


class HandArmTaskBase(Task):
    family = "dexterous_manipulation"
    task_id = "base"
    physics_dt = float(ROBOT.physics_dt)
    control_dt = float(ROBOT.control_dt)
    rho_step = float(ROBOT.rho_step)          # metric step per control step (configs/robots)
    rho_game = 1.0
    failure_budget = 2
    x_range = (0.15, 0.85)
    y_range = (-0.45, 0.55)
    z_range = (TABLE_H - 0.05, TABLE_H + 0.65)
    table_center = (0.45, 0.0)
    table_half = (0.6, 0.65)
    table_rgba = (0.16, 0.16, 0.17, 1)      # black anodised workbench top
    base_pos = (0.0, 0.0, TABLE_H)

    def __init__(self, seed: int, hand: str = "leap", **variant):
        super().__init__(seed, **variant)
        if hand not in HANDS:
            raise ValueError(f"hand must be one of {sorted(HANDS)}")
        self.hand_name = hand
        self.hand_model = HANDS[hand]
        self.name = f"{hand}_{self.task_id}"
        self.robot = f"UR5e + {self.hand_model.display}"

    # ----------------------------------------------------------------- scene
    def make_builder(self, name: str | None = None, *, link_friction: str = "plastic") -> SceneBuilder:
        sb = SceneBuilder(name or self.name, timestep=self.physics_dt)
        sb.table("table", self.table_center, TABLE_H, self.table_half, rgba=self.table_rgba)
        attach_hand_arm(sb, self.hand_model, base_pos=self.base_pos, link_friction=link_friction)
        sb.static_box("pedestal", (self.base_pos[0], self.base_pos[1], TABLE_H + 0.004), (0.085, 0.085, 0.004),
                      rgba=(0.25, 0.25, 0.27, 1), contype=0, conaffinity=0)
        return sb

    def finish_build(self, sb: SceneBuilder, base_mat) -> None:
        self.model, self.data = sb.compile()
        self.base_mat = np.asarray(base_mat, float)
        self.ha = HandArm(self.model, self.data, self.hand_model, self.base_mat, Q_HOME)
        self.space = hand_arm_space(self.x_range, self.y_range, self.z_range)
        self.table_geoms = [gid(self.model, "table_top")]
        m = self.model
        self._flex_adr = {f: np.array([m.jnt_qposadr[m.joint(self.ha.hp + j).id] for j in js])
                          for f, js in FLEX_JOINTS[self.hand_name].items()}

    def contact_monitor(self, objects: dict[str, list[str]], pairs: dict[str, tuple] | None = None,
                        default_pairs: bool = True) -> ContactMonitor:
        """Hand/arm groups (see module doc) + ``table`` + one group per object (body names).
        Default pairs: tip_<f>_<obj>, hand_<obj>, arm_<obj>, palm_<obj>, arm_table, hand_table."""
        m = self.model
        groups = dict(self.ha.contact_groups())
        groups["table"] = self.table_geoms
        P = {}
        if default_pairs:
            P.update({"arm_table": ("arm", "table"), "hand_table": ("hand", "table")})
        for key, bodies in objects.items():
            groups[key] = body_geoms(m, bodies)
            if default_pairs:
                P[f"hand_{key}"] = ("hand", key)
                P[f"arm_{key}"] = ("arm", key)
                P[f"palm_{key}"] = ("palm", key)
                for f in self.ha.hand.fingers:
                    P[f"tip_{f}_{key}"] = (f"dist_{f}", key)
        P.update(pairs or {})
        return ContactMonitor(m, groups, P)

    # ------------------------------------------------------------- kinematics
    def solve_ready(self, pos, mat, q0=Q_HOME, iters: int = 400, tol: float = 1e-4) -> np.ndarray:
        """PRE-EPISODE: arm joint configuration putting the grasp site at (pos, mat)
        (kinematic shadow only; nothing is written to the simulation)."""
        ik = DLSIK(self.model, self.ha.arm_joints, self.ha.hp + "grasp", posture=np.asarray(q0, float),
                   posture_gain=0.02, max_step=0.05, iterations=1, rot_weight=1.0, damping=0.01)
        qpos = self.data.qpos.copy()
        q = np.asarray(q0, float).copy()
        ik.reset(q)
        for _ in range(iters):
            qpos[self.ha.qadr] = q
            q = ik.solve(qpos, pos, mat, measured_q=q, resync_tol=1e9)
            ik.shadow.qpos[self.ha.qadr] = q
            mujoco.mj_kinematics(self.model, ik.shadow)
            p, R = ik.site_pose(ik.shadow)
            if np.linalg.norm(p - pos) < tol and np.linalg.norm(R - np.asarray(mat)) < 10 * tol:
                break
        return q

    def finger_flexion(self, f: str) -> float:
        return float(self.data.qpos[self._flex_adr[f]].sum())

    # ----------------------------------------------------------------- reset
    def reset(self) -> None:
        self.ha.set_configuration(self.q_ready, self.s_ready)
        self.settle(0.8)

    s_ready = (0.0, 0.0, 0.0)

    def settle(self, seconds: float) -> None:
        """Pre-episode settling under gravity with the controllers holding the reset pose."""
        for _ in range(int(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0

    # --------------------------------------------------------------- command
    def cmd(self, pos, mat, s=(0.0, 0.0, 0.0)) -> np.ndarray:
        return self.ha.cmd(self.space, pos, mat, s)

    def initial_command(self) -> np.ndarray:
        return self.ha.encode(self.space)

    def apply_command(self, e) -> None:
        self.ha.command_embedding(self.space, e)

    def achieved(self, obs: dict) -> np.ndarray:
        return obs["achieved"]

    def base_obs(self) -> dict:
        ha = self.ha
        return {"t": self.time, "q": ha.q(), "dq": ha.dq(), "grasp": ha.grasp_pos(), "grasp_mat": ha.grasp_mat(),
                "synergy": ha.synergy(), "hand_len": ha.actuator_lengths(),
                "tips": {f: ha.tip_pos(f) for f in ha.hand.fingers},
                "flex": {f: self.finger_flexion(f) for f in FLEX_JOINTS[self.hand_name]},
                "achieved": ha.encode(self.space)}

    def rotation_error(self, R_target) -> float:
        """Angle [rad] between the measured hand frame and ``R_target``."""
        R = self.ha.grasp_mat().T @ np.asarray(R_target)
        return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def stiff_pad_transport_test(hand: str = "leap", *, speed: float = 0.3, distance: float = 0.25,
                             stiff: bool = True, object_mass: float = 0.20, seed: int = 0,
                             obj_offset=(0.0, 0.0, 0.0), power: float = 1.0, snapshot=None) -> dict:
    """Stiff-pad rule check (DEVELOPER_GUIDE 2.4): power-grasp a 4 x 4 x 14 cm bar (0.2 kg, plastic)
    from the side, lift it 6 cm and carry it ``distance`` at ``speed`` (trapezoidal profile,
    2 m/s^2 ramps).  Returns the object's drift in the hand (grasp-site) frame between the
    settled poses before and after the transport, and the maximum drift during it.
    ``stiff=False`` restores MuJoCo's default soft contact (solref 0.02 s) on the hand for comparison."""
    from eddies.envs.framework.physics import FRICTION
    hm = HANDS[hand]

    class _Probe(HandArmTaskBase):
        task_id = "pad_probe"

    probe = _Probe(seed, hand=hand)
    sb = probe.make_builder("pad_probe_" + hand)
    obj_c = np.array([0.50, 0.0, TABLE_H + 0.10 + 0.07])
    sb.static_box("stand", (0.50, 0.0, TABLE_H + 0.05), (0.03, 0.03, 0.05), rgba=(0.3, 0.3, 0.32, 1))
    sb.free_box("bar", obj_c + [0, 0, 0.0005], (0.02, 0.02, 0.07), mass=object_mass, rgba=(0.85, 0.4, 0.1, 1),
                friction=FRICTION["plastic"])
    if not stiff:
        for g in sb.spec.geoms:
            if g.name.startswith(hm.name + "/"):
                g.solref = np.array([0.02, 1.0])
                g.solimp = np.array([0.9, 0.95, 0.001, 0.5, 2.0])
    R = hand_frame((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))       # fingers forward, palm facing +y, thumb up
    probe.finish_build(sb, base_mat=R)
    m, d, ha = probe.model, probe.data, probe.ha
    bar = m.body("bar").id
    g_at = obj_c - R @ np.asarray(obj_offset, float)      # grasp-site target (object centre = grasp + offset)
    pre = g_at - 0.08 * R[:, 0]
    q = probe.solve_ready(pre, R)
    ha.set_configuration(q, (0.0, 0.0, 0.0))
    probe.settle(0.5)
    dec = int(round(probe.control_dt / m.opt.timestep))

    def run(duration, path, synergy):
        n = int(round(duration / probe.control_dt))
        for i in range(n):
            ha.command(path((i + 1) / n), R, synergy((i + 1) / n))
            for _ in range(dec):
                mujoco.mj_step(m, d)

    def in_hand():
        return ha.grasp_mat().T @ (d.xpos[bar] - ha.grasp_pos())

    def snap(tag):
        if snapshot is not None:
            snapshot(tag, m, d, in_hand())

    grasp_at = g_at.copy()
    snap("0_pregrasp")
    run(1.0, lambda s: pre + s * (grasp_at - pre), lambda s: (0, 0, 0))
    run(0.6, lambda s: grasp_at, lambda s: (0, 0, 0))          # let the arm converge before closing
    snap("1_at_object")
    run(0.8, lambda s: grasp_at, lambda s: (power * min(1.0, 1.4 * s), 0, 0))
    run(0.6, lambda s: grasp_at, lambda s: (power, 0, 0))
    snap("2_closed")
    lift = grasp_at + np.array([0, 0, 0.06])
    run(0.8, lambda s: grasp_at + s * (lift - grasp_at), lambda s: (power, 0, 0))
    run(0.5, lambda s: lift, lambda s: (power, 0, 0))
    snap("3_lifted")
    p0 = in_hand()
    # trapezoidal transport along +y
    acc, v = 2.0, speed
    t_r = v / acc
    T = distance / v + t_r

    def pos(t):
        if t < t_r:
            s = 0.5 * acc * t * t
        elif t < T - t_r:
            s = 0.5 * acc * t_r * t_r + v * (t - t_r)
        else:
            s = distance - 0.5 * acc * max(T - t, 0.0) ** 2
        return lift + np.array([0.0, min(s, distance), 0.0])

    n = int(round(T / probe.control_dt))
    max_drift, vmax = 0.0, 0.0
    for i in range(n):
        ha.command(pos((i + 1) * probe.control_dt), R, (power, 0, 0))
        for _ in range(dec):
            mujoco.mj_step(m, d)
        max_drift = max(max_drift, float(np.linalg.norm(in_hand() - p0)))
        vmax = max(vmax, float(np.linalg.norm(d.cvel[bar][3:])))
    run(0.5, lambda s: pos(T), lambda s: (power, 0, 0))
    snap("4_transported")
    p1 = in_hand()
    if snapshot is not None:
        snapshot("drift_vector", m, d, p1 - p0)
    held = bool(d.xpos[bar][2] > obj_c[2] + 0.03)
    return {"hand": hand, "stiff_pads": stiff, "speed_cmd_m_s": speed, "object_peak_speed_m_s": vmax,
            "drift_settled_m": float(np.linalg.norm(p1 - p0)), "drift_max_m": max_drift, "held": held,
            "object_lift_m": float(d.xpos[bar][2] - obj_c[2])}


REOPEN_TOL = 0.15        # [rad] per-finger flexion-sum tolerance w.r.t. the settled open posture
REOPEN_ABD_TOL = 0.10    # [rad] Aero thumb abduction tolerance


def synergy_reopen_test(hand: str = "leap", closed=(1.0, 0.0, 0.0), *, close_time: float = 1.2,
                        open_time: float = 1.5, palm_down: bool = True, seed: int = 0) -> dict:
    """Release self-test: can the hand physically RE-OPEN after a synergy?  The arm holds the hand
    in free air (default palm DOWN, fingers forward: gravity then pulls the fingers closed, the worst
    case for a spring-returned hand).  Sequence: settle in the open posture (baseline flexion per
    finger), command ``closed`` = (power, pinch, point) for ``close_time`` s, then command open.
    ``t_reopen_s`` = first time after the open command at which every finger's measured flexion sum
    is within ``REOPEN_TOL`` of its open baseline (and, for the Aero, the thumb abduction within
    ``REOPEN_ABD_TOL``), held to the end of ``open_time``; ``None`` if it never re-opens.  Also
    reports the thumb-index pad-end gap.  Run it for both hands after any change to postures,
    springs or actuator limits (``python -m eddies.envs.tasks.hand_base`` runs the full matrix)."""

    class _Probe(HandArmTaskBase):
        task_id = "reopen_probe"

    probe = _Probe(seed, hand=hand)
    sb = probe.make_builder("reopen_probe_" + hand)
    R = hand_frame((1.0, 0.0, 0.0), (0.0, 0.0, -1.0) if palm_down else (0.0, 1.0, 0.0))
    probe.finish_build(sb, base_mat=R)
    m, d, ha = probe.model, probe.data, probe.ha
    g = np.array([0.45, 0.0, TABLE_H + 0.30])
    q = probe.solve_ready(g, R)
    ha.set_configuration(q, (0.0, 0.0, 0.0))
    probe.settle(0.6)
    fingers = list(FLEX_JOINTS[hand])
    abd = (m.jnt_qposadr[m.joint(ha.hp + "right_thumb_cmc_abd").id] if hand == "aero" else None)

    def flex():
        return np.array([probe.finger_flexion(f) for f in fingers])

    def gap():
        return float(np.linalg.norm(ha.tip_pos("th") - ha.tip_pos("if")))

    f_open, abd_open, gap_open = flex(), (float(d.qpos[abd]) if abd is not None else 0.0), gap()
    dec = int(round(probe.control_dt / m.opt.timestep))
    n_close = int(round(close_time / probe.control_dt))
    for i in range(n_close):
        ha.command(g, R, np.asarray(closed, float) * min(1.0, (i + 1) / (0.3 / probe.control_dt)))
        for _ in range(dec):
            mujoco.mj_step(m, d)
    f_closed, gap_closed = flex(), gap()
    t_ok, at_half = None, None
    n_open = int(round(open_time / probe.control_dt))
    for i in range(n_open):
        ha.command(g, R, (0.0, 0.0, 0.0))
        for _ in range(dec):
            mujoco.mj_step(m, d)
        t = (i + 1) * probe.control_dt
        ok = bool(np.all(np.abs(flex() - f_open) < REOPEN_TOL)
                  and (abd is None or abs(float(d.qpos[abd]) - abd_open) < REOPEN_ABD_TOL))
        if ok and t_ok is None:
            t_ok = t
        elif not ok:
            t_ok = None
        if at_half is None and t >= 0.5 - 1e-9:
            at_half = flex()
    f_end = flex()
    return {"hand": hand, "closed_synergy": tuple(float(x) for x in closed), "palm_down": palm_down,
            "t_reopen_s": t_ok, "fingers": fingers,
            "flex_open_rad": np.round(f_open, 3).tolist(), "flex_closed_rad": np.round(f_closed, 3).tolist(),
            "flex_after_0.5s_rad": np.round(at_half, 3).tolist(), "flex_end_rad": np.round(f_end, 3).tolist(),
            "thumb_index_gap_open_m": gap_open, "thumb_index_gap_closed_m": gap_closed,
            "thumb_index_gap_end_m": gap(), "max_flex_residual_rad": float(np.max(np.abs(f_end - f_open)))}


def _self_test() -> int:
    """Release matrix: power/pinch/point -> open on both hands; pass = re-open within 0.5 s."""
    bad = 0
    for hand in ("leap", "aero"):
        for name, s in (("power", (1, 0, 0)), ("pinch", (0, 1, 0)), ("point", (0, 0, 1))):
            r = synergy_reopen_test(hand, s)
            ok = r["t_reopen_s"] is not None and r["t_reopen_s"] <= 0.5
            bad += not ok
            print(f"{hand:5s} {name:6s}->open  t_reopen={r['t_reopen_s']}  closed={r['flex_closed_rad']}  "
                  f"0.5s={r['flex_after_0.5s_rad']}  end={r['flex_end_rad']}  open={r['flex_open_rad']}  "
                  f"gap {1000 * r['thumb_index_gap_closed_m']:.1f}->{1000 * r['thumb_index_gap_end_m']:.1f} mm  "
                  f"{'PASS' if ok else 'FAIL'}")
    return bad


if __name__ == "__main__":
    raise SystemExit(_self_test())


__all__ = ["HandArmTaskBase", "TABLE_H", "Q_HOME", "FLEX_JOINTS", "hand_frame", "ARM_JOINTS", "ARM_PREFIX",
           "stiff_pad_transport_test", "synergy_reopen_test"]
