"""Task 1 — Franka opens a drawer, takes a box out of it and puts it on the table.

Scene: a two-drawer wooden chest (``assets/furniture.add_drawer_cabinet``) stands
on the 0.75 m table in front of the Panda, its front facing the robot.  The TOP
drawer (1.6-2.6 kg hollow box on a slide joint: damping 5-15 N s/m, breakaway
friction 1-4 N, travel 0.28 m with hard stops) holds a small cardboard box
(4.5 x 4.5 x 7 cm, 0.08-0.18 kg).  The robot

1. grasps the drawer's metal bar handle (16 mm, 40 mm projection) with a PITCHED
   pinch (approach axis 60 deg below horizontal, fingers closing in the vertical
   plane across the bar).  On the Panda's natural arm branch a horizontal hand
   cannot reach poses closer than ~0.88 m (joint-6 range), so a pitched grasp is
   the kinematically sound choice; the pull is then mostly form closure (the
   upper finger pushes the bar), not only friction;
2. pulls the drawer open (measured slide-joint opening >= 0.20 m), releases and
   backs off from the handle;
3. picks the box top-down (fingers closing parallel to the drawer front so the
   20 cm long hand housing clears the front panel), lifts it out and places it
   on the table inside a goal disc, releases and retreats.

Everything is physical: the drawer moves only through finger contact forces on
the handle; grasps hold only by pad friction; all labels/outcomes are measured
(slide-joint encoder/vision opening, mj_contactForce contact forces, poses).

Failure outcomes in the contracts: handle missed (gripper closes past the bar),
handle slip while pulling (-> re-grasp), drawer found (pushed) closed again
(-> re-open), object grasp miss, object dropped (-> re-pick from wherever it
rests).  Stress test: the drawer is pushed back in by 8-12 cm shortly after the
handle is released, and the gripper force drops out for 0.4 s while the box is
carried (it falls), both triggered by measured state.

Decoder (shared by every method, ``ArticulatedFrankaBase``): the gripper aperture
reference is rate-limited to the Franka Hand's 0.05 m/s per finger (an unbounded
reference jump drove the light menagerie fingers at ~2 m/s into the object, giving
non-physical kN contact impulses).  'bad' includes finger over-force: pad-object or
pad-handle normal force > 300 N for >= 2 consecutive physics steps.

LTLf:  G !bad & F(opened & F(grasped & F(placed & released)))
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from eddies.core.online import RecursiveLeastSquares
from eddies.envs.assets.furniture import add_drawer_cabinet
from eddies.envs.controllers.franka import HOME, MAX_APERTURE
from eddies.envs.controllers.ik import DLSIK, rotation_error_mat
from eddies.envs.framework.contacts import ContactMonitor
from eddies.envs.framework.perception import PoseEstimator, tilt_of_quat, wrap_to, yaw_of_quat
from eddies.envs.framework.scene import body_geoms, gid, yaw_quat
from eddies.envs.framework.task import OK, Skill, SkillSpec, OutcomeSpec, word
from eddies.envs.tasks.franka_base import TABLE_H, FrankaTaskBase

# ----------------------------------------------------------------- helpers
DOWN = np.diag([1.0, -1.0, -1.0])        # top-down hand, fingers closing along world y


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def pinch_rotation(heading: float, pitch: float, fingers: str = "vertical") -> np.ndarray:
    """Hand rotation whose approach axis (hand z) points along horizontal ``heading``
    tilted ``pitch`` below the horizontal.  ``fingers='vertical'``: the closing axis
    lies in the vertical plane of the approach (pinch across a HORIZONTAL bar);
    ``'horizontal'``: the closing axis is horizontal (pinch around a VERTICAL bar)."""
    if fingers == "vertical":
        return rot_z(heading) @ rot_y(-np.pi / 2 + pitch) @ rot_z(np.pi / 2) @ DOWN
    return rot_z(heading) @ rot_y(-np.pi / 2 + pitch) @ DOWN


def rot_err(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(Rot.from_matrix(Ra @ Rb.T).magnitude())


def top_down(yaw: float) -> np.ndarray:
    return rot_z(yaw) @ DOWN


def box_grasp_yaw(box_yaw: float, prefer: float = 0.0) -> float:
    """Of the two finger-closing axes perpendicular to box faces, the hand yaw closest to ``prefer``."""
    return float(wrap_to(box_yaw - prefer, np.pi / 2) + prefer)


class WeightedDLSIK(DLSIK):
    """Joint-weighted damped-least-squares IK (same interface/reference integrator as DLSIK).

    dq = W^-1 J^T (J W^-1 J^T + lambda^2 I)^-1 e  (+ weighted null-space posture term).
    Weighting joint 3 (and slightly joint 5) makes the solver avoid the large opposite
    q1/q3 swings that plain DLS produces near the Panda shoulder singularity (q2 ~ 0, joint
    1 and 3 axes collinear), which occurs when the pitched hand reaches into a raised
    cavity.  Shared by every method of the task (it is part of the task's decoder).
    """

    def __init__(self, model, joint_names, site, *, weights, **kw):
        super().__init__(model, joint_names, site, **kw)
        self.winv = 1.0 / np.asarray(weights, float)

    def solve(self, live_qpos, target_pos, target_mat=None, *, resync_tol: float = 0.15,
              measured_q=None):
        m, d = self.model, self.shadow
        q_meas = live_qpos[self.qadr] if measured_q is None else measured_q
        if self.q_ref is None or np.max(np.abs(self.q_ref - q_meas)) > resync_tol:
            self.q_ref = q_meas.copy()
        d.qpos[:] = live_qpos
        q = self.q_ref.copy()
        n = len(q)
        Wi = np.diag(self.winv)
        for _ in range(self.iterations):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos, R = self.site_pose(d)
            err_p = np.asarray(target_pos, float) - pos
            mujoco.mj_jacSite(m, d, self.jacp, self.jacr, self.site)
            if target_mat is None:
                J, err = self.jacp[:, self.dadr], err_p
            else:
                err_r = rotation_error_mat(R, np.asarray(target_mat, float).reshape(3, 3))
                J = np.vstack([self.jacp[:, self.dadr], self.rot_weight * self.jacr[:, self.dadr]])
                err = np.concatenate([err_p, self.rot_weight * err_r])
            A = J @ Wi @ J.T + (self.damping ** 2) * np.eye(J.shape[0])
            Jp = Wi @ J.T @ np.linalg.inv(A)
            dq = Jp @ err
            if self.posture is not None:
                dq += (np.eye(n) - Jp @ J) @ (self.posture_gain * (self.posture - q))
            mx = np.max(np.abs(dq))
            if mx > self.max_step:
                dq *= self.max_step / mx
            q = np.clip(q + dq, self.lower, self.upper)
        self.q_ref = q
        return q.copy()


class ArticulatedFrankaBase(FrankaTaskBase):
    """Shared machinery of the Franka articulated-furniture tasks (drawer, microwave)."""

    def use_weighted_ik(self, weights=(1.0, 1.0, 4.0, 1.0, 2.0, 1.0, 1.0)) -> None:
        """Install the joint-weighted DLS IK in the arm decoder (same gains as FrankaArm's)."""
        self.arm.ik = WeightedDLSIK(self.model, self.arm.joints, f"{self.arm.p}tcp", posture=HOME,
                                    posture_gain=0.05, max_step=0.04, iterations=2, weights=weights)
        self.ik_weights = tuple(float(w) for w in weights)

    # ------------------------------------------------- joint-reference interpolation
    # The IK runs at the 50 Hz control rate; a raw 50 Hz staircase reference makes
    # the menagerie wrist servos (kp 2000, kv 200, |tau| <= 12 N m) bang-bang between
    # the torque limits whenever a wrist joint must move at ~0.5-1 rad/s (each
    # 20 ms step asks for > 12 N m).  Like the real Franka controller, which
    # interpolates set-points at 1 kHz, the decoder can ramp the joint reference
    # linearly over the control period (first-order hold).  Opt-in per task;
    # shared by every method of that task.
    interp_ref = False

    # ------------------------------------------------- gripper speed limit (Franka Hand)
    # The menagerie finger servo (kp 3000 N/m on a ~15 g finger) closes from fully open
    # in ~10 ms at ~2 m/s per finger when the aperture reference jumps; the real Franka
    # Hand is speed-limited to ~0.05 m/s per finger (0.10 m/s aperture) by its
    # firmware.  An unbounded jump made the fingers strike the object at 2 m/s and
    # tunnel into its collision planks (10 ms, 20 kN contact impulses).  The decoder
    # therefore ramps the aperture reference at <= GRIP_RATE every physics step (shared
    # by all methods; it only shapes the reference -- the 140 N force limit is unchanged).
    GRIP_RATE = 0.10                 # m/s aperture (= 0.05 m/s per finger, Franka Hand spec)
    _grip_target = None

    def apply_command(self, e) -> None:
        act, g = self.arm.act_arm, self.arm.act_grip
        prev = self.data.ctrl[act].copy()
        grip_now = float(self.data.ctrl[g])
        self.arm.command_embedding(self.space, e)
        self._grip_target = float(self.data.ctrl[g])
        self.data.ctrl[g] = grip_now                 # rate-limited in inner_loop
        if not self.interp_ref:
            return
        self._ref_new = self.data.ctrl[act].copy()
        self._ref_old = prev if getattr(self, "_ref_new_init", False) else self._ref_new.copy()
        self._ref_new_init = True
        self._ref_k = 0
        self._ref_n = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        self.data.ctrl[act] = self._ref_old

    def inner_loop(self) -> None:
        if self._grip_target is not None:
            g = self.arm.act_grip
            step = self.GRIP_RATE / MAX_APERTURE * 255.0 * self.model.opt.timestep
            cur = float(self.data.ctrl[g])
            self.data.ctrl[g] = cur + float(np.clip(self._grip_target - cur, -step, step))
        if not self.interp_ref or getattr(self, "_ref_new", None) is None:
            return
        self._ref_k = min(self._ref_k + 1, self._ref_n)
        s = self._ref_k / self._ref_n
        self.data.ctrl[self.arm.act_arm] = self._ref_old + s * (self._ref_new - self._ref_old)

    # ------------------------------------------------- finger over-force ('bad')
    # A Franka Hand grasps with <= 140 N; pad-object (or pad-handle) normal forces above
    # CRUSH_N sustained for >= CRUSH_STEPS physics steps are a genuine safety violation
    # (crushing the object / overloading the fingers), whatever produced them.
    CRUSH_N = 300.0
    CRUSH_STEPS = 2

    def crush_violation(self, f: dict) -> bool:
        """Call once per labels() evaluation; counts consecutive over-force physics steps."""
        t = float(self.data.time)
        if getattr(self, "_crush_t", None) != t:
            self._crush_t = t
            over = max(f["L_obj"] + f["R_obj"], f["L_h"] + f["R_h"]) > self.CRUSH_N
            self._crush_n = getattr(self, "_crush_n", 0) + 1 if over else 0
            if self._crush_n >= self.CRUSH_STEPS:
                self.crush_events = getattr(self, "crush_events", 0) + (self._crush_n == self.CRUSH_STEPS)
        return getattr(self, "_crush_n", 0) >= self.CRUSH_STEPS

    # ------------------------------------------------- open-hand impact ('bad')
    # With the hand OPEN (aperture > object width + 1 cm) the pads exert no squeeze, so a
    # large pad-object force is an unintended collision (e.g. a fingertip landing on a cup
    # rim during the approach).  One pad above IMPACT_N -- more than the hand's own 140 N
    # maximum grasp force -- for >= IMPACT_STEPS physics steps is a safety violation.
    IMPACT_N = 150.0
    IMPACT_STEPS = 2

    def open_pad_force(self, f: dict, open_width: float) -> float:
        """Largest single-pad object force while the hand is open (0 when it is not open)."""
        return max(f["L_obj"], f["R_obj"]) if self.arm.aperture() > open_width else 0.0

    def open_impact_violation(self, f: dict, open_width: float) -> bool:
        """Call once per labels() evaluation; counts consecutive open-hand impact steps."""
        t = float(self.data.time)
        if getattr(self, "_imp_t", None) != t:
            self._imp_t = t
            pad = self.open_pad_force(f, open_width)
            self.max_open_pad_force = max(getattr(self, "max_open_pad_force", 0.0), pad)
            self._imp_n = getattr(self, "_imp_n", 0) + 1 if pad > self.IMPACT_N else 0
            if self._imp_n == self.IMPACT_STEPS:
                self.impact_events = getattr(self, "impact_events", 0) + 1
        return getattr(self, "_imp_n", 0) >= self.IMPACT_STEPS

    def cmd_R(self, pos, R, aperture: float, force: float = 0.0) -> np.ndarray:
        """Normalized command for TCP position ``pos`` and hand rotation matrix ``R``."""
        return self.space.normalize(np.r_[np.asarray(pos, float), self.arm.chart.encode_matrix(R),
                                          float(aperture), float(force)])

    def make_contacts(self, handle_bodies, obj_bodies, furniture_bodies):
        """Contact groups: fingers L/R, arm links, table, handle, object, furniture, environment."""
        m = self.model
        L, R = self.arm.finger_geoms["left"], self.arm.finger_geoms["right"]
        fingers = L + R
        arm_only = [g for g in self.arm.robot_geoms if g not in fingers and
                    (m.geom_contype[g] or m.geom_conaffinity[g])]
        handle = body_geoms(m, handle_bodies)
        obj = body_geoms(m, obj_bodies)
        furn = [g for g in body_geoms(m, furniture_bodies) if g not in handle and
                (m.geom_contype[g] or m.geom_conaffinity[g])]
        env = sorted(set(furn) | set(handle) | set(self.table_geoms) | set(self.extra_env_geoms()))
        groups = {"L": L, "R": R, "fingers": fingers, "arm": arm_only, "table": self.table_geoms,
                  "handle": handle, "obj": obj, "furn": furn, "env": env}
        pairs = {"L_obj": ("L", "obj"), "R_obj": ("R", "obj"), "L_h": ("L", "handle"),
                 "R_h": ("R", "handle"), "arm_furn": ("arm", "furn"), "fing_furn": ("fingers", "furn"),
                 "arm_handle": ("arm", "handle"), "arm_table": ("arm", "table"),
                 "fing_table": ("fingers", "table"), "obj_env": ("obj", "env"),
                 "obj_table": ("obj", "table"), "arm_obj": ("arm", "obj")}
        return ContactMonitor(m, groups, pairs)

    def extra_env_geoms(self) -> list:
        return [gid(self.model, "floor")]

    def forces(self) -> dict:
        if self._contacts.get("t") != self.data.time:
            self._contacts = self.cm.measure(self.data)
            self._contacts["t"] = self.data.time
        return self._contacts

    def hand_rot_err(self, R) -> float:
        return rot_err(self.arm.tcp_mat(), R)


# ----------------------------------------------------------------- constants
CAB_DEPTH = 0.38
CAB_WIDTH = 0.44
DRAWER_H = (0.12, 0.12)
TRAVEL = 0.28
BAR_R = 0.008
BAR_STANDOFF = 0.040
BAR_FRAC = 0.75
PITCH_H = np.radians(60.0)       # handle pinch: approach 60 deg below horizontal
AP_HANDLE = 0.045                # pre-grasp aperture around the 16 mm bar
PRE_HANDLE = 0.08                # pre-grasp distance back along the approach axis
OPEN_MIN = 0.18                  # 'opened': box can be picked top-down
OPEN_TARGET = 0.20               # pull token goal (measured opening)
PULL_CMD = 0.222                 # commanded pull (arm PD lag ~1.5-2 cm)
OBJ_HALF = np.array([0.0225, 0.0225, 0.035])
GRASP_DZ = 0.008                 # TCP above the box centre when grasping
H_ABOVE = 0.12                   # pre-grasp height above the box centre
LIFT = 0.15                      # lift height (box bottom clears the drawer walls)
GOAL_NOM = np.array([0.33, -0.40])
GOAL_TOL = 0.04
OPEN_HAND_W = 2 * OBJ_HALF[0] + 0.01       # aperture above which the hand counts as open (open-hand impact rule)


class FrankaDrawerTask(ArticulatedFrankaBase):
    name = "franka_drawer"
    description = ("Franka Panda opens the top drawer of a two-drawer cabinet by its bar handle (pitched "
                   "pinch; measured slide-joint opening), takes a small box out of the drawer and places it "
                   "on the table in a goal region (friction grasps only).")
    horizon = 80.0
    success_hold = 1.0
    formula = "G !bad & F(opened & F(grasped & F(placed & released)))"
    terminal_cells = ("done",)
    failure_budget = 3
    rho_game = 1.5
    main_camera = "main"
    detail_camera = "detail"

    def __init__(self, seed: int, **variant):
        super().__init__(seed, **variant)
        rng = self.rng
        # ------------------------------------------------ randomized parameters
        self.front_x = rng.uniform(0.625, 0.655)              # carcass front plane (world x)
        self.cab_y = rng.uniform(-0.03, 0.03)
        self.cab_yaw = np.pi + rng.uniform(-0.04, 0.04)        # front faces the robot (-x)
        self.drawer_mass = rng.uniform(1.6, 2.6)
        self.drawer_damping = rng.uniform(5.0, 15.0)
        self.drawer_fric = rng.uniform(1.0, 4.0)
        self.obj_depth = rng.uniform(0.075, 0.095)             # behind the front panel's inner face
        self.obj_lat = rng.uniform(-0.04, 0.04)
        self.obj_yaw0 = rng.uniform(-0.25, 0.25)
        self.obj_mass = rng.uniform(0.08, 0.18)
        self.obj_mu = rng.uniform(0.45, 0.65)
        self.goal = GOAL_NOM + rng.uniform(-0.03, 0.03, 2)
        # ------------------------------------------------------------- scene
        sb = self.make_builder(self.name)
        n_out = np.array([np.cos(self.cab_yaw), np.sin(self.cab_yaw), 0.0])
        center = np.array([self.front_x, self.cab_y, 0.0]) - (CAB_DEPTH / 2) * n_out
        self.cab_center = center + [0, 0, TABLE_H]
        info = add_drawer_cabinet(
            sb, "cabinet", self.cab_center, self.cab_yaw, depth=CAB_DEPTH, width=CAB_WIDTH,
            drawer_heights=DRAWER_H, travel=TRAVEL,
            drawer_mass=[2.0, self.drawer_mass], damping=[8.0, self.drawer_damping],
            frictionloss=[2.0, self.drawer_fric], handle_radius=BAR_R, handle_standoff=BAR_STANDOFF,
            handle_height_frac=BAR_FRAC)
        self.cab_info = info
        self.top = info["drawers"][-1]
        Rc = rot_z(self.cab_yaw)
        self.n_out = n_out                                     # opening direction (world)
        floor = self.top["interior_floor_local"]
        obj_local = np.array([self.top["front_inner_local_x"] - self.obj_depth, self.obj_lat,
                              floor[2] + OBJ_HALF[2] + 0.0005])
        self.obj0 = self.cab_center + Rc @ obj_local
        self.floor_z0 = TABLE_H + floor[2]
        self.front_top_z = TABLE_H + self.top["front_top_local_z"]
        self.cab_top_z = TABLE_H + info["size"][2]
        body = sb.free_box("box", self.obj0, OBJ_HALF, mass=self.obj_mass, rgba=(0.92, 0.48, 0.10, 1),
                           quat=yaw_quat(self.cab_yaw - np.pi + self.obj_yaw0),
                           friction=(self.obj_mu, 0.005, 1e-4))
        body.geoms[0].priority = 1
        # goal marker (visual only)
        sb.spec.worldbody.add_geom(name="goal_marker", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                   pos=[*self.goal, TABLE_H + 0.0005],
                                   size=[GOAL_TOL, 0.0005, 0], rgba=[0.1, 0.75, 0.25, 0.45],
                                   contype=0, conaffinity=0, group=1, mass=0.0)
        hz = TABLE_H + self.top["handle_closed_local"][2]
        sb.camera("main", (0.12, -1.22, TABLE_H + 0.86), (0.42, -0.06, TABLE_H + 0.26), fovy=50)
        sb.camera("detail", (0.30, 0.34, hz + 0.36), (self.front_x - 0.10, self.cab_y - 0.02, hz - 0.03), fovy=50)
        sb.camera("top", (0.5, 0.0, TABLE_H + 1.5), (0.5, 0.0, TABLE_H), fovy=55)
        self.finish_build(sb)
        m = self.model
        self.jq_drawer = int(m.jnt_qposadr[m.joint(self.top["joint"]).id])
        self.jd_drawer = int(m.jnt_dofadr[m.joint(self.top["joint"]).id])
        self.drawer_bid = m.body(self.top["body"]).id
        self.obj_bid = m.body("box").id
        cab_bodies = [info["body"]] + [d["body"] for d in info["drawers"]] + \
                     [d["handle_body"] for d in info["drawers"][:-1]]
        self._contacts = {}
        self.cm = self.make_contacts([self.top["handle_body"]], ["box"], cab_bodies)
        # perception (shared by all methods; same per seed)
        self.est_obj = PoseEstimator(rng)
        self.est_handle = PoseEstimator(rng, bias_far=0.003, sigma_far=0.001)
        self.q_noise = 0.001
        self._build_skills()
        # stress test schedule (drawn even when unused so the nominal RNG stream is unchanged)
        self._push_delay = rng.uniform(0.15, 0.45)
        self._push_dist = rng.uniform(0.08, 0.12)
        self._drop_delay = rng.uniform(0.0, 0.15)
        self._pert = {"drawer_push": None, "grip_dropout": None}
        self._grip_range = self.model.actuator_forcerange[self.arm.act_grip].copy()
        self._handle_free_since = None
        self.max_opening = 0.0

    # ================================================================ state
    def opening(self) -> float:
        return float(self.data.qpos[self.jq_drawer])

    def handle_pos(self) -> np.ndarray:
        return self.data.site(self.top["handle_site"]).xpos.copy()

    def support_z(self, p) -> float:
        """Height of the surface under point ``p``: open drawer floor, cabinet top or table."""
        Rc = rot_z(self.cab_yaw)
        rel = Rc.T @ (np.asarray(p, float) - self.data.site(self.top["interior_site"]).xpos)
        hx, hy = self.top["interior_half"]
        if abs(rel[0]) < hx and abs(rel[1]) < hy:
            return self.floor_z0
        relc = Rc.T @ (np.asarray(p, float) - self.cab_center)
        if abs(relc[0]) < CAB_DEPTH / 2 and abs(relc[1]) < CAB_WIDTH / 2:
            return self.cab_top_z
        return TABLE_H

    def _geometry(self) -> dict:
        d = self.data
        f = self.forces()
        obj = d.xpos[self.obj_bid].copy()
        oq = d.xquat[self.obj_bid].copy()
        ov = d.cvel[self.obj_bid][3:].copy()
        hold_obj = f["L_obj"] > 1.0 and f["R_obj"] > 1.0
        hold_handle = f["L_h"] > 1.0 and f["R_h"] > 1.0
        in_air = f["obj_env"] < 0.05
        tilt = tilt_of_quat(oq)
        # resting height of the box on the table: half height of the face it stands on
        Rb = d.xmat[self.obj_bid].reshape(3, 3)
        up = np.abs(Rb[2, :])
        half_z = float(OBJ_HALF[int(np.argmax(up))])
        on_table = (f["obj_table"] > 0.2 and abs(obj[2] - (TABLE_H + half_z)) < 0.008)
        lift = obj[2] - half_z - self.support_z(obj)
        in_goal = np.linalg.norm(obj[:2] - self.goal) < GOAL_TOL
        flat = up.max() > np.cos(0.3)
        return {"obj": obj, "oq": oq, "ov": ov, "hold_obj": hold_obj, "hold_handle": hold_handle,
                "in_air": in_air, "lifted": bool(in_air and lift > 0.008), "lift": lift, "tilt": tilt, "on_table": on_table, "in_goal": in_goal,
                "placed": bool(on_table and in_goal and flat), "half_z": half_z, "f": f,
                "opening": self.opening()}

    def observe(self) -> dict:
        obs = self.base_obs()
        g = self._geometry()
        tcp = obs["tcp"]
        op, oy, on = self.est_obj.estimate(g["obj"], yaw_of_quat(g["oq"]), tcp)
        hp_true = self.handle_pos()
        hp, hy, hn = self.est_handle.estimate(hp_true, self.cab_yaw, tcp)
        f = g["f"]
        obs.update({
            "obj_est": op, "obj_yaw": wrap_to(oy, np.pi / 2), "obj_near_est": on,
            "obj_true": g["obj"], "obj_half_z": g["half_z"],
            "handle_est": hp, "handle_true": hp_true, "cab_yaw_est": hy,
            "opening": g["opening"], "opening_est": g["opening"] + self.rng.normal(0, self.q_noise),
            "contact_L": f["L_obj"], "contact_R": f["R_obj"], "handle_L": f["L_h"], "handle_R": f["R_h"],
            "grasp": g["hold_obj"], "hold_handle": g["hold_handle"], "in_air": g["in_air"],
            "placed": g["placed"], "on_table": g["on_table"], "obj_speed": float(np.linalg.norm(g["ov"])),
        })
        return obs

    def labels(self) -> set:
        g = self._geometry()
        f = g["f"]
        out = set()
        if g["opening"] >= OPEN_MIN:
            out.add("opened")
        if g["hold_obj"] and g["lifted"]:
            out.add("grasped")
        if g["placed"]:
            out.add("placed")
        if f["L_obj"] + f["R_obj"] < 0.3:
            out.add("released")
        fell = g["obj"][2] < TABLE_H - 0.05
        impact = self.open_impact_violation(f, OPEN_HAND_W)     # evaluated every step
        smash = (impact or self.crush_violation(f) or f["arm_table"] > 25.0 or f["fing_table"] > 60.0 or f["arm_furn"] > 60.0
                 or f["fing_furn"] > 80.0 or f["arm_handle"] > 60.0 or f["arm_obj"] > 40.0)
        if fell or smash:
            out.add("bad")
        return out

    def task_success(self) -> bool:
        g = self._geometry()
        f = g["f"]
        tcp = self.arm.tcp_pos()
        return bool(g["placed"] and f["L_obj"] + f["R_obj"] < 0.3 and np.linalg.norm(g["ov"]) < 0.03
                    and tcp[2] > g["obj"][2] + g["half_z"] + 0.05)

    # ============================================================ abstraction
    def approach_axis(self, yaw_est: float | None = None) -> np.ndarray:
        return self.handle_R(yaw_est)[:, 2]

    def handle_R(self, yaw_est: float | None = None) -> np.ndarray:
        yaw = self.cab_yaw if yaw_est is None else yaw_est
        heading = yaw + np.pi                 # approach INTO the front (opposite of the outward normal)
        return pinch_rotation(heading, PITCH_H, "vertical")

    def cell(self, obs: dict, labels: frozenset) -> str:
        obj, tcp, ap = obs["obj_true"], obs["tcp"], obs["aperture"]
        grasp, op = obs["grasp"], obs["opening"]
        if obj[2] < TABLE_H - 0.05:
            return "lost"
        if obs["placed"] and not (obs["contact_L"] > 1.0 and obs["contact_R"] > 1.0):
            if obs["contact_L"] + obs["contact_R"] < 0.3:
                return "done" if tcp[2] > obj[2] + obs["obj_half_z"] + 0.08 else "on_goal"
        if obs["placed"] and grasp:
            return "placed_held"
        if grasp:
            if obs["in_air"]:
                over = (np.linalg.norm(obj[:2] - self.goal) < 0.012 and obj[2] > TABLE_H + 2 * OBJ_HALF[2] + 0.02)
                if over:
                    return "over_goal"
                if obj[2] > self.floor_z0 + OBJ_HALF[2] + 0.10:
                    return "lifted"
            return "obj_grasped"
        h = obs["handle_true"]
        dh = np.linalg.norm(tcp - h)
        if obs["hold_handle"]:
            return "h_open" if op >= OPEN_TARGET else "h_grasped"
        if op < OPEN_MIN:
            if dh < 0.012 and ap > 0.03:
                return "at_handle"
            pre = h - PRE_HANDLE * self.approach_axis()
            if np.linalg.norm(tcp - pre) < 0.02 and ap > 0.03:
                return "pre_handle"
            return "closed"
        if dh < 0.05:
            return "at_handle_open"
        dxy = np.linalg.norm(tcp[:2] - obj[:2])
        if ap > 0.065 and dxy < 0.015 and abs(tcp[2] - (obj[2] + GRASP_DZ)) < 0.012:
            return "at_obj"
        if ap > 0.065 and dxy < 0.02 and 0.06 < tcp[2] - obj[2] < 0.18:
            return "above_obj"
        return "open"

    # ================================================================ skills
    def _build_skills(self):
        S = {}
        R_h_nom = self.handle_R()
        a_nom = R_h_nom[:, 2]
        h_nom = self.cab_center + rot_z(self.cab_yaw) @ self.top["handle_closed_local"]

        def near(obs, pos, tol):
            return np.linalg.norm(obs["tcp"] - pos) < tol

        def R_handle_obs(obs):
            return self.handle_R(obs["cab_yaw_est"])

        # 1 reach the pre-grasp pose in front of the handle (re-planned from the estimate)
        def pre_pose(obs):
            R = R_handle_obs(obs)
            return obs["handle_est"] - PRE_HANDLE * R[:, 2], R

        def t_reach_h(obs, st):
            p, R = pre_pose(obs)
            return self.cmd_R(p, R, AP_HANDLE)

        def o_reach_h(obs, labels, st):
            p, R = pre_pose(obs)
            ok = near(obs, p, 0.01) and self.hand_rot_err(R) < 0.06 and obs["aperture"] > 0.035
            return OK if ok else None

        # 2 move along the approach axis onto the bar
        def t_appr_h(obs, st):
            return self.cmd_R(obs["handle_est"], R_handle_obs(obs), AP_HANDLE)

        def o_appr_h(obs, labels, st):
            if obs["opening"] >= OPEN_MIN:
                return "already_open"
            return OK if near(obs, obs["handle_est"], 0.006) and self.hand_rot_err(R_handle_obs(obs)) < 0.05 else None

        # 3 close the gripper on the bar (decided by MEASURED pad contact forces)
        def e_hold(obs, st):
            st.memory["hold"] = obs["tcp"].copy()
            st.memory["R"] = self.arm.tcp_mat()

        def t_grasp_h(obs, st):
            return self.cmd_R(st.memory["hold"], st.memory["R"], 0.0)

        def o_grasp_h(obs, labels, st):
            if obs["handle_L"] > 2.0 and obs["handle_R"] > 2.0 and st.elapsed(obs["t"]) > 0.4:
                return OK
            if obs["aperture"] < 2 * BAR_R - 0.004 and not obs["hold_handle"]:
                return "missed"
            return None

        # 4 pull along the drawer axis (nominal: estimated front normal; Streamer online:
        #   RLS identification of the handle's motion line p(q) = p0 + q n from the
        #   measured TCP while holding and the measured opening q)
        def e_pull(obs, st):
            e_hold(obs, st)
            st.memory["q0"] = obs["opening_est"]
            n = np.array([np.cos(obs["cab_yaw_est"]), np.sin(obs["cab_yaw_est"]), 0.0])
            st.memory["n"] = n
            if st.memory.get("online"):
                rls = RecursiveLeastSquares(2, 3, ridge=1.0, forgetting=0.995)
                p0 = obs["tcp"] - obs["opening_est"] * n
                rls.theta = np.column_stack([p0, n])
                rls.P = np.diag([1e-4, 1e-2])
                st.memory["rls"] = rls
            st.memory["lost"] = None

        def pull_target(obs, st):
            if st.memory.get("rls") is not None:
                rls = st.memory["rls"]
                if obs["hold_handle"]:
                    rls.update(np.array([1.0, obs["opening_est"]]), obs["tcp"])
                return rls.predict(np.array([1.0, PULL_CMD]))
            return st.memory["hold"] + (PULL_CMD - st.memory["q0"]) * st.memory["n"]

        def t_pull(obs, st):
            return self.cmd_R(pull_target(obs, st), st.memory["R"], 0.0)

        def o_pull(obs, labels, st):
            if obs["hold_handle"]:
                st.memory["lost"] = None
                if obs["opening"] >= OPEN_TARGET:
                    return OK
                return None
            st.memory["lost"] = st.memory.get("lost") or obs["t"]
            gone = np.linalg.norm(obs["tcp"] - obs["handle_true"]) > 0.015
            if obs["t"] - st.memory["lost"] > 0.15 and gone:
                return "slipped_open" if obs["opening"] >= OPEN_MIN else "slipped"
            return None

        # 5 open the gripper (release the handle)
        def t_rel_h(obs, st):
            return self.cmd_R(st.memory["hold"], st.memory["R"], AP_HANDLE)

        def o_rel_h(obs, labels, st):
            ok = obs["handle_L"] + obs["handle_R"] < 0.3 and obs["aperture"] > 0.035
            return OK if ok else None

        # 6 back off from the bar along the approach axis and up
        def e_retreat_h(obs, st):
            e_hold(obs, st)
            st.memory["target"] = obs["tcp"] - 0.07 * st.memory["R"][:, 2] + [0, 0, 0.05]

        def t_retreat_h(obs, st):
            return self.cmd_R(st.memory["target"], st.memory["R"], AP_HANDLE)

        def o_retreat_h(obs, labels, st):
            ok = near(obs, st.memory["target"], 0.015) and np.linalg.norm(obs["tcp"] - obs["handle_true"]) > 0.06
            return OK if ok else None

        # 7 reach above the box (top-down, fingers closing parallel to the drawer front)
        def grasp_yaw(obs):
            prefer = self.cab_yaw - np.pi
            return box_grasp_yaw(obs["obj_yaw"], prefer)

        def closed_again(obs):
            return obs["opening"] < OPEN_MIN and np.linalg.norm(obs["obj_true"][:2] - obs["tcp"][:2]) < 0.25 \
                and obs["obj_true"][2] > TABLE_H + 0.05

        def t_reach_o(obs, st):
            return self.cmd_R(obs["obj_est"] + [0, 0, H_ABOVE], top_down(grasp_yaw(obs)), MAX_APERTURE)

        def o_reach_o(obs, labels, st):
            if closed_again(obs):
                return "closed_again"
            ok = near(obs, obs["obj_est"] + [0, 0, H_ABOVE], 0.008) and obs["aperture"] > 0.07
            return OK if ok else None

        # 8 descend around the box
        def t_desc_o(obs, st):
            return self.cmd_R(obs["obj_est"] + [0, 0, GRASP_DZ], top_down(grasp_yaw(obs)), MAX_APERTURE)

        def o_desc_o(obs, labels, st):
            if closed_again(obs):
                return "closed_again"
            return OK if near(obs, obs["obj_est"] + [0, 0, GRASP_DZ], 0.005) else None

        # 9 close on the box (grasp decided by measured pad contact)
        def t_grasp_o(obs, st):
            return self.cmd_R(st.memory["hold"], st.memory["R"], 0.0)

        def o_grasp_o(obs, labels, st):
            if obs["grasp"] and obs["contact_L"] > 3 and obs["contact_R"] > 3 and st.elapsed(obs["t"]) > 0.5:
                return OK
            if obs["aperture"] < 2 * OBJ_HALF[0] - 0.012 and not obs["grasp"]:
                return "missed"
            return None

        # 10 lift the box out of the drawer
        def dropped(obs, st):
            held = (obs["contact_L"] + obs["contact_R"] > 0.2 or
                    np.linalg.norm(obs["obj_true"] - obs["tcp"]) < 0.04)
            if held:
                st.memory["lost"] = None
                return False
            st.memory["lost"] = st.memory.get("lost") or obs["t"]
            return obs["t"] - st.memory["lost"] > 0.1

        def e_lift(obs, st):
            e_hold(obs, st)
            st.memory["lost"] = None

        def t_lift(obs, st):
            return self.cmd_R(st.memory["hold"] + [0, 0, LIFT], st.memory["R"], 0.0)

        def o_lift(obs, labels, st):
            if dropped(obs, st):
                return "dropped"
            ok = obs["in_air"] and obs["obj_true"][2] > self.floor_z0 + OBJ_HALF[2] + 0.12
            return OK if ok else None

        # 11 carry above the goal (box centre over the goal centre)
        def carry_tcp(obs, clearance):
            off = obs["obj_true"] - obs["tcp"]
            z = TABLE_H + obs["obj_half_z"] + clearance - off[2]
            return np.r_[self.goal - off[:2], z]

        def e_carry(obs, st):
            e_hold(obs, st)
            st.memory["lost"] = None
            st.memory["z"] = max(obs["tcp"][2], TABLE_H + 0.25)

        def t_carry(obs, st):
            p = carry_tcp(obs, 0.0)
            p[2] = st.memory["z"]
            return self.cmd_R(p, st.memory["R"], 0.0)

        def o_carry(obs, labels, st):
            if dropped(obs, st):
                return "dropped"
            p = carry_tcp(obs, 0.0)
            p[2] = st.memory["z"]
            ok = (np.linalg.norm(obs["obj_true"][:2] - self.goal) < 0.006 and near(obs, p, 0.008))
            return OK if ok else None

        # 12 lower onto the table until physical contact
        def t_lower(obs, st):
            return self.cmd_R(carry_tcp(obs, -0.002), st.memory["R"], 0.0)

        def o_lower(obs, labels, st):
            if dropped(obs, st):
                return "dropped"
            return OK if obs["placed"] and st.elapsed(obs["t"]) > 0.3 else None

        # 13 open the gripper
        def t_rel_o(obs, st):
            return self.cmd_R(st.memory["hold"] + [0, 0, 0.001], st.memory["R"], MAX_APERTURE)

        def o_rel_o(obs, labels, st):
            return OK if obs["aperture"] > 0.055 and obs["contact_L"] + obs["contact_R"] < 0.3 else None

        # 14 retreat upward
        def t_retreat(obs, st):
            return self.cmd_R(st.memory["hold"] + [0, 0, 0.14], st.memory["R"], MAX_APERTURE)

        def o_retreat(obs, labels, st):
            return OK if obs["tcp"][2] > obs["obj_true"][2] + obs["obj_half_z"] + 0.10 else None

        # ---- anchors (nominal goal of each token) and trust radii
        obj_nom = self.obj0.copy()
        obj_open = obj_nom + OPEN_TARGET * self.n_out
        pull_end = h_nom + PULL_CMD * self.n_out
        retreat_end = pull_end - 0.07 * a_nom + [0, 0, 0.05]
        Rd = top_down(self.cab_yaw - np.pi)
        goal_c = np.r_[self.goal, TABLE_H + OBJ_HALF[2] + GRASP_DZ]
        carry_z = obj_open[2] + GRASP_DZ + LIFT
        nominal = {
            "reach_handle": self.cmd_R(h_nom - PRE_HANDLE * a_nom, R_h_nom, AP_HANDLE),
            "approach_handle": self.cmd_R(h_nom, R_h_nom, AP_HANDLE),
            "grasp_handle": self.cmd_R(h_nom, R_h_nom, 0.0),
            "pull_drawer": self.cmd_R(pull_end, R_h_nom, 0.0),
            "release_handle": self.cmd_R(pull_end, R_h_nom, AP_HANDLE),
            "retreat_handle": self.cmd_R(retreat_end, R_h_nom, AP_HANDLE),
            "reach_object": self.cmd_R(obj_open + [0, 0, H_ABOVE], Rd, MAX_APERTURE),
            "descend_object": self.cmd_R(obj_open + [0, 0, GRASP_DZ], Rd, MAX_APERTURE),
            "grasp_object": self.cmd_R(obj_open + [0, 0, GRASP_DZ], Rd, 0.0),
            "lift_object": self.cmd_R(obj_open + [0, 0, GRASP_DZ + LIFT], Rd, 0.0),
            "carry_to_goal": self.cmd_R(np.r_[self.goal, carry_z], Rd, 0.0),
            "lower_object": self.cmd_R(goal_c, Rd, 0.0),
            "release_object": self.cmd_R(goal_c, Rd, MAX_APERTURE),
            "retreat": self.cmd_R(goal_c + [0, 0, 0.14], Rd, MAX_APERTURE),
        }
        home = self.cmd([0.474, 0.0, TABLE_H + 0.41], 0, 0.08)
        prev = home
        deltas = {}
        for name in self.nominal_plan():
            deltas[name] = self.space.norm(nominal[name] - prev) + 0.32
            prev = nominal[name]
        spec = [
            ("reach_handle", t_reach_h, o_reach_h, None, 12.0, 3.0),
            ("approach_handle", t_appr_h, o_appr_h, None, 6.0, 1.5),
            ("grasp_handle", t_grasp_h, o_grasp_h, e_hold, 3.0, 1.0),
            ("pull_drawer", t_pull, o_pull, e_pull, 8.0, 2.5),
            ("release_handle", t_rel_h, o_rel_h, e_hold, 3.0, 0.8),
            ("retreat_handle", t_retreat_h, o_retreat_h, e_retreat_h, 6.0, 1.2),
            ("reach_object", t_reach_o, o_reach_o, None, 12.0, 3.0),
            ("descend_object", t_desc_o, o_desc_o, None, 6.0, 1.5),
            ("grasp_object", t_grasp_o, o_grasp_o, e_hold, 3.0, 1.2),
            ("lift_object", t_lift, o_lift, e_lift, 6.0, 1.5),
            ("carry_to_goal", t_carry, o_carry, e_carry, 10.0, 2.5),
            ("lower_object", t_lower, o_lower, e_hold, 6.0, 1.8),
            ("release_object", t_rel_o, o_rel_o, e_hold, 3.0, 1.0),
            ("retreat", t_retreat, o_retreat, e_hold, 6.0, 1.5),
        ]
        for name, tgt, out, enter, timeout, dur in spec:
            S[name] = Skill(name, nominal[name], deltas[name], tgt, out, timeout=timeout,
                            nominal_duration=dur, on_enter=enter)
        self._skills = S

    def skills(self) -> dict:
        return self._skills

    def nominal_plan(self) -> list:
        return ["reach_handle", "approach_handle", "grasp_handle", "pull_drawer", "release_handle",
                "retreat_handle", "reach_object", "descend_object", "grasp_object", "lift_object",
                "carry_to_goal", "lower_object", "release_object", "retreat"]

    def skill_graph(self) -> list:
        R = "released"
        OR = {"opened", "released"}
        O = "opened"
        OG = {"opened", "grasped"}
        OP = {"opened", "placed"}
        OPR = {"opened", "placed", "released"}
        return [
            SkillSpec("reach_handle", "closed", (OutcomeSpec(OK, "pre_handle", word(R)),)),
            SkillSpec("approach_handle", "pre_handle", (
                OutcomeSpec(OK, "at_handle", word(R)),
                OutcomeSpec("already_open", "open", word(R, OR), failure=True))),
            SkillSpec("grasp_handle", "at_handle", (
                OutcomeSpec(OK, "h_grasped", word(R)),
                OutcomeSpec("missed", "closed", word(R), failure=True))),
            SkillSpec("pull_drawer", "h_grasped", (
                OutcomeSpec(OK, "h_open", word(R, OR)),
                OutcomeSpec("slipped", "closed", word(R), failure=True),
                OutcomeSpec("slipped_open", "at_handle_open", word(R, OR), failure=True))),
            SkillSpec("release_handle", "h_open", (OutcomeSpec(OK, "at_handle_open", word(OR)),)),
            SkillSpec("retreat_handle", "at_handle_open", (OutcomeSpec(OK, "open", word(OR)),)),
            SkillSpec("reach_object", "open", (
                OutcomeSpec(OK, "above_obj", word(OR)),
                OutcomeSpec("closed_again", "closed", word(OR, R), failure=True))),
            SkillSpec("descend_object", "above_obj", (
                OutcomeSpec(OK, "at_obj", word(OR)),
                OutcomeSpec("closed_again", "closed", word(OR, R), failure=True))),
            SkillSpec("grasp_object", "at_obj", (
                OutcomeSpec(OK, "obj_grasped", word(OR, O)),
                OutcomeSpec("missed", "open", word(OR), failure=True))),
            SkillSpec("lift_object", "obj_grasped", (
                OutcomeSpec(OK, "lifted", word(O, OG)),
                OutcomeSpec("dropped", "open", word(O, OR), failure=True),
                OutcomeSpec("dropped", "open", word(O, OG, OR), failure=True))),   # slipped after lift-off
            SkillSpec("carry_to_goal", "lifted", (
                OutcomeSpec(OK, "over_goal", word(OG)),
                OutcomeSpec("dropped", "open", word(OG, OR), failure=True))),
            SkillSpec("lower_object", "over_goal", (
                OutcomeSpec(OK, "placed_held", word(OG, OP)),
                OutcomeSpec("dropped", "open", word(OG, OR), failure=True))),
            SkillSpec("release_object", "placed_held", (OutcomeSpec(OK, "on_goal", word(OP, OPR)),)),
            SkillSpec("retreat", "on_goal", (OutcomeSpec(OK, "done", word(OPR)),)),
        ]

    # ========================================================== stress test
    def disturb(self) -> None:
        """(1) Drawer pushed back in: after the robot has released the handle (measured:
        no pad-handle contact for 0.2 s, TCP > 5 cm from the bar, drawer open), a 15 N
        push along the closing axis acts on the drawer until the measured displacement
        plus its predicted viscous coast reaches 8-12 cm (per seed), i.e. the drawer is
        pushed PARTIALLY closed (box no longer reachable); (2) gripper force dropout to
        +-0.5 N for 0.4 s just after the box has been lifted 2 cm out of the drawer floor
        (a grasp slip: it falls back into the drawer).  Triggered by measured state with
        per-seed delays; identical for every method."""
        d, t = self.data, self.data.time
        d.xfrc_applied[self.drawer_bid, :3] = 0.0
        P = self._pert
        f = self.forces()
        tcp = self.arm.tcp_pos()
        op = self.opening()
        # ---- (1) drawer push
        released = f["L_h"] + f["R_h"] < 0.1
        if P["drawer_push"] is None:
            if released and op >= OPEN_MIN and np.linalg.norm(tcp - self.handle_pos()) > 0.05:
                self._handle_free_since = self._handle_free_since or t
                if t - self._handle_free_since > 0.2:
                    P["drawer_push"] = {"t_trigger": t, "t_start": t + self._push_delay, "force_N": 15.0,
                                        "target_m": float(self._push_dist), "q_start": None}
            else:
                self._handle_free_since = None
        elif P["drawer_push"].get("done") is None:
            p = P["drawer_push"]
            if t >= p["t_start"]:
                if p["q_start"] is None:
                    p["q_start"] = op
                moved = p["q_start"] - op
                v0 = max(-float(self.data.qvel[self.jd_drawer]), 0.0)
                m_eff, c, fc = self.drawer_mass + self.obj_mass + 0.02, self.drawer_damping, self.drawer_fric
                # exact free-coast distance of m x'' = -c x' - fc (viscous + Coulomb runner friction)
                coast = (m_eff / c) * v0 - (fc * m_eff / c ** 2) * np.log1p(c * v0 / fc)
                if moved + coast < p["target_m"] and t < p["t_start"] + 1.0:
                    d.xfrc_applied[self.drawer_bid, :3] = -p["force_N"] * self.n_out
                else:
                    p["done"] = True
                    p["t_end"] = t
                    p["moved_m"] = float(moved)
        elif P["drawer_push"].get("q_final") is None and t > P["drawer_push"]["t_end"] + 1.0:
            P["drawer_push"]["q_final"] = op
        # ---- (2) grip dropout while lifting the box out of the drawer (it slips back in)
        g = self._geometry()
        if P["grip_dropout"] is None and g["hold_obj"] and g["in_air"] and g["lift"] > 0.02:
            P["grip_dropout"] = {"t_trigger": t, "t_start": t + self._drop_delay, "duration_s": 0.4,
                                 "force_range_N": [-0.5, 0.5]}
        drop = P["grip_dropout"]
        lim = self.model.actuator_forcerange[self.arm.act_grip]
        if drop is not None and drop["t_start"] <= t < drop["t_start"] + drop["duration_s"]:
            lim[:] = drop["force_range_N"]
            drop.setdefault("obj_z_at_start", float(g["obj"][2]))
        else:
            lim[:] = self._grip_range

    def perturbation_info(self) -> dict:
        return self._pert

    # ========================================================= diagnostics
    def step_diagnostics(self) -> dict:
        f = self.forces()
        self.max_opening = max(self.max_opening, self.opening())
        return {"arm_table_force_N": f["arm_table"], "finger_table_force_N": f["fing_table"],
                "arm_furniture_force_N": f["arm_furn"], "finger_furniture_force_N": f["fing_furn"],
                "finger_handle_force_N": f["L_h"] + f["R_h"], "finger_box_force_N": f["L_obj"] + f["R_obj"],
                "open_hand_pad_force_N": self.open_pad_force(f, OPEN_HAND_W),
                "drawer_opening_m": self.opening()}

    def log_signals(self) -> dict:
        return {"box": self.data.xpos[self.obj_bid].copy(), "tcp": self.arm.tcp_pos(),
                "aperture": self.arm.aperture(), "drawer_opening": self.opening(),
                "handle": self.handle_pos()}

    def episode_metrics(self) -> dict:
        obj = self.data.xpos[self.obj_bid]
        return {"final_box_goal_xy_error_m": float(np.linalg.norm(obj[:2] - self.goal)),
                "final_box_height_m": float(obj[2] - TABLE_H),
                "final_drawer_opening_m": self.opening(), "max_drawer_opening_m": self.max_opening,
                "finger_overforce_events": int(getattr(self, "crush_events", 0)),
                "open_hand_impact_events": int(getattr(self, "impact_events", 0)),
                "max_open_hand_pad_force_N": float(getattr(self, "max_open_pad_force", 0.0))}

    def variant_info(self) -> dict:
        return {"cabinet_front_x": self.front_x, "cabinet_y": self.cab_y, "cabinet_yaw": self.cab_yaw,
                "drawer_mass_kg": self.drawer_mass, "drawer_damping_Ns_per_m": self.drawer_damping,
                "drawer_frictionloss_N": self.drawer_fric, "box_init": self.obj0, "box_yaw": self.obj_yaw0,
                "box_mass_kg": self.obj_mass, "box_friction": self.obj_mu, "goal_xy": self.goal,
                "perception_bias_box_m": self.est_obj.b_far, "perception_bias_handle_m": self.est_handle.b_far}

    def overlay_lines(self) -> list:
        g = self._geometry()
        state = ("in hand" if g["hold_obj"] else "placed" if g["placed"] else "on table" if g["on_table"]
                 else "in drawer")
        return [f"drawer opening: {g['opening'] * 100:5.1f} cm   box: {state}   "
                f"box-goal: {np.linalg.norm(g['obj'][:2] - self.goal) * 100:5.1f} cm"]
