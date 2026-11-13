"""Task 14 - turn a steering wheel with a dexterous hand on a UR5e (LEAP / Aero variants).

A steering-wheel test rig stands on the black workbench: a steel column (static,
clamped to the table) carries a three-spoke steering wheel on a HINGE joint whose
axis is tilted 15-30 deg from vertical toward the robot (bus/truck-style wheel).
The wheel is built procedurally: rim = ring of capsule segments (centre-line radius
0.15-0.19 m, tube radius 15-18 mm, i.e. a 3.0-3.6 cm grip like a car wheel; leather/PU
friction), three capsule spokes dished down to an ellipsoidal hub; total mass
1.0-1.6 kg.  The column bearing has viscous damping (0.05-0.15 N m s/rad), Coulomb
friction (frictionloss 0.30-0.70 N m), a weak centring spring (0-0.12 N m/rad,
always weaker than the friction at the target so the wheel stays where it is left)
and +-150 deg end stops.  (No flat cylinder face is ever touched by the hand:
MuJoCo 3.15's native CCD flips the normal of a box corner on a flat cylinder face.)

The hand hook/power-grasps the rim from the outside (palm outside the rim facing the hub,
fingers pointing down along the wheel axis and wrapping under the tube toward the hub, thumb
along the rim tangent), the arm follows the rim ARC about the (estimated) wheel
axis so the wheel is turned counter-clockwise (left) by a randomized target angle
of 70-110 deg, the hand opens and the arm retreats.  The wheel angle is MEASURED
by the column encoder (hinge joint position).  The arc follower is driven by the MEASURED hand
position on the rim (its arc angle about the estimated wheel axis), so it keeps pushing forward even
if the wheel lags the hand.  The IN-HAND SLIP (hand arc angle - grip point - encoder angle) is measured
all the time: if the rim slips out of the hand (grasp lost) or slides through the closed hand by more
than 8 deg ('slipped'), the hand lets go (token 'let_go': open, back off outward/up) and the supervisor
re-grasps the rim at the planned grip point (now rotated with the wheel) and turns again.

LTLf:  G !bad & F(grasped_rim & F(turned & F released))

* grasped_rim  >= 3 digits (>= 2 of them long fingers) touch the wheel with >= 0.3 N each, total
               hand-wheel force >= 3 N, and the rim tube centre lies within 2.5 cm of the grip point
* turned       |encoder angle - target| <= 5 deg
* released     hand-wheel contact force < 0.2 N
* slip         (measured, not a label) wrap(hand arc angle - grip angle - encoder angle); |slip| > 8 deg
               while grasped and not yet 'turned' is the abstract cell 'slip_held'
* bad          hand-wheel force > 200 N, hand/arm crash into the column/base (> 20 N),
               arm-table > 20 N, hand-table > 30 N, arm-wheel > 20 N, or the wheel driven
               into its +-150 deg end stop.  Contact forces are averaged over a 10 ms window
               (5 physics steps), like the low-pass of a real force/torque safety monitor: a
               single 2 ms contact-impulse step of the stiff fingertip contacts (finger impact on
               the rim) is a soft-constraint artefact, not a sustained squeeze.
"""
from __future__ import annotations

from collections import deque

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from eddies.envs.controllers.hand_arm import hand_frame
from eddies.envs.framework.perception import PoseEstimator
from eddies.envs.framework.scene import yaw_quat
from eddies.envs.framework.task import OK, Skill, SkillSpec, OutcomeSpec, word
from eddies.envs.tasks.hand_base import TABLE_H, HandArmTaskBase

# ---------------------------------------------------------------- geometry
N_RIM = 40                         # capsule segments of the rim
DISH = 0.035                       # hub below the rim plane (dished wheel) [m]
HUB_R = 0.050                      # hub radius [m]
SPOKE_R = 0.009                    # spoke capsule radius [m]
SPOKE_ANGLES = (0.0, np.pi / 2, -np.pi / 2)     # body-frame spoke angles (far, left, right)
# nominal hub centre.  HUB_NOTE: the hub stands 0.57-0.63 m in front of the UR5e base.  With the earlier
# 0.52 m the near rim (psi ~ 180 deg) and the pre-grasp/let-go poses 5 cm outside it put the UR5e close to
# its shoulder singularity (arm Jacobian sigma_min 0.006 at x ~ 0.29 m): a re-grasp there flipped the DLS
# IK to another arm branch whose forearm then hit the rim.  At 0.60 m sigma_min >= 0.07 on every path
# (nominal >= 0.11), measured with dev/hubscan.py.
HUB_NOM = np.array([0.60, 0.0, TABLE_H + 0.30])
TILT_NOM = np.radians(22.0)
R_NOM = 0.17
STOP = np.radians(150.0)           # end stops
# ------------------------------------------------------------------ logic
TARGET_NOM = np.radians(90.0)
TOL_TURNED = np.radians(5.0)       # label "turned"
TOL_TURN_DONE = np.radians(1.5)    # turn skill completion
LEAD = np.radians(10.0)            # carrot lead along the arc
PRE_OUT, PRE_UP = 0.05, 0.03       # pre-grasp offset of the grasp site: outward (+r) and up (+n) [m]
H_RELEASE = 0.015                  # palm backs off outward while opening [m]
H_RETREAT = 0.10                   # retreat height above the rim [m]
D_AWAY = 0.06                      # hand counted as away from the rim [m]
F_BAD_WHEEL = 200.0                # squeeze limit (nominal grip 35-75 N, few-ms contact transients to ~130 N)
F_WIN = 0.010                      # force-limit averaging window for 'bad' [s]
SLIP_TOL = np.radians(8.0)         # in-hand slip of the rim that aborts the turn (nominal |slip| < 3 deg)
LETGO_OUT, LETGO_UP = 0.05, 0.03   # let_go: palm backs off outward / up once the fingers have opened [m]
POWER = (1.0, 0.0, 0.0)
OPEN = (0.0, 0.0, 0.0)
# tube centre relative to the grasp site, in the hand frame (x_h palm normal, y_h thumb, z_h fingers)
# (measured with dev/grasp_scan.py: all fingers + thumb on the rim, 30-50 N total squeeze)
GRASP_OFF = {"leap": np.array([0.0, 0.0, 0.0]), "aero": np.array([0.015, 0.0, 0.005])}
# Aero finger cables: extension force allowed to the servo, emulating the hand's return springs (see
# _aero_return_springs).  0 = the shared pull-only cable of controllers/hand_arm.py.
AERO_RETURN_N = 10.0
# --perturb in-grasp kick (road-reaction kickback) torque range [N m], per hand: drawn from just above the
# torque at which that hand's measured power grasp starts to slip on the rim (dev/kick_probe.py sweep,
# Streamer grasp: LEAP ~3.0-3.5 N m, Aero ~7-7.5 N m; the Aero cable hand squeezes ~60 N vs ~42 N) to ~1.4x
# it, so the kick really slides the rim through the hand (10-70 deg) without throwing the wheel into its
# end stops (>= 10 N m on the Aero / >= 6 N m on the LEAP did, an unrecoverable 'bad' by the disturbance itself)
KICK_NM = {"leap": (3.5, 5.0), "aero": (7.5, 9.0)}


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def rot_about(axis, angle) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(axis, float) / np.linalg.norm(axis) * angle).as_matrix()


def wheel_frame(yaw: float, tilt: float) -> np.ndarray:
    """Columns (u, v, n): n = wheel axis (up, tilted toward the robot at -x), u = in-plane
    direction away from the robot (zero-angle mark), v = n x u (robot's left)."""
    Rz = rot_about((0, 0, 1), yaw)
    Ry = rot_about((0, 1, 0), -tilt)        # tilts +z toward -x
    return Rz @ Ry


def rim_frame(W: np.ndarray, psi: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit radial vector and hand frame for an over-hand power grasp at wheel-plane angle psi."""
    u, v, n = W[:, 0], W[:, 1], W[:, 2]
    r = np.cos(psi) * u + np.sin(psi) * v
    # palm outside the rim facing the hub (x_h = -r), fingers down along the wheel axis (z_h = -n),
    # thumb along the rim tangent: the fingers wrap under the tube toward the hub (hook/power grasp)
    return r, hand_frame(-n, -r)


class HandWheelTask(HandArmTaskBase):
    task_id = "wheel"
    description = ("UR5e with a dexterous hand power-grasps a steering-wheel rim, turns the hinged wheel "
                   "counter-clockwise to a randomized target angle (70-110 deg, column encoder), releases "
                   "it and retreats.")
    horizon = 45.0
    success_hold = 1.0
    formula = "G !bad & F(grasped_rim & F(turned & F released))"
    terminal_cells = ("done",)
    # the --perturb protocol injects two disturbances that can each cost one failure (in-grasp kick ->
    # 'slipped', push-back -> 'moved'); one unit of margin for a failure during the re-grasp
    failure_budget = 3
    main_camera = "main"
    detail_camera = "detail"

    def __init__(self, seed: int, hand: str = "leap", **variant):
        super().__init__(seed, hand=hand, **variant)
        self._fhist = deque()              # (t, contact-force dict) inside the F_WIN window
        rng = self.rng
        # ------------------------------------------------ randomized physics / instruction
        self.hub = HUB_NOM + np.r_[rng.uniform(-0.03, 0.03), rng.uniform(-0.04, 0.04), rng.uniform(-0.02, 0.02)]
        self.rig_yaw = rng.uniform(-0.15, 0.15)
        self.tilt = np.radians(rng.uniform(15.0, 30.0))
        self.R_rim = rng.uniform(0.15, 0.19)
        self.r_tube = rng.uniform(0.015, 0.018)
        self.m_rim = rng.uniform(0.50, 0.80)
        self.m_hub = rng.uniform(0.40, 0.70)
        self.m_spoke = rng.uniform(0.04, 0.07)
        self.damping = rng.uniform(0.05, 0.15)       # [N m s/rad]
        self.frictionloss = rng.uniform(0.30, 0.70)  # [N m]
        self.k_center = rng.uniform(0.0, 0.12)       # [N m/rad]
        self.mu_rim = rng.uniform(0.75, 1.0)         # leather / PU grip
        self.target = np.radians(rng.uniform(70.0, 110.0))
        self.W = wheel_frame(self.rig_yaw, self.tilt)
        # grip point on the rim (wheel-body angle): arc centred on the near side of the wheel
        self.phi_grip = np.pi - self.target / 2
        # ------------------------------------------------------- scene
        sb = self.make_builder()
        self._build_rig(sb)
        c = self.hub
        # main: from the far side of the rig, looking back at the robot (the palm faces the hub, so the grasp
        # is seen through the wheel); detail: above the far rim, looking down at the near half of the rim
        sb.camera("main", (1.38, -0.80, TABLE_H + 0.98), (0.37, 0.0, TABLE_H + 0.18), fovy=42)
        near = c - 0.9 * self.R_rim * self.W[:, 0]
        sb.camera("detail", c + np.array([0.16, -0.10, 0.36]), near + np.array([0.0, 0.0, -0.02]), fovy=55)
        sb.camera("top", c + np.array([-0.05, 0.0, 0.85]), c, fovy=50)
        _, self.R_nom = rim_frame(wheel_frame(0.0, TILT_NOM), np.pi)
        # MuJoCo 3.15's native CCD returns contact normals pointing ALONG the rim tube (opposite signs on
        # two neighbouring rim capsules) for the LEAP thumb-tip mesh at a capsule junction: the two
        # contacts squeeze the pad with ~150-200 N while the net force is small.  The libccd (MPR/GJK)
        # path gives consistent normals; it is used for this scene (variant nativeccd=True restores it).
        if not self.variant.get("nativeccd", False):
            sb.spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        self.finish_build(sb, base_mat=self.R_nom)
        m = self.model
        self.cm = self.contact_monitor(
            {"wheel": ["wheel"], "rig": ["column"]},
            pairs={**{f"finger_{f}_wheel": (f"finger_{f}", "wheel") for f in self.ha.hand.fingers},
                   "fingers_wheel": ("fingers", "wheel")})
        self.jw = m.joint("wheel_hinge").id
        self.w_qadr = int(m.jnt_qposadr[self.jw])
        self.w_dadr = int(m.jnt_dofadr[self.jw])
        self.bid_wheel = m.body("wheel").id
        self.off = GRASP_OFF[self.hand_name]
        if self.hand_name == "aero":
            self._aero_return_springs()
        # ------------------------------------------------ perception (hub pose, axis tilt, rim radius)
        self.est = PoseEstimator(rng, near_distance=0.30)
        self.tilt_bias = np.radians(rng.normal(0, 1.5, 2))       # far, near-scale drawn once
        self.rad_bias = rng.normal(0, 0.0015)
        # ------------------------------------------------ reset configuration
        self.q_ready = self.solve_ready(*self.ready_pose())
        self._contacts = {}
        self._grasp_since = None
        self._build_skills()
        # ------------------------------------------------ stress test schedule (always drawn)
        self._turn_push_frac = rng.uniform(0.35, 0.65)
        self._turn_push_tau = rng.uniform(*KICK_NM[hand])
        self._back_delay = rng.uniform(0.1, 0.4)
        self._back_angle = np.radians(rng.uniform(20.0, 35.0))
        self._pert = {"turn_push": None, "push_back": None}

    # ----------------------------------------------------------------- scene
    def _build_rig(self, sb) -> None:
        sb.material("rim_mat", (0.06, 0.06, 0.07, 1), specular=0.25, shininess=0.3)
        sb.material("spoke_mat", (0.62, 0.63, 0.66, 1), specular=0.7, shininess=0.8)
        sb.material("hub_mat", (0.18, 0.18, 0.2, 1), specular=0.4, shininess=0.5)
        sb.material("column_mat", (0.30, 0.31, 0.33, 1), specular=0.5, shininess=0.5)
        G = mujoco.mjtGeom
        W, c = self.W, self.hub
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, W.reshape(-1))
        col = sb.spec.worldbody.add_body(name="column", pos=list(c), quat=list(q))
        # steering column (static, clamped to the table) along -n down to the base plate
        h_top = c[2] - TABLE_H
        L = (h_top - 0.02) / np.cos(self.tilt)
        self.col_L = L
        col.add_geom(name="column_tube", type=G.mjGEOM_CYLINDER, fromto=[0, 0, -DISH - 0.03, 0, 0, -L],
                     size=[0.028, 0, 0], material="column_mat", friction=[0.35, 0.003, 1e-4], condim=3)
        col.add_geom(name="column_shroud", type=G.mjGEOM_BOX, pos=[0, 0, -DISH - 0.075],
                     size=[0.045, 0.045, 0.04], material="column_mat", friction=[0.35, 0.003, 1e-4], condim=3)
        bottom = c - L * W[:, 2]
        sb.static_box("rig_base", (bottom[0], bottom[1], TABLE_H + 0.012), (0.11, 0.11, 0.012),
                      rgba=(0.22, 0.22, 0.24, 1), quat=yaw_quat(self.rig_yaw), friction="metal")
        # wheel on a hinge about the column axis
        wh = col.add_body(name="wheel", pos=[0, 0, 0])
        wh.add_joint(name="wheel_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                     range=[-np.degrees(STOP), np.degrees(STOP)],      # mjSpec compiler angle unit: degree
                     limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                     damping=self.damping, frictionloss=self.frictionloss, stiffness=self.k_center,
                     springref=0.0, armature=0.002, solref_limit=[0.004, 1.0],
                     solref_friction=[0.005, 1.0])      # stiff dry friction: no creep under the spring
        R, rt = self.R_rim, self.r_tube
        fr = [self.mu_rim, 0.008, 1e-4]
        for i in range(N_RIM):
            a0, a1 = 2 * np.pi * i / N_RIM, 2 * np.pi * (i + 1) / N_RIM
            wh.add_geom(name=f"rim_{i}", type=G.mjGEOM_CAPSULE, size=[rt, 0, 0],
                        fromto=[R * np.cos(a0), R * np.sin(a0), 0, R * np.cos(a1), R * np.sin(a1), 0],
                        mass=self.m_rim / N_RIM, material="rim_mat", friction=fr, condim=4)
        for k, a in enumerate(SPOKE_ANGLES):
            ca, sa = np.cos(a), np.sin(a)
            wh.add_geom(name=f"spoke_{k}", type=G.mjGEOM_CAPSULE, size=[SPOKE_R, 0, 0],
                        fromto=[0.8 * HUB_R * ca, 0.8 * HUB_R * sa, -DISH, (R - rt) * ca, (R - rt) * sa, 0.0],
                        mass=self.m_spoke, material="spoke_mat", friction=[0.35, 0.003, 1e-4], condim=3)
        wh.add_geom(name="hub", type=G.mjGEOM_ELLIPSOID, size=[HUB_R, HUB_R, 0.022], pos=[0, 0, -DISH],
                    mass=self.m_hub, material="hub_mat", friction=[0.4, 0.004, 1e-4], condim=3)
        wh.add_geom(name="hub_emblem", type=G.mjGEOM_CYLINDER, size=[0.018, 0.001, 0], pos=[0, 0, -DISH + 0.021],
                    rgba=[0.75, 0.75, 0.78, 1], contype=0, conaffinity=0, mass=0.0)
        # top-centre marker (visual): shows the wheel angle in the video
        for s in (-1, 1):
            a = s * np.radians(3.5)
            wh.add_geom(name=f"marker_{s}", type=G.mjGEOM_CAPSULE, size=[rt + 0.0015, 0, 0],
                        fromto=[R, 0, 0, R * np.cos(a), R * np.sin(a), 0], rgba=[0.95, 0.45, 0.05, 1],
                        contype=0, conaffinity=0, mass=0.0)

    def _aero_return_springs(self) -> None:
        """WORKAROUND (documented deviation).  controllers/hand_arm.py makes the Aero finger cables
        pull-only ([-75, 0] N) and relies on the model's spring tendons to re-extend the fingers, but in
        the pinned menagerie model those springs do not do it: the MCP spring tendon (k 352 N/m, rest
        length 11.4 mm) is at equilibrium near 1.2 rad of flexion and the joints have 0.02 N m friction, so
        after any power closure the fingers stay curled (measured: flexion 1.0-1.6 rad per finger after
        1.5 s of 'open'; hand_base.synergy_reopen_test('aero') also returns t_reopen=None).  The real hand
        re-opens.  Here the cable servo may push with at most AERO_RETURN_N = 10 N (the menagerie
        actuator pushes without limit), which re-opens the fingers in ~1.5 s; pulling is unchanged."""
        m = self.model
        for a in self.ha.act_hand:
            if "abd" not in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a)):
                m.actuator_forcerange[a, 1] = AERO_RETURN_N

    def grip_point(self) -> np.ndarray:
        """Where the rim tube centre sits in a correct grasp (grasp site + hand-frame offset), measured."""
        return self.ha.grasp_pos() + self.ha.grasp_mat() @ self.off

    # ------------------------------------------------------------------ state
    def _forces(self) -> dict:
        if self._contacts.get("t") != self.data.time:
            self._contacts = self.cm.measure(self.data)
            t = float(self.data.time)
            self._contacts["t"] = t
            h = self._fhist
            if h and t < h[-1][0]:
                h.clear()
            h.append((t, self._contacts))
            while h[0][0] <= t - F_WIN + 1e-9:
                h.popleft()
        return self._contacts

    def _forces_avg(self) -> dict:
        """Contact forces averaged over the last F_WIN seconds (safety-monitor low-pass)."""
        self._forces()
        keys = ("hand_wheel", "arm_wheel", "hand_rig", "arm_rig", "arm_table", "hand_table")
        return {k: float(np.mean([f[k] for _, f in self._fhist])) for k in keys}

    def angle(self) -> float:
        return float(self.data.qpos[self.w_qadr])

    def omega(self) -> float:
        return float(self.data.qvel[self.w_dadr])

    def _grip_pose(self, hub, W, R_rim, psi) -> tuple[np.ndarray, np.ndarray]:
        """Grasp-site target and hand frame for an over-hand grasp of the rim at plane angle psi."""
        r, Rh = rim_frame(W, psi)
        tube = hub + R_rim * r
        return tube - Rh @ self.off, Rh

    def rim_distance(self, p) -> tuple[float, float]:
        """Distance of point p from the TRUE rim centre-line and its plane angle."""
        W, c = self.W, self.hub
        d = np.asarray(p, float) - c
        x, y, z = d @ W[:, 0], d @ W[:, 1], d @ W[:, 2]
        rho = np.hypot(x, y)
        return float(np.hypot(rho - self.R_rim, z)), float(np.arctan2(y, x))

    def _estimate(self, sensor) -> dict:
        pos, yaw, near = self.est.estimate(self.hub, self.rig_yaw, sensor)
        tilt = self.tilt + (0.25 if near else 1.0) * self.tilt_bias[0]
        rad = self.R_rim + (0.25 if near else 1.0) * self.rad_bias
        return {"hub": pos, "W": wheel_frame(yaw, tilt), "R": rad, "near": near}

    def _grasp_measure(self, f: dict) -> tuple[bool, int]:
        # power/hook grasp: >= 3 digits on the wheel, >= 2 of them long fingers wrapped round the tube
        # (single fingers and the Aero thumb lose contact intermittently while the grasp holds)
        n_long = sum(f[f"finger_{fi}_wheel"] > 0.3 for fi in self.ha.hand.fingers if fi != "th")
        n = n_long + int(f["finger_th_wheel"] > 0.3)
        d, _ = self.rim_distance(self.grip_point())
        ok = bool(n >= 3 and n_long >= 2 and f["hand_wheel"] >= 3.0 and d < 0.025)
        return ok, int(n)

    def observe(self) -> dict:
        obs = self.base_obs()
        f = self._forces()
        est = self._estimate(obs["grasp"])
        grasped, nf = self._grasp_measure(f)
        d_rim, psi_hand = self.rim_distance(self.grip_point())
        th = self.angle()
        # hand arc angle about the ESTIMATED wheel axis (drives the turn), in-hand slip (rig geometry)
        W, dg = est["W"], self.grip_point() - est["hub"]
        psi_hand_est = float(np.arctan2(dg @ W[:, 1], dg @ W[:, 0]))
        obs.update({"psi_hand_est": psi_hand_est, "slip": wrap(psi_hand - self.phi_grip - th),
            "theta": th, "omega": self.omega(), "theta_target": self.target, "grip_phi": self.phi_grip,
            "hub_est": est["hub"], "W_est": est["W"], "R_est": est["R"], "est_near": est["near"],
            "f_wheel": f["hand_wheel"] + f["arm_wheel"], "f_rig": f["hand_rig"] + f["arm_rig"],
            "n_fingers": nf, "grasped": grasped, "d_rim": d_rim, "psi_hand": psi_hand,
            "turned": abs(th - self.target) <= TOL_TURNED, "power": float(obs["synergy"][0]),
            "hub_true": self.hub.copy()})
        return obs

    def labels(self) -> set:
        f = self._forces()
        out = set()
        grasped, _ = self._grasp_measure(f)
        if grasped:
            out.add("grasped_rim")
        th = self.angle()
        if abs(th - self.target) <= TOL_TURNED:
            out.add("turned")
        if f["hand_wheel"] + f["arm_wheel"] < 0.2:
            out.add("released")
        fa = self._forces_avg()
        crash = (fa["hand_rig"] > 20.0 or fa["arm_rig"] > 20.0 or fa["arm_table"] > 20.0
                 or fa["hand_table"] > 30.0 or fa["arm_wheel"] > 20.0)
        if fa["hand_wheel"] > F_BAD_WHEEL or crash or abs(th) > STOP - np.radians(1.0):
            out.add("bad")
        return out

    def task_success(self) -> bool:
        f = self._forces()
        d, _ = self.rim_distance(self.grip_point())
        return bool(abs(self.angle() - self.target) <= TOL_TURNED and f["hand_wheel"] + f["arm_wheel"] < 0.2
                    and d > D_AWAY - 0.01 and abs(self.omega()) < 0.05)

    def cell(self, obs: dict, labels: frozenset) -> str:
        turned, fw = obs["turned"], obs["f_wheel"]
        if obs["grasped"]:
            if turned:
                return "turned_held"
            # the rim slid through the closed hand: the hand no longer holds the planned grip point
            return "slip_held" if abs(obs["slip"]) > SLIP_TOL else "grasped"
        if turned and fw < 0.2 and obs["power"] < 0.3:
            return "done" if obs["d_rim"] > D_AWAY else "released"
        if fw < 0.2 and obs["power"] < 0.2:
            psi = self.phi_grip + obs["theta"]
            p_g, R_g = self._grip_pose(self.hub, self.W, self.R_rim, psi)
            rot = self.rotation_error(R_g)
            if obs["d_rim"] < 0.012 and rot < 0.25 and abs(obs["slip"]) < SLIP_TOL:
                return "at_rim"
            if np.linalg.norm(obs["grasp"] - (p_g + self._pre(self.W, R_g))) < 0.025 and rot < 0.25:
                return "pregrasp"
        return "free"

    @staticmethod
    def _pre(W, Rh) -> np.ndarray:
        """Pre-grasp offset: outward (-x_h = +r) and up along the wheel axis."""
        return -PRE_OUT * Rh[:, 0] + PRE_UP * W[:, 2]

    # ----------------------------------------------------------------- skills
    def _build_skills(self):
        def grip(obs, psi_extra=0.0):
            psi = obs["grip_phi"] + obs["theta"] + psi_extra
            return self._grip_pose(obs["hub_est"], obs["W_est"], obs["R_est"], psi)

        def near(obs, p, R, tol, rtol=0.08):
            return np.linalg.norm(obs["grasp"] - p) < tol and self.rotation_error(R) < rtol

        def still(obs, v=0.05):
            return np.linalg.norm(obs["dq"]) < v

        # 1 reach the pre-grasp pose outside/above the grip point (open hand)
        def t_pre(obs, st):
            p, R = grip(obs)
            return self.cmd(p + self._pre(obs["W_est"], R), R, OPEN)

        def o_pre(obs, labels, st):
            p, R = grip(obs)
            ok = near(obs, p + self._pre(obs["W_est"], R), R, 0.006) and still(obs)
            return OK if ok and obs["f_wheel"] < 0.2 else None

        # 2 approach: move the open hand onto the rim (tube centre to the grasp point)
        def t_app(obs, st):
            p, R = grip(obs)
            return self.cmd(p, R, OPEN)

        def o_app(obs, labels, st):
            if obs["f_wheel"] > 5.0:
                st.memory["bump"] = st.memory.get("bump") or obs["t"]
                if obs["t"] - st.memory["bump"] > 0.1:
                    return "bumped"
            else:
                st.memory["bump"] = None
            p, R = grip(obs)
            ok = near(obs, p, R, 0.003) and still(obs, 0.03) and st.elapsed(obs["t"]) > 0.2
            return OK if ok else None

        # 3 close the power grasp (hold the pose)
        def e_grasp(obs, st):
            st.memory["hold"] = grip(obs)

        def t_grasp(obs, st):
            p, R = st.memory["hold"]
            return self.cmd(p, R, POWER)

        def o_grasp(obs, labels, st):
            if obs["grasped"]:
                st.memory["g"] = st.memory.get("g") or obs["t"]
                if obs["t"] - st.memory["g"] > 0.3 and st.elapsed(obs["t"]) > 0.6:
                    return OK
            else:
                st.memory["g"] = None
            if obs["power"] > 0.85 and not obs["grasped"] and st.elapsed(obs["t"]) > 1.0:
                return "missed"
            return None

        # 4 turn: follow the rim arc about the estimated axis; the carrot leads the MEASURED hand arc angle
        # (not the encoder: if the rim slides in the hand, an encoder-based carrot would fall behind the
        # hand and drag the wheel backwards), the remaining encoder error sets the (bounded) lead
        def e_turn(obs, st):
            phys = self.space.denormalize(st.start_command)
            st.memory["p0"] = phys[:3].copy()
            st.memory["R0"] = self.ha.chart.decode_matrix(phys[3:6])
            st.memory["ph0"] = obs["psi_hand_est"]

        def t_turn(obs, st):
            th = obs["theta"]
            err = obs["theta_target"] - th
            # integral action removes the steady lag of the compliant fingers / arm near the target
            if abs(err) < np.radians(6.0):
                st.memory["I"] = float(np.clip(st.memory.get("I", 0.0) + 1.5 * err * self.control_dt,
                                               -np.radians(6.0), np.radians(6.0)))
            a = wrap(obs["psi_hand_est"] - st.memory["ph0"]) + float(np.clip(1.5 * err + st.memory.get("I", 0.0),
                                                                             -LEAD, LEAD))
            Rot = rot_about(obs["W_est"][:, 2], a)
            c = obs["hub_est"]
            p = c + Rot @ (st.memory["p0"] - c)
            return self.cmd(p, Rot @ st.memory["R0"], POWER)

        def o_turn(obs, labels, st):
            if not obs["grasped"]:
                st.memory["lost"] = st.memory.get("lost") or obs["t"]
                if obs["t"] - st.memory["lost"] > 0.2:
                    return "slipped"
            else:
                st.memory["lost"] = None
            # rim slid through the closed hand (measured in-hand slip), persistent for 0.1 s
            if obs["grasped"] and not obs["turned"] and abs(obs["slip"]) > SLIP_TOL:
                st.memory["sl"] = st.memory.get("sl") or obs["t"]
                if obs["t"] - st.memory["sl"] > 0.1:
                    return "slipped"
            else:
                st.memory["sl"] = None
            err = abs(obs["theta"] - obs["theta_target"])
            slow = abs(obs["omega"]) < 0.05
            # settled close to the target, or stalled for 1.5 s well inside the 'turned' band
            st.memory["slow"] = (st.memory.get("slow") or obs["t"]) if (slow and err < 0.6 * TOL_TURNED) else None
            ok = (err < TOL_TURN_DONE and slow) or (st.memory["slow"] is not None and obs["t"] - st.memory["slow"] > 1.5)
            return OK if ok and obs["grasped"] and st.elapsed(obs["t"]) > 0.3 else None

        # 5 release: open the hand; the palm backs off outward once the fingers have opened
        def e_rel(obs, st):
            phys = self.space.denormalize(st.start_command)
            st.memory["p"] = phys[:3].copy()
            st.memory["R"] = self.ha.chart.decode_matrix(phys[3:6])

        def t_rel(obs, st):
            s = float(np.clip((st.elapsed(obs["t"]) - 0.3) / 0.4, 0.0, 1.0))
            return self.cmd(st.memory["p"] - s * H_RELEASE * st.memory["R"][:, 0], st.memory["R"], OPEN)

        # 'moved' is decided once the pushed wheel has come to rest (|omega| < 0.05 rad/s), so the successor
        # cell (free / pregrasp / at_rim relative to the rotated grip point) is the settled one
        def o_rel(obs, labels, st):
            if not obs["turned"] and obs["f_wheel"] < 0.2 and abs(obs["omega"]) < 0.05:
                return "moved"
            ok = obs["turned"] and obs["f_wheel"] < 0.2 and obs["power"] < 0.2 and st.elapsed(obs["t"]) > 0.4
            return OK if ok else None      # (a wheel still moving out of the band: wait for 'moved')

        # 6 retreat: outward and up, away from the rim
        def t_ret(obs, st):
            R = st.memory["R"]
            return self.cmd(st.memory["p"] - 0.05 * R[:, 0] + H_RETREAT * obs["W_est"][:, 2], R, OPEN)

        def o_ret(obs, labels, st):
            if not obs["turned"] and abs(obs["omega"]) < 0.05:
                return "moved"
            return OK if obs["turned"] and obs["d_rim"] > D_AWAY + 0.005 and obs["f_wheel"] < 0.2 else None

        # 7 let_go (recovery after an in-hand slip): open the hand where it is, then back the palm off
        # outward and up so the fingers clear the tube; the re-grasp follows (reach_pregrasp ...)
        def t_let(obs, st):
            s = float(np.clip((st.elapsed(obs["t"]) - 0.3) / 0.5, 0.0, 1.0))
            R = st.memory["R"]
            return self.cmd(st.memory["p"] - s * LETGO_OUT * R[:, 0] + s * LETGO_UP * obs["W_est"][:, 2], R, OPEN)

        def o_let(obs, labels, st):
            ok = obs["f_wheel"] < 0.2 and obs["power"] < 0.2 and obs["d_rim"] > 0.045 and st.elapsed(obs["t"]) > 0.6
            return OK if ok else None

        # ------------------------------------------------------------ anchors / trust radii
        W0 = wheel_frame(0.0, TILT_NOM)
        n0 = W0[:, 2]
        g_s = self._grip_pose(HUB_NOM, W0, R_NOM, np.pi - TARGET_NOM / 2)
        g_e = self._grip_pose(HUB_NOM, W0, R_NOM, np.pi + TARGET_NOM / 2)
        nominal = {
            "reach_pregrasp": self.cmd(g_s[0] + self._pre(W0, g_s[1]), g_s[1], OPEN),
            "approach": self.cmd(g_s[0], g_s[1], OPEN),
            "grasp": self.cmd(g_s[0], g_s[1], POWER),
            "turn": self.cmd(g_e[0], g_e[1], POWER),
            "release": self.cmd(g_e[0] - H_RELEASE * g_e[1][:, 0], g_e[1], OPEN),
            "retreat": self.cmd(g_e[0] - 0.05 * g_e[1][:, 0] + H_RETREAT * n0, g_e[1], OPEN),
        }
        prev = self.cmd(*self.ready_pose(), OPEN)
        deltas = {}
        for name in self.nominal_plan():
            deltas[name] = self.space.norm(nominal[name] - prev) + 0.30
            prev = nominal[name]
        # a re-grasp (after a slip or a push-back) happens at the same rim point, rotated by the current
        # wheel angle (0 .. target + 30 deg): the approach tokens' trust balls cover that whole arc
        g_far = self._grip_pose(HUB_NOM, W0, R_NOM, np.pi + TARGET_NOM / 2 + np.radians(30))
        for name in ("reach_pregrasp", "approach", "grasp"):
            sv = POWER if name == "grasp" else OPEN
            off = self._pre(W0, g_far[1]) if name == "reach_pregrasp" else 0.0
            far = self.cmd(g_far[0] + off, g_far[1], sv)
            deltas[name] = max(deltas[name], self.space.norm(far - nominal[name]) + 0.20)
        # let_go starts wherever the slip left the closed hand on the arc (start .. end of the nominal turn):
        # anchor at mid-arc, open hand backed off; the trust ball covers the arc and the start command
        g_m = self._grip_pose(HUB_NOM, W0, R_NOM, np.pi)
        nominal["let_go"] = self.cmd(g_m[0] - LETGO_OUT * g_m[1][:, 0] + LETGO_UP * n0, g_m[1], OPEN)
        deltas["let_go"] = max(self.space.norm(self.cmd(g[0], g[1], POWER) - nominal["let_go"])
                               for g in (g_s, g_e, g_far)) + 0.20
        spec = [
            ("reach_pregrasp", t_pre, o_pre, None, 10.0, 2.5),
            ("approach", t_app, o_app, None, 6.0, 1.5),
            ("grasp", t_grasp, o_grasp, e_grasp, 4.0, 1.5),
            ("turn", t_turn, o_turn, e_turn, 15.0, 4.5),        # measured mean closed-loop turn 4.1-4.8 s
            ("release", t_rel, o_rel, e_rel, 4.0, 1.0),
            ("retreat", t_ret, o_ret, e_rel, 6.0, 1.5),
            ("let_go", t_let, o_let, e_rel, 4.0, 1.0),
        ]
        self._skills = {name: Skill(name, nominal[name], deltas[name], tgt, out, timeout=to, nominal_duration=dur,
                                    on_enter=ent) for name, tgt, out, ent, to, dur in spec}

    def ready_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Pre-episode hand pose: 8 cm above / 6 cm outside the nominal pre-grasp pose."""
        W0 = wheel_frame(0.0, TILT_NOM)
        p0, R0 = self._grip_pose(HUB_NOM, W0, R_NOM, np.pi - TARGET_NOM / 2)
        return p0 + self._pre(W0, R0) + 0.08 * W0[:, 2] - 0.06 * R0[:, 0], R0

    def skills(self) -> dict:
        return self._skills

    def nominal_plan(self) -> list:
        return ["reach_pregrasp", "approach", "grasp", "turn", "release", "retreat"]

    def skill_graph(self) -> list:
        rl, g, t = "released", "grasped_rim", "turned"
        return [
            SkillSpec("reach_pregrasp", "free", (OutcomeSpec(OK, "pregrasp", word("", rl)),)),
            SkillSpec("approach", "pregrasp", (OutcomeSpec(OK, "at_rim", word(rl)),
                                               OutcomeSpec("bumped", "free", word(rl, ""), failure=True))),
            SkillSpec("grasp", "at_rim", (OutcomeSpec(OK, "grasped", word(rl, "", g)),
                                          OutcomeSpec("missed", "free", word(rl, ""), failure=True))),
            SkillSpec("turn", "grasped", (OutcomeSpec(OK, "turned_held", word(g, {g, t})),
                                          # rim out of the hand / rim slid through the closed hand
                                          OutcomeSpec("slipped", "free", word(g, ""), failure=True),
                                          OutcomeSpec("slipped", "slip_held", word(g), failure=True))),
            # recovery: open the hand where it is and back off; the re-grasp starts from there
            SkillSpec("let_go", "slip_held", tuple(OutcomeSpec(OK, c, word(g, "", rl)) for c in ("free", "pregrasp"))),
            # 'moved': the released wheel was pushed out of the target band (external disturbance); the open
            # hand is then wherever the rim left it relative to the grip point
            SkillSpec("release", "turned_held", (OutcomeSpec(OK, "released", word({g, t}, t, {t, rl})),
                                                 *(OutcomeSpec("moved", c, word({g, t}, t, {t, rl}, rl), failure=True)
                                                   for c in ("free", "pregrasp", "at_rim")))),
            SkillSpec("retreat", "released", (OutcomeSpec(OK, "done", word({t, rl})),
                                              *(OutcomeSpec("moved", c, word({t, rl}, rl), failure=True)
                                                for c in ("free", "pregrasp", "at_rim")))),
        ]

    # ------------------------------------------------------------- stress test
    def disturb(self) -> None:
        """(1) While the hand turns the wheel (measured angle past 35-65 % of the target) a counter-torque
        pulse (KICK_NM: LEAP 3.5-5.0 N m, Aero 7.5-9.0 N m, i.e. above the hand's measured slip torque) for
        0.15 s pushes the wheel back against the grasp (a road-reaction kick): the rim slides 10-70 deg through
        the closed hand -> 'slipped' -> let_go -> re-grasp;
        (2) after the hand has let go at the target (no hand contact, wheel 'turned') the wheel is pushed
        back by a 0.1 s torque pulse sized from the wheel inertia, damping and friction for a 20-35 deg
        back-rotation.  Torques are external (xfrc_applied on the wheel body, about the column axis),
        triggered by measured state, identical for all methods."""
        d, t = self.data, float(self.data.time)
        b = self.bid_wheel
        d.xfrc_applied[b, 3:] = 0.0
        n = self.W[:, 2]
        th = self.angle()
        P = self._pert
        f = self._forces()
        if P["turn_push"] is None:
            if th > self._turn_push_frac * self.target and f["hand_wheel"] > 1.0:
                P["turn_push"] = {"t_start": t, "theta_start": th, "torque_Nm": float(self._turn_push_tau),
                                  "duration_s": 0.15}
        elif not P["turn_push"].get("done"):
            p = P["turn_push"]
            if t < p["t_start"] + 0.15:
                d.xfrc_applied[b, 3:] = -self._turn_push_tau * n
            else:
                p["done"] = True
                p["theta_end"] = th
        if P["push_back"] is None:
            if (P["turn_push"] is not None and P["turn_push"].get("done") and abs(th - self.target) <= TOL_TURNED
                    and f["hand_wheel"] + f["arm_wheel"] < 0.2 and self.ha.synergy()[0] < 0.3):
                P["push_back"] = {"t_trigger": t, "t_start": t + self._back_delay, "theta_trigger": th}
        elif not P["push_back"].get("done"):
            p = P["push_back"]
            T = 0.10
            if "torque_Nm" not in p:
                p["torque_Nm"] = self._pulse_torque(self._back_angle, T, p["theta_trigger"])
            tau = p["torque_Nm"]
            p.update({"target_back_rad": float(self._back_angle), "duration_s": T})
            if p["t_start"] <= t < p["t_start"] + T:
                d.xfrc_applied[b, 3:] = -tau * n
            elif t >= p["t_start"] + T + 0.8:
                p["done"] = True
                p["theta_end"] = th
                p["back_rotation_rad"] = float(p["theta_trigger"] - th)

    def _pulse_torque(self, angle: float, T: float, theta0: float) -> float:
        """Torque of a T-second pulse that rotates the free wheel back by ``angle``: bisection on a 1-D
        model of the column (inertia + armature, viscous damping, Coulomb friction, centring spring)."""
        I = float(self.model.dof_armature[self.w_dadr] + self._wheel_inertia())
        b, fl, k = self.damping, self.frictionloss, self.k_center
        dt = 0.001

        def travel(tau):
            x, w, t = 0.0, 0.0, 0.0
            while t < 5.0:
                drive = (tau if t < T else 0.0) + k * (theta0 - x)
                if w <= 1e-9 and abs(drive) <= fl and t > 0:
                    break
                w = max(0.0, w + dt * (drive - fl - b * w) / I)
                x += dt * w
                t += dt
            return x

        lo, hi = 0.0, 20.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if travel(mid) < angle else (lo, mid)
        return float(0.5 * (lo + hi))

    def _wheel_inertia(self) -> float:
        m = self.model
        b = self.bid_wheel
        # inertia about the hinge axis (body frame z) from the compiled body inertia (principal frame)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.body_iquat[b])
        R = R.reshape(3, 3)
        Ib = R @ np.diag(m.body_inertia[b]) @ R.T
        off = m.body_ipos[b]
        return float(Ib[2, 2] + m.body_mass[b] * (off[0] ** 2 + off[1] ** 2))

    def perturbation_info(self) -> dict:
        return self._pert

    # ---------------------------------------------------------- diagnostics
    def step_diagnostics(self) -> dict:
        f = self._forces()
        fa = self._forces_avg()
        return {"hand_wheel_force_N": f["hand_wheel"], "hand_wheel_force_10ms_N": fa["hand_wheel"], "rig_force_N": f["hand_rig"] + f["arm_rig"],
                "arm_table_force_N": f["arm_table"], "hand_table_force_N": f["hand_table"],
                "wheel_angle_deg": float(np.degrees(self.angle()))}

    def log_signals(self) -> dict:
        f = self._forces()
        return {"grasp": self.ha.grasp_pos(), "theta": self.angle(), "omega": self.omega(),
                "wheel_force": f["hand_wheel"], "synergy": self.ha.synergy()}

    def episode_metrics(self) -> dict:
        return {"final_angle_deg": float(np.degrees(self.angle())), "target_deg": float(np.degrees(self.target)),
                "final_error_deg": float(np.degrees(self.angle() - self.target))}

    def variant_info(self) -> dict:
        return {"hand": self.hand_name, "hub": self.hub, "rig_yaw": self.rig_yaw, "tilt_deg": np.degrees(self.tilt),
                "rim_radius_m": self.R_rim, "tube_radius_m": self.r_tube, "rim_mass_kg": self.m_rim,
                "hub_mass_kg": self.m_hub, "spoke_mass_kg": self.m_spoke, "damping_Nms": self.damping,
                "frictionloss_Nm": self.frictionloss, "centering_Nm_per_rad": self.k_center,
                "rim_friction": self.mu_rim, "target_deg": np.degrees(self.target),
                "grip_phi_deg": np.degrees(self.phi_grip), "wheel_inertia_kgm2": self._wheel_inertia(),
                "perception_bias_far_m": self.est.b_far, "perception_bias_near_m": self.est.b_near,
                "tilt_bias_deg": np.degrees(self.tilt_bias[0]), "radius_bias_m": self.rad_bias}

    def overlay_lines(self) -> list:
        f = self._forces()
        return [f"wheel angle {np.degrees(self.angle()):6.1f} deg  (target {np.degrees(self.target):5.1f})"
                f"   hand-wheel force {f['hand_wheel']:5.1f} N"]
