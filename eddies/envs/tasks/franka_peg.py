"""Task 2 - Franka peg-in-hole insertion (take the peg from the table, insert >= 4 cm, release).

Physical set-up (no welds, teleports, mocap or pose writes after ``reset``):

* **Peg**: anodized-aluminium square bar 24 x 24 x 90 mm (2700 kg/m^3 -> 0.140 kg, mass
  randomized +-6 %) standing upright on the laminate table.  Its friction coefficient
  mu in [0.30, 0.45] governs its contacts with the table and the block (geom priority).
* **Hole block**: 90 x 90 x 62 mm clear-acrylic fixture bolted to the table (static
  geometry), blind square hole 50 mm deep, diametral clearance c in [1.0, 2.0] mm, 45-degree
  entry chamfer with leg in [2.0, 3.0] mm (``assets/peg_board.py``).
* **Robot**: menagerie Franka Panda with the exemplar's realistic hand edits
  (``controllers/franka.py``: ~36 N on the finger tendon = ~18 N per finger at a 24 mm part,
  stiff rubber pads).  The grasp holds only through pad friction (in-hand drift <= 0.8 mm
  during a 0.3 m/s carry).

**Shared decoder with the force channel** (identical for every method): the 8-D command
(TCP position, yaw chart, aperture, force f*) is decoded by the exemplar IK; f* is the
LIMIT of the measured upward reaction force at the wrist.  For f* > 0.5 N an admittance
runs at physics rate (500 Hz) on top of the IK joint reference:

    o_z  <- clip(o_z + dt * K_Z * (F_up - f*), 0, O_Z_MAX)        (force-limited descent)
    o_xy <- o_xy + dt * (K_XY * F_xy - o_xy / TAU_XY), |o_xy| <= O_XY_MAX   (lateral yield)
    q_ctrl = q_IK + J_pos^+ o

so the hand presses with at most f* and yields sideways along the measured contact force
(like a remote-centre-compliance wrist).  f* = 0 is stiff position control (the offsets
decay with 0.1 s).  The wrist force is the measured contact force acting on hand+fingers
(what a wrist F/T sensor reads) with nominal-payload compensation once a grasp is detected,
0.15 N sensor noise and a 10 ms low-pass.

Skills: reach_above_peg, descend_peg, grasp_peg, lift_peg, move_over_hole, insert
(force-limited, jam detection = no depth progress + high measured force), back_off and
spiral_insert (recovery after a jam: lift 8 mm, re-centre on the wrist-camera hole
estimate, Archimedean spiral with light pressing force until the peg drops in), release,
retreat.  Outcomes are decided from measured state: every success outcome of a token that
carries the peg (lift, move_over_hole, insert/spiral 'seated', back_off) and every 'jammed'
requires the peg to be measurably held (both pad forces > 1 N); a peg that slipped out is
classified once it has come to rest as ``dropped`` (upright on the table/block),
``dropped_in_hole`` (fell into the hole out of the closed hand -> cell
``in_hole_unreleased``: open the hand, retreat) or ``toppled`` (lying/leaning).

Stress test (``--perturb``): the standing peg is shoved 2-4 cm along the table (force at
10 mm above its base, sized from mass/friction so it slides without toppling) while the
hand hovers above it, and the gripper force range collapses to +-1 N for ~0.3 s during the
carry, so the peg slips out and falls.  A peg that lands upright is re-grasped (budgeted
``dropped`` outcome); one that falls into the hole is released and left there (budgeted
``dropped_in_hole``); one that topples (lies flat or leans, tilt > 10 deg at rest) cannot
be stood up again with top-down grasps: ``toppled`` is an unlisted outcome and the Streamer stops
(not in the winning set).

LTLf:  G !bad & F(grasped & F(inserted & released))
"""
from __future__ import annotations

import mujoco
import numpy as np

from eddies.envs.assets.peg_board import add_hole_block, add_square_peg
from eddies.envs.controllers.franka import MAX_APERTURE
from eddies.envs.framework.contacts import ContactMonitor
from eddies.envs.framework.perception import PoseEstimator, wrap_to, yaw_of_quat
from eddies.envs.framework.physics import DENSITY
from eddies.envs.framework.scene import body_geoms, subtree_geoms
from eddies.envs.framework.task import OK, OutcomeSpec, Skill, SkillSpec, word
from eddies.envs.tasks.franka_base import TABLE_H, FrankaTaskBase

# ----------------------------------------------------------------- geometry [m]
PEG_SIDE = 0.024
PEG_LEN = 0.090
PEG_MASS_NOM = DENSITY["aluminium"] * PEG_SIDE * PEG_SIDE * PEG_LEN     # 0.140 kg
BLOCK_W = 0.090
BLOCK_H = 0.062
BLOCK_FLOOR = 0.012
HOLE_DEPTH = BLOCK_H - BLOCK_FLOOR                                      # 0.050
INSERT_MIN = 0.040              # required insertion depth (task statement: >= 4 cm)
GRASP_BELOW_TOP = 0.015         # TCP (pad centre) below the peg's top face when grasping
TIP_BELOW_TCP = PEG_LEN - GRASP_BELOW_TOP                               # 0.075 nominal
STAND_TILT = np.radians(10.0)   # a peg on its base is ~vertical; more tilt = lying/leaning (not graspable top-down)
PEG_NOM = np.array([0.50, -0.13])
HOLE_NOM = np.array([0.50, 0.13])
# ----------------------------------------------------------------- skills
H_ABOVE = 0.10                  # TCP above the peg top before descending
LIFT = 0.10                     # lift height of the peg
PRE_INSERT = 0.030              # peg tip above the block top before inserting
BACKOFF = 0.008                 # peg tip above the block top after backing off a jam
V_INSERT = 0.03                 # descent rate of the insertion reference [m/s]
V_SEARCH = 0.02
V_SLOW = 0.012                  # through the chamfer/entry zone and the last 6 mm before the floor
F_INSERT = 10.0                 # force limit (N) commanded during insertion
F_SEARCH = 6.0                  # lighter pressing force during the spiral search
SPIRAL_PITCH = 0.0025           # radial growth per revolution [m]
SPIRAL_PERIOD = 1.0             # s per revolution
SPIRAL_RMAX = 0.0075
# ----------------------------------------------------------------- decoder admittance
K_Z = 0.012                     # m / (N s): vertical force-limiting admittance gain
O_Z_MAX = 0.07
K_XY = 0.002                    # m / (N s): lateral admittance gain
TAU_XY = 1.0                    # s: lateral spring-back time constant
O_XY_MAX = 0.006
F_ACTIVE = 0.5                  # N: force channel below this = stiff position control
YAW_MAX = 1.7                   # rad: commanded hand yaw kept well inside the SO(3) chart
FT_NOISE = 0.15                 # N, wrist force sensor noise (1 sigma per axis)
FT_TAU = 0.010                  # s, sensor low-pass


class FrankaPegInsertionTask(FrankaTaskBase):
    name = "franka_peg_insertion"
    description = ("Franka Panda picks a 24 mm square aluminium peg standing on the table and inserts it "
                   ">= 4 cm into a chamfered blind square hole (1-2 mm clearance) of a fixture block with a "
                   "measured-force-limited compliant descent, releases it and retreats.")
    horizon = 60.0
    success_hold = 1.0
    formula = "G !bad & F(grasped & F(inserted & released))"
    terminal_cells = ("done",)
    main_camera = "main"
    detail_camera = "detail"

    def __init__(self, seed: int, **variant):
        # test-only hook (never set by the registry/runner): extra bias of the hole estimate,
        # used to exercise the jam -> back_off -> spiral_insert recovery path.
        self._hole_bias_test = np.asarray(variant.pop("hole_estimate_bias", (0.0, 0.0)), float)
        super().__init__(seed, **variant)
        rng = self.rng
        self.peg0 = PEG_NOM + np.r_[rng.uniform(-0.05, 0.05), rng.uniform(-0.04, 0.04)]
        self.hole0 = HOLE_NOM + np.r_[rng.uniform(-0.05, 0.05), rng.uniform(-0.03, 0.04)]
        self.peg_yaw0 = rng.uniform(-0.6, 0.6)
        self.hole_yaw0 = rng.uniform(-0.35, 0.35)
        self.peg_mass = PEG_MASS_NOM * rng.uniform(0.94, 1.06)
        self.mu = rng.uniform(0.30, 0.45)
        self.clearance = rng.uniform(0.0010, 0.0020)
        self.chamfer = rng.uniform(0.0020, 0.0030)
        sb = self.make_builder(self.name)
        self.hb = add_hole_block(sb, "hole_block", self.hole0, self.hole_yaw0, TABLE_H,
                                 hole_side=PEG_SIDE + self.clearance, outer=BLOCK_W, height=BLOCK_H,
                                 bottom=BLOCK_FLOOR, chamfer=self.chamfer)
        add_square_peg(sb, "peg", np.r_[self.peg0, TABLE_H], self.peg_yaw0, side=PEG_SIDE, length=PEG_LEN,
                       mass=self.peg_mass, friction=(self.mu, 0.003, 1e-4))
        sb.camera("main", (1.55, -1.05, TABLE_H + 1.0), (0.38, 0.0, TABLE_H + 0.22), fovy=45)
        c = self.hb.entrance
        # close-up of the hole from a low angle: the peg is visible through the clear acrylic walls
        sb.camera("detail", c + np.array([0.16, -0.20, 0.065]), c + np.array([0.0, 0.0, 0.010]), fovy=42)
        sb.camera("top", (0.5, 0.0, TABLE_H + 1.3), (0.5, 0.0, TABLE_H), fovy=50)
        self.finish_build(sb)
        m = self.model
        self.peg_bid = m.body("peg").id
        self.block_geoms = body_geoms(m, ["hole_block"])
        self._build_contact_monitor()
        self.hand_mask = np.zeros(m.ngeom, bool)
        self.hand_mask[[g for g in subtree_geoms(m, self.arm.hand_body)
                        if m.geom_contype[g] or m.geom_conaffinity[g]]] = True
        # perception: static scene camera (far) + wrist camera (near); the peg estimate is
        # of its TIP (bottom-face centre), the hole estimate of its entrance centre.
        self.est_peg = PoseEstimator(rng, near_distance=0.20)     # wrist camera: hand within 20 cm of the peg base
        self.est_hole = PoseEstimator(rng, near_distance=0.15)
        self._contacts = {}
        self._geo = {}
        self._build_skills()
        # stress-test schedule (drawn even when unused so nominal RNG streams match)
        self._push_dir = rng.uniform(-np.pi, np.pi)
        self._push_delay = rng.uniform(0.0, 0.25)
        self._push_slide = rng.uniform(0.02, 0.04)
        self._drop_delay = rng.uniform(0.1, 0.5)
        self._drop_dur = rng.uniform(0.25, 0.35)
        self._pert = {"push": None, "dropout": None}
        self._grip_range = self.model.actuator_forcerange[self.arm.act_grip].copy()
        # decoder state
        self._ft_rng = np.random.default_rng(900_000 + self.seed)
        self._shadow = mujoco.MjData(m)
        self._jacp = np.zeros((3, m.nv))
        self._f6 = np.zeros(6)
        self._reset_decoder()

    # ============================================================ scene helpers
    def _build_contact_monitor(self):
        m = self.model
        fingers = self.arm.finger_geoms["left"] + self.arm.finger_geoms["right"]
        arm_only = [g for g in self.arm.robot_geoms if g not in fingers and (m.geom_contype[g] or m.geom_conaffinity[g])]
        groups = {"L": self.arm.finger_geoms["left"], "R": self.arm.finger_geoms["right"], "fingers": fingers,
                  "arm": arm_only, "table": self.table_geoms, "peg": body_geoms(m, ["peg"]),
                  "block": self.block_geoms}
        pairs = {"L_peg": ("L", "peg"), "R_peg": ("R", "peg"), "arm_peg": ("arm", "peg"),
                 "fingers_table": ("fingers", "table"), "arm_table": ("arm", "table"),
                 "fingers_block": ("fingers", "block"), "arm_block": ("arm", "block"),
                 "peg_block": ("peg", "block"), "peg_table": ("peg", "table")}
        self.cm = ContactMonitor(m, groups, pairs)

    def _reset_decoder(self):
        self._inhand_last = np.array([0.0, 0.0, -TIP_BELOW_TCP])
        self._inhand_yaw_last = 0.0
        self._q_ik = None
        self._Jp = np.zeros((7, 3))
        self._f_lim = 0.0
        self._adm = np.zeros(3)
        self._ft = np.zeros(3)
        self._ft_peak = 0.0
        self._rest_s = 0.0
        self._rest_t = None

    def reset(self) -> None:
        super().reset()
        self._reset_decoder()
        self._q_ik = self.arm.q()
        self._contacts, self._geo = {}, {}

    # ================================================================ state
    def _forces(self):
        if self._contacts.get("t") != self.data.time:
            self._contacts = self.cm.measure(self.data)
            self._contacts["t"] = self.data.time
        return self._contacts

    def _geometry(self):
        if self._geo.get("t") == self.data.time:
            return self._geo
        b = self.data.body("peg")
        R = b.xmat.reshape(3, 3)
        axis = R[:, 2]
        if axis[2] < 0:                       # the peg is symmetric end-to-end
            axis = -axis
        center = b.xpos.copy()
        tip = center - axis * PEG_LEN / 2     # lower end face centre
        top = center + axis * PEG_LEN / 2
        f = self._forces()
        grasp = f["L_peg"] > 1.0 and f["R_peg"] > 1.0
        hb = self.hb
        loc = hb.to_local(tip)
        depth = hb.top_z - tip[2]
        lat_tol = (hb.hole_side - PEG_SIDE) / 2 + 0.002
        tilt = float(np.arccos(np.clip(axis[2], -1.0, 1.0)))
        in_hole_xy = abs(loc[0]) < lat_tol and abs(loc[1]) < lat_tol
        inserted = depth >= INSERT_MIN and in_hole_xy and tilt < np.radians(6.0)
        vel = self.data.cvel[self.peg_bid][3:].copy()
        angvel = self.data.cvel[self.peg_bid][:3].copy()
        self._geo = {"t": self.data.time, "center": center, "quat": b.xquat.copy(), "tip": tip, "top": top,
                     "axis": axis, "tilt": tilt, "vel": vel, "angvel": angvel, "grasp": grasp, "f": f, "depth": float(depth),
                     "loc": loc, "inserted": bool(inserted), "in_hole_xy": bool(in_hole_xy)}
        return self._geo

    def _wrist_force_raw(self) -> np.ndarray:
        """Contact force acting ON the hand + fingers (world frame), i.e. a wrist F/T reading."""
        d, m = self.data, self.model
        F = np.zeros(3)
        n = d.ncon
        if n == 0:
            return F
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        h1, h2 = self.hand_mask[g1], self.hand_mask[g2]
        frames = d.contact.frame
        for c in np.flatnonzero(h1 ^ h2):
            mujoco.mj_contactForce(m, d, int(c), self._f6)
            fw = frames[c].reshape(3, 3).T @ self._f6[:3]       # force on geom2
            F += fw if h2[c] else -fw
        return F

    def _update_rest(self, g) -> float:
        """Seconds the peg has been at rest (|v| < 2 cm/s and |w| < 0.3 rad/s), from measured state."""
        t = float(self.data.time)
        dt = 0.0 if self._rest_t is None else t - self._rest_t
        self._rest_t = t
        at_rest = np.linalg.norm(g["vel"]) < 0.02 and np.linalg.norm(g["angvel"]) < 0.3
        self._rest_s = self._rest_s + dt if at_rest else 0.0
        return float(self._rest_s)

    def observe(self) -> dict:
        obs = self.base_obs()
        g = self._geometry()
        tcp = obs["tcp"]
        tip_e, peg_yaw, near_p = self.est_peg.estimate(g["tip"], yaw_of_quat(g["quat"]), tcp)
        hole_e, hole_yaw, near_h = self.est_hole.estimate(self.hb.entrance, self.hb.yaw, tcp)
        hole_e = hole_e + np.r_[self._hole_bias_test, 0.0]
        f = g["f"]
        tcp_yaw = self.tcp_yaw()
        if g["grasp"]:                         # wrist camera sees the peg in the hand
            self._inhand_last = tip_e - tcp
            self._inhand_yaw_last = wrap_to(peg_yaw - tcp_yaw, np.pi / 2)
        obs.update({"tip_est": tip_e, "peg_yaw": wrap_to(peg_yaw, np.pi / 2), "peg_near": near_p,
                    "hole_est": hole_e, "hole_yaw": wrap_to(hole_yaw, np.pi / 2), "hole_near": near_h,
                    "contact_L": f["L_peg"], "contact_R": f["R_peg"], "grasp": g["grasp"],
                    "tip_true": g["tip"], "top_true": g["top"], "peg_tilt": g["tilt"],
                    "peg_speed": float(np.linalg.norm(g["vel"])), "peg_rest_s": self._update_rest(g),
                    "depth": g["depth"],
                    "inserted": g["inserted"], "in_hole_xy": g["in_hole_xy"], "hole_true": self.hb.entrance,
                    "ft": self._ft.copy(), "f_up": float(self._ft[2]), "tcp_yaw": tcp_yaw,
                    "inhand": self._inhand_last.copy(), "inhand_yaw": float(self._inhand_yaw_last)})
        obs["achieved"] = self.arm.encode(self.space, force=float(np.clip(self._ft[2], 0.0, 40.0)))
        return obs

    def labels(self) -> set:
        g = self._geometry()
        f = g["f"]
        out = set()
        if g["grasp"] and g["tip"][2] > TABLE_H + 0.006:
            out.add("grasped")
        if g["inserted"]:
            out.add("inserted")
        if self._released(f["L_peg"] + f["R_peg"], self.arm.aperture()):
            out.add("released")
        fell = g["center"][2] < TABLE_H - 0.02
        smash = (f["arm_table"] > 25.0 or f["fingers_table"] > 60.0 or f["arm_block"] > 25.0
                 or f["fingers_block"] > 60.0 or f["arm_peg"] > 40.0 or f["peg_block"] > 150.0)
        if fell or smash:
            out.add("bad")
        return out

    @staticmethod
    def _released(pad_force_sum: float, aperture: float) -> bool:
        """Measured release: no pad force and the gripper open wider than the peg."""
        return pad_force_sum < 0.3 and aperture > PEG_SIDE + 0.006

    def task_success(self) -> bool:
        g = self._geometry()
        f = g["f"]
        tcp = self.arm.tcp_pos()
        # 'released' (gripper measured open, no pad force) is part of the physical goal, as in
        # the formula: a peg that fell into the hole out of a closed hand is not yet released.
        return bool(g["inserted"] and not g["grasp"] and self._released(f["L_peg"] + f["R_peg"], self.arm.aperture())
                    and tcp[2] > g["top"][2] + 0.05 and np.linalg.norm(g["vel"]) < 0.03)

    def cell(self, obs: dict, labels: frozenset) -> str:
        tip, top, tcp = obs["tip_true"], obs["top_true"], obs["tcp"]
        grasp, ap = obs["grasp"], obs["aperture"]
        hole = obs["hole_true"]
        if tip[2] < TABLE_H - 0.02 and top[2] < TABLE_H - 0.02:
            return "peg_lost"
        dz = tip[2] - hole[2]
        dxy = np.linalg.norm(tip[:2] - hole[:2])
        if obs["inserted"]:
            if grasp:
                return "inserted_held"
            if not self._released(obs["contact_L"] + obs["contact_R"], ap):
                return "in_hole_unreleased"      # peg fell into the hole out of a closed hand
            return "done" if tcp[2] > top[2] + 0.05 else "released_in_hole"
        if grasp:
            loc = self.hb.to_local(tip)
            over_block = max(abs(loc[0]), abs(loc[1])) < BLOCK_W / 2
            if over_block and -HOLE_DEPTH - 0.005 < dz <= 0.003:     # on the top face / chamfer / in the hole
                return "jammed"
            if dxy < 0.006 and dz <= 0.020:
                return "near_hole"
            if dxy < 0.006 and dz <= 0.080:
                return "over_hole"
            if tip[2] > TABLE_H + 0.05:
                return "lifted"
            return "grasped"
        if obs["peg_tilt"] > STAND_TILT:
            return "peg_down"
        pxy = tip[:2]
        dxy_tcp = np.linalg.norm(tcp[:2] - pxy)
        grasp_z = top[2] - GRASP_BELOW_TOP
        if ap > 0.065 and dxy_tcp < 0.015 and abs(tcp[2] - grasp_z) < 0.012:
            return "at_peg"
        if ap > 0.065 and dxy_tcp < 0.02 and 0.06 < tcp[2] - top[2] < 0.16:
            return "above_peg"
        return "free"

    # ============================================================== decoder
    def initial_command(self) -> np.ndarray:
        return self.arm.encode(self.space)

    def apply_command(self, e) -> None:
        info = self.arm.command_embedding(self.space, e)          # IK -> joint reference
        self._q_ik = self.data.ctrl[self.arm.act_arm].copy()
        self._f_lim = float(info["force"])
        # damped pseudo-inverse of the TCP position Jacobian at the measured configuration
        sh = self._shadow
        sh.qpos[:] = self.data.qpos
        mujoco.mj_kinematics(self.model, sh)
        mujoco.mj_comPos(self.model, sh)
        mujoco.mj_jacSite(self.model, sh, self._jacp, None, self.arm.tcp)
        J = self._jacp[:, self.arm.dadr]
        self._Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(3))
        self.data.ctrl[self.arm.act_arm] = self._q_ik + self._Jp @ self._adm

    def inner_loop(self) -> None:
        """Physics-rate wrist F/T filter + force-limiting / lateral admittance."""
        dt = self.model.opt.timestep
        raw = self._wrist_force_raw()
        g = self._geometry()
        if g["grasp"]:
            raw = raw + np.array([0.0, 0.0, PEG_MASS_NOM * 9.81])     # payload compensation
        raw = raw + self._ft_rng.normal(0.0, FT_NOISE, 3)
        a = dt / (FT_TAU + dt)
        self._ft += a * (raw - self._ft)
        F = self._ft
        self._ft_peak = max(self._ft_peak, float(np.linalg.norm(F)))
        o = self._adm
        if self._f_lim > F_ACTIVE:
            o[2] = float(np.clip(o[2] + dt * K_Z * (F[2] - self._f_lim), 0.0, O_Z_MAX))
            o[:2] += dt * (K_XY * F[:2] - o[:2] / TAU_XY)
            n = np.linalg.norm(o[:2])
            if n > O_XY_MAX:
                o[:2] *= O_XY_MAX / n
        else:
            o *= np.exp(-dt / 0.1)
        if self._q_ik is not None:
            self.data.ctrl[self.arm.act_arm] = self._q_ik + self._Jp @ o

    def achieved(self, obs: dict) -> np.ndarray:
        return obs["achieved"]

    # =============================================================== skills
    def _inhand(self, obs) -> np.ndarray:
        """Offset TCP -> peg tip: wrist-camera measurement while the peg is in the hand, else the
        last in-hand measurement (initially the nominal grasp offset)."""
        return obs["inhand"]

    def _insert_yaw(self, obs) -> float:
        """ABSOLUTE hand yaw aligning the held peg's faces with the hole (mod 90 deg), the
        equivalent closest to the current hand yaw.  The in-hand peg rotation is the latched
        measurement, so the target never chases the hand's own rotation."""
        desired = obs["hole_yaw"] - obs["inhand_yaw"]
        y = obs["tcp_yaw"] + wrap_to(desired - obs["tcp_yaw"], np.pi / 2)
        return float(np.clip(y, -YAW_MAX, YAW_MAX))

    def _over_hole_tcp(self, obs, tip_height: float) -> np.ndarray:
        off = self._inhand(obs)
        h = obs["hole_est"]
        return np.r_[h[:2] - off[:2], h[2] + tip_height - off[2]]

    def _build_skills(self):
        S = {}

        def near(obs, pos, tol):
            return np.linalg.norm(obs["tcp"] - pos) < tol

        def grasp_yaw(obs):
            return wrap_to(obs["peg_yaw"], np.pi / 2)

        def peg_top_est(obs):
            return obs["tip_est"] + np.array([0.0, 0.0, PEG_LEN])

        def toppled(obs):
            return (obs["peg_tilt"] > STAND_TILT and not obs["grasp"]
                    and (obs["peg_rest_s"] > 0.25 or obs["peg_tilt"] > np.radians(60)))

        # 1 reach above the standing peg (re-planned from the live estimate)
        def t_reach(obs, st):
            return self.cmd(peg_top_est(obs) + [0, 0, H_ABOVE], grasp_yaw(obs), MAX_APERTURE)

        def o_reach(obs, labels, st):
            if toppled(obs):
                return "toppled"
            p = peg_top_est(obs) + [0, 0, H_ABOVE]
            return OK if near(obs, p, 0.008) and obs["aperture"] > 0.07 else None

        # 2 descend around the peg top
        def t_descend(obs, st):
            return self.cmd(peg_top_est(obs) - [0, 0, GRASP_BELOW_TOP], grasp_yaw(obs), MAX_APERTURE)

        def o_descend(obs, labels, st):
            if toppled(obs):
                return "toppled"
            return OK if near(obs, peg_top_est(obs) - [0, 0, GRASP_BELOW_TOP], 0.005) else None

        # 3 close the gripper (grasp decided by MEASURED pad contact forces)
        def e_hold(obs, st):
            st.memory["hold"] = obs["tcp"].copy()
            st.memory["yaw"] = obs["tcp_yaw"]

        def t_grasp(obs, st):
            return self.cmd(st.memory["hold"], st.memory["yaw"], 0.0)

        def o_grasp(obs, labels, st):
            if obs["grasp"] and obs["contact_L"] > 3 and obs["contact_R"] > 3 and st.elapsed(obs["t"]) > 0.5:
                return OK
            if obs["aperture"] < PEG_SIDE - 0.008 and not obs["grasp"]:
                return "missed"
            return None

        def dropped(obs, st):
            """The peg physically left the fingers: no pad contact AND tip away from the hand
            (robust to transient force dips); decided once it has settled."""
            held = (obs["contact_L"] + obs["contact_R"] > 0.2 or
                    np.linalg.norm(obs["tip_true"] - obs["tcp"]) < TIP_BELOW_TCP + 0.03)
            if held:
                st.memory["lost"] = None
                return False
            st.memory["lost"] = st.memory.get("lost") or obs["t"]
            return obs["t"] - st.memory["lost"] > 0.1 and (obs["peg_rest_s"] > 0.25 or obs["t"] - st.memory["lost"] > 2.0)

        def drop_outcome(obs, st):
            """Classify a drop from the MEASURED resting state of the peg."""
            if obs["inserted"]:
                return "dropped_in_hole"     # slipped out over the hole and fell in (hand still closed)
            if obs["peg_tilt"] > STAND_TILT:
                return "toppled"
            return "dropped" if st.memory.get("seen_grasped") else "slipped"

        def in_hand(obs):
            """Every success outcome of a token that carries the peg requires the peg to be
            measurably held (both pad forces > 1 N): a peg that has slipped out is never 'ok',
            even while it is still falling along the commanded path (dropped() then decides)."""
            return bool(obs["grasp"])

        # 4 lift
        def t_lift(obs, st):
            return self.cmd(st.memory["hold"] + [0, 0, LIFT], st.memory["yaw"], 0.0)

        def o_lift(obs, labels, st):
            if "grasped" in labels:
                st.memory["seen_grasped"] = True
            if dropped(obs, st):
                return drop_outcome(obs, st)
            return OK if in_hand(obs) and obs["tip_true"][2] > TABLE_H + LIFT - 0.012 else None

        # 5 carry over the hole, peg faces aligned with the hole (wrist-camera estimates)
        def t_move(obs, st):
            return self.cmd(self._over_hole_tcp(obs, PRE_INSERT), self._insert_yaw(obs), 0.0)

        def o_move(obs, labels, st):
            st.memory["seen_grasped"] = True
            if dropped(obs, st):
                return drop_outcome(obs, st)
            tip_e, h = obs["tip_est"], obs["hole_est"]
            yaw_err = abs(wrap_to(obs["peg_yaw"] - obs["hole_yaw"], np.pi / 2))
            ok = (in_hand(obs) and np.linalg.norm(tip_e[:2] - h[:2]) < 0.0015 and yaw_err < np.radians(2.0)
                  and near(obs, self._over_hole_tcp(obs, PRE_INSERT), 0.004))
            return OK if ok else None

        # 6 compliant insertion: force-limited descent (f* = F_INSERT) with jam detection
        def e_insert(obs, st):
            st.memory["zr"] = float(obs["tcp"][2])
            st.memory["tl"] = float(obs["t"])
            st.memory["hist"] = []
            st.memory["seen_grasped"] = True

        def insert_xy_z(obs, st, v_fast):
            """Descending reference (tip 3 mm below the floor at the end; the force limit stops
            it physically).  Slower through the chamfer/entry zone and before the floor
            (measured tip depth) to keep impact forces low."""
            p = self._over_hole_tcp(obs, -HOLE_DEPTH - 0.003)
            d = obs["depth"]
            v = V_SLOW if (-0.006 < d < 0.004 or d > HOLE_DEPTH - 0.006) else v_fast
            t = float(obs["t"])
            st.memory["zr"] = max(p[2], st.memory["zr"] - v * (t - st.memory["tl"]))
            st.memory["tl"] = t
            return np.r_[p[:2], st.memory["zr"]]

        def t_insert(obs, st):
            return self.cmd(insert_xy_z(obs, st, V_INSERT), self._insert_yaw(obs), 0.0, F_INSERT)

        def progress_state(obs, st, f_lim):
            """Returns 'seated' / 'jammed' / None from MEASURED depth progress and force."""
            t, depth = obs["t"], obs["depth"]
            hist = st.memory["hist"]
            hist.append((t, depth))
            while hist and hist[0][0] < t - 0.6:
                hist.pop(0)
            span = hist[-1][0] - hist[0][0] if hist else 0.0
            stalled = span > 0.45 and (max(d for _, d in hist) - hist[0][1]) < 0.001
            if obs["inserted"] and (depth > HOLE_DEPTH - 0.003 or (stalled and depth > INSERT_MIN + 0.004)):
                return "seated"
            if stalled and obs["f_up"] > 0.6 * f_lim and depth < INSERT_MIN:
                return "jammed"
            return None

        def o_insert(obs, labels, st):
            if dropped(obs, st):
                return drop_outcome(obs, st)
            s = progress_state(obs, st, F_INSERT)
            if s == "seated" and in_hand(obs):
                return OK
            if s == "jammed" and in_hand(obs):
                return "jammed"
            return None

        # 7 back off a jam: lift the tip 8 mm above the block top, re-centre (stiff)
        def t_backoff(obs, st):
            return self.cmd(self._over_hole_tcp(obs, BACKOFF), self._insert_yaw(obs), 0.0, 0.0)

        def o_backoff(obs, labels, st):
            if dropped(obs, st):
                return drop_outcome(obs, st)
            ok = (in_hand(obs) and obs["tip_true"][2] - obs["hole_true"][2] > 0.004
                  and near(obs, self._over_hole_tcp(obs, BACKOFF), 0.004))
            return OK if ok and st.elapsed(obs["t"]) > 0.3 else None

        # 8 spiral search with light pressing force until the peg drops in, then insert
        def e_spiral(obs, st):
            e_insert(obs, st)
            st.memory["xy_eng"] = None

        def t_spiral(obs, st):
            el = st.elapsed(obs["t"])
            p = insert_xy_z(obs, st, V_SEARCH if st.memory["xy_eng"] is None else V_INSERT)
            if st.memory["xy_eng"] is None and obs["depth"] > 0.004:      # dropped into the hole
                st.memory["xy_eng"] = obs["tcp"][:2].copy()
            if st.memory["xy_eng"] is not None:
                xy = st.memory["xy_eng"]
                f = F_INSERT
            else:
                th = 2 * np.pi * el / SPIRAL_PERIOD
                r = min(SPIRAL_RMAX, SPIRAL_PITCH * el / SPIRAL_PERIOD)
                xy = p[:2] + r * np.array([np.cos(th), np.sin(th)])
                f = F_SEARCH
            return self.cmd(np.r_[xy, p[2]], self._insert_yaw(obs), 0.0, f)

        def o_spiral(obs, labels, st):
            if dropped(obs, st):
                return drop_outcome(obs, st)
            el = st.elapsed(obs["t"])
            if st.memory["xy_eng"] is None:
                if el > SPIRAL_PERIOD * (SPIRAL_RMAX / SPIRAL_PITCH + 1.5) and in_hand(obs):
                    return "jammed"
                progress_state(obs, st, F_SEARCH)
                return None
            s = progress_state(obs, st, F_INSERT)
            if s == "seated" and in_hand(obs):
                return OK
            if s == "jammed" and in_hand(obs):
                return "jammed"
            return None

        # 9 open the gripper (peg stays in the hole)
        def t_release(obs, st):
            return self.cmd(st.memory["hold"], st.memory["yaw"], MAX_APERTURE, 0.0)

        def o_release(obs, labels, st):
            ok = obs["aperture"] > PEG_SIDE + 0.012 and obs["contact_L"] + obs["contact_R"] < 0.3
            return OK if ok else None

        # 10 retreat upward
        def t_retreat(obs, st):
            return self.cmd(st.memory["hold"] + [0, 0, 0.12], st.memory["yaw"], MAX_APERTURE, 0.0)

        def o_retreat(obs, labels, st):
            return OK if obs["tcp"][2] > obs["top_true"][2] + 0.08 else None

        # ------------------------------------------------ anchors / trust radii
        peg_top = np.r_[PEG_NOM, TABLE_H + PEG_LEN]
        hole_top = np.r_[HOLE_NOM, TABLE_H + BLOCK_H]
        grasp_p = peg_top - [0, 0, GRASP_BELOW_TOP]
        nominal = {
            "reach_above_peg": self.cmd(peg_top + [0, 0, H_ABOVE], 0, MAX_APERTURE),
            "descend_peg": self.cmd(grasp_p, 0, MAX_APERTURE),
            "grasp_peg": self.cmd(grasp_p, 0, 0.0),
            "lift_peg": self.cmd(grasp_p + [0, 0, LIFT], 0, 0.0),
            "move_over_hole": self.cmd(hole_top + [0, 0, PRE_INSERT + TIP_BELOW_TCP], 0, 0.0),
            "insert": self.cmd(hole_top + [0, 0, -HOLE_DEPTH - 0.003 + TIP_BELOW_TCP], 0, 0.0, F_INSERT),
            "back_off": self.cmd(hole_top + [0, 0, BACKOFF + TIP_BELOW_TCP], 0, 0.0, 0.0),
            "spiral_insert": self.cmd(hole_top + [0, 0, -HOLE_DEPTH - 0.003 + TIP_BELOW_TCP], 0, 0.0, F_SEARCH),
            "release": self.cmd(hole_top + [0, 0, -HOLE_DEPTH + TIP_BELOW_TCP], 0, MAX_APERTURE, 0.0),
            "retreat": self.cmd(hole_top + [0, 0, -HOLE_DEPTH + TIP_BELOW_TCP + 0.12], 0, MAX_APERTURE, 0.0),
        }
        home = self.cmd([0.474, 0.0, TABLE_H + 0.41], 0, MAX_APERTURE)
        prev = home
        deltas = {}
        for name in self.nominal_plan():
            deltas[name] = self.space.norm(nominal[name] - prev) + 0.32
            prev = nominal[name]
        deltas["back_off"] = self.space.norm(nominal["back_off"] - nominal["insert"]) + 0.32
        deltas["spiral_insert"] = self.space.norm(nominal["spiral_insert"] - nominal["back_off"]) + 0.32
        spec = [
            ("reach_above_peg", t_reach, o_reach, None, 10.0, 2.5),
            ("descend_peg", t_descend, o_descend, None, 6.0, 1.5),
            ("grasp_peg", t_grasp, o_grasp, e_hold, 3.0, 1.2),
            ("lift_peg", t_lift, o_lift, e_hold, 6.0, 1.5),
            ("move_over_hole", t_move, o_move, None, 10.0, 2.5),
            ("insert", t_insert, o_insert, e_insert, 10.0, 3.5),
            ("back_off", t_backoff, o_backoff, None, 4.0, 1.0),
            ("spiral_insert", t_spiral, o_spiral, e_spiral, 14.0, 6.0),
            ("release", t_release, o_release, e_hold, 3.0, 1.0),
            ("retreat", t_retreat, o_retreat, e_hold, 6.0, 1.5),
        ]
        for name, tgt, out, enter, timeout, dur in spec:
            S[name] = Skill(name, nominal[name], deltas[name], tgt, out, timeout=timeout,
                            nominal_duration=dur, on_enter=enter)
        self._skills = S
        self._nominal_anchors = nominal

    def skills(self) -> dict:
        return self._skills

    def nominal_plan(self) -> list:
        return ["reach_above_peg", "descend_peg", "grasp_peg", "lift_peg", "move_over_hole",
                "insert", "release", "retreat"]

    def skill_graph(self) -> list:
        e = word("")
        r = word("released")
        g = word("grasped")
        gi = {"grasped", "inserted"}
        # A peg that slips out of the hand (measured: no pad force, away from the hand, at rest)
        # while it is carried over / pressed into the hole lands upright on the table or block
        # ('dropped' -> free, re-graspable) or falls into the hole with the hand still closed
        # ('dropped_in_hole' -> in_hole_unreleased: open the hand, retreat).  A peg that lands
        # lying/leaning ('toppled') stays unlisted (no top-down recovery exists).
        drops = (OutcomeSpec("dropped", "free", word("grasped", ""), failure=True),
                 OutcomeSpec("dropped_in_hole", "in_hole_unreleased", word("grasped", "", "inserted"), failure=True))
        return [
            SkillSpec("reach_above_peg", "free", (OutcomeSpec(OK, "above_peg", word("", "released")),)),
            SkillSpec("reach_above_peg", "at_peg", (OutcomeSpec(OK, "above_peg", r),)),
            SkillSpec("descend_peg", "above_peg", (OutcomeSpec(OK, "at_peg", r),)),
            SkillSpec("grasp_peg", "at_peg", (OutcomeSpec(OK, "grasped", word("released", "")),
                                              OutcomeSpec("missed", "free", word("released", ""), failure=True))),
            SkillSpec("lift_peg", "grasped", (OutcomeSpec(OK, "lifted", word("", "grasped")),
                                              OutcomeSpec("dropped", "free", word("", "grasped", ""), failure=True),
                                              OutcomeSpec("slipped", "free", e, failure=True))),
            SkillSpec("move_over_hole", "lifted", (OutcomeSpec(OK, "over_hole", g),) + drops),
            SkillSpec("insert", "over_hole", (OutcomeSpec(OK, "inserted_held", word("grasped", gi)),
                                              OutcomeSpec("jammed", "jammed", g, failure=True)) + drops),
            SkillSpec("back_off", "jammed", (OutcomeSpec(OK, "near_hole", g),) + drops),
            SkillSpec("spiral_insert", "near_hole", (OutcomeSpec(OK, "inserted_held", word("grasped", gi)),
                                                     OutcomeSpec("jammed", "jammed", g, failure=True)) + drops),
            SkillSpec("release", "inserted_held",
                      (OutcomeSpec(OK, "released_in_hole", word(gi, "inserted", {"inserted", "released"})),)),
            # the fallen peg's top may already be > 5 cm below the hand, so opening can end in 'done'
            SkillSpec("release", "in_hole_unreleased",
                      (OutcomeSpec(OK, "released_in_hole", word("inserted", {"inserted", "released"})),
                       OutcomeSpec(OK, "done", word("inserted", {"inserted", "released"})))),
            SkillSpec("retreat", "released_in_hole", (OutcomeSpec(OK, "done", word({"inserted", "released"})),)),
        ]

    # ========================================================== stress test
    def disturb(self) -> None:
        """(1) 'peg bumped before grasp': while the open gripper hovers/descends above the
        peg, a horizontal shove at 10 mm above the peg's base (force + equivalent torque about
        the CoM) slides the standing peg 2-4 cm.  The two-level profile (1.6 mu m g, then
        0.6 mu m g while it decelerates) is sized from the peg's mass and friction so that it
        slides without toppling (a free slide of a 24x90 mm peg with mu > b/h = 0.27 would tip).
        (2) 'grasp force dropout during transport': the gripper force range collapses to
        +-1 N for 0.25-0.35 s while the lifted peg is carried to the hole (transient
        actuator fault).  Both triggered by measured state, identical for all methods."""
        d, t = self.data, self.data.time
        d.xfrc_applied[self.peg_bid, :] = 0.0
        g = self._geometry()
        tcp = self.arm.tcp_pos()
        P = self._pert
        if P["push"] is None:
            if (not g["grasp"] and np.linalg.norm(tcp[:2] - g["tip"][:2]) < 0.02
                    and 0.04 < tcp[2] - g["top"][2] < 0.12 and self.arm.aperture() > 0.06):
                P["push"] = {"t_trigger": t, "t_start": t + self._push_delay, "dir": self._push_dir,
                             "start_xy": g["tip"][:2].copy()}
        elif P["push"].get("done") is None:
            p = P["push"]
            mu, m, gr = self.mu, self.peg_mass, 9.81
            a1, a2 = 0.6 * mu * gr, 0.4 * mu * gr
            T1 = np.sqrt(2 * self._push_slide / (a1 * a1 * (1 / a1 + 1 / a2)))
            T2 = a1 * T1 / a2
            p.update({"target_slide_m": float(self._push_slide), "T1_s": float(T1), "T2_s": float(T2),
                      "force_N": float(1.6 * mu * m * gr), "force_tail_N": float(0.6 * mu * m * gr),
                      "applied_at_height_m": 0.010})
            tau = t - p["t_start"]
            if 0.0 <= tau < T1 + T2:
                F = (1.6 if tau < T1 else 0.6) * mu * m * gr
                u = np.array([np.cos(p["dir"]), np.sin(p["dir"]), 0.0])
                r = np.array([0.0, 0.0, 0.010 - PEG_LEN / 2])     # application point - CoM
                d.xfrc_applied[self.peg_bid, :3] = F * u
                d.xfrc_applied[self.peg_bid, 3:] = np.cross(r, F * u)
            elif tau >= T1 + T2:
                p["done"] = True
                p["t_end"] = t
                p["displacement_m"] = float(np.linalg.norm(g["tip"][:2] - p["start_xy"]))
                p["tilt_after_deg"] = float(np.degrees(g["tilt"]))
        elif P["push"].get("settled") is None and t > P["push"]["t_end"] + 0.5:
            P["push"]["settled"] = True
            P["push"]["displacement_settled_m"] = float(np.linalg.norm(g["tip"][:2] - P["push"]["start_xy"]))
            P["push"]["tilt_settled_deg"] = float(np.degrees(g["tilt"]))
        if P["dropout"] is None and g["grasp"] and g["tip"][2] > TABLE_H + 0.08:
            P["dropout"] = {"t_trigger": t, "t_start": t + self._drop_delay, "duration_s": float(self._drop_dur),
                            "force_range_N": 1.0, "tip_offset_before_m": None}
        drop = P["dropout"]
        lim = self.model.actuator_forcerange[self.arm.act_grip]
        if drop is not None and drop["t_start"] <= t < drop["t_start"] + drop["duration_s"]:
            if drop["tip_offset_before_m"] is None:
                drop["tip_offset_before_m"] = float(tcp[2] - g["tip"][2])
            lim[:] = (-1.0, 1.0)
        else:
            lim[:] = self._grip_range
            if drop is not None and t >= drop["t_start"] + drop["duration_s"] and "tip_offset_after_m" not in drop:
                drop["tip_offset_after_m"] = float(tcp[2] - g["tip"][2])
                drop["held_after"] = bool(g["grasp"])

    def perturbation_info(self) -> dict:
        return self._pert

    # ========================================================== diagnostics
    def step_diagnostics(self) -> dict:
        f = self._forces()
        g = self._geometry()
        return {"arm_table_force_N": f["arm_table"], "finger_table_force_N": f["fingers_table"],
                "finger_block_force_N": f["fingers_block"], "arm_block_force_N": f["arm_block"],
                "peg_block_force_N": f["peg_block"], "wrist_force_N": float(np.linalg.norm(self._ft)),
                "wrist_force_up_N": float(self._ft[2]),
                "peg_depth_in_hole_m": g["depth"] if g["in_hole_xy"] else 0.0,
                "admittance_offset_z_m": float(self._adm[2]),
                "admittance_offset_xy_m": float(np.linalg.norm(self._adm[:2]))}

    def log_signals(self) -> dict:
        g = self._geometry()
        return {"peg_tip": g["tip"], "tcp": self.arm.tcp_pos(), "aperture": self.arm.aperture(),
                "wrist_force": self._ft.copy(), "depth": g["depth"], "adm_offset": self._adm.copy(),
                "force_limit": self._f_lim}

    def episode_metrics(self) -> dict:
        g = self._geometry()
        # depth below the block top is only meaningful when the peg tip is in the hole column
        return {"final_depth_in_hole_m": g["depth"] if g["in_hole_xy"] else None,
                "final_peg_tilt_deg": float(np.degrees(g["tilt"])),
                "final_tip_offset_in_hole_m": float(np.linalg.norm(g["loc"][:2])),
                "final_inserted": g["inserted"], "peak_wrist_force_N": self._ft_peak}

    def variant_info(self) -> dict:
        return {"peg_init_xy": self.peg0, "hole_xy": self.hole0, "peg_yaw": self.peg_yaw0,
                "hole_yaw": self.hole_yaw0, "peg_mass_kg": self.peg_mass, "peg_friction": self.mu,
                "peg_side_m": PEG_SIDE, "peg_length_m": PEG_LEN, "hole_side_m": self.hb.hole_side,
                "diametral_clearance_m": self.clearance, "chamfer_m": self.chamfer,
                "hole_depth_m": HOLE_DEPTH, "perception_bias_peg_m": self.est_peg.b_far,
                "perception_bias_hole_m": self.est_hole.b_far,
                "perception_bias_peg_near_m": self.est_peg.b_near,
                "perception_bias_hole_near_m": self.est_hole.b_near,
                "test_hole_estimate_bias_m": self._hole_bias_test}

    def overlay_lines(self) -> list:
        g = self._geometry()
        over_block = max(abs(g["loc"][0]), abs(g["loc"][1])) < BLOCK_W / 2
        depth = f"{1e3 * g['depth']:6.1f} mm" if over_block else "   -   "
        return [f"peg depth {depth} (need >= {1e3 * INSERT_MIN:.0f} mm)   "
                f"wrist F_up {self._ft[2]:5.1f} N (limit {self._f_lim:4.1f} N)",
                f"admittance dz {1e3 * self._adm[2]:4.1f} mm  dxy {1e3 * np.linalg.norm(self._adm[:2]):3.1f} mm   "
                f"clearance {1e3 * self.clearance:.2f} mm  chamfer {1e3 * self.chamfer:.1f} mm"]
