"""Task 15 - tighten a small threaded peg on a black construction with a dexterous hand (UR5e + LEAP / Aero).

SCENE.  A black anodised construction (base plate + 4.4 cm square column, static: it
is bolted to the black workbench) carries a small threaded peg: a steel bolt with a
red knurled plastic KNOB (16-gon convex prism, diameter 23-25 mm, 20 mm high) on a
10 mm neck and a washer.  The bolt is ONE rigid body with two joints along its
axis: a HINGE (thread rotation) and a SLIDE (axial advance), coupled by a MuJoCo
joint EQUALITY  slide = lead * hinge  (lead = pitch / 2 pi, pitch 2-3 mm/turn).
Thread friction is the hinge ``frictionloss`` (running torque 0.015-0.03 N m,
self-locking: gravity alone cannot turn it) plus viscous ``damping``; the slide has
a 0.5 N Coulomb friction.  qpos = 0 is the SEATED pose (washer face on the column
top); the seating stop is the hinge lower limit (stiff, 4 ms), the slide's own
lower limit sits 1 mm below as a backstop.  At reset the bolt is unscrewed
1.05-1.30 turns (pre-episode joint write, allowed in ``reset``).

Numerics of the thread: MuJoCo's constraint
regularizer for a joint equality uses ``dof_invweight0`` of BOTH joints, not scaled by
the polynomial slope, so a realistic 2e-6 kg m^2 knob made the thread compliant
(3.8 mm axial violation under 10 N).  The equality uses solimp 0.9999 and the hinge
an armature of 2e-4 kg m^2: violation ~0.08 mm per 10 N axial load.  The extra
inertia costs < 2e-3 N m at the turning accelerations (< 10 % of the thread friction).

GRASP.  Both hands' thumb-index pinch synergies close ALONG the finger axis (the
index curls back onto the thumb), so a vertical knob cannot be pinched across its
diameter with the fingers pointing down.  The hand therefore pinches the knob
FROM ABOVE with the palm DOWN and the fingers horizontal (index on the far side,
thumb on the near side); the knob is turned about the PALM NORMAL, i.e. the hand
frame is rotated about the vertical knob axis through the pinch centre.  For the
UR5e this rotation is mainly ``wrist_2`` (flange horizontal), with small arm
compensation - not ``wrist_3`` as in a fingers-down tripod.
One regrasp cycle = pinch -> turn (hand heading psi_start -> psi_start - TURN, clockwise seen
from above = tightening a right-hand thread; LEAP 35 -> -55 deg, Aero 30 -> -50 deg) -> release
(open to the pre-shape) -> unwind (rotate the open hand back about the knob axis; after a slip
the hand first opens where it is) -> pinch ...  Turn / unwind targets: a heading reference that
moves toward the goal at a per-hand rate and stays within a per-hand ``lead`` of the MEASURED
heading; the commanded position keeps the pinch centre on the (estimated) knob axis.  Speed
matters physically: the shared IK/servo lag moves the pinch centre off the knob axis, and a
lateral offset d of the pinching hand ROLLS the knob by ~d / r_knob (opposite pad tangential
forces).  Aero (15 cm grasp-site lever): 25 deg/s -> 8-12 mm offset, knob/hand ratio 0.5;
16 deg/s -> 2-4 mm, ratio 0.9-1.15 (dev sweep, seeds 0-3).  A turn that rolls the knob BACK by
> 10 deg is aborted as ``slipped``.

PROGRESS ABSTRACTION.  The supervisor needs a rank-decreasing game, so the abstract
cells carry r = remaining thread rotation in 45 deg units (measured hinge encoder:
hinge = rotation left to the seat), with a +-5 deg hysteresis so that the few
degrees a pinch/release can jiggle the knob rarely flip r (a forward drift during
release is a modelled outcome, a backward jiggle or knock-back a budgeted failure).  A successful turn
(>= 58 deg measured knob rotation) decreases r by 1-3; a slip (< 58 deg or the
pinch lost) is a budgeted failure.  Cells: free_r, above_r, at_r, held_r, turned_r,
near_r, seated_held, seated_open, done.

LTLf:  G !bad & F(seated & F released)

* seated    slide encoder < 0.06 mm AND hinge within 6 deg of the seat stop
* released  total hand-knob contact force < 0.2 N
* bad       (contact forces low-pass filtered, tau 10 ms) hand/arm force on the knob > 60 N,
            palm/arm on the construction > 25 N, fingers on the construction > 40 N, arm on
            the table > 20 N, hand on the table > 30 N, arm-hand self-collision > 20 N
SUCCESS (measured): seated, >= 1 full turn of measured rotation in the episode, no
hand-knob contact, all fingertips >= 3 cm above the knob top (held 0.5 s).
"""
from __future__ import annotations

import mujoco
import numpy as np

from eddies.envs.controllers.hand_arm import hand_frame
from eddies.envs.framework.perception import PoseEstimator
from eddies.envs.framework.physics import FRICTION
from eddies.envs.framework.task import OK, Skill, SkillSpec, OutcomeSpec, word
from eddies.envs.tasks.hand_base import TABLE_H, HandArmTaskBase

# ---------------------------------------------------------------- geometry
PLATE_HALF = (0.075, 0.075, 0.006)     # construction base plate half sizes [m]
COL_HALF = (0.022, 0.022)              # column half sizes (x, y) [m]
WASHER_T = 0.002                       # washer thickness [m]
NECK = 0.010                           # bolt neck between the washer and the knob [m]
KNOB_H = 0.020                         # knob height [m]
NOM_XY = np.array([0.50, 0.08])        # nominal construction centre on the table
# ---------------------------------------------------------------- turning
ANG_TOL = np.radians(8.0)              # heading tolerance of the abstract cells
UNIT = np.radians(45.0)                # progress unit of the abstraction [rad]
HYST = np.radians(5.0)                 # hysteresis of the progress index [rad]
TURN_OK = np.radians(58.0)             # minimum measured knob rotation of a successful turn (> UNIT + 2 HYST)
BACKSPIN = np.radians(10.0)            # knob turned BACK (loosened) this much during a turn -> slipped
R_MAX = 11                             # largest progress index (1.30 turns = 468 deg -> 11)
# ---------------------------------------------------------------- logic
SEAT_SLIDE = 0.00006                   # [m]
SEAT_ANGLE = np.radians(6.0)           # [rad]
F_PINCH = 0.5                          # per-finger contact force of a pinch [N]
F_FREE = 0.2                           # "released" [N]
H_ABOVE = 0.05                         # pinch centre above the knob centre for "above" [m]
H_READY = 0.12
TIPS_CLEAR = 0.03                      # fingertips above the knob top for "done" [m]
F_BAD_KNOB = 60.0
# "bad" contact-force limits [N] (filtered with BAD_TAU): hand/arm on the knob, palm/arm on the construction,
# fingers on the construction, arm on the table, hand on the table, arm-hand self-collision
BAD_LIMITS = {"knob": F_BAD_KNOB, "palm_constr": 25.0, "fingers_constr": 40.0, "arm_table": 20.0,
              "hand_table": 30.0, "arm_hand": 20.0}
BAD_TAU = 0.010                        # [s]
# per hand: pinch centre ``c`` in the hand (grasp-site) frame [m] (calibrated so that the pads meet the knob
# around mid-height), pre-shape ``s_pre`` and closed pinch ``s_close`` (pinch synergy), the turning window
# (``psi_start``, span ``turn`` [deg]) and the UR5e IK branch seed.  Turn / unwind heading reference: moves at
# ``w_turn`` / ``w_unwind`` [deg/s] (1000 = unlimited) and stays within ``lead`` / ``lead_unwind`` of the MEASURED
# heading; the commanded position keeps the pinch centre on the knob axis for the heading
# psi_meas + alpha (psi_ref - psi_meas).  LEAP: knob/hand ratio 0.95-1.08 at lead 20 deg (unlimited rate);
# Aero: 16 deg/s, lead 10 deg (ratio 0.9-1.15; at 25 deg/s only 0.5, see the module doc).
# Windows (kinematic sweep of the IK solution along the turn, wrist_2 > 0 branch): LEAP singular near -75 deg
# (smallest Jacobian singular value 0.011), so 35 -> -55 deg (sv >= 0.17); the Aero palm (17 deg tilted,
# 15 cm lever) COLLIDES with the UR5e wrist_1 link between -55 and -65 deg (measured 600 N self-contact during
# an unwind before this was found) and the arm is weakest near +45 deg (sv 0.07), so 30 -> -50 deg (80 deg,
# clearance >= 29 mm, sv >= 0.09).
Q_W2POS = (-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, 0.0)
HAND_CFG = {
    "leap": {"c": (0.071, 0.044, 0.007), "s_pre": 0.75, "s_close": 1.0, "psi_start": np.radians(35.0),
             "q_seed": Q_W2POS, "lead": np.radians(20.0), "alpha": 1.0,
             "w_turn": 1000.0, "w_unwind": 1000.0, "turn": 90.0,
             "lead_unwind": np.radians(20.0)},
    "aero": {"c": (0.033, 0.027, 0.022), "s_pre": 0.55, "s_close": 1.0, "psi_start": np.radians(30.0),
             "q_seed": Q_W2POS, "lead": np.radians(10.0), "alpha": 1.0,
             "w_turn": 16.0, "w_unwind": 1000.0, "turn": 80.0,
             "lead_unwind": np.radians(10.0)},
}


def prism_mesh(radius: float, height: float, n: int = 16, chamfer: float = 0.0015):
    """Convex n-gon prism (knurled knob) with a chamfered top edge, base at z = 0."""
    verts = []
    for z, r in ((0.0, radius), (height - chamfer, radius), (height, radius - chamfer)):
        for i in range(n):
            a = 2 * np.pi * (i + 0.5) / n
            verts += [r * np.cos(a), r * np.sin(a), z]
    return verts


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


R_PALM_DOWN = hand_frame((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))   # fingers +x, palm down, thumb +y


def turn_frame(psi: float) -> np.ndarray:
    """Hand frame with the palm down and the fingers pointing at world yaw ``psi``."""
    return rot_z(psi) @ R_PALM_DOWN


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class HandPegTightenTask(HandArmTaskBase):
    task_id = "peg_tighten"
    description = ("UR5e with a dexterous hand pinches the knob of a threaded peg on a black construction from "
                   "above and tightens it by >= 1 turn in regrasp cycles (pinch, turn 90 deg, release, unwind) "
                   "until it is seated, then releases it and retreats.")
    horizon = 90.0
    success_hold = 0.5
    formula = "G !bad & F(seated & F released)"
    terminal_cells = ("done",)
    failure_budget = 3
    main_camera = "main"
    detail_camera = "detail"

    def __init__(self, seed: int, hand: str = "leap", **variant):
        super().__init__(seed, hand=hand, **variant)
        rng = self.rng
        # ------------------------------------------------ randomized physics
        self.cxy = NOM_XY + np.r_[rng.uniform(-0.03, 0.03), rng.uniform(-0.04, 0.04)]
        self.col_h = rng.uniform(0.085, 0.095)                 # column height [m]
        self.pitch = rng.uniform(0.002, 0.003)                 # thread pitch [m / turn]
        self.turns0 = rng.uniform(1.05, 1.30)                  # turns from the seat at the start
        self.knob_r = rng.uniform(0.0115, 0.0125)              # knob radius [m]
        self.shaft_r = 0.006                                   # M12 shank
        self.bolt_mass = rng.uniform(0.020, 0.030)             # [kg]
        self.thread_fric = rng.uniform(0.015, 0.030)           # running torque (frictionloss) [N m]
        self.thread_damp = rng.uniform(0.002, 0.004)           # [N m s / rad]
        self.cfg = HAND_CFG[self.hand_name]
        self.c_pinch = np.asarray(self.cfg["c"], float)
        self.s_pre = float(self.cfg["s_pre"])
        self.s_close = float(self.cfg["s_close"])
        self.psi_start = float(self.cfg["psi_start"])
        self.turn = float(np.radians(self.cfg["turn"]))
        self.psi_end = self.psi_start - self.turn
        self.lead = float(self.cfg["lead"])
        self.lead_unwind = float(self.cfg["lead_unwind"])
        self.alpha = float(self.cfg["alpha"])
        self.w_turn = np.radians(float(self.cfg["w_turn"]))
        self.w_unwind = np.radians(float(self.cfg["w_unwind"]))
        # ------------------------------------------------------- scene
        sb = self.make_builder()
        self._build_construction(sb)
        kc = self.knob_center_nominal()
        sb.camera("main", (1.25, -1.00, TABLE_H + 0.60), (0.25, 0.08, TABLE_H + 0.25), fovy=42)
        sb.camera("detail", kc + np.array([0.19, -0.21, 0.035]), kc + np.array([0.0, 0.0, 0.005]), fovy=38)
        sb.camera("side", kc + np.array([0.0, -0.30, 0.02]), kc, fovy=40)
        self.R_base = turn_frame(self.psi_start - self.turn / 2)
        self.finish_build(sb, base_mat=self.R_base)
        m = self.model
        self.j_slide = m.joint("bolt_slide").id
        self.j_hinge = m.joint("bolt_hinge").id
        self.q_slide = int(m.jnt_qposadr[self.j_slide])
        self.q_hinge = int(m.jnt_qposadr[self.j_hinge])
        self.bid_bolt = m.body("bolt").id
        self.sid_knob = m.site("knob_center").id
        self.cm = self.contact_monitor({"knob": ["bolt"], "constr": ["construction"]},
                                       pairs={"fingers_constr": ("fingers", "constr"),
                                              "arm_hand": ("arm", "hand")})
        self.est = PoseEstimator(rng, near_distance=0.15)
        # ------------------------------------------------ reset configuration
        self.h0 = 2 * np.pi * self.turns0
        self.q_ready = self.solve_ready(self._grasp_for(kc + [0, 0, H_READY], self.psi_start),
                                        turn_frame(self.psi_start), q0=np.asarray(self.cfg["q_seed"], float))
        self.n_cycles = int(np.ceil(self.h0 / (0.9 * self.turn)))   # nominal regrasp cycles (knob/hand ratio ~0.9)
        self._contacts = {}
        self._r = None
        self._max_knob_force = 0.0
        self._ff, self._ff_max, self._ff_t = {}, {}, None
        self._build_skills()
        # ------------------------------------------------ stress test schedule (always drawn)
        self._knock_delay = rng.uniform(0.1, 0.4)
        self._knock_angle = np.radians(rng.uniform(10.0, 20.0))
        self._drop_delay = rng.uniform(0.2, 0.6)
        self._drop_level = rng.uniform(0.10, 0.20)
        self._pert = {"knock_back": None, "grip_dropout": None}
        self._hand_range = self.model.actuator_forcerange[self.ha.act_hand].copy()

    # ----------------------------------------------------------------- scene
    def knob_center_nominal(self) -> np.ndarray:
        z = TABLE_H + 2 * PLATE_HALF[2] + self.col_h + WASHER_T + NECK + KNOB_H / 2 + self.turns0 * self.pitch
        return np.r_[self.cxy, z]

    def _build_construction(self, sb) -> None:
        spec = sb.spec
        sb.material("constr_mat", (0.05, 0.05, 0.055, 1), specular=0.4, shininess=0.5)
        sb.material("knob_mat", (0.85, 0.20, 0.05, 1), specular=0.5, shininess=0.6)
        sb.material("steel_mat", (0.70, 0.71, 0.73, 1), specular=0.8, shininess=0.8)
        body = spec.worldbody.add_body(name="construction", pos=[self.cxy[0], self.cxy[1], TABLE_H])
        px, py, pz = PLATE_HALF
        body.add_geom(name="constr_plate", type=mujoco.mjtGeom.mjGEOM_BOX, size=[px, py, pz], pos=[0, 0, pz],
                      material="constr_mat", friction=list(FRICTION["metal"]), condim=3)
        for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):   # visual screw heads on the plate
            body.add_geom(name=f"constr_screw{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.005, 0.002, 0],
                          pos=[sx * (px - 0.012), sy * (py - 0.012), 2 * pz + 0.002], material="steel_mat",
                          contype=0, conaffinity=0, mass=0.0)
        cx, cy = COL_HALF
        z0 = 2 * pz
        body.add_geom(name="constr_column", type=mujoco.mjtGeom.mjGEOM_BOX, size=[cx, cy, self.col_h / 2],
                      pos=[0, 0, z0 + self.col_h / 2], material="constr_mat", friction=list(FRICTION["metal"]),
                      condim=3)
        top = z0 + self.col_h
        # bolt: hinge (thread rotation) + slide (axial advance) coupled by a joint equality (lead = pitch / 2 pi);
        # qpos = 0 is the SEATED pose; the hinge lower limit is the seating stop (slide limit = 1 mm backstop)
        bolt = body.add_body(name="bolt", pos=[0, 0, top])
        bolt.add_joint(name="bolt_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                       range=[-0.001, 0.02], limited=mujoco.mjtLimited.mjLIMITED_TRUE, frictionloss=0.5,
                       solref_limit=[0.004, 1.0])
        bolt.add_joint(name="bolt_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                       frictionloss=self.thread_fric, damping=self.thread_damp, armature=2e-4,
                       range=[0.0, 720.0 if spec.compiler.degree else 4 * np.pi],
                       limited=mujoco.mjtLimited.mjLIMITED_TRUE, solref_limit=[0.004, 1.0])
        # threaded shank (visual; it runs inside the column bore)
        bolt.add_geom(name="bolt_shank", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[self.shaft_r, 0.012, 0],
                      pos=[0, 0, -0.012], material="steel_mat", contype=0, conaffinity=0,
                      mass=self.bolt_mass * 0.5)
        bolt.add_geom(name="bolt_washer", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[self.knob_r + 0.002, WASHER_T / 2, 0], pos=[0, 0, WASHER_T / 2], material="steel_mat",
                      contype=0, conaffinity=0, mass=0.002)
        bolt.add_geom(name="bolt_neck", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[self.shaft_r, NECK / 2, 0],
                      pos=[0, 0, WASHER_T + NECK / 2], material="steel_mat", contype=0, conaffinity=0, mass=0.002)
        spec.add_mesh(name="knob_mesh", uservert=prism_mesh(self.knob_r, KNOB_H))
        bolt.add_geom(name="knob", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="knob_mesh",
                      pos=[0, 0, WASHER_T + NECK], material="knob_mat", friction=list(FRICTION["plastic"]),
                      condim=4, mass=self.bolt_mass * 0.5 - 0.004)
        # white index mark across the knob top (visual) so that the rotation is visible
        bolt.add_geom(name="knob_mark", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[self.knob_r * 0.85, 0.0018, 0.0004], pos=[0, 0, WASHER_T + NECK + KNOB_H + 0.0002],
                      rgba=[0.95, 0.95, 0.95, 1], contype=0, conaffinity=0, mass=0.0)
        bolt.add_site(name="knob_center", pos=[0, 0, WASHER_T + NECK + KNOB_H / 2], size=[0.003] * 3,
                      rgba=[1, 0, 0, 0])
        lead = self.pitch / (2 * np.pi)
        spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="bolt_slide", name2="bolt_hinge",
                          data=[0.0, lead, 0, 0, 0, 0, 0, 0, 0, 0, 0], solref=[0.004, 1.0],
                          solimp=[0.9999, 0.9999, 0.001, 0.5, 2.0])

    # ----------------------------------------------------------------- reset
    def reset(self) -> None:
        # pre-episode: unscrew the bolt by turns0 (consistent slide/hinge pair), then the base reset
        self.data.qpos[self.q_hinge] = self.h0
        self.data.qpos[self.q_slide] = self.turns0 * self.pitch
        super().reset()
        self._r = None
        self._max_knob_force = 0.0
        self._ff, self._ff_max, self._ff_t = {}, {}, None

    # ------------------------------------------------------------------ state
    def _forces(self) -> dict:
        if self._contacts.get("t") != self.data.time:
            self._contacts = self.cm.measure(self.data)
            self._contacts["t"] = self.data.time
        return self._contacts

    def hinge(self) -> float:
        """Measured thread angle left to the seat [rad] (hinge encoder)."""
        return float(self.data.qpos[self.q_hinge])

    def slide(self) -> float:
        """Measured axial height above the seat [m] (slide encoder)."""
        return float(self.data.qpos[self.q_slide])

    def seated(self) -> bool:
        return bool(self.slide() < SEAT_SLIDE and self.hinge() < SEAT_ANGLE)

    def knob_center(self) -> np.ndarray:
        return self.data.site_xpos[self.sid_knob].copy()

    def heading(self) -> float:
        """World yaw of the measured finger direction (z_h is horizontal with the palm down)."""
        z = self.ha.grasp_mat()[:, 2]
        return float(np.arctan2(z[1], z[0]))

    def pinch_center(self) -> np.ndarray:
        return self.ha.grasp_pos() + self.ha.grasp_mat() @ self.c_pinch

    def _grasp_for(self, pinch_target, psi: float) -> np.ndarray:
        return np.asarray(pinch_target, float) - turn_frame(psi) @ self.c_pinch

    def _progress(self) -> int:
        """Remaining-rotation index r (45 deg units, +-5 deg hysteresis) from the hinge encoder; 0 when seated."""
        h = self.hinge()
        if self.seated():
            self._r = 0
            return 0
        r_new = max(1, int(np.ceil(h / UNIT - 1e-9)))
        r = self._r
        if r is None or r == 0 or not ((r - 1) * UNIT - HYST < h <= r * UNIT + HYST):
            r = r_new
        self._r = min(r, R_MAX)
        return self._r

    def _pinched(self, f) -> bool:
        return bool(f["tip_th_knob"] > F_PINCH and f["tip_if_knob"] > F_PINCH)

    def observe(self) -> dict:
        obs = self.base_obs()
        f = self._forces()
        r = self._progress()
        kc = self.knob_center()
        est, _, near = self.est.estimate(kc, 0.0, obs["grasp"])
        pc = self.pinch_center()
        tips_z = min(p[2] for p in obs["tips"].values())
        obs.update({
            "knob_est": est, "knob_near": near, "knob_true": kc, "pinch_center": pc, "pc_rel": pc - kc,
            "heading": self.heading(), "hinge": self.hinge(), "slide": self.slide(), "seated": self.seated(),
            "r": r, "f_knob": f["hand_knob"] + f["arm_knob"],
            "f_th": f["tip_th_knob"], "f_if": f["tip_if_knob"], "pinched": self._pinched(f),
            "f_fingers_constr": f["fingers_constr"],
            "tips_clear": bool(tips_z > kc[2] + KNOB_H / 2 + TIPS_CLEAR),
            "s_pinch": float(obs["synergy"][1])})
        return obs

    def _bad_forces(self, f) -> dict:
        """Contact forces checked by ``bad`` (raw, this physics step)."""
        return {"knob": f["hand_knob"] + f["arm_knob"], "palm_constr": max(f["palm_constr"], f["arm_constr"]),
                "fingers_constr": f["fingers_constr"], "arm_table": f["arm_table"], "hand_table": f["hand_table"],
                "arm_hand": f["arm_hand"]}

    def labels(self) -> set:
        f = self._forces()
        self._progress()
        out = set()
        if self.seated():
            out.add("seated")
        fk = f["hand_knob"] + f["arm_knob"]
        self._max_knob_force = max(self._max_knob_force, fk)
        if fk < F_FREE:
            out.add("released")
        # safety forces are low-pass filtered (tau = 10 ms, like a force sensor's bandwidth) so that a
        # one-step impact spike of a light fingertip (e.g. a limp finger bouncing on the column) is not a
        # crash, while any sustained over-force (> ~30 ms) is
        t = float(self.data.time)
        if self._ff_t != t:
            a = min(1.0, self.model.opt.timestep / BAD_TAU)
            for k, v in self._bad_forces(f).items():
                self._ff[k] = self._ff.get(k, 0.0) + a * (v - self._ff.get(k, 0.0))
                self._ff_max[k] = max(self._ff_max.get(k, 0.0), self._ff[k])
            self._ff_t = t
        if any(self._ff.get(k, 0.0) > lim for k, lim in BAD_LIMITS.items()):
            out.add("bad")
        return out

    def task_success(self) -> bool:
        f = self._forces()
        kc = self.knob_center()
        tips_z = min(self.ha.tip_pos(fn)[2] for fn in self.ha.hand.fingers)
        return bool(self.seated() and self.h0 - self.hinge() >= 2 * np.pi
                    and f["hand_knob"] + f["arm_knob"] < F_FREE
                    and tips_z > kc[2] + KNOB_H / 2 + TIPS_CLEAR)

    def cell(self, obs: dict, labels: frozenset) -> str:
        if obs["seated"]:
            if obs["pinched"] or obs["f_knob"] >= F_FREE:
                return "seated_held"
            return "done" if obs["tips_clear"] else "seated_open"
        r = obs["r"]
        psi = obs["heading"]
        if obs["pinched"]:
            return f"turned_{r}" if psi <= self.psi_end + ANG_TOL else f"held_{r}"
        rel = obs["pc_rel"]
        dxy, dz = float(np.linalg.norm(rel[:2])), float(rel[2])
        is_open = obs["s_pinch"] < self.s_pre + 0.12 and obs["f_knob"] < F_FREE
        at_start = abs(wrap(psi - self.psi_start)) < ANG_TOL
        if is_open and at_start and dxy < 0.006 and abs(dz) < 0.006:
            return f"at_{r}"
        if is_open and at_start and dxy < 0.012 and 0.035 < dz < 0.08:
            return f"above_{r}"
        if np.linalg.norm(rel) < 0.035:
            return f"near_{r}"
        return f"free_{r}"

    # ----------------------------------------------------------------- skills
    def _build_skills(self):
        S_PRE, S_CLOSE = self.s_pre, self.s_close

        def pinch_cmd(pc, psi, s):
            return self.cmd(self._grasp_for(pc, psi), turn_frame(psi), (0.0, s, 0.0))

        def near_pc(obs, target, tol):
            return np.linalg.norm(obs["pinch_center"] - target) < tol

        # 1 reach: pre-shape the pinch and hover 5 cm above the knob at the start heading
        def t_reach(obs, st):
            return pinch_cmd(obs["knob_est"] + [0, 0, H_ABOVE], self.psi_start, S_PRE)

        def o_reach(obs, labels, st):
            ok = (near_pc(obs, obs["knob_est"] + [0, 0, H_ABOVE], 0.006)
                  and abs(wrap(obs["heading"] - self.psi_start)) < np.radians(4)
                  and abs(obs["s_pinch"] - S_PRE) < 0.2 and np.linalg.norm(obs["dq"]) < 0.15)
            return OK if ok else None

        # 2 descend: pinch centre onto the knob centre (open pre-shape around the knob)
        def bumped(obs, st):
            if obs["f_knob"] > 1.0 or obs["f_fingers_constr"] > 1.0:
                st.memory["bump"] = st.memory.get("bump") or obs["t"]
                return obs["t"] - st.memory["bump"] > 0.1
            st.memory["bump"] = None
            return False

        def t_desc(obs, st):
            return pinch_cmd(obs["knob_est"], self.psi_start, S_PRE)

        def o_desc(obs, labels, st):
            if bumped(obs, st):
                return "bumped"
            ok = (near_pc(obs, obs["knob_est"], 0.003) and np.linalg.norm(obs["dq"]) < 0.1
                  and st.elapsed(obs["t"]) > 0.3)
            return OK if ok else None

        # 3 pinch: close thumb and index on the knob (grasp decided by MEASURED pad forces)
        def e_pinch(obs, st):
            st.memory["psi"] = obs["heading"]

        def t_pinch(obs, st):
            return pinch_cmd(obs["knob_est"], st.memory["psi"], S_CLOSE)

        def o_pinch(obs, labels, st):
            if obs["pinched"]:
                st.memory["ok"] = st.memory.get("ok") or obs["t"]
                if obs["t"] - st.memory["ok"] > 0.25 and st.elapsed(obs["t"]) > 0.5:
                    return OK
                return None
            st.memory["ok"] = None
            if obs["s_pinch"] > 0.95 or st.elapsed(obs["t"]) > 2.5:
                return "missed"
            return None

        # 4 turn: rotate the hand about the knob axis (rate-limited heading reference, see carrot)
        def e_turn(obs, st):
            st.memory["h0"] = obs["hinge"]
            st.memory["hold"] = (obs["grasp"].copy(), obs["grasp_mat"].copy())

        def carrot(obs, st, psi_goal, rate, lead):
            """Heading reference moving toward psi_goal at ``rate`` [rad/s] and kept within ``lead`` of the
            MEASURED heading; the commanded position keeps the pinch centre on the (estimated) knob axis for
            the heading psi_meas + alpha (psi_ref - psi_meas)."""
            psi = obs["heading"]
            ref = st.memory.get("psi_ref", psi)
            ref += float(np.clip(psi_goal - ref, -rate * self.control_dt, rate * self.control_dt))
            ref = float(np.clip(ref, psi - lead, psi + lead))
            st.memory["psi_ref"] = ref
            return self._grasp_for(obs["knob_est"], psi + self.alpha * (ref - psi)), turn_frame(ref)

        def t_turn(obs, st):
            if obs["seated"]:
                g, R = st.memory["hold"]
            else:
                g, R = carrot(obs, st, self.psi_end, self.w_turn, self.lead)
                st.memory["hold"] = (obs["grasp"].copy(), obs["grasp_mat"].copy())
            return self.cmd(g, R, (0.0, S_CLOSE, 0.0))

        def o_turn(obs, labels, st):
            if obs["seated"]:
                st.memory["seat"] = st.memory.get("seat") or obs["t"]
                return OK if obs["t"] - st.memory["seat"] > 0.1 else None
            st.memory["seat"] = None
            if obs["hinge"] - st.memory["h0"] > BACKSPIN:     # the knob is being rolled back: stop the turn
                return "slipped"
            if not obs["pinched"]:
                st.memory["lost"] = st.memory.get("lost") or obs["t"]
                if obs["t"] - st.memory["lost"] > 0.15:
                    return "slipped"
            else:
                st.memory["lost"] = None
            if obs["heading"] <= self.psi_end + np.radians(4):
                # end of the turn: let the arm settle on the final pose before the grasp is released
                st.memory["end"] = st.memory.get("end") or obs["t"]
                if obs["t"] - st.memory["end"] > 0.2 and np.linalg.norm(obs["dq"]) < 0.1:
                    return OK if st.memory["h0"] - obs["hinge"] >= TURN_OK else "slipped"
            return None

        # 5 release: open to the pre-shape where the hand is
        def e_rel(obs, st):
            st.memory["g"] = obs["grasp"].copy()
            st.memory["R"] = obs["grasp_mat"].copy()

        def t_rel(obs, st):
            return self.cmd(st.memory["g"], st.memory["R"], (0.0, S_PRE, 0.0))

        def o_rel(obs, labels, st):
            if obs["f_knob"] < F_FREE and obs["s_pinch"] < S_PRE + 0.12:
                st.memory["free"] = st.memory.get("free") or obs["t"]
                return OK if obs["t"] - st.memory["free"] > 0.15 else None
            st.memory["free"] = None
            return None

        # 6 unwind: rotate the OPEN hand back about the knob axis to the start heading (carrot)
        def t_unwind(obs, st):
            if "hold" not in st.memory:
                st.memory["hold"] = (obs["grasp"].copy(), obs["grasp_mat"].copy())
            if obs["s_pinch"] > S_PRE + 0.12 or obs["f_knob"] >= F_FREE and not st.memory.get("opened"):
                g, R = st.memory["hold"]          # (after a slip) open where the hand is before turning back
                return self.cmd(g, R, (0.0, S_PRE, 0.0))
            st.memory["opened"] = True
            g, R = carrot(obs, st, self.psi_start, self.w_unwind, self.lead_unwind)
            return self.cmd(g, R, (0.0, S_PRE, 0.0))

        def o_unwind(obs, labels, st):
            if bumped(obs, st):
                return "bumped"
            ok = (abs(wrap(obs["heading"] - self.psi_start)) < np.radians(4) and near_pc(obs, obs["knob_est"], 0.004)
                  and obs["s_pinch"] < S_PRE + 0.12 and np.linalg.norm(obs["dq"]) < 0.15)
            return OK if ok else None

        # 7 retreat: open fully and lift above the knob
        def e_ret(obs, st):
            st.memory["psi"] = obs["heading"]

        def t_ret(obs, st):
            return pinch_cmd(obs["knob_est"] + [0, 0, 0.09], st.memory["psi"], 0.0)

        def o_ret(obs, labels, st):
            return OK if obs["tips_clear"] and obs["f_knob"] < F_FREE else None

        kn = self.knob_center_nominal()
        mid = 0.5 * (self.psi_start + self.psi_end)
        nominal = {
            "reach_above": pinch_cmd(kn + [0, 0, H_ABOVE], self.psi_start, S_PRE),
            "descend": pinch_cmd(kn, self.psi_start, S_PRE),
            "pinch": pinch_cmd(kn, self.psi_start, S_CLOSE),
            "turn": pinch_cmd(kn, self.psi_end, S_CLOSE),
            "release": pinch_cmd(kn, self.psi_end, S_PRE),
            "unwind": pinch_cmd(kn, self.psi_start, S_PRE),
            "retreat": pinch_cmd(kn + [0, 0, 0.09], mid, 0.0),
        }
        home = pinch_cmd(kn + [0, 0, H_READY], self.psi_start, 0.0)
        # trust radius: largest distance from any predecessor's nominal goal (cycles) + randomization margin
        preds = {"reach_above": [home], "descend": [nominal["reach_above"]],
                 "pinch": [nominal["descend"], nominal["unwind"]], "turn": [nominal["pinch"]],
                 "release": [nominal["turn"], pinch_cmd(kn, mid, S_CLOSE)],
                 "unwind": [nominal["release"]],
                 "retreat": [nominal["release"], pinch_cmd(kn, self.psi_start, S_PRE)]}
        deltas = {k: max(self.space.norm(nominal[k] - p) for p in preds[k]) + 0.30 for k in nominal}
        spec = [
            ("reach_above", t_reach, o_reach, None, 10.0, 2.5),
            ("descend", t_desc, o_desc, None, 6.0, 1.5),
            ("pinch", t_pinch, o_pinch, e_pinch, 4.0, 1.2),
            ("turn", t_turn, o_turn, e_turn, 12.0, 5.0),
            ("release", t_rel, o_rel, e_rel, 3.0, 0.8),
            ("unwind", t_unwind, o_unwind, None, 12.0, 5.0),
            ("retreat", t_ret, o_ret, e_ret, 6.0, 1.5),
        ]
        self._skills = {name: Skill(name, nominal[name], deltas[name], tgt, out, timeout=to, nominal_duration=dur,
                                    on_enter=ent) for name, tgt, out, ent, to, dur in spec}

    def skills(self) -> dict:
        return self._skills

    def nominal_plan(self) -> list:
        plan = ["reach_above", "descend"]
        for k in range(self.n_cycles):
            plan += ["pinch", "turn", "release"] + (["unwind"] if k < self.n_cycles - 1 else [])
        return plan + ["retreat"]

    def skill_graph(self) -> list:
        rl = "released"
        sr = {"seated", "released"}
        pre = word(rl)                      # words before seating (the residual cannot change)
        specs = []
        for r in range(1, R_MAX + 1):
            lower = [q for q in (r - 1, r - 2, r - 3) if q >= 1]
            fail_r = [q for q in (r, r - 1, r - 2, r - 3) if q >= 1]
            specs += [
                SkillSpec("reach_above", f"free_{r}", (OutcomeSpec(OK, f"above_{r}", pre),)),
                SkillSpec("descend", f"above_{r}", (OutcomeSpec(OK, f"at_{r}", pre),
                                                    OutcomeSpec("bumped", f"near_{r}", word(rl, ""), failure=True))),
                # closing the pinch can jiggle the knob back across a progress boundary (budgeted failure)
                SkillSpec("pinch", f"at_{r}", (OutcomeSpec(OK, f"held_{r}", word(rl, "")),
                                               OutcomeSpec("missed", f"near_{r}", word(rl, ""), failure=True))
                          + ((OutcomeSpec("jiggled", f"held_{r + 1}", word(rl, ""), failure=True),)
                             if r < R_MAX else ())),
                SkillSpec("turn", f"held_{r}",
                          tuple(OutcomeSpec(OK, f"turned_{q}", word("")) for q in lower)
                          + ((OutcomeSpec(OK, "seated_held", word("", "seated")),) if r <= 3 else ())
                          + tuple(OutcomeSpec("slipped", f"turned_{q}", word(""), failure=True) for q in fail_r)
                          + tuple(OutcomeSpec("slipped", f"near_{q}", word("", rl), failure=True) for q in fail_r)),
                # the opening fingers may carry the knob a few degrees further (progress, never back)
                SkillSpec("release", f"turned_{r}", (OutcomeSpec(OK, f"near_{r}", word("", rl)),)
                          + ((OutcomeSpec(OK, f"near_{r - 1}", word("", rl)),) if r > 1 else ())
                          # a released knob can be knocked back (loosened) across a boundary: budgeted failure
                          + ((OutcomeSpec("knocked", f"near_{r + 1}", word("", rl), failure=True),)
                             if r < R_MAX else ())),
                SkillSpec("unwind", f"near_{r}", (OutcomeSpec(OK, f"at_{r}", pre),
                                                  OutcomeSpec("bumped", f"near_{r}", word(rl, ""), failure=True))
                          + (tuple(OutcomeSpec("knocked", f"{c}_{r + 1}", word(rl, ""), failure=True)
                                   for c in ("at", "near")) if r < R_MAX else ())),
            ]
        specs += [
            SkillSpec("release", "seated_held", (OutcomeSpec(OK, "seated_open", word("seated", sr)),)),
            SkillSpec("retreat", "seated_open", (OutcomeSpec(OK, "done", word(sr)),)),
        ]
        return specs

    # ------------------------------------------------------------- stress test
    def disturb(self) -> None:
        """(1) knock-back: once the hand has released the knob after its first turn, a short LOOSENING torque
        pulse about the bolt axis (0.06 s, sized from the thread friction and the bolt inertia for a 10-20 deg
        knock-back) - as if the peg were bumped; (2) a 0.3 s grip-force dropout (hand actuator force limits cut
        to 10-20 %) during the first turn that starts after >= 150 deg of rotation.  Triggered by measured
        state, identical for all methods."""
        d, t = self.data, float(self.data.time)
        b = self.bid_bolt
        d.xfrc_applied[b, 3:6] = 0.0
        P = self._pert
        f = self._forces()
        fk = f["hand_knob"] + f["arm_knob"]
        rot = self.h0 - self.hinge()
        if P["knock_back"] is None:
            if rot > np.radians(60) and fk < F_FREE and not self.seated():
                P["knock_back"] = {"t_trigger": t, "t_start": t + self._knock_delay,
                                   "target_deg": float(np.degrees(self._knock_angle)), "hinge0": self.hinge()}
        elif not P["knock_back"].get("done"):
            p = P["knock_back"]
            T = 0.06
            inertia = float(self.model.dof_armature[self.model.jnt_dofadr[self.j_hinge]]) + 2e-6
            # constant torque for T, then coasting against friction: theta = a T^2/2 (1 + a/a_f), a = (tau - f)/I
            a_f = self.thread_fric / inertia
            a = 0.5 * (-a_f + np.sqrt(a_f ** 2 + 8 * a_f * self._knock_angle / T ** 2))
            tau = self.thread_fric + inertia * a + self.thread_damp * a * T
            p["torque_Nm"], p["duration_s"] = float(tau), T
            if p["t_start"] <= t < p["t_start"] + T:
                d.xfrc_applied[b, 5] = tau            # +z = loosening
            elif t >= p["t_start"] + T + 0.3:
                p["done"] = True
                p["knock_back_deg"] = float(np.degrees(self.hinge() - p["hinge0"]))
        lim = self.model.actuator_forcerange
        if P["grip_dropout"] is None:
            if rot > np.radians(150) and self._pinched(f) and not self.seated():
                P["grip_dropout"] = {"t_trigger": t, "t_start": t + self._drop_delay, "duration_s": 0.3,
                                     "force_fraction": float(self._drop_level), "hinge0": self.hinge()}
        drop = P["grip_dropout"]
        if drop is not None and drop["t_start"] <= t < drop["t_start"] + 0.3:
            lim[self.ha.act_hand] = self._hand_range * self._drop_level
        else:
            lim[self.ha.act_hand] = self._hand_range
            if drop is not None and t >= drop["t_start"] + 0.3 and "pinched_after" not in drop:
                drop["pinched_after"] = self._pinched(f)

    def perturbation_info(self) -> dict:
        return self._pert

    # ---------------------------------------------------------- diagnostics
    def step_diagnostics(self) -> dict:
        f = self._forces()
        return {"knob_force_N": f["hand_knob"] + f["arm_knob"], "fingers_constr_force_N": f["fingers_constr"],
                "palm_constr_force_N": f["palm_constr"], "arm_table_force_N": f["arm_table"],
                "arm_hand_self_contact_N": f["arm_hand"],
                "thread_violation_m": abs(self.slide() - self.pitch / (2 * np.pi) * self.hinge())}

    def log_signals(self) -> dict:
        f = self._forces()
        return {"hinge": self.hinge(), "slide": self.slide(), "heading": self.heading(),
                "knob_force": f["hand_knob"] + f["arm_knob"], "f_th": f["tip_th_knob"], "f_if": f["tip_if_knob"],
                "grasp": self.ha.grasp_pos(), "synergy": self.ha.synergy()}

    def episode_metrics(self) -> dict:
        return {"rotation_total_deg": float(np.degrees(self.h0 - self.hinge())),
                "remaining_deg": float(np.degrees(self.hinge())), "final_slide_mm": 1000 * self.slide(),
                "seated": self.seated(), "max_knob_force_N": float(self._max_knob_force),
                "max_filtered_bad_forces_N": dict(self._ff_max)}

    def variant_info(self) -> dict:
        return {"hand": self.hand_name, "construction_xy": self.cxy, "column_height_m": self.col_h,
                "pitch_m_per_turn": self.pitch, "initial_turns": self.turns0, "knob_radius_m": self.knob_r,
                "bolt_mass_kg": self.bolt_mass, "thread_friction_Nm": self.thread_fric,
                "thread_damping_Nms": self.thread_damp, "nominal_cycles": self.n_cycles,
                "perception_bias_far_m": self.est.b_far, "perception_bias_near_m": self.est.b_near}

    def overlay_lines(self) -> list:
        f = self._forces()
        return [f"knob turned {np.degrees(self.h0 - self.hinge()):6.1f} deg   left to seat "
                f"{np.degrees(self.hinge()):6.1f} deg   above seat {1000 * self.slide():4.2f} mm   "
                f"seated: {self.seated()}   pinch th {f['tip_th_knob']:4.1f} N  if {f['tip_if_knob']:4.1f} N"]
