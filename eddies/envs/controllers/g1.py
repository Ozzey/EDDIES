"""Unitree G1 humanoid (29 DoF) with Dex3-1 hands: reusable whole-body controller.

Robot model: menagerie ``unitree_g1/g1_with_hands.xml`` (pinned commit), floating
base, 43 position actuators (12 legs, 3 waist, 2x7 arms, 2x7 Dex3-1 fingers),
attached with the prefix ``"g1/"`` by :func:`attach_g1`.  The robot stands on
its own feet under physics; nothing is fixed or welded.

Edits applied to the menagerie model (``edit_g1``; all physical, documented here):

* **Dex3-1 collision geometry = primitives** (:data:`HAND_COLLIDERS`): the
  menagerie convex-hull collision meshes of the palm/wrist and of the thumb_0,
  thumb_2, index_0/1 and middle_0/1 links are deleted and replaced by fitted
  boxes/capsules/one cylinder (thumb_1 already is a menagerie box).  Reason: the
  hulls produced wrong-depth contacts against the cylindrical peg (MuJoCo native
  CCD/EPA near the cylinder rim) -> 1.4-2.2 kN hand-peg spikes that threw the peg
  in early runs; with the primitives every peak is <= 0.92 kN and all
  remaining transients are explained by the finger dynamics (see ``power`` below).
  Inertias are unchanged (explicit menagerie ``<inertial>``; total mass 34.39 kg).
* **Dex3-1 finger/palm contacts are stiff skin pads** (DEVELOPER_GUIDE §2.4):
  every hand collision geom (palm/wrist primitives, thumb_0..2, index_0..1,
  middle_0..1) gets ``priority=2, condim=4, solref=(0.004, 1),
  solimp=(0.95, 0.99, 0.001, 0.5, 2)`` and silicone-skin friction
  ``FRICTION["skin"] = 0.9``.  Menagerie default: mu = 1.0, condim 3, 20 ms
  contact time constant (lets friction grasps creep).
* Sites added on each ``{side}_wrist_yaw_link`` (hand frame, see below):
  ``{side}_grasp`` (power-grasp centre, where a held 3 cm cylinder settles),
  ``{side}_palm`` (palm surface centre) and ``{side}_{thumb,index,middle}_tip``.
* Actuators are UNCHANGED: position servos kp = 500 Nm/rad, dampratio 1, torque
  limits from the menagerie ``actuatorfrcrange`` (Unitree spec values): hip
  pitch/yaw 88, hip roll/knee 139, ankle 50, waist yaw 88, waist roll/pitch 50,
  shoulder/elbow/wrist-roll 25, wrist pitch/yaw 5, Dex3 thumb_0 2.45, other
  finger joints 1.4 Nm.  These are JOINT-level limits (``jnt_actfrcrange``,
  MuJoCo clamps ``qfrc_actuator``); ``data.actuator_force`` reports the unclamped
  servo output and may read e.g. 30 Nm on a finger pressing an object.  Joint
  armature 0.01 kg m^2 and friction loss 0.3 Nm on every joint (menagerie).
  Feet: menagerie sole spheres (r 5 mm, mu 0.6).

Hand frame (``{side}_wrist_yaw_link``, all wrist angles 0 and elbow 0 => the hand
frame is parallel to the torso frame): +x = finger direction, +z = up (index
finger on top at z=+2.85 cm, middle finger below at z=-2.85 cm, thumb between
them at z=0), palm normal = +y for the RIGHT hand / -y for the LEFT hand (the
palm faces the body midline).  With the hand frame equal to the world frame the
right hand holds a vertical cylinder like a glass ("lateral power grasp").

Finger synergies (:data:`SYNERGIES`, joint order thumb_0, thumb_1, thumb_2,
index_0, index_1, middle_0, middle_1, RIGHT-hand signs; the left hand is the
mirror image, see :func:`synergy_posture`).  A synergy command is a closure
s in [0, 1]: q = preshape + s (closed - preshape), preshape = ``open`` unless
listed in :data:`PRESHAPE`.  Closed targets lie beyond contact, so a closed
finger presses with its joint torque limit (kp = 500 saturates within 3 mrad).

* ``open``  - fingers straight, thumb swung back; a 3 cm cylinder standing at
  ``grasp + PREGRASP_OFFSET`` (hand frame) is collision-free during a vertical
  approach;
* ``power`` - lateral power grasp: index + middle wrap, thumb opposes, object
  pressed into the palm.  3 cm peg: summed hand-peg normal force while holding
  240-300 N in the Streamer grasp pose, 400-480 N in a slightly different pose
  (shorter contact levers); in both cases every finger joint
  is quasi-static at its 1.4 Nm limit (constraint torque = actuator torque,
  qacc ~ 0), i.e. the squeeze is what the torque-limited fingers sustain through
  the grasp geometry (palm up to ~330 N); held peg drift <= 0.71 mm at
  0.3 m/s transport, axis change <= 1.04 deg (seeds 0-4).  Closure transient:
  the fingers close at up to 4 rad/s and carry the menagerie's 0.01 kg m^2
  reflected motor inertia per joint; stopping them on the rigid peg through the
  stiff pads gives a 20-40 ms impact of 0.45-0.92 kN summed normal force
  (finger-joint constraint torque up to ~9 Nm, deceleration ~300-650 rad/s^2).
  Slower closure (<= 2 rad/s) turned most grasps into misses, so 4 rad/s is kept;
* ``pinch`` - thumb pad against the index pad (thumb_0 pre-tilted, middle tucked;
  empty closure: tip sites 11 mm apart = pads pressing);
* ``hook``  - index + middle curled ~70/75 deg, thumb open (drawer/door handles);
* ``point`` - index straight, middle and thumb curled (buttons).

Whole-body balance (``G1WholeBody.inner_loop`` at the physics rate):

* legs: joint references from a numerical leg IK that places the pelvis at a
  commanded offset above the (fixed) feet with an upright pelvis (stance 0.77 m,
  slight knee bend away from the straight-knee singularity); waist held at 0;
* CoM regulation: the horizontal pelvis offset integrates the error between the
  MEASURED centre of mass and the centre of the support polygon (k = 1.5 1/s,
  limit +-7 cm), so arm motion/loads never drive the CoM toward the polygon edge
  (CoM margin 7-8.5 cm during all reaches);
* sway damping: ankle pitch/roll references get CoM-velocity feedback
  (0.6 rad per m/s).  The kp = 500 ankles alone leave the whole-body sway mode at
  zeta ~0.05; measured lateral push capability (right arm extended, 0.2 s push):
  no damping -> falls at 50 N, with damping -> rejects 60 N (12 N s), falls at 70 N;
* gravity feed-forward for the subtrees hanging from the pelvis (waist, arms,
  fingers): servo target offset tau_g / kp from ``qfrc_bias`` (counts against the
  torque limits; held objects are not compensated);
* arms/fingers: 7-DoF DLS IK of the ``{side}_grasp`` site (controllers/ik.py, 3
  iterations of <= 0.012 rad at the control rate) followed by a critically damped
  second-order reference filter with velocity saturation at the physics rate (arms
  1.5 rad/s, fingers 4 rad/s): an abrupt arm target step would otherwise knock the
  robot over.  Steady-state grasp-site error < 1 mm.

Balance monitor: :meth:`G1WholeBody.balance_state` returns pelvis height/tilt,
CoM, centre of pressure (from measured foot contact forces), the support
polygon and the CoM stability margin; ``fallen`` = pelvis below 0.55 m, pelvis
tilt above 30 deg, or more than 5 N between a non-foot robot geom and the floor.
"""
from __future__ import annotations

import mujoco
import numpy as np

from eddies.common.config import load

ROBOT = load("robots/g1")

from eddies.envs.controllers.ik import DLSIK, rotation_error_mat
from eddies.envs.framework.embedding import EmbeddingSpace, RotationChart
from eddies.envs.framework.physics import FRICTION
from eddies.envs.framework.scene import SceneBuilder, aid, bid, qadr, sid, subtree_geoms

G1_XML = "unitree_g1/g1_with_hands.xml"
PREFIX = "g1/"
SIDES = ("left", "right")
LEG_NAMES = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
LEG_JOINTS = {s: [f"{s}_{j}_joint" for j in LEG_NAMES] for s in SIDES}
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM_NAMES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll",
             "wrist_pitch", "wrist_yaw")
ARM_JOINTS = {s: [f"{s}_{j}_joint" for j in ARM_NAMES] for s in SIDES}
FINGER_NAMES = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
HAND_JOINTS = {s: [f"{s}_hand_{j}_joint" for j in FINGER_NAMES] for s in SIDES}
FINGER_LINKS = {"thumb": ("thumb_0", "thumb_1", "thumb_2"), "index": ("index_0", "index_1"),
                "middle": ("middle_0", "middle_1")}

# 'stand' keyframe of the menagerie model (pelvis height 0.79 m, straight legs)
STAND_ARM = {"left": np.array([0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0]),
             "right": np.array([0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0])}
# arm posture used as IK null-space reference while manipulating in front of the body
READY_ARM = {"left": np.array([-0.35, 0.25, 0.0, 0.55, 0.0, 0.0, 0.0]),
             "right": np.array([-0.35, -0.25, 0.0, 0.55, 0.0, 0.0, 0.0])}

# Dex3-1 synergy postures (RIGHT-hand signs, order FINGER_NAMES)
SYNERGIES = {
    "open": np.array([0.0, 0.70, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "power": np.array([0.0, -0.60, -1.00, 1.20, 1.60, 1.20, 1.60]),
    "pinch": np.array([-0.30, -0.80, -0.29, 1.10, 0.65, 1.50, 1.70]),   # thumb tilted to the index
    "hook": np.array([0.0, 0.70, 0.0, 1.20, 1.30, 1.20, 1.30]),
    "point": np.array([0.0, -0.90, -1.50, 0.0, 0.0, 1.50, 1.70]),
}
# per-synergy pre-shape at s = 0 (default: SYNERGIES["open"]); the pinch pre-tilts the
# thumb towards the index so the closing thumb does not sweep through the index finger
PRESHAPE = {"pinch": np.array([-0.30, 0.70, 0.0, 0.0, 0.0, 0.0, 0.0])}
# mirror signs (left = sign * right): rotations about the hand y-axis keep their sign
MIRROR = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

# hand-frame sites (RIGHT hand; y is mirrored for the left hand)
GRASP_SITE = np.array([0.1015, 0.0307, 0.0])      # where a power-grasped 3 cm cylinder settles
PREGRASP_OFFSET = np.array([0.0135, 0.0083, 0.0]) # object centre rel. to grasp site before closing
PALM_SITE = np.array([0.085, 0.018, 0.0])
TIP_SITES = {"thumb": ("thumb_2", np.array([-0.002, 0.050, 0.0])),
             "index": ("index_1", np.array([0.050, -0.002, 0.0])),
             "middle": ("middle_1", np.array([0.050, -0.002, 0.0]))}

PAD_SOLREF = (0.004, 1.0)
PAD_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)

# Primitive collision geometry of the Dex3-1 hand (RIGHT-hand body frames, metres; the
# left hand is the y-mirror, verified against the left meshes).  ``edit_g1`` DELETES the
# menagerie convex-hull collision meshes of the palm/wrist (``*_wrist_yaw_link``: wrist
# and palm meshes) and of the thumb_0, thumb_2, index_0/1 and middle_0/1 links and adds
# these primitives instead (thumb_1 already is a menagerie box and is kept).
#
# Why: MuJoCo's native CCD (EPA) mis-resolves mesh-hull vs cylinder contacts near the
# cylinder rim.  Sweep of the index_0 hull against a 3 cm cylinder (300 random poses x 46
# gaps): 1339 of 8484 contacts (16 %) reported a penetration up to 15 mm deeper than the
# true distance (normal flipped to the cylinder end face) -> 1-10 kN solver spikes on
# the peg.  Same sweep with capsules/boxes: 0 wrong contacts.  Episode evidence
# (g1_peg_insertion, seeds 0-4): hand-peg force peaks with the hulls up to 2.2 kN and
# failures after a thrown/tilted peg; with these primitives every peak is <= 0.92 kN (finger-impact
# transients, see ``power`` in the module docstring) and all those runs succeed.
#
# Fit to the menagerie collision-mesh vertices (both hands): finger links - < 2.6 % of
# the vertices more than 1 mm outside the primitives (p95 <= 0.8 mm), primitive surface
# beyond the mesh hull p95 <= 2.9 mm (rounded tips/edges); palm - palmar face matched
# within ~1 mm (y = +16.0..+16.5 mm proximal, +15.4 mm distal), 2.5 % of the vertices
# > 1 mm outside (box corners vs rounded dorsal edges, a thin lip at the finger bases).
# The finger capsules give the rounded silicone pads of the real hand.
#   kind "capsule": (p0, p1, radius);  "box": (centre, half sizes);
#   "cylinder": (centre, radius, half length along the body z axis)
_DISTAL = ((0.0, 0.0, 0.0042), (0.0430, 0.0, 0.0042), (0.0, 0.0, -0.0042), (0.0430, 0.0, -0.0042))
HAND_COLLIDERS = {
    "wrist_yaw_link": [            # wrist motor (cylinder about the yaw axis) + flange + palm
        ("wrist", "cylinder", ((0.0, 0.0, 0.0), 0.0180, 0.0305)),
        ("wrist_flange", "box", ((0.0310, -0.0030, 0.0), (0.0105, 0.0270, 0.0265))),
        ("palm_a", "box", ((0.0498, -0.0035, 0.0), (0.0083, 0.0195, 0.0300))),
        ("palm_b", "box", ((0.0665, -0.0040, 0.0), (0.0085, 0.0200, 0.0350))),
        ("palm_c", "box", ((0.0840, -0.0035, 0.0), (0.0090, 0.0205, 0.0410))),
        ("palm_d", "box", ((0.1045, -0.0045, 0.0), (0.0115, 0.0199, 0.0435))),
        ("palm_front", "box", ((0.1215, -0.0105, 0.0), (0.0055, 0.0115, 0.0200))),
        ("knuckles", "box", ((0.1220, -0.0150, 0.0), (0.0055, 0.0065, 0.0380))),
    ],
    "thumb_0_link": [("thumb_0", "box", ((-0.0007, 0.0142, 0.0), (0.0108, 0.0142, 0.0095)))],
    # distal links: rounded-rectangle pad (two capsules), thumb_2 points along +y
    "thumb_2_link": [("thumb_2a", "capsule", ((0.0, 0.0, 0.0042), (0.0, 0.0430, 0.0042), 0.0088)),
                     ("thumb_2b", "capsule", ((0.0, 0.0, -0.0042), (0.0, 0.0430, -0.0042), 0.0088))],
    # proximal links: joint housing (capsule along the joint axis z) + phalanx box
    "index_0_link": [("index_0_joint", "capsule", ((0.0, 0.0, -0.0059), (0.0, 0.0, 0.0059), 0.0070)),
                     ("index_0", "box", ((0.0320, 0.0010, 0.0), (0.0180, 0.0136, 0.0129)))],
    "index_1_link": [("index_1a", "capsule", (_DISTAL[0], _DISTAL[1], 0.0088)),
                     ("index_1b", "capsule", (_DISTAL[2], _DISTAL[3], 0.0088))],
    "middle_0_link": [("middle_0_joint", "capsule", ((0.0, 0.0, -0.0059), (0.0, 0.0, 0.0059), 0.0070)),
                      ("middle_0", "box", ((0.0320, 0.0010, 0.0), (0.0180, 0.0136, 0.0129)))],
    "middle_1_link": [("middle_1a", "capsule", (_DISTAL[0], _DISTAL[1], 0.0088)),
                      ("middle_1b", "capsule", (_DISTAL[2], _DISTAL[3], 0.0088))],
}

PELVIS_MIN_HEIGHT = 0.55       # fallen below this pelvis height [m]
PELVIS_MAX_TILT = np.radians(30.0)


def synergy_posture(side: str, kind: str, s: float) -> np.ndarray:
    """Finger joint targets (order FINGER_NAMES) for closure ``s`` of synergy ``kind``."""
    s = float(np.clip(s, 0.0, 1.0))
    o = PRESHAPE.get(kind, SYNERGIES["open"])
    q = o + s * (SYNERGIES[kind] - o)
    return q if side == "right" else MIRROR * q


def mirror_y(v, side: str) -> np.ndarray:
    v = np.asarray(v, float).copy()
    if side == "left":
        v[1] = -v[1]
    return v


def _hand_body_name(side: str, body: str) -> str:
    return f"{side}_{body}" if body == "wrist_yaw_link" else f"{side}_hand_{body}"


def _add_hand_colliders(spec: mujoco.MjSpec) -> None:
    """Replace the Dex3 convex-hull collision meshes by :data:`HAND_COLLIDERS`."""
    G = mujoco.mjtGeom
    for side in SIDES:
        f = np.array([1.0, 1.0 if side == "right" else -1.0, 1.0])
        for body, prims in HAND_COLLIDERS.items():
            b = spec.body(_hand_body_name(side, body))
            for g in list(b.geoms):
                if g.type == G.mjGEOM_MESH and (g.contype or g.conaffinity):
                    spec.delete(g)
            for name, kind, a in prims:
                kw = dict(name=f"{side}_{name}_col", group=3, rgba=[0.8, 0.3, 0.3, 0.5])
                if kind == "capsule":
                    b.add_geom(type=G.mjGEOM_CAPSULE, fromto=np.r_[f * a[0], f * a[1]], size=[a[2], 0, 0], **kw)
                elif kind == "box":
                    b.add_geom(type=G.mjGEOM_BOX, pos=f * np.asarray(a[0]), size=list(a[1]), **kw)
                elif kind == "cylinder":
                    b.add_geom(type=G.mjGEOM_CYLINDER, pos=f * np.asarray(a[0]), size=[a[1], a[2], 0], **kw)
                else:
                    raise ValueError(kind)


def edit_g1(spec: mujoco.MjSpec) -> None:
    """Menagerie G1 edits: primitive hand colliders (replacing the convex-hull meshes),
    stiff skin finger/palm pads, hand sites; actuators and inertias unchanged."""
    _add_hand_colliders(spec)
    hand_bodies = [f"{s}_hand_{f}_link" for s in SIDES for f in FINGER_NAMES] + \
                  [f"{s}_wrist_yaw_link" for s in SIDES]
    for name in hand_bodies:
        for g in spec.body(name).geoms:
            if g.contype or g.conaffinity:
                g.friction = np.array(FRICTION["skin"])
                g.condim = 4
                g.priority = 2
                g.solref = np.array(PAD_SOLREF)
                g.solimp = np.array(PAD_SOLIMP)
    for side in SIDES:
        wy = spec.body(f"{side}_wrist_yaw_link")
        wy.add_site(name=f"{side}_grasp", pos=mirror_y(GRASP_SITE, side), size=[0.006] * 3,
                    rgba=[1, 0.2, 0.2, 0.0])
        wy.add_site(name=f"{side}_palm", pos=mirror_y(PALM_SITE, side), size=[0.006] * 3,
                    rgba=[0.2, 1, 0.2, 0.0])
        for tip, (link, p) in TIP_SITES.items():
            spec.body(f"{side}_hand_{link}_link").add_site(
                name=f"{side}_{tip}_tip", pos=mirror_y(p, side), size=[0.005] * 3, rgba=[0.2, 0.2, 1, 0.0])


def attach_g1(sb: SceneBuilder, pos=(0.0, 0.0, 0.0), yaw: float = 0.0, prefix: str = PREFIX) -> None:
    """Attach the edited G1 so that its feet stand on the floor at ``pos`` (z = floor)."""
    q = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    sb.attach_robot(G1_XML, prefix, pos=pos, quat=q, edit=edit_g1)


def g1_space(x_range=None, y_range=None, z_range=None,
             synergy_weight: float | None = None, extra: list | None = None) -> EmbeddingSpace:
    """7-D (+extra) hand embedding: grasp-site position (m, world), rotation-vector chart
    of the hand frame about the "glass" orientation (rad), synergy closure (1).

    ``extra``: optional list of (name, lower, upper, weight, unit) appended dims.
    The three position spans default to 0.8 m each (0.4 m per normalized unit), so a
    metric step of 0.0075 per 50 Hz control step is a 0.15 m/s hand speed.
    """
    EMB = ROBOT.embedding
    x_range = tuple(EMB.ranges.x) if x_range is None else x_range
    y_range = tuple(EMB.ranges.y) if y_range is None else y_range
    z_range = tuple(EMB.ranges.z) if z_range is None else z_range
    synergy_weight = float(EMB.weights[6]) if synergy_weight is None else synergy_weight
    names = list(EMB.names)
    lo = [x_range[0], y_range[0], z_range[0], -np.pi, -np.pi, -np.pi, 0.0]
    hi = [x_range[1], y_range[1], z_range[1], np.pi, np.pi, np.pi, 1.0]
    w = [float(x) for x in EMB.weights[:6]] + [synergy_weight]
    nlo = [-1, -1, -1, -1, -1, -1, 0]
    nhi = [1, 1, 1, 1, 1, 1, 1]
    units = ["m", "m", "m", "rad", "rad", "rad", "1"]
    for (n, a, b, wt, u) in (extra or []):
        names.append(n), lo.append(a), hi.append(b), w.append(wt), nlo.append(-1), nhi.append(1), units.append(u)
    return EmbeddingSpace(names=names, phys_lower=lo, phys_upper=hi, weights=w,
                          norm_lower=nlo, norm_upper=nhi, units=units)


def _hull(points: np.ndarray) -> np.ndarray:
    """2-D convex hull (Andrew monotone chain), counter-clockwise."""
    pts = sorted(map(tuple, np.round(points, 6)))
    if len(pts) <= 2:
        return np.asarray(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def polygon_margin(poly: np.ndarray, p) -> float:
    """Signed distance from p to the boundary of a CCW convex polygon (>0 inside)."""
    p = np.asarray(p, float)[:2]
    n = len(poly)
    if n < 3:
        return -float(np.min(np.linalg.norm(poly - p, axis=1))) if n else -np.inf
    d = np.inf
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        nrm = np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-12)   # inward normal (CCW)
        d = min(d, float(nrm @ (p - a)))
    return d


class JointSmoother:
    """Critically damped 2nd-order reference filter with velocity saturation."""

    def __init__(self, n: int, omega: float = 25.0, v_max: float = 1.5):
        self.q = np.zeros(n)
        self.v = np.zeros(n)
        self.omega, self.v_max = float(omega), np.broadcast_to(np.asarray(v_max, float), (n,)).copy()

    def reset(self, q) -> None:
        self.q = np.asarray(q, float).copy()
        self.v[:] = 0.0

    def step(self, goal, dt: float) -> np.ndarray:
        w = self.omega
        a = w * w * (np.asarray(goal, float) - self.q) - 2.0 * w * self.v
        self.v = np.clip(self.v + a * dt, -self.v_max, self.v_max)
        self.q = self.q + self.v * dt
        return self.q


class LegIK:
    """Numerical leg IK on a kinematic shadow: pelvis pose given, feet fixed in the world."""

    def __init__(self, model: mujoco.MjModel, prefix: str = PREFIX):
        self.m = model
        self.d = mujoco.MjData(model)
        self.base_qadr = qadr(model, f"{prefix}floating_base_joint")
        self.legs = {}
        for s in SIDES:
            joints = [f"{prefix}{j}" for j in LEG_JOINTS[s]]
            ids = [model.joint(j).id for j in joints]
            self.legs[s] = {"qadr": np.array([model.jnt_qposadr[i] for i in ids]),
                            "dadr": np.array([model.jnt_dofadr[i] for i in ids]),
                            "lo": model.jnt_range[ids, 0] + 0.01, "hi": model.jnt_range[ids, 1] - 0.01,
                            "foot": bid(model, f"{prefix}{s}_ankle_roll_link")}
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def foot_poses(self, qpos) -> dict:
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        return {s: (self.d.xpos[L["foot"]].copy(), self.d.xmat[L["foot"]].reshape(3, 3).copy())
                for s, L in self.legs.items()}

    def solve(self, qpos_template, pelvis_pos, pelvis_quat, feet: dict, q_init: dict,
              iterations: int = 4, damping: float = 1e-3) -> dict:
        d = self.d
        d.qpos[:] = qpos_template
        d.qpos[self.base_qadr:self.base_qadr + 3] = pelvis_pos
        d.qpos[self.base_qadr + 3:self.base_qadr + 7] = pelvis_quat
        out = {}
        for s, L in self.legs.items():
            d.qpos[L["qadr"]] = q_init[s]
        for _ in range(iterations):
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            for s, L in self.legs.items():
                p_t, R_t = feet[s]
                err = np.concatenate([p_t - d.xpos[L["foot"]],
                                      rotation_error_mat(d.xmat[L["foot"]].reshape(3, 3), R_t)])
                mujoco.mj_jacBody(self.m, d, self.jacp, self.jacr, L["foot"])
                J = np.vstack([self.jacp[:, L["dadr"]], self.jacr[:, L["dadr"]]])
                dq = np.linalg.solve(J.T @ J + damping * np.eye(6), J.T @ err)
                d.qpos[L["qadr"]] = np.clip(d.qpos[L["qadr"]] + dq, L["lo"], L["hi"])
        for s, L in self.legs.items():
            out[s] = d.qpos[L["qadr"]].copy()
        return out


class G1WholeBody:
    """Runtime handle: state, decoders and balance for an attached G1 (prefix ``g1/``).

    Typical use inside a task (see ``tasks/g1_base.py``)::

        g1 = G1WholeBody(model, data)
        g1.reset_standing()                       # pre-episode only (joint writes)
        g1.command_hand("right", pos, mat)        # control rate (IK)
        g1.command_synergy("right", s, "power")   # control rate
        g1.inner_loop()                           # every physics step (smoothing + balance)
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, prefix: str = PREFIX, *,
                 stance_height: float = 0.77, com_offset=(0.0, 0.0), arm_v_max: float = 1.5,
                 finger_v_max: float = 4.0, active_sides=("right",)):
        self.model, self.data, self.p = model, data, prefix
        m = model
        self.dt = float(m.opt.timestep)
        self.base_qadr = qadr(m, f"{prefix}floating_base_joint")
        self.base_dadr = int(m.jnt_dofadr[m.joint(f"{prefix}floating_base_joint").id])
        self.pelvis = bid(m, f"{prefix}pelvis")
        self.torso = bid(m, f"{prefix}torso_link")

        def jinfo(names):
            ids = [m.joint(prefix + n).id for n in names]
            return (np.array([m.jnt_qposadr[i] for i in ids]), np.array([m.jnt_dofadr[i] for i in ids]),
                    np.array([aid(m, prefix + n) for n in names]))
        self.leg = {s: jinfo(LEG_JOINTS[s]) for s in SIDES}
        self.waist = jinfo(WAIST_JOINTS)
        self.arm = {s: jinfo(ARM_JOINTS[s]) for s in SIDES}
        self.hand = {s: jinfo(HAND_JOINTS[s]) for s in SIDES}
        self.grasp_site = {s: sid(m, f"{prefix}{s}_grasp") for s in SIDES}
        self.palm_site = {s: sid(m, f"{prefix}{s}_palm") for s in SIDES}
        self.tip_site = {s: {t: sid(m, f"{prefix}{s}_{t}_tip") for t in TIP_SITES} for s in SIDES}
        self.hand_body = {s: bid(m, f"{prefix}{s}_wrist_yaw_link") for s in SIDES}
        self.active_sides = tuple(active_sides)
        self.ik = {s: DLSIK(m, [prefix + j for j in ARM_JOINTS[s]], f"{prefix}{s}_grasp",
                            posture=READY_ARM[s], posture_gain=0.08, rot_weight=0.6,
                            damping=0.03, max_step=0.012, iterations=3, limit_margin=0.03)
                   for s in SIDES}
        self.leg_ik = LegIK(m, prefix)
        self.chart = RotationChart([1.0, 0.0, 0.0, 0.0])
        # geoms
        self.robot_geoms = [g for g in subtree_geoms(m, f"{prefix}pelvis")
                            if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.hand_geoms = {}
        for s in SIDES:
            parts = {"palm": [g for g in subtree_geoms(m, f"{prefix}{s}_wrist_yaw_link")
                              if m.geom_bodyid[g] == self.hand_body[s]]}
            for f, links in FINGER_LINKS.items():
                parts[f] = [g for ln in links for g in subtree_geoms(m, f"{prefix}{s}_hand_{ln}_link")
                            if m.geom_bodyid[g] == m.body(f"{prefix}{s}_hand_{ln}_link").id]
            parts = {k: [g for g in v if m.geom_contype[g] or m.geom_conaffinity[g]] for k, v in parts.items()}
            parts["all"] = sorted(set(sum(parts.values(), [])))
            self.hand_geoms[s] = parts
        feet_bodies = {bid(m, f"{prefix}{s}_ankle_roll_link") for s in SIDES}
        self.foot_geoms = [g for g in self.robot_geoms if m.geom_bodyid[g] in feet_bodies
                           and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        self._foot_mask = np.zeros(m.ngeom, bool)
        self._foot_mask[self.foot_geoms] = True
        self._robot_mask = np.zeros(m.ngeom, bool)
        self._robot_mask[self.robot_geoms] = True
        self.floor_geom = m.geom("floor").id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0 else -1
        # controller state
        self.stance_height = float(stance_height)
        self.com_offset = np.asarray(com_offset, float)
        self.arm_smooth = {s: JointSmoother(7, omega=25.0, v_max=arm_v_max) for s in SIDES}
        self.hand_smooth = {s: JointSmoother(7, omega=40.0, v_max=finger_v_max) for s in SIDES}
        self.leg_smooth = JointSmoother(12, omega=30.0, v_max=1.0)
        self.arm_goal = {s: STAND_ARM[s].copy() for s in SIDES}
        self.hand_goal = {s: synergy_posture(s, "open", 0.0) for s in SIDES}
        self.synergy = {s: ("power", 0.0) for s in SIDES}
        self.hand_target = {s: None for s in SIDES}
        self.pelvis_offset = np.zeros(2)          # CoM regulator output (pelvis xy rel. stance)
        self.pelvis_offset_limit = 0.07
        self.k_com = 1.5                          # CoM integral gain [1/s]
        self.k_sway = 0.6                         # ankle rad per (m/s) CoM velocity (lateral push
                                                  # probe: 0 -> falls at 50 N x 0.2 s; 0.6 -> rejects 60 N)
        self.balance_enabled = True
        self.feet = None
        self.stance_xy = np.zeros(2)
        self.q_leg_goal = None
        self._jac_com = np.zeros((3, m.nv))
        self._f6 = np.zeros(6)
        self._leg_counter = 0
        self.leg_ik_decimation = max(1, int(round(0.01 / self.dt)))   # 100 Hz leg IK
        self._bal = None
        self.kp = m.actuator_gainprm[:, 0].copy()

    # ------------------------------------------------------------------ joints
    def q(self, idx) -> np.ndarray:
        return self.data.qpos[idx[0]].copy()

    def arm_q(self, side: str) -> np.ndarray:
        return self.data.qpos[self.arm[side][0]].copy()

    def hand_q(self, side: str) -> np.ndarray:
        return self.data.qpos[self.hand[side][0]].copy()

    def joint_limit_margin(self, side: str = "right") -> float:
        m = self.model
        ids = [m.joint(self.p + j).id for j in ARM_JOINTS[side]]
        q = self.arm_q(side)
        return float(np.min(np.minimum(q - m.jnt_range[ids, 0], m.jnt_range[ids, 1] - q)))

    # ------------------------------------------------------------------- reset
    def reset_standing(self, arm_q: dict | None = None, settle: float = 0.0) -> None:
        """PRE-EPISODE ONLY: standing configuration (feet flat on the floor), matching
        controls, controller state; optional settling under physics with balance on."""
        m, d = self.model, self.data
        b = self.base_qadr
        d.qpos[b:b + 3] = (0.0, 0.0, 0.79)
        d.qpos[b + 3:b + 7] = (1.0, 0.0, 0.0, 0.0)
        for s in SIDES:
            d.qpos[self.leg[s][0]] = 0.0
            d.qpos[self.arm[s][0]] = (arm_q or {}).get(s, STAND_ARM[s])
            d.qpos[self.hand[s][0]] = synergy_posture(s, "open", 0.0)
        d.qpos[self.waist[0]] = 0.0
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(m, d)
        # put the lowest foot sphere on the floor
        mujoco.mj_forward(m, d)
        zmin = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in self.foot_geoms)
        d.qpos[b + 2] -= zmin
        mujoco.mj_forward(m, d)
        self.feet = self.leg_ik.foot_poses(d.qpos)
        self.stance_xy = 0.5 * (self.feet["left"][0][:2] + self.feet["right"][0][:2])
        base = d.qpos[b:b + 3].copy()
        # crouch: lower the pelvis by (0.79 - stance_height) with the feet fixed
        self._pelvis_nominal = np.array([base[0], base[1], base[2] - (0.79 - self.stance_height)])
        self.pelvis_offset[:] = 0.0
        bent = np.array([-0.25, 0.0, 0.0, 0.5, -0.25, 0.0])      # seed away from the straight-knee singularity
        self.q_leg_goal = np.r_[bent, bent]
        for _ in range(10):
            q_leg = self._solve_legs(d.qpos.copy())
            self.q_leg_goal = np.concatenate([q_leg["left"], q_leg["right"]])
        # write the crouched stance and drop the feet onto the floor again
        d.qpos[b:b + 3] = self._pelvis_nominal
        for s in SIDES:
            d.qpos[self.leg[s][0]] = q_leg[s]
        mujoco.mj_forward(m, d)
        zmin = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in self.foot_geoms)
        d.qpos[b + 2] -= zmin
        self._pelvis_nominal[2] = d.qpos[b + 2]
        mujoco.mj_forward(m, d)
        self.feet = self.leg_ik.foot_poses(d.qpos)
        self.q_leg_goal = np.concatenate([q_leg["left"], q_leg["right"]])
        self.leg_smooth.reset(self.q_leg_goal)
        for s in SIDES:
            self.arm_goal[s] = d.qpos[self.arm[s][0]].copy()
            self.arm_smooth[s].reset(self.arm_goal[s])
            self.hand_goal[s] = d.qpos[self.hand[s][0]].copy()
            self.hand_smooth[s].reset(self.hand_goal[s])
            self.ik[s].reset(self.arm_goal[s])
            self.hand_target[s] = None
        self._write_ctrl(np.zeros(2))
        if settle > 0:
            for _ in range(int(round(settle / self.dt))):
                self.inner_loop()
                mujoco.mj_step(m, d)

    def _solve_legs(self, qpos) -> dict:
        b = self.base_qadr
        p = self._pelvis_nominal.copy()
        p[:2] += self.pelvis_offset
        q_init = {"left": self.q_leg_goal[:6], "right": self.q_leg_goal[6:]}
        return self.leg_ik.solve(qpos, p, np.array([1.0, 0, 0, 0]), self.feet, q_init)

    # ----------------------------------------------------------------- command
    def command_hand(self, side: str, pos, mat) -> None:
        """Control rate: IK of the ``{side}_grasp`` site to a world pose."""
        q_ref = self.ik[side].solve(self.data.qpos, np.asarray(pos, float), np.asarray(mat, float),
                                    measured_q=self.arm_q(side))
        self.arm_goal[side] = q_ref
        self.hand_target[side] = (np.asarray(pos, float).copy(), np.asarray(mat, float).copy())

    def command_arm_joints(self, side: str, q) -> None:
        self.arm_goal[side] = np.asarray(q, float).copy()
        self.ik[side].reset(self.arm_goal[side])

    def command_synergy(self, side: str, s: float, kind: str = "power") -> None:
        self.synergy[side] = (kind, float(np.clip(s, 0.0, 1.0)))
        self.hand_goal[side] = synergy_posture(side, kind, s)

    # -------------------------------------------------------------- inner loop
    def inner_loop(self) -> None:
        """Physics rate: reference smoothing, CoM regulation, sway damping, ctrl write."""
        m, d = self.model, self.data
        dt = self.dt
        com = d.subtree_com[self.pelvis]
        mujoco.mj_jacSubtreeCom(m, d, self._jac_com, self.pelvis)
        vcom = self._jac_com @ d.qvel
        damp = np.zeros(2)
        if self.balance_enabled and self.feet is not None:
            target = self.support_center() + self.com_offset
            err = com[:2] - target
            self.pelvis_offset = np.clip(self.pelvis_offset - self.k_com * err * dt,
                                         -self.pelvis_offset_limit, self.pelvis_offset_limit)
            damp = self.k_sway * vcom[:2]
            self._leg_counter += 1
            if self._leg_counter >= self.leg_ik_decimation:
                self._leg_counter = 0
                q_leg = self._solve_legs(d.qpos)
                self.q_leg_goal = np.concatenate([q_leg["left"], q_leg["right"]])
        self.leg_smooth.step(self.q_leg_goal, dt)
        for s in SIDES:
            self.arm_smooth[s].step(self.arm_goal[s], dt)
            self.hand_smooth[s].step(self.hand_goal[s], dt)
        self._write_ctrl(damp)

    def _write_ctrl(self, damp) -> None:
        d = self.data
        # Gravity feed-forward for the subtrees hanging from the pelvis (waist, arms,
        # fingers): the servo target is offset by tau_g / kp so that the joint torque
        # holds the link weights (counted against the actuator torque limits, as a
        # real joint controller's feed-forward torque).  Held objects are NOT
        # compensated.  Not applied to the legs: their holding torques come from the
        # ground reaction, which qfrc_bias of a floating-base tree does not represent.
        bias = d.qfrc_bias
        q_leg = self.leg_smooth.q.copy()
        # ankle pitch (+ leans the body back) / roll (+ leans the body to +y)
        for i0 in (0, 6):
            q_leg[i0 + 4] += damp[0]
            q_leg[i0 + 5] -= damp[1]
        d.ctrl[self.leg["left"][2]] = q_leg[:6]
        d.ctrl[self.leg["right"][2]] = q_leg[6:]
        d.ctrl[self.waist[2]] = bias[self.waist[1]] / self.kp[self.waist[2]]
        for s in SIDES:
            d.ctrl[self.arm[s][2]] = self.arm_smooth[s].q + bias[self.arm[s][1]] / self.kp[self.arm[s][2]]
            d.ctrl[self.hand[s][2]] = self.hand_smooth[s].q + bias[self.hand[s][1]] / self.kp[self.hand[s][2]]

    # ------------------------------------------------------------------- state
    def grasp_pos(self, side: str = "right") -> np.ndarray:
        return self.data.site_xpos[self.grasp_site[side]].copy()

    def grasp_mat(self, side: str = "right") -> np.ndarray:
        return self.data.site_xmat[self.grasp_site[side]].reshape(3, 3).copy()

    def palm_pos(self, side: str = "right") -> np.ndarray:
        return self.data.site_xpos[self.palm_site[side]].copy()

    def tip_pos(self, side: str, finger: str) -> np.ndarray:
        return self.data.site_xpos[self.tip_site[side][finger]].copy()

    def synergy_closure(self, side: str = "right", kind: str | None = None) -> float:
        """Measured closure: projection of the finger encoders onto the synergy direction."""
        kind = kind or self.synergy[side][0]
        o = synergy_posture(side, kind, 0.0)
        c = synergy_posture(side, kind, 1.0)
        dvec = c - o
        return float(np.clip((self.hand_q(side) - o) @ dvec / (dvec @ dvec), 0.0, 1.0))

    def com(self) -> np.ndarray:
        return self.data.subtree_com[self.pelvis].copy()

    def com_vel(self) -> np.ndarray:
        mujoco.mj_jacSubtreeCom(self.model, self.data, self._jac_com, self.pelvis)
        return self._jac_com @ self.data.qvel

    def foot_points(self) -> np.ndarray:
        return self.data.geom_xpos[self.foot_geoms][:, :2].copy()

    def support_center(self) -> np.ndarray:
        return self.foot_points().mean(axis=0)

    def pelvis_height(self) -> float:
        return float(self.data.xpos[self.pelvis][2])

    def pelvis_tilt(self) -> float:
        R = self.data.xmat[self.pelvis].reshape(3, 3)
        return float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))

    def floor_contacts(self) -> tuple[float, float, np.ndarray | None]:
        """(foot force, non-foot robot force against the floor, centre of pressure)."""
        m, d = self.model, self.data
        n = d.ncon
        foot_f, other_f = 0.0, 0.0
        cop_num = np.zeros(2)
        if n == 0 or self.floor_geom < 0:
            return 0.0, 0.0, None
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        for c in np.flatnonzero((g1 == self.floor_geom) | (g2 == self.floor_geom)):
            other = g2[c] if g1[c] == self.floor_geom else g1[c]
            if not self._robot_mask[other]:
                continue
            mujoco.mj_contactForce(m, d, int(c), self._f6)
            fn = max(0.0, float(self._f6[0]))
            if self._foot_mask[other]:
                foot_f += fn
                cop_num += fn * d.contact.pos[c][:2]
            else:
                other_f += fn
        cop = cop_num / foot_f if foot_f > 1e-6 else None
        return foot_f, other_f, cop

    def balance_state(self) -> dict:
        """Measured balance quantities (cached per simulation time)."""
        t = self.data.time
        if self._bal is not None and self._bal["t"] == t:
            return self._bal
        poly = _hull(self.foot_points())
        com = self.com()
        foot_f, other_f, cop = self.floor_contacts()
        h, tilt = self.pelvis_height(), self.pelvis_tilt()
        margin = polygon_margin(poly, com)
        cop_margin = polygon_margin(poly, cop) if cop is not None else -np.inf
        fallen = bool(h < PELVIS_MIN_HEIGHT or tilt > PELVIS_MAX_TILT or other_f > 5.0)
        self._bal = {"t": t, "pelvis_height": h, "pelvis_tilt": tilt, "com": com, "cop": cop,
                     "com_margin": margin, "cop_margin": cop_margin, "foot_force": foot_f,
                     "nonfoot_floor_force": other_f, "fallen": fallen, "polygon": poly}
        return self._bal

    # ----------------------------------------------------------------- encode
    def encode(self, space: EmbeddingSpace, side: str = "right", pos=None, mat=None,
               closure: float | None = None, extra=()) -> np.ndarray:
        pos = self.grasp_pos(side) if pos is None else pos
        mat = self.grasp_mat(side) if mat is None else mat
        closure = self.synergy_closure(side) if closure is None else closure
        return space.normalize(np.r_[pos, self.chart.encode_matrix(mat), closure, list(extra)])

    def decode(self, space: EmbeddingSpace, e) -> dict:
        phys = space.denormalize(e)
        return {"pos": phys[:3], "mat": self.chart.decode_matrix(phys[3:6]), "grasp": float(phys[6]),
                "extra": phys[7:]}
