"""UR5e arm + dexterous hand decoder (LEAP Hand or Tetheria Aero Hand Open).

Shared by the three dexterous-hand tasks (push button, steering wheel, peg
tightening) and both hands.  Robots come from the pinned menagerie checkout:

* arm  ``universal_robots_ur5e/ur5e.xml`` (6 position servos, 150/28 N m limits)
* hand ``leap_hand/right_hand.xml``  (16 joint position servos, kp = 3 N m/rad) or
  ``tetheria_aero_hand_open/right_hand.xml`` (7 actuators: 4 finger-flexion
  tendons, thumb abduction, 2 thumb tendons; extension by the hand's springs).

MOUNT ("forearm" mount, as on teleoperated UR + dexterous-hand set-ups).  Each
hand is attached at the UR5e ``attachment_site`` (ISO 9409 flange, z out of the
flange) through a 12 mm aluminium adapter disc (r = 31.5 mm, 0.10 kg) with its
WRIST face flush on the adapter and the fingers pointing along the flange z
axis.  Flange frame convention for both hands: +z = finger direction,
+x = palm normal (palmar side), +y = z x x (thumb side).

* LEAP: wrist face of ``palm`` (centre (-0.100, -0.023, -0.012) in the palm
  frame); palm-frame axes (x, y, z) -> flange (z, y, -x).
* Aero: 4-hole plate of ``tetheria_mount`` (centre (-0.0223, 0.0018, 0.011) in
  the mount frame); mount-frame axes (x, y, z) -> flange (z, -y, x).  The Aero
  palm is built 17 deg dorsiflexed w.r.t. its mount; the anatomical hand frame
  below compensates for it.

HAND FRAME / GRASP SITE.  ``<hand>/grasp`` (site on the hand root body) is the
IK point of the arm and the position part of the embedding: the centre of the
closed power grasp: the point where a 4 cm bar settles after the fingers close
(they pull an object toward the palm), measured with
``tasks.hand_base.stiff_pad_transport_test``; ~2.5 cm in front of the palm.  Its axes are anatomical: z_h = extended-finger direction,
x_h = palm normal, y_h = toward the thumb side (LEAP: flange axes; Aero: flange
axes rotated -17 deg about y).  ``hand_frame(z_axis, x_hint)`` builds a world
rotation of this frame; the rotation chart is only valid within pi - 0.1 rad of
the task's base frame, so pick the base near the orientations the task uses.

EMBEDDING (``hand_arm_space``, 9-D, normalized):

    0-2  x, y, z        grasp-site position, world [m]            -> [-1, 1]
    3-5  rx, ry, rz     rotation vector of R_h R_base^T (world)   -> [-1, 1] (= +-pi)
                        R_base is a TASK-chosen nominal hand frame (``RotationChart``)
    6    power          power-grasp synergy  (0 open .. 1 closed) -> [0, 1]
    7    pinch          thumb-index pinch    (0 .. 1)             -> [0, 1]
    8    point          index-pointing fist  (0 .. 1)             -> [0, 1]

    metric weights (1, 1, 1, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05).

SYNERGIES.  Each hand defines four actuator-space postures (``HandModel.postures``:
open / power / pinch / point).  Command: ctrl = clip(u_open + sum_k s_k (u_k - u_open)).
They are realised PHYSICALLY by the hand's own actuators (LEAP joint servos, Aero
tendon servos + springs); the measured synergy (``HandArm.synergy()``) is the
least-squares fit of the measured actuator lengths (joint angles / tendon lengths).
Use ``HandArm.cmd(...)`` / ``HandArmTaskBase.cmd(...)`` to build a command.

FINGERTIPS.  Sites ``<hand>/<f>_tip_site`` at the END of each fingertip pad
(f in th, if, mf, rf [+ pf for the Aero]); ``HandArm.tip_pos(f)``.
Fingertip pad geoms (``HandModel.tip_geoms``) follow the stiff-pad rule:
priority 2, condim 4, solref (0.004, 1), solimp (0.95, 0.99, 0.001, 0.5, 2),
silicone/TPU "skin" friction 0.9.  All other hand collision geoms are stiff too
(same solref/solimp, priority 1) with 3D-print "plastic" friction 0.4
(``link_friction`` argument).

CONTACT GROUPS (``HandArm.contact_groups()`` -> name -> geom ids, for
``framework.contacts.ContactMonitor``): ``tip_<f>`` (pad geom), ``dist_<f>``
(all distal-link geoms), ``finger_<f>`` (all geoms of finger f), ``fingers``,
``palm`` (palm/mount/adapter), ``hand`` (all hand geoms), ``arm`` (UR5e
links + adapter).

ACTUATOR LIMITS (documented before -> after):
* UR5e unchanged (menagerie kp 2000/500, 150 N m shoulder/elbow, 28 N m wrists;
  IK reference speed capped at 3 rad/s, below the UR5e's 180 deg/s joint limit);
  gravity compensation of the arm and of the hand payload is routed through
  the arm actuators (``actuatorgravcomp``) and clamped by joint
  ``actuatorfrcrange`` = the same 150/28 N m (as the UR controller's payload
  model does; grasped objects are NOT compensated).
* LEAP: menagerie servos have NO force limit -> forcerange +-0.5 N m
  (XC330-M288-T: 0.93 N m stall at 5 V; LEAP API current limit 550 mA ~ 0.29 N m;
  0.5 N m ~ 1 A short-term).  kp = 3 N m/rad kept.
* Aero: menagerie tendon servos push AND pull with unlimited force (+28 N push
  measured at full extension) -> pull-only cable, forcerange [-T, 0] with
  T = AERO_TENDON_MAX (calibrated so that a blocked index fingertip reaches the
  published ~10 N); thumb abduction servo +-0.5 N m (was unlimited).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from eddies.common.config import load

ROBOT = load("robots/leap")      # LEAP and Aero share the arm and the embedding

from eddies.envs.controllers.ik import DLSIK
from eddies.envs.framework.embedding import EmbeddingSpace, RotationChart
from eddies.envs.framework.physics import FRICTION
from eddies.envs.framework.scene import SceneBuilder, aid, jid, qadr, sid, subtree_geoms
from eddies.envs.paths import menagerie

UR5E_XML = "universal_robots_ur5e/ur5e.xml"
ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
ARM_TORQUE = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)      # UR5e max joint torques [N m]
ARM_PREFIX = "ur5e/"
ADAPTER_T = 0.012          # adapter disc thickness [m]
ADAPTER_R = 0.0315         # UR5e flange radius [m]
ADAPTER_MASS = 0.10        # aluminium disc [kg]
PAD_SOLREF = (0.004, 1.0)
PAD_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
LEAP_TORQUE_MAX = 0.5      # [N m]
AERO_TENDON_MAX = 75.0     # [N] pull-only cable tension limit (blocked index tip ~10 N, see doc)
AERO_ABD_TORQUE_MAX = 0.5  # [N m]
SYNERGIES = ("power", "pinch", "point")


def _rot(cols) -> np.ndarray:
    """Rotation matrix whose columns are the images of the root x, y, z axes."""
    return np.array(cols, float).T


def _mat2quat(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def _quat2mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float) / np.linalg.norm(q))
    return R.reshape(3, 3)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@dataclass(frozen=True)
class HandModel:
    """Static description of one menagerie hand (all names unprefixed)."""
    name: str
    xml: str
    display: str
    root: str
    mount_point: tuple                 # wrist/mount face centre in the root frame [m]
    root_axes_in_flange: tuple         # images of root x, y, z in the flange frame
    hand_tilt: float                   # anatomical frame = flange frame rotated about y [rad]
    grasp_in_flange: tuple             # grasp-site position in the flange frame [m]
    fingers: tuple
    tip_geoms: dict
    distal_bodies: dict
    tip_end_local: dict                # tip-end point in the distal body frame [m]
    finger_bodies: dict
    palm_bodies: tuple
    actuators: tuple                   # hand actuator names in synergy order
    postures: dict = field(default_factory=dict)   # name -> actuator targets
    gravcomp_fingers: bool = True

    @property
    def R_root(self) -> np.ndarray:
        return _rot(self.root_axes_in_flange)

    @property
    def R_hand(self) -> np.ndarray:
        """Anatomical hand frame expressed in the flange frame."""
        return rot_y(self.hand_tilt)

    def root_pose_in_flange(self, adapter: float = ADAPTER_T):
        R = self.R_root
        p = np.array([0.0, 0.0, adapter]) - R @ np.asarray(self.mount_point, float)
        return p, R


_LEAP_FINGER_JOINTS = {
    "if": ("if_mcp", "if_rot", "if_pip", "if_dip"),
    "mf": ("mf_mcp", "mf_rot", "mf_pip", "mf_dip"),
    "rf": ("rf_mcp", "rf_rot", "rf_pip", "rf_dip"),
    "th": ("th_cmc", "th_axl", "th_mcp", "th_ipl"),
}


def _leap_posture(**joints) -> np.ndarray:
    """Actuator targets (if, mf, rf: mcp, rot, pip, dip; th: cmc, axl, mcp, ipl)."""
    order = [j for f in ("if", "mf", "rf", "th") for j in _LEAP_FINGER_JOINTS[f]]
    return np.array([float(joints.get(j, 0.0)) for j in order])


LEAP = HandModel(
    name="leap", xml="leap_hand/right_hand.xml", display="LEAP Hand (right)", root="palm",
    mount_point=(-0.100, -0.023, -0.012),
    root_axes_in_flange=((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    hand_tilt=0.0,
    grasp_in_flange=(0.0506, -0.0138, 0.0776),     # where a power-grasped bar settles (grasp test)
    fingers=("th", "if", "mf", "rf"),
    tip_geoms={"th": "th_tip", "if": "if_tip", "mf": "mf_tip", "rf": "rf_tip"},
    distal_bodies={"th": "th_ds", "if": "if_ds", "mf": "mf_ds", "rf": "rf_ds"},
    tip_end_local={"th": (-0.0049, -0.0614, -0.0141), "if": (-0.0038, -0.0494, 0.0127),
                   "mf": (-0.0038, -0.0494, 0.0127), "rf": (-0.0038, -0.0494, 0.0127)},
    finger_bodies={"th": ("th_mp", "th_bs", "th_px", "th_ds"),
                   "if": ("if_bs", "if_px", "if_md", "if_ds"),
                   "mf": ("mf_bs", "mf_px", "mf_md", "mf_ds"),
                   "rf": ("rf_bs", "rf_px", "rf_md", "rf_ds")},
    palm_bodies=("palm",),
    actuators=tuple(f"{j}_act" for f in ("if", "mf", "rf", "th") for j in _LEAP_FINGER_JOINTS[f]),
    postures={
        "open": _leap_posture(th_cmc=0.25, th_axl=0.20),
        # tight fist (collision-free without an object): an enclosed object is squeezed by the servos
        "power": _leap_posture(if_mcp=1.40, if_pip=1.40, if_dip=1.00, mf_mcp=1.40, mf_pip=1.40, mf_dip=1.00,
                               rf_mcp=1.40, rf_pip=1.40, rf_dip=1.00, th_cmc=0.90, th_axl=0.60,
                               th_mcp=0.40, th_ipl=0.70),
        # pad-to-pad thumb-index pinch (kinematic optimisation, pads meet at s = 1)
        "pinch": _leap_posture(if_mcp=1.146, if_pip=0.680, if_dip=0.472, th_cmc=1.114, th_axl=0.826,
                               th_mcp=0.502, th_ipl=0.440),
        # index extended, middle/ring curled, thumb over the middle finger (collision-free fist)
        "point": _leap_posture(mf_mcp=1.50, mf_pip=1.50, mf_dip=0.90, rf_mcp=1.50, rf_pip=1.50, rf_dip=0.90,
                               th_cmc=1.10, th_axl=0.40, th_mcp=0.95, th_ipl=0.70),
    },
    gravcomp_fingers=True,
)

_AERO_FINGERS = {"if": "index", "mf": "middle", "rf": "ring", "pf": "pinky"}


def _aero_posture(i, m, r, p, abd, th1, th2) -> np.ndarray:
    """Actuator targets: index/middle/ring/pinky tendon length [m], thumb abduction [rad],
    thumb tendons 1/2 length [m].  Long tendon = extended (springs), short = flexed."""
    return np.array([i, m, r, p, abd, th1, th2], float)


AERO = HandModel(
    name="aero", xml="tetheria_aero_hand_open/right_hand.xml", display="Tetheria Aero Hand Open (right)",
    root="tetheria_mount",
    mount_point=(-0.0223, 0.0018, 0.011),
    root_axes_in_flange=((0, 0, 1), (0, -1, 0), (1, 0, 0)),
    hand_tilt=np.radians(-17.0),
    grasp_in_flange=(-0.0075, 0.000, 0.1531),      # where a power-grasped bar settles (grasp test)
    fingers=("th", "if", "mf", "rf", "pf"),
    tip_geoms={"th": "th_tip", "if": "if_tip", "mf": "mf_tip", "rf": "rf_tip", "pf": "pf_tip"},
    distal_bodies={"th": "right_thumb_distal_link", **{f: f"right_{n}_distal_link" for f, n in _AERO_FINGERS.items()}},
    tip_end_local={"th": (0.0, -0.002, 0.027), **{f: (0.0, -0.003, 0.024) for f in _AERO_FINGERS}},
    finger_bodies={"th": ("right_t_link", "right_thumb_mcp_link", "right_thumb_proximal_link",
                          "right_thumb_distal_link"),
                   **{f: (f"right_{n}_f_link", f"right_{n}_proximal_link", f"right_{n}_middle_link",
                          f"right_{n}_distal_link") for f, n in _AERO_FINGERS.items()}},
    palm_bodies=("tetheria_mount", "palm"),
    actuators=("right_index_A_tendon", "right_middle_A_tendon", "right_ring_A_tendon",
               "right_pinky_A_tendon", "right_thumb_A_cmc_abd", "right_th1_A_tendon", "right_th2_A_tendon"),
    postures={
        "open": _aero_posture(0.1104, 0.1104, 0.1104, 0.1104, 0.30, 0.0384, 0.1121),
        # fingers flexed onto the object; thumb cables stop short of their 75 N limit on a ~4 cm object
        # (a saturated, force-limited thumb ratchets inward under transport loads and drags the object)
        "power": _aero_posture(0.0760, 0.0760, 0.0760, 0.0760, 1.20, 0.0340, 0.1000),
        "pinch": _aero_posture(0.0800, 0.1104, 0.1104, 0.1104, 1.10, 0.0297, 0.1004),   # grid search
        # flexed cables commanded just above their joint-limit lengths (0.062-0.064 m) so the
        # curled fingers are held at moderate tension instead of fighting the limits
        "point": _aero_posture(0.1104, 0.0650, 0.0650, 0.0665, 1.10, 0.0300, 0.0900),
    },
    gravcomp_fingers=False,
)

HANDS = {"leap": LEAP, "aero": AERO}


# ------------------------------------------------------------------ model edits
def _match_options(child: mujoco.MjSpec, parent: mujoco.MjSpec) -> None:
    """The world options win (hand files set their own timestep/integrator); copying
    them first only silences the attach-conflict warnings."""
    for k in ("timestep", "impratio", "iterations", "ls_iterations", "integrator", "cone",
              "noslip_iterations", "density", "viscosity"):
        setattr(child.option, k, getattr(parent.option, k))
    child.option.gravity = parent.option.gravity


def edit_ur5e(spec: mujoco.MjSpec, *, adapter: float = ADAPTER_T) -> None:
    """Gravity compensation through the actuators + flange adapter disc."""
    for light in list(spec.lights):
        spec.delete(light)
    for b in spec.bodies:
        if b.name and b.name != "world":
            b.gravcomp = 1.0
    for name, lim in zip(ARM_JOINTS, ARM_TORQUE):
        j = spec.joint(name)
        j.actgravcomp = True
        j.actfrcrange = np.array([-lim, lim])
        j.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
    if adapter > 0:
        site = spec.site("attachment_site")
        R = _quat2mat(site.quat)
        pos = np.asarray(site.pos, float) + R @ np.array([0, 0, adapter / 2])
        spec.body("wrist_3_link").add_geom(
            name="flange_adapter", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[ADAPTER_R, adapter / 2, 0],
            pos=list(pos), quat=list(_mat2quat(R)), mass=ADAPTER_MASS, rgba=[0.12, 0.12, 0.13, 1],
            friction=list(FRICTION["metal"]), group=1)


def make_hand_editor(hand: HandModel, parent: mujoco.MjSpec, *, adapter: float = ADAPTER_T,
                     tip_friction: str = "skin", link_friction: str = "plastic"):
    """Return the mjSpec edit function applied to the menagerie hand before attaching."""

    def edit(spec: mujoco.MjSpec) -> None:
        _match_options(spec, parent)
        for light in list(spec.lights):
            spec.delete(light)
        root = spec.body(hand.root)
        root.pos = [0.0, 0.0, 0.0]
        root.quat = [1.0, 0.0, 0.0, 0.0]
        p, R = hand.root_pose_in_flange(adapter)
        Rt = R.T
        # anatomical grasp site (pose given in the flange frame -> root frame)
        g = Rt @ (np.asarray(hand.grasp_in_flange, float) - p)
        Rg = Rt @ hand.R_hand
        root.add_site(name="grasp", pos=list(g), quat=list(_mat2quat(Rg)), size=[0.006] * 3,
                      rgba=[1, 0.3, 0.1, 0.0], group=4)
        for f in hand.fingers:
            spec.body(hand.distal_bodies[f]).add_site(name=f"{f}_tip_site", pos=list(hand.tip_end_local[f]),
                                                     size=[0.003] * 3, rgba=[0.1, 0.9, 0.2, 0.0], group=4)
        if hand.name == "aero":   # cosmetic only: hide cable/spring paths, grey pulleys
            for t in spec.tendons:
                t.group = 3
            for gm in spec.geoms:
                if not (gm.contype or gm.conaffinity) and list(gm.rgba) == [0.0, 1.0, 0.0, 1.0]:
                    gm.rgba = np.array([0.55, 0.55, 0.57, 1.0])
        tips = set(hand.tip_geoms.values())
        for gm in spec.geoms:
            if not (gm.contype or gm.conaffinity):
                continue
            gm.solref = np.array(PAD_SOLREF)
            gm.solimp = np.array(PAD_SOLIMP)
            gm.condim = 4
            if gm.name in tips:
                gm.priority = 2
                gm.friction = np.array(FRICTION[tip_friction])
            else:
                gm.priority = 1
                gm.friction = np.array(FRICTION[link_friction])
        for b in spec.bodies:
            if b.name in hand.palm_bodies:
                b.gravcomp = 1.0
            elif b.name and b.name != "world":
                b.gravcomp = 1.0 if hand.gravcomp_fingers else 0.0
        if hand.name == "leap":
            for a in spec.actuators:
                a.forcerange = np.array([-LEAP_TORQUE_MAX, LEAP_TORQUE_MAX])
                a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            for j in spec.joints:
                j.actgravcomp = True
        else:
            for a in spec.actuators:
                if a.name == "right_thumb_A_cmc_abd":
                    a.forcerange = np.array([-AERO_ABD_TORQUE_MAX, AERO_ABD_TORQUE_MAX])
                else:   # cable: pull only (negative actuator force = tension)
                    a.forcerange = np.array([-AERO_TENDON_MAX, 0.0])
                a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        # MuJoCo 3.15: MjSpec.attach silently DROPS spatial tendons that wrap geoms (and the
        # actuators/sensors on them) unless the child spec has been compiled once, which
        # resolves the wrap references.  The Aero hand's 19 cable/spring tendons need this.
        spec.compile()

    return edit


def attach_hand_arm(sb: SceneBuilder, hand: HandModel | str, *, base_pos, base_yaw: float = -np.pi / 2,
                    adapter: float = ADAPTER_T, tip_friction: str = "skin",
                    link_friction: str = "plastic") -> str:
    """Attach the UR5e at ``base_pos`` (yaw so that the menagerie 'home' faces +x) and
    the hand at its flange.  Returns the hand prefix (e.g. ``"leap/"``)."""
    hand = HANDS[hand] if isinstance(hand, str) else hand
    yaw = np.array([np.cos(base_yaw / 2), 0, 0, np.sin(base_yaw / 2)])
    sb.attach_robot(UR5E_XML, ARM_PREFIX, pos=base_pos, quat=yaw,
                    edit=lambda s: edit_ur5e(s, adapter=adapter))
    w3 = sb.spec.body(ARM_PREFIX + "wrist_3_link")
    site = mujoco.MjSpec.from_file(menagerie(UR5E_XML)).site("attachment_site")   # pose in wrist_3_link
    Rs, ps = _quat2mat(site.quat), np.asarray(site.pos, float)
    p, R = hand.root_pose_in_flange(adapter)
    prefix = hand.name + "/"
    sb.attach_robot(hand.xml, prefix, pos=ps + Rs @ p, quat=_mat2quat(Rs @ R),
                    edit=make_hand_editor(hand, sb.spec, adapter=adapter, tip_friction=tip_friction,
                                          link_friction=link_friction), parent=w3)
    return prefix


# ------------------------------------------------------------------ embedding
def hand_arm_space(x_range, y_range, z_range) -> EmbeddingSpace:
    return EmbeddingSpace(
        names=list(ROBOT.embedding.names),
        phys_lower=[x_range[0], y_range[0], z_range[0], -np.pi, -np.pi, -np.pi, 0.0, 0.0, 0.0],
        phys_upper=[x_range[1], y_range[1], z_range[1], np.pi, np.pi, np.pi, 1.0, 1.0, 1.0],
        weights=[float(w) for w in ROBOT.embedding.weights],
        norm_lower=[-1, -1, -1, -1, -1, -1, 0, 0, 0],
        norm_upper=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        units=["m", "m", "m", "rad", "rad", "rad", "-", "-", "-"])


def hand_frame(z_axis, x_hint) -> np.ndarray:
    """Rotation matrix of a hand frame with finger direction ``z_axis`` and palm normal
    as close as possible to ``x_hint`` (columns x_h, y_h, z_h in world coordinates)."""
    z = np.asarray(z_axis, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(x_hint, float) - np.dot(x_hint, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


# ------------------------------------------------------------------ runtime
class HandArm:
    """Runtime handle for the UR5e + hand assembly (see the module docstring)."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, hand: HandModel | str,
                 base_mat: np.ndarray, q_posture: np.ndarray, *, arm_prefix: str = ARM_PREFIX):
        self.model, self.data = model, data
        self.hand = HANDS[hand] if isinstance(hand, str) else hand
        self.hp = self.hand.name + "/"
        ap = arm_prefix
        self.arm_joints = [ap + j for j in ARM_JOINTS]
        self.qadr = np.array([qadr(model, j) for j in self.arm_joints])
        self.dadr = np.array([model.jnt_dofadr[jid(model, j)] for j in self.arm_joints])
        self.act_arm = np.array([aid(model, ap + a) for a in ARM_ACTUATORS])
        self.act_hand = np.array([aid(model, self.hp + a) for a in self.hand.actuators])
        self.ctrl_lo = model.actuator_ctrlrange[self.act_hand, 0].copy()
        self.ctrl_hi = model.actuator_ctrlrange[self.act_hand, 1].copy()
        self.grasp = sid(model, self.hp + "grasp")
        self.flange = sid(model, ap + "attachment_site")
        self.wrist_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ap + "wrist_3_link"))
        self.tips = {f: sid(model, f"{self.hp}{f}_tip_site") for f in self.hand.fingers}
        self.q_posture = np.asarray(q_posture, float)
        # 2 iterations x 0.03 rad per 20 ms control step = 3 rad/s < UR5e joint speed limit (pi rad/s)
        self.ik = DLSIK(model, self.arm_joints, self.hp + "grasp", posture=self.q_posture,
                        posture_gain=0.05, max_step=0.03, iterations=2, rot_weight=0.6)
        self.chart = RotationChart(_mat2quat(base_mat))
        # synergy basis in actuator space
        P = self.hand.postures
        self.u_open = np.asarray(P["open"], float)
        self.B = np.column_stack([np.asarray(P[k], float) - self.u_open for k in SYNERGIES])
        self._scale = np.maximum(self.ctrl_hi - self.ctrl_lo, 1e-6)
        self._Bn = self.B / self._scale[:, None]
        # geoms
        m = model
        self.arm_geoms = [g for g in subtree_geoms(m, ap + "base") if (m.geom_contype[g] or m.geom_conaffinity[g])
                          and not mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])).startswith(self.hp)]
        self.hand_geoms = [g for g in subtree_geoms(m, self.hp + self.hand.root)
                           if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.last_cmd = None

    # ------------------------------------------------------------------ state
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def dq(self) -> np.ndarray:
        return self.data.qvel[self.dadr].copy()

    def grasp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.grasp].copy()

    def grasp_mat(self) -> np.ndarray:
        return self.data.site_xmat[self.grasp].reshape(3, 3).copy()

    def flange_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.flange].copy()

    def tip_pos(self, finger: str) -> np.ndarray:
        return self.data.site_xpos[self.tips[finger]].copy()

    def tip_in_hand(self, finger: str) -> np.ndarray:
        """Fingertip end position in the grasp-site (hand) frame (measured)."""
        return self.grasp_mat().T @ (self.tip_pos(finger) - self.grasp_pos())

    def actuator_lengths(self) -> np.ndarray:
        """Measured actuator lengths: joint angles (LEAP) / tendon lengths + abduction (Aero)."""
        return self.data.actuator_length[self.act_hand].copy()

    def synergy(self) -> np.ndarray:
        """Least-squares synergy coordinates of the MEASURED hand configuration."""
        r = (self.actuator_lengths() - self.u_open) / self._scale
        s, *_ = np.linalg.lstsq(self._Bn, r, rcond=None)
        return np.clip(s, 0.0, 1.0)

    def synergy_ctrl(self, s) -> np.ndarray:
        u = self.u_open + self.B @ np.clip(np.asarray(s, float), 0.0, 1.0)
        return np.clip(u, self.ctrl_lo, self.ctrl_hi)

    def posture_error(self, s) -> float:
        """Max normalized deviation of measured actuator lengths from the synergy target."""
        return float(np.max(np.abs(self.actuator_lengths() - self.synergy_ctrl(s)) / self._scale))

    # ------------------------------------------------------------------ reset
    def set_configuration(self, q_arm, s=(0.0, 0.0, 0.0)) -> None:
        """PRE-EPISODE ONLY: write the arm joints, matching arm and hand controls."""
        d = self.data
        d.qpos[self.qadr] = q_arm
        d.qvel[self.dadr] = 0
        d.ctrl[self.act_arm] = q_arm
        d.ctrl[self.act_hand] = self.synergy_ctrl(s)
        if self.hand.name == "leap":     # joint servos: start at the commanded posture
            for a, u in zip(self.act_hand, d.ctrl[self.act_hand]):
                j = int(self.model.actuator_trnid[a, 0])
                d.qpos[self.model.jnt_qposadr[j]] = u
        mujoco.mj_forward(self.model, d)
        self.ik.reset(np.asarray(q_arm, float))

    # ---------------------------------------------------------------- command
    def command(self, pos, mat, s) -> None:
        q_ref = self.ik.solve(self.data.qpos, pos, mat, measured_q=self.q())
        self.data.ctrl[self.act_arm] = q_ref
        self.data.ctrl[self.act_hand] = self.synergy_ctrl(s)

    def command_embedding(self, space: EmbeddingSpace, e) -> dict:
        phys = space.denormalize(e)
        mat = self.chart.decode_matrix(phys[3:6])
        self.command(phys[:3], mat, phys[6:9])
        self.last_cmd = {"pos": phys[:3], "mat": mat, "synergy": phys[6:9]}
        return self.last_cmd

    def encode(self, space: EmbeddingSpace, pos=None, mat=None, s=None) -> np.ndarray:
        pos = self.grasp_pos() if pos is None else pos
        mat = self.grasp_mat() if mat is None else mat
        s = self.synergy() if s is None else s
        return space.normalize(np.r_[pos, self.chart.encode_matrix(mat), s])

    def cmd(self, space: EmbeddingSpace, pos, mat, s) -> np.ndarray:
        """Normalized command for a grasp-site pose (world rotation matrix) and synergy."""
        return space.normalize(np.r_[np.asarray(pos, float), self.chart.encode_matrix(mat), np.asarray(s, float)])

    def joint_limit_margin(self) -> float:
        rng = self.model.jnt_range[[jid(self.model, j) for j in self.arm_joints]]
        q = self.q()
        return float(np.min(np.minimum(q - rng[:, 0], rng[:, 1] - q)))

    def contact_groups(self) -> dict:
        m = self.model
        groups = {"arm": list(self.arm_geoms), "hand": list(self.hand_geoms)}
        hp = self.hp
        fingers_all = []
        for f in self.hand.fingers:
            bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, hp + b) for b in self.hand.finger_bodies[f]}
            gs = [g for g in self.hand_geoms if int(m.geom_bodyid[g]) in bodies]
            groups[f"finger_{f}"] = gs
            fingers_all += gs
            db = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, hp + self.hand.distal_bodies[f])
            groups[f"dist_{f}"] = [g for g in self.hand_geoms if int(m.geom_bodyid[g]) == db]
            groups[f"tip_{f}"] = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, hp + self.hand.tip_geoms[f])]
        groups["fingers"] = fingers_all
        groups["palm"] = [g for g in self.hand_geoms if g not in set(fingers_all)]
        adapter = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, ARM_PREFIX + "flange_adapter")
        if adapter >= 0:
            groups["palm"].append(adapter)
        return groups
