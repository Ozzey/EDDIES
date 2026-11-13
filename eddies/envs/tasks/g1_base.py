"""Shared Unitree G1 (+ Dex3-1 hands) humanoid manipulation setup.

Reusable base for every G1 task (peg insertion, drawer, microwave, ...).  The
robot, its decoder and its balance controller live in ``controllers/g1.py``;
this module adds the scene conventions, the embedding, contact groups, the fall
label and the stress-test helpers.

Scene conventions
-----------------
* The G1 stands at the world origin facing +x (feet flat on the floor, soles at
  x in [-0.05, 0.12] m, y = +-0.119 m), pelvis 0.77 m (slight knee bend).  It is a
  free-floating robot balanced by its own controller; nothing is fixed.
* ``make_builder(name)`` -> SceneBuilder with the robot attached.  Put furniture
  in FRONT of the robot: front edge at x >= 0.22 m (knees/toes are at x <= 0.13).
* ``add_table(sb, height, x_front, depth, y_center, width)`` adds a static table
  (default top 0.76 m, front edge 0.24 m).  MEASURED right-hand workspace with
  the full controller (hand frame = world frame, "glass" orientation; steady
  state error < 2 mm, no self-contact, CoM margin 8.5 cm):
      x in [0.28, 0.38] m, y in [-0.30, -0.08] m, z in [0.88, 1.02] m
  (x = 0.43 only for y >= -0.20; at y >= -0.03 the right arm collides with the
  torso, 150-240 N).  The left hand is the mirror image (y -> -y).  Lower or
  farther targets need a tilted hand; verify new layouts with ``reach_check``
  (kinematic) AND a closed-loop probe (self-contact, torque limits).

Hand frames / synergies (see ``controllers/g1.py``)
---------------------------------------------------
* ``{side}_grasp`` site: where a power-grasped 3 cm cylinder settles; the
  embedding commands this site.  Hand frame axes: +x fingers, +z up (index on
  top), palm normal +y (right hand) / -y (left hand).
* ``PREGRASP_OFFSET`` (hand frame): object centre relative to the grasp site
  for a collision-free approach with the ``open`` synergy.
* Synergies: ``open``, ``power``, ``pinch``, ``hook``, ``point``; the task picks
  one kind per command dimension (``synergy_kind``), the embedding carries the
  closure s in [0, 1].

Embedding (``g1_space``): [x, y, z] grasp-site position (world, m; 0.8 m spans),
[rx, ry, rz] rotation vector of the hand frame relative to the world frame
(rad), [grasp] synergy closure.  Metric diag [1, 1, 1, .3, .3, .3, .05];
``rho_step = 0.0075`` per 50 Hz control step = 0.15 m/s hand speed.

Contact groups (``contact_monitor(objects)``) - physical normal forces:
``thumb_<obj>``, ``index_<obj>``, ``middle_<obj>``, ``palm_<obj>``, ``hand_<obj>``
(active hand parts vs each object), ``body_<obj>`` (any other robot geom vs the object),
``hand_table``, ``body_table`` (robot vs table), plus ``#n`` contact counts.

Balance / fall
--------------
``self.g1.balance_state()``: pelvis height and tilt, CoM, centre of pressure
from the measured foot contacts, support polygon, CoM margin.  ``fallen()`` ->
pelvis < 0.55 m, tilt > 30 deg, or any non-foot robot geom on the floor.  Tasks
put ``fallen()`` into their ``bad`` label.

Stress test helpers
-------------------
``torso_push(active, force_xyz)`` applies/clears an external force on the
torso (``data.xfrc_applied``), e.g. a lateral push during the reach (the
controller rejects 60 N x 0.2 s lateral pushes with the arm extended; 70 N
topples it - the robot does not step); ``object_push(body, force_xy, point)``
pushes an object at a chosen point (e.g. at its base so that it slides instead
of tipping); ``pair_friction(geom_a, geom_b)`` returns the sliding friction
MuJoCo actually uses for a pair (higher priority wins, max if equal) - size
pushes with it.

Writing a G1 task (see ``tasks/g1_peg.py``)
-------------------------------------------
1. ``sb = self.make_builder(name)``; ``self.add_table(sb, ...)`` / furniture;
   objects; cameras (main: whole robot incl. feet - a side profile works well
   and leaves the top-right free for the video inset; detail: the hand).
2. ``self.finish_build(sb)`` (compiles, creates ``self.g1`` and ``self.space``),
   ``self.cm = self.contact_monitor({...})``, perception estimators, skills.
3. Override ``ready_hand_pose()`` -> (grasp-site position, hand matrix) written
   at reset by IK (pre-episode joint write), or ``ready_arm()`` for joint angles.
4. Use ``self.cmd(pos, rotvec, grasp)`` for skill targets/anchors, put
   ``self.fallen()`` into ``bad``, add ``self.balance_diagnostics()`` to
   ``step_diagnostics`` and call ``super().overlay_lines()``.
Lessons from the peg task: give rigid manipulated parts stiff contacts
(solref 0.004 like the pads; the 20 ms default lets a wedged part sink ~1 mm
into walls); the Dex3 hands collide through fitted primitives
(``controllers/g1.py`` HAND_COLLIDERS) - the menagerie convex-hull meshes gave
wrong-depth contacts against cylinders and 1-2 kN spikes, so do not re-enable
them; expect a 20-40 ms finger-impact transient (0.45-0.92 kN summed normal
force) when a synergy closes at 4 rad/s on a rigid object (0.01 kg m^2 finger
armature), and keep forbidden-contact thresholds on non-hand parts; verify a
layout with a closed-loop workspace probe, not only kinematic IK.
"""
from __future__ import annotations

import mujoco
import numpy as np

from eddies.common.config import load

ROBOT = load("robots/g1")   # physical constants of the robot

from eddies.envs.controllers.g1 import PREFIX, G1WholeBody, attach_g1, g1_space
from eddies.envs.framework.contacts import ContactMonitor
from eddies.envs.framework.scene import SceneBuilder, body_geoms
from eddies.envs.framework.task import Task

TABLE_H = 0.76
TABLE_FRONT = 0.24


class G1TaskBase(Task):
    robot = "Unitree G1 humanoid (29 DoF) + Dex3-1 hands"
    family = "humanoid_manipulation"
    physics_dt = float(ROBOT.physics_dt)
    control_dt = float(ROBOT.control_dt)
    rho_step = float(ROBOT.rho_step)          # metric step per control step (configs/robots)
    rho_game = 3.0
    failure_budget = 2
    hand = "right"
    synergy_kind = "power"
    settle_time = 1.5          # pre-episode settling with the balance controller on [s]

    # ----------------------------------------------------------------- scene
    def make_builder(self, name: str) -> SceneBuilder:
        sb = SceneBuilder(name, timestep=self.physics_dt)
        attach_g1(sb)
        return sb

    def add_table(self, sb: SceneBuilder, height: float = TABLE_H, x_front: float = TABLE_FRONT,
                  depth: float = 0.6, y_center: float = -0.1, width: float = 0.9,
                  name: str = "table") -> None:
        sb.table(name, (x_front + depth / 2, y_center), height, (depth / 2, width / 2))
        self.table_name = name
        self.table_height = float(height)

    def finish_build(self, sb: SceneBuilder, **g1_kwargs) -> None:
        self.model, self.data = sb.compile()
        self.g1 = G1WholeBody(self.model, self.data, active_sides=(self.hand,), **g1_kwargs)
        self.space = g1_space()
        m = self.model
        self.table_geoms = (body_geoms(m, [self.table_name]) if hasattr(self, "table_name") else [])
        self.torso_body = m.body(PREFIX + "torso_link").id

    def contact_monitor(self, objects: dict[str, list[str]]) -> ContactMonitor:
        m = self.model
        hg = self.g1.hand_geoms[self.hand]
        hand_all = set(hg["all"])
        body = [g for g in self.g1.robot_geoms if g not in hand_all and g not in self.g1.foot_geoms]
        groups = {"thumb": hg["thumb"], "index": hg["index"], "middle": hg["middle"],
                  "palm": hg["palm"], "hand": hg["all"], "body": body, "table": self.table_geoms}
        pairs = {"hand_table": ("hand", "table"), "body_table": ("body", "table")}
        for key, bodies in objects.items():
            groups[key] = body_geoms(m, bodies)
            for part in ("thumb", "index", "middle", "palm", "hand", "body"):
                pairs[f"{part}_{key}"] = (part, key)
        return ContactMonitor(m, groups, pairs)

    # ----------------------------------------------------------------- reset
    def ready_arm(self) -> dict:
        """Arm joint configurations {side: q(7)} written at reset (optional override)."""
        return {}

    def ready_hand_pose(self):
        """(grasp-site position, hand rotation matrix) of the active hand at reset, solved
        by IK and written pre-episode; None keeps ``ready_arm`` / the stand pose."""
        return None

    def reset(self) -> None:
        g1 = self.g1
        arm = dict(self.ready_arm())
        pose = self.ready_hand_pose()
        if pose is not None:
            g1.reset_standing(arm_q=arm, settle=0.0)          # body state for the IK
            arm[self.hand] = self.solve_arm_ik(pose[0], pose[1])
        g1.reset_standing(arm_q=arm, settle=self.settle_time)
        self.data.time = 0.0
        self.data.xfrc_applied[:] = 0.0

    def solve_arm_ik(self, pos, mat, side: str | None = None, q0=None, iters: int = 300) -> np.ndarray:
        """Offline IK (pre-episode) for a grasp-site pose with the current body state."""
        side = side or self.hand
        ik = self.g1.ik[side]
        q = np.asarray(q0 if q0 is not None else ik.posture, float).copy()
        saved = ik.max_step, ik.iterations
        ik.max_step, ik.iterations = 0.05, 1
        ik.reset(q)
        qpos = self.data.qpos.copy()
        for _ in range(iters):
            q = ik.solve(qpos, pos, mat, measured_q=ik.q_ref)
        ik.max_step, ik.iterations = saved
        return q

    def reach_check(self, pos, mat=np.eye(3), side: str | None = None) -> float:
        """Kinematic reach error [m] of the grasp site for a target pose (design tool)."""
        side = side or self.hand
        q = self.solve_arm_ik(pos, mat, side)
        sh = self.g1.ik[side].shadow
        sh.qpos[:] = self.data.qpos
        sh.qpos[self.g1.arm[side][0]] = q
        mujoco.mj_kinematics(self.model, sh)
        return float(np.linalg.norm(sh.site_xpos[self.g1.grasp_site[side]] - np.asarray(pos)))

    # --------------------------------------------------------------- control
    def inner_loop(self) -> None:
        self.g1.inner_loop()

    def cmd(self, pos, rotvec=(0.0, 0.0, 0.0), grasp: float = 0.0, extra=()) -> np.ndarray:
        """Normalized command: grasp-site position, hand rotation vector, synergy closure."""
        return self.space.normalize(np.r_[np.asarray(pos, float), np.asarray(rotvec, float), grasp,
                                          list(extra)])

    def initial_command(self) -> np.ndarray:
        return self.g1.encode(self.space, self.hand)

    def apply_command(self, e) -> None:
        dec = self.g1.decode(self.space, self.space.clip(e))
        self.g1.command_hand(self.hand, dec["pos"], dec["mat"])
        self.g1.command_synergy(self.hand, dec["grasp"], self.synergy_kind)

    def achieved(self, obs: dict) -> np.ndarray:
        return obs["achieved"]

    def base_obs(self) -> dict:
        g1, s = self.g1, self.hand
        bal = g1.balance_state()
        return {"t": self.time, "q": g1.arm_q(s), "dq": self.data.qvel[g1.arm[s][1]].copy(),
                "hand_q": g1.hand_q(s), "grasp_pos": g1.grasp_pos(s), "grasp_mat": g1.grasp_mat(s),
                "closure": g1.synergy_closure(s, self.synergy_kind),
                "achieved": g1.encode(self.space, s),
                "pelvis_height": bal["pelvis_height"], "pelvis_tilt": bal["pelvis_tilt"],
                "com_margin": bal["com_margin"], "fallen": bal["fallen"]}

    def fallen(self) -> bool:
        return bool(self.g1.balance_state()["fallen"])

    # ------------------------------------------------------------ disturbance
    def torso_push(self, active: bool, force_xyz=(0.0, 0.0, 0.0)) -> None:
        f = self.data.xfrc_applied[self.torso_body]
        f[:] = 0.0
        if active:
            f[:3] = force_xyz

    def pair_friction(self, geom_a: str, geom_b: str) -> float:
        """Sliding friction MuJoCo uses for a geom pair: the higher-priority geom's
        coefficient, or the element-wise maximum when the priorities are equal."""
        m = self.model
        ga, gb = m.geom(geom_a).id, m.geom(geom_b).id
        pa, pb = m.geom_priority[ga], m.geom_priority[gb]
        fa, fb = m.geom_friction[ga][0], m.geom_friction[gb][0]
        if pa != pb:
            return float(fa if pa > pb else fb)
        return float(max(fa, fb))

    def object_push(self, body: str, force_xy=None, point=None) -> None:
        """External force on an object at a world point (clears it when force_xy is None)."""
        b = self.model.body(body).id
        f = self.data.xfrc_applied[b]
        f[:] = 0.0
        if force_xy is None:
            return
        F = np.r_[np.asarray(force_xy, float)[:2], 0.0]
        f[:3] = F
        if point is not None:
            r = np.asarray(point, float) - self.data.xipos[b]
            f[3:] = np.cross(r, F)

    # ------------------------------------------------------------ diagnostics
    def balance_diagnostics(self) -> dict:
        bal = self.g1.balance_state()
        return {"pelvis_tilt_deg": float(np.degrees(bal["pelvis_tilt"])),
                "com_margin_deficit_m": float(-bal["com_margin"]),
                "pelvis_drop_m": float(0.77 - bal["pelvis_height"])}

    def overlay_lines(self) -> list:
        bal = self.g1.balance_state()
        return [f"balance: pelvis {bal['pelvis_height']:.3f} m  tilt {np.degrees(bal['pelvis_tilt']):4.1f} deg"
                f"  CoM margin {100 * bal['com_margin']:4.1f} cm"]
