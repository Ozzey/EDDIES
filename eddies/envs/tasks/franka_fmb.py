"""FMB-style insertion (Functional Manipulation Benchmark, Luo et al., IJRR 2024): pick, place on
fixture, regrasp, rotate/align, insert, push to seat.

Physical set-up (no welds, teleports, mocap or pose writes after ``reset``):

* **Object**: 24 x 24 x 90 mm square aluminium peg (FMB-style square prism, 1-2 mm mating
  tolerance as in FMB), standing at a random pose in a 14 x 10 cm pick region (yaw +-0.6 rad).
* **Fixture**: a loose chamfered pocket (6 mm clearance, 20 mm deep, 4 mm chamfer) at a fixed,
  known pose.  As in FMB, the object is placed on the fixture and regrasped from it.
* **Board**: chamfered square hole (diametral clearance 1-2 mm), pose randomized over
  12 x 10 cm and +-15 deg yaw (FMB randomizes the board pose and rotates it up to 15 deg).
  The hole ends in a **spring-loaded latch**: a 10 mm plunger on a slide joint (12 mm travel),
  spring 500 N/m preloaded to 12 N against its top stop (18 N at the bottom).  The peg's
  weight (1.4 N) and the 10 N insertion force limit cannot push it down; seating requires
  pushing on the peg's top with the closed fingers (force-limited push, 25 N).  ``seated``
  is a latched switch: the plunger bottomed out with the peg in the hole (like a snap-fit
  click); the spring then returns the plunger.

Skills (the pick/regrasp tokens are the SAME skills in two contexts; the supervisor's LTLf
residual tells which phase the product state is in):
    reach_above_peg, descend_peg, grasp_peg, lift_peg            (pick from table / regrasp from fixture)
    move_over_fixture, place_in_fixture, release                 (place on fixture)
    move_over_hole, insert, back_off, spiral_insert, release     (align, insert, recovery)
    push_to_seat, retreat                                        (push home, leave)

LTLf:  G !bad & F(grasped & F(staged & F(grasped & F(inserted & F(seated & released)))))

Stress test (two designed disturbances, identical for every method on a seed): the standing
peg is shoved 2-4 cm before the FIRST grasp (inherited), and the grasp force collapses for
0.25-0.35 s during the final approach to the BOARD (regrasped peg, tip < 5 cm above the board):
the peg drops onto the board or into the hole and must be regrasped / released and pushed.
"""
from __future__ import annotations

import mujoco
import numpy as np

from eddies.envs.assets.peg_board import add_hole_block, add_square_peg
from eddies.envs.controllers.franka import MAX_APERTURE
from eddies.envs.framework.perception import PoseEstimator
from eddies.envs.framework.scene import body_geoms, subtree_geoms
from eddies.envs.framework.task import OK, OutcomeSpec, Skill, SkillSpec, word
from eddies.envs.tasks import franka_peg as P
from eddies.envs.tasks.franka_base import TABLE_H, FrankaTaskBase
from eddies.envs.tasks.franka_peg import FrankaPegInsertionTask

# ------------------------------------------------------------------ geometry [m]
PEG_SIDE, PEG_LEN = P.PEG_SIDE, P.PEG_LEN
BOARD_W = 0.090
BOARD_FLOOR = 0.012
LATCH_TRAVEL = 0.012
LATCH_THICK = 0.010
LATCH_DEPTH0 = 0.047                    # hole top -> plunger top with the latch UP
BOARD_H = BOARD_FLOOR + LATCH_TRAVEL + LATCH_THICK + LATCH_DEPTH0         # 0.081
SEAT_DEPTH = LATCH_DEPTH0 + LATCH_TRAVEL                                   # 0.059 when latched
LATCH_K = 500.0                         # N/m spring of the plunger (a position actuator with a
LATCH_PRELOAD = 12.0                    # constant set-point above its travel = preloaded spring)
LATCH_SETPOINT = LATCH_PRELOAD / LATCH_K                                   # 0.024 m above the top stop
LATCH_FRICTION = 1.0                    # N, small joint friction loss
LATCH_DAMPING = 30.0                    # N s / m
FIX_CLEAR = 0.006
FIX_H, FIX_FLOOR, FIX_CHAMFER = 0.032, 0.012, 0.004
FIX_DEPTH = FIX_H - FIX_FLOOR           # 0.020
FIX_XY = np.array([0.40, 0.0])          # fixed, known fixture pose
PEG_NOM = np.array([0.50, -0.16])
HOLE_NOM = np.array([0.52, 0.15])
F_PUSH = 25.0                           # N force limit of the push
V_PUSH = 0.015                          # m/s descent of the push reference
FMB_YAW = np.radians(15.0)
BLOCKED = {"t_start": float("inf"), "duration_s": 0.0, "blocked_until_staged": True}


class FrankaFMBTask(FrankaPegInsertionTask):
    name = "franka_fmb"
    description = ("FMB-style insertion: the Franka picks a square peg from the table, places it on a "
                   "fixture, regrasps it, aligns it with a randomly posed board (+-15 deg), inserts it "
                   "with force-limited compliance (1-2 mm clearance) and pushes it home against a latch.")
    horizon = 90.0
    success_hold = 1.0
    formula = "G !bad & F(grasped & F(staged & F(grasped & F(inserted & F(seated & released)))))"
    terminal_cells = ("done",)
    failure_budget = 2

    def __init__(self, seed: int, **variant):
        self._hole_bias_test = np.asarray(variant.pop("hole_estimate_bias", (0.0, 0.0)), float)
        FrankaTaskBase.__init__(self, seed, **variant)
        rng = self.rng
        self.peg0 = PEG_NOM + np.r_[rng.uniform(-0.07, 0.07), rng.uniform(-0.05, 0.05)]
        self.hole0 = HOLE_NOM + np.r_[rng.uniform(-0.06, 0.06), rng.uniform(-0.05, 0.05)]
        self.peg_yaw0 = rng.uniform(-0.6, 0.6)
        self.hole_yaw0 = rng.uniform(-FMB_YAW, FMB_YAW)
        self.peg_mass = P.PEG_MASS_NOM * rng.uniform(0.94, 1.06)
        self.mu = rng.uniform(0.30, 0.45)
        self.clearance = rng.uniform(0.0010, 0.0020)
        self.chamfer = rng.uniform(0.0020, 0.0030)
        sb = self.make_builder(self.name)
        self.hb = add_hole_block(sb, "board", self.hole0, self.hole_yaw0, TABLE_H,
                                 hole_side=PEG_SIDE + self.clearance, outer=BOARD_W, height=BOARD_H,
                                 bottom=BOARD_FLOOR, chamfer=self.chamfer,
                                 rgba=(0.30, 0.55, 0.85, 0.40))
        self.fb = add_hole_block(sb, "fixture", FIX_XY, 0.0, TABLE_H, hole_side=PEG_SIDE + FIX_CLEAR,
                                 outer=0.070, height=FIX_H, bottom=FIX_FLOOR, chamfer=FIX_CHAMFER,
                                 rgba=(0.85, 0.75, 0.35, 0.55))
        # latch plunger inside the board hole: slide joint along the hole axis
        board = sb.spec.body("board")
        z_up = BOARD_FLOOR + LATCH_TRAVEL + LATCH_THICK / 2
        latch = board.add_body(name="latch", pos=[0.0, 0.0, z_up])
        latch.add_joint(name="latch_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                        range=[-LATCH_TRAVEL, 0.0], limited=True, frictionloss=LATCH_FRICTION,
                        damping=LATCH_DAMPING, armature=0.01, solref_limit=[0.002, 1.0])
        spring = sb.spec.add_actuator()
        spring.name = "latch_spring"
        spring.target = "latch_slide"
        spring.trntype = mujoco.mjtTrn.mjTRN_JOINT
        spring.set_to_position(kp=LATCH_K, kv=0.0)
        spring.ctrllimited = False
        half = (PEG_SIDE + self.clearance) / 2 - 0.0012
        latch.add_geom(name="latch_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, LATCH_THICK / 2],
                       mass=0.02, friction=[0.3, 0.003, 1e-4], condim=3, rgba=[0.9, 0.2, 0.2, 1.0])
        add_square_peg(sb, "peg", np.r_[self.peg0, TABLE_H], self.peg_yaw0, side=PEG_SIDE, length=PEG_LEN,
                       mass=self.peg_mass, friction=(self.mu, 0.003, 1e-4))
        sb.camera("main", (1.55, -1.05, TABLE_H + 1.0), (0.42, 0.0, TABLE_H + 0.18), fovy=45)
        c = self.hb.entrance
        sb.camera("detail", c + np.array([0.16, -0.20, 0.065]), c + np.array([0.0, 0.0, 0.010]), fovy=42)
        sb.camera("top", (0.48, 0.0, TABLE_H + 1.3), (0.48, 0.0, TABLE_H), fovy=50)
        self.finish_build(sb)
        m = self.model
        self.peg_bid = m.body("peg").id
        self.latch_qadr = int(m.jnt_qposadr[m.joint("latch_slide").id])
        self.latch_act = int(m.actuator("latch_spring").id)
        self._seated_flag = False
        self.block_geoms = body_geoms(m, ["board", "latch"])
        self.fixture_geoms = body_geoms(m, ["fixture"])
        self._build_contact_monitor()
        self.hand_mask = np.zeros(m.ngeom, bool)
        self.hand_mask[[g for g in subtree_geoms(m, self.arm.hand_body)
                        if m.geom_contype[g] or m.geom_conaffinity[g]]] = True
        self.est_peg = PoseEstimator(rng, near_distance=0.20)
        self.est_hole = PoseEstimator(rng, near_distance=0.15)
        self._contacts, self._geo = {}, {}
        self._plan_mode = "parent"
        self._build_skills()
        self._push_dir = rng.uniform(-np.pi, np.pi)
        self._push_delay = rng.uniform(0.0, 0.25)
        self._push_slide = rng.uniform(0.02, 0.04)
        self._drop_delay = rng.uniform(0.1, 0.5)
        self._drop_dur = rng.uniform(0.25, 0.35)
        self._pert = {"push": None, "dropout": dict(BLOCKED)}
        self._grip_range = self.model.actuator_forcerange[self.arm.act_grip].copy()
        self._ft_rng = np.random.default_rng(900_000 + self.seed)
        self._shadow = mujoco.MjData(m)
        self._jacp = np.zeros((3, m.nv))
        self._f6 = np.zeros(6)
        self._staged_once = False
        self._reset_decoder()

    # ============================================================ scene helpers
    def _build_contact_monitor(self):
        from eddies.envs.framework.contacts import ContactMonitor
        m = self.model
        fingers = self.arm.finger_geoms["left"] + self.arm.finger_geoms["right"]
        arm_only = [g for g in self.arm.robot_geoms if g not in fingers and (m.geom_contype[g] or m.geom_conaffinity[g])]
        groups = {"L": self.arm.finger_geoms["left"], "R": self.arm.finger_geoms["right"], "fingers": fingers,
                  "arm": arm_only, "table": self.table_geoms, "peg": body_geoms(m, ["peg"]),
                  "block": list(self.block_geoms) + list(self.fixture_geoms)}
        pairs = {"L_peg": ("L", "peg"), "R_peg": ("R", "peg"), "arm_peg": ("arm", "peg"),
                 "fingers_table": ("fingers", "table"), "arm_table": ("arm", "table"),
                 "fingers_block": ("fingers", "block"), "arm_block": ("arm", "block"),
                 "peg_block": ("peg", "block"), "peg_table": ("peg", "table")}
        self.cm = ContactMonitor(m, groups, pairs)

    def reset(self) -> None:
        self.data.ctrl[self.latch_act] = LATCH_SETPOINT        # constant spring set-point
        super().reset()
        self._staged_once = False
        self._seated_flag = False

    # ================================================================ state
    def latch_pos(self) -> float:
        return float(self.data.qpos[self.latch_qadr])            # 0 = up, -LATCH_TRAVEL = latched

    def _geometry(self):
        if self._geo.get("t") == self.data.time:
            return self._geo
        g = super()._geometry()
        # pushing on the top with closed fingers loads both pads too: a GRASP also needs the
        # hand open around the peg (aperture close to the peg width)
        g["grasp"] = bool(g["grasp"] and self.arm.aperture() > PEG_SIDE - 0.006)
        tip = g["tip"]
        fl = self.fb.to_local(tip)
        in_fix = (max(abs(fl[0]), abs(fl[1])) < (self.fb.hole_side - PEG_SIDE) / 2 + 0.003
                  and self.fb.top_z - tip[2] > FIX_DEPTH - 0.006 and g["tilt"] < np.radians(10))
        q = self.latch_pos()
        bottomed = bool(g["in_hole_xy"] and q < -LATCH_TRAVEL + 0.002 and g["depth"] > SEAT_DEPTH - 0.004)
        self._seated_flag = self._seated_flag or bottomed           # latched switch
        seated = bool(self._seated_flag and g["inserted"])
        g.update({"in_fixture": bool(in_fix), "latch_q": q, "seated": seated, "bottomed": bottomed})
        return g

    def observe(self) -> dict:
        obs = super().observe()
        g = self._geometry()
        obs.update({"in_fixture": g["in_fixture"], "seated": g["seated"], "latch_q": g["latch_q"],
                    "fixture_top": np.r_[FIX_XY, TABLE_H + FIX_H]})
        return obs

    def labels(self) -> set:
        out = super().labels()
        g = self._geometry()
        f = g["f"]
        released = self._released(f["L_peg"] + f["R_peg"], self.arm.aperture())
        if g["in_fixture"] and not g["grasp"] and released:
            out.add("staged")
            self._staged_once = True
        if g["seated"]:
            out.add("seated")
        return out

    def task_success(self) -> bool:
        g = self._geometry()
        f = g["f"]
        tcp = self.arm.tcp_pos()
        return bool(g["seated"] and not g["grasp"] and self._released(f["L_peg"] + f["R_peg"], self.arm.aperture())
                    and tcp[2] > g["top"][2] + 0.05 and np.linalg.norm(g["vel"]) < 0.03)

    def cell(self, obs: dict, labels: frozenset) -> str:
        tip, top, tcp = obs["tip_true"], obs["top_true"], obs["tcp"]
        grasp, ap = obs["grasp"], obs["aperture"]
        if obs["seated"]:
            if grasp:
                return "inserted_held"
            closed_on_top = ap < 0.02 and np.linalg.norm(tcp[:2] - top[:2]) < 0.02 and tcp[2] - top[2] < 0.06
            if closed_on_top:
                return "pushed"
            return "done" if tcp[2] > top[2] + 0.05 and ap > PEG_SIDE + 0.006 else "pushed"
        if obs["in_fixture"]:
            if grasp:
                return "in_fixture_held"
            # open hand around / above the standing peg: generic pick cells (regrasp context)
            base = super().cell(obs, labels)
            return base if base in ("at_peg", "above_peg") else "staged"
        if grasp and not obs["inserted"]:
            fl = self.fb.to_local(tip)
            dxy = np.linalg.norm(tip[:2] - self.fb.center_xy)
            if dxy < 0.006 and 0.0 < tip[2] - self.fb.top_z <= 0.035:
                return "over_fixture"
            if max(abs(fl[0]), abs(fl[1])) < 0.035 and tip[2] - self.fb.top_z <= 0.0:
                return "fixture_jam"
        base = super().cell(obs, labels)
        if base == "released_in_hole" or (base == "done" and not obs["seated"]):
            return "released_in_hole"
        return base

    # =============================================================== skills
    def nominal_plan(self) -> list:
        if getattr(self, "_plan_mode", "fmb") == "parent":
            return super().nominal_plan()
        return ["reach_above_peg", "descend_peg", "grasp_peg", "lift_peg", "move_over_fixture",
                "place_in_fixture", "release", "reach_above_peg", "descend_peg", "grasp_peg", "lift_peg",
                "move_over_hole", "insert", "release", "push_to_seat", "retreat"]

    def _build_skills(self):
        self._plan_mode = "parent"
        super()._build_skills()                    # pick / insert / recovery / release / retreat
        self._plan_mode = "fmb"
        S = self._skills
        fix_top = np.r_[FIX_XY, TABLE_H + FIX_H]
        fix_floor = TABLE_H + FIX_FLOOR

        def near(obs, pos, tol):
            return np.linalg.norm(obs["tcp"] - pos) < tol

        def over_fixture_tcp(obs, tip_height):
            off = self._inhand(obs)
            return np.r_[fix_top[:2] - off[:2], fix_top[2] + tip_height - off[2]]

        def dropped(obs, st):
            held = (obs["contact_L"] + obs["contact_R"] > 0.2 or
                    np.linalg.norm(obs["tip_true"] - obs["tcp"]) < P.TIP_BELOW_TCP + 0.03)
            if held:
                st.memory["lost"] = None
                return False
            st.memory["lost"] = st.memory.get("lost") or obs["t"]
            return obs["t"] - st.memory["lost"] > 0.1 and (obs["peg_rest_s"] > 0.25 or obs["t"] - st.memory["lost"] > 2.0)

        def drop_outcome(obs, st):
            if obs["in_fixture"]:
                return "dropped_in_fixture"
            if obs["peg_tilt"] > P.STAND_TILT:
                return "toppled"
            return "dropped"

        # move the lifted peg over the fixture pocket (faces aligned with the pocket, yaw 0)
        def t_move_fix(obs, st):
            yaw = obs["tcp_yaw"] + P.wrap_to(-obs["inhand_yaw"] - obs["tcp_yaw"], np.pi / 2)
            return self.cmd(over_fixture_tcp(obs, 0.025), float(np.clip(yaw, -P.YAW_MAX, P.YAW_MAX)), 0.0)

        def o_move_fix(obs, labels, st):
            if dropped(obs, st):
                return drop_outcome(obs, st)
            ok = (obs["grasp"] and np.linalg.norm(obs["tip_true"][:2] - fix_top[:2]) < 0.004
                  and near(obs, over_fixture_tcp(obs, 0.025), 0.005))
            return OK if ok else None

        # lower into the loose pocket until the tip rests on the fixture floor
        def e_place(obs, st):
            st.memory["zr"] = float(obs["tcp"][2])
            st.memory["tl"] = float(obs["t"])
            st.memory["yaw"] = obs["tcp_yaw"]

        def t_place(obs, st):
            off = self._inhand(obs)
            target_z = fix_floor - 0.004 - off[2]
            t = float(obs["t"])
            st.memory["zr"] = max(target_z, st.memory["zr"] - 0.03 * (t - st.memory["tl"]))
            st.memory["tl"] = t
            return self.cmd(np.r_[fix_top[:2] - off[:2], st.memory["zr"]], st.memory["yaw"], 0.0, 8.0)

        def o_place(obs, labels, st):
            if dropped(obs, st):
                return drop_outcome(obs, st)
            if obs["grasp"] and obs["in_fixture"] and obs["tip_true"][2] < fix_floor + 0.004:
                return OK
            if obs["grasp"] and st.elapsed(obs["t"]) > 4.0 and not obs["in_fixture"]:
                return "missed_fixture"
            return None

        # push the peg home against the latch with the closed fingers (force-limited)
        def e_push(obs, st):
            st.memory["phase"] = "rise"
            st.memory["xy"] = obs["top_true"][:2].copy()
            st.memory["zr"] = float(max(obs["tcp"][2], obs["top_true"][2] + 0.03))   # rise target
            st.memory["tl"] = float(obs["t"])
            st.memory["yaw"] = obs["tcp_yaw"]
            st.memory["hist"] = []

        def t_push(obs, st):
            t = float(obs["t"])
            xy = st.memory["xy"]
            if st.memory["phase"] == "rise":           # open hand clears the peg top first
                if obs["tcp"][2] > obs["top_true"][2] + 0.024:
                    st.memory["phase"] = "close"
                    st.memory["t_close"] = t
                st.memory["tl"] = t
                return self.cmd(np.r_[xy, st.memory["zr"]], st.memory["yaw"], MAX_APERTURE, 0.0)
            if st.memory["phase"] == "close":
                if obs["aperture"] < 0.004 or t - st.memory["t_close"] > 1.2:
                    st.memory["phase"] = "press"
                st.memory["tl"] = t
                return self.cmd(np.r_[xy, st.memory["zr"]], st.memory["yaw"], 0.0, 0.0)
            st.memory["zr"] = max(obs["top_true"][2] - 0.03, st.memory["zr"] - V_PUSH * (t - st.memory["tl"]))
            st.memory["tl"] = t
            return self.cmd(np.r_[xy, st.memory["zr"]], st.memory["yaw"], 0.0, F_PUSH)

        def o_push(obs, labels, st):
            if obs["seated"] and st.memory["phase"] == "press" and obs["latch_q"] < -LATCH_TRAVEL + 0.003:
                return OK
            hist = st.memory["hist"]
            hist.append((obs["t"], obs["latch_q"]))
            while hist and hist[0][0] < obs["t"] - 1.0:
                hist.pop(0)
            if (st.memory["phase"] == "press" and hist and hist[-1][0] - hist[0][0] > 0.9
                    and abs(hist[-1][1] - hist[0][1]) < 0.0005 and obs["f_up"] > 0.6 * F_PUSH):
                return "stuck"
            return None

        # retreat after the push: open and move up
        def e_hold(obs, st):
            st.memory["hold"] = obs["tcp"].copy()
            st.memory["yaw"] = obs["tcp_yaw"]

        def t_retreat2(obs, st):
            return self.cmd(st.memory["hold"] + [0, 0, 0.12], st.memory["yaw"], MAX_APERTURE, 0.0)

        def o_retreat2(obs, labels, st):
            return OK if obs["tcp"][2] > obs["top_true"][2] + 0.08 and obs["aperture"] > PEG_SIDE + 0.01 else None

        # clearance-aware grasp yaw: of the two face-aligned yaws (square peg, 90 deg apart)
        # take the one whose open fingers stay farthest from the board and the fixture
        def finger_clearance(center_xy, yaw, tip_z):
            axis = np.array([-np.sin(yaw), np.cos(yaw)])          # finger closing axis
            worst = np.inf
            for sgn in (-1.0, 1.0):
                f = center_xy + sgn * (MAX_APERTURE / 2 + 0.012) * axis
                for blk, half in ((self.hb, BOARD_W / 2), (self.fb, 0.035)):
                    if tip_z > blk.top_z + 0.005:                 # fingertips pass above this block
                        continue
                    loc = blk.to_local(np.r_[f, 0.0])[:2]
                    out = np.maximum(np.abs(loc) - half, 0.0)
                    worst = min(worst, float(np.linalg.norm(out)) if out.any() else -float(np.min(half - np.abs(loc))))
            return worst

        def safe_yaw(obs):
            y0 = P.wrap_to(obs["peg_yaw"], np.pi / 2)
            c = obs["tip_est"][:2]
            tip_z = obs["tip_est"][2] + PEG_LEN - P.GRASP_BELOW_TOP - 0.012     # fingertip height at the grasp
            if finger_clearance(c, y0, tip_z) > 0.01:
                return float(y0)                                   # default grasp is collision-free
            cands = [y for y in (y0 + np.pi / 2, y0 - np.pi / 2) if abs(y) <= P.YAW_MAX] + [y0]
            return float(max(cands, key=lambda y: finger_clearance(c, y, tip_z)))

        def peg_top_est(obs):
            return obs["tip_est"] + np.array([0.0, 0.0, PEG_LEN])

        def t_reach_safe(obs, st):
            if "gy" not in st.memory:
                st.memory["gy"] = safe_yaw(obs)
            return self.cmd(peg_top_est(obs) + [0, 0, P.H_ABOVE], st.memory["gy"], MAX_APERTURE)

        def t_descend_safe(obs, st):
            if "gy" not in st.memory:
                st.memory["gy"] = safe_yaw(obs)
            return self.cmd(peg_top_est(obs) - [0, 0, P.GRASP_BELOW_TOP], st.memory["gy"], MAX_APERTURE)

        S["reach_above_peg"].target = t_reach_safe
        S["descend_peg"].target = t_descend_safe

        hole_top = np.r_[HOLE_NOM, TABLE_H + BOARD_H]
        nominal = dict(self._nominal_anchors)
        nominal.update({
            "move_over_fixture": self.cmd(fix_top + [0, 0, 0.025 + P.TIP_BELOW_TCP], 0, 0.0),
            "place_in_fixture": self.cmd(np.r_[FIX_XY, fix_floor + P.TIP_BELOW_TCP], 0, 0.0, 8.0),
            "move_over_hole": self.cmd(hole_top + [0, 0, P.PRE_INSERT + P.TIP_BELOW_TCP], 0, 0.0),
            "insert": self.cmd(hole_top + [0, 0, -LATCH_DEPTH0 + P.TIP_BELOW_TCP], 0, 0.0, P.F_INSERT),
            "back_off": self.cmd(hole_top + [0, 0, P.BACKOFF + P.TIP_BELOW_TCP], 0, 0.0, 0.0),
            "spiral_insert": self.cmd(hole_top + [0, 0, -LATCH_DEPTH0 + P.TIP_BELOW_TCP], 0, 0.0, P.F_SEARCH),
            "release": self.cmd(hole_top + [0, 0, -LATCH_DEPTH0 + P.TIP_BELOW_TCP], 0, MAX_APERTURE, 0.0),
            "push_to_seat": self.cmd(hole_top + [0, 0, PEG_LEN - SEAT_DEPTH], 0, 0.0, F_PUSH),
            "retreat": self.cmd(hole_top + [0, 0, PEG_LEN - SEAT_DEPTH + 0.12], 0, MAX_APERTURE, 0.0),
        })
        # hand-set trust radii (as in every task): distance to the previous anchor + 0.32
        deltas = {n: s.delta for n, s in S.items()}
        prev = self.cmd([0.474, 0.0, TABLE_H + 0.41], 0, MAX_APERTURE)
        for name in self.nominal_plan():
            deltas[name] = max(deltas.get(name, 0.0), self.space.norm(nominal[name] - prev) + 0.32)
            prev = nominal[name]
        for name in ("release", "retreat", "reach_above_peg"):     # used in several contexts
            deltas[name] = max(deltas[name], 1.2)
        new = [("move_over_fixture", t_move_fix, o_move_fix, None, 10.0, 2.5),
               ("place_in_fixture", t_place, o_place, e_place, 6.0, 1.5),
               ("push_to_seat", t_push, o_push, e_push, 8.0, 2.5),
               ("retreat", t_retreat2, o_retreat2, e_hold, 6.0, 1.5)]
        for name, tgt, out, enter, timeout, dur in new:
            S[name] = Skill(name, nominal[name], deltas[name], tgt, out, timeout=timeout,
                            nominal_duration=dur, on_enter=enter)
        for name, sk in S.items():
            sk.anchor = np.asarray(nominal[name], float)
            sk.delta = float(deltas[name])
        self._nominal_anchors = nominal

    def skill_graph(self) -> list:
        e, r, g = word(""), word("released"), word("grasped")
        st = word("staged")
        gi = {"grasped", "inserted"}
        drops = (OutcomeSpec("dropped", "free", word("grasped", ""), failure=True),
                 OutcomeSpec("dropped_in_fixture", "staged", word("grasped", ""), failure=True))
        drops_hole = (OutcomeSpec("dropped", "free", word("grasped", ""), failure=True),
                      OutcomeSpec("dropped_in_hole", "in_hole_unreleased", word("grasped", "", "inserted"), failure=True))
        return [
            # pick from the table / regrasp from the fixture (the same tokens)
            SkillSpec("reach_above_peg", "free", (OutcomeSpec(OK, "above_peg", word("", "released")),)),
            SkillSpec("reach_above_peg", "staged", (OutcomeSpec(OK, "above_peg", word({"staged", "released"})),
                                                    OutcomeSpec(OK, "above_peg", word("", {"staged", "released"})))),
            SkillSpec("reach_above_peg", "at_peg", (OutcomeSpec(OK, "above_peg", r),)),
            SkillSpec("descend_peg", "above_peg", (OutcomeSpec(OK, "at_peg", r),)),
            SkillSpec("grasp_peg", "at_peg", (OutcomeSpec(OK, "grasped", word("released", "")),
                                              OutcomeSpec(OK, "in_fixture_held", word("released", "", "grasped")),
                                              OutcomeSpec("missed", "free", word("released", ""), failure=True),
                                              OutcomeSpec("missed", "staged", word("released", ""), failure=True))),
            SkillSpec("lift_peg", "grasped", (OutcomeSpec(OK, "lifted", word("", "grasped")),
                                              OutcomeSpec("dropped", "free", word("", "grasped", ""), failure=True),
                                              OutcomeSpec("slipped", "free", e, failure=True))),
            SkillSpec("lift_peg", "in_fixture_held", (OutcomeSpec(OK, "lifted", g),
                                                      OutcomeSpec("dropped_in_fixture", "staged", word("", "grasped", ""), failure=True),
                                                      OutcomeSpec("slipped", "staged", e, failure=True))),
            # place on the fixture
            SkillSpec("move_over_fixture", "lifted", (OutcomeSpec(OK, "over_fixture", g),) + drops),
            SkillSpec("place_in_fixture", "over_fixture", (OutcomeSpec(OK, "in_fixture_held", g),
                                                           OutcomeSpec("missed_fixture", "fixture_jam", g, failure=True)) + drops),
            SkillSpec("move_over_fixture", "fixture_jam", (OutcomeSpec(OK, "over_fixture", g),) + drops),
            SkillSpec("release", "in_fixture_held", (OutcomeSpec(OK, "staged", word("", {"staged", "released"})),
                                                     OutcomeSpec(OK, "above_peg", word("", {"staged", "released"})),
                                                     OutcomeSpec(OK, "at_peg", word("", {"staged", "released"})))),
            # align, insert, recover
            SkillSpec("move_over_hole", "lifted", (OutcomeSpec(OK, "over_hole", g),) + drops_hole),
            SkillSpec("insert", "over_hole", (OutcomeSpec(OK, "inserted_held", word("grasped", gi)),
                                              OutcomeSpec("jammed", "jammed", g, failure=True)) + drops_hole),
            SkillSpec("back_off", "jammed", (OutcomeSpec(OK, "near_hole", g),) + drops_hole),
            SkillSpec("spiral_insert", "near_hole", (OutcomeSpec(OK, "inserted_held", word("grasped", gi)),
                                                     OutcomeSpec("jammed", "jammed", g, failure=True)) + drops_hole),
            SkillSpec("release", "inserted_held",
                      (OutcomeSpec(OK, "released_in_hole", word(gi, "inserted", {"inserted", "released"})),)),
            SkillSpec("release", "in_hole_unreleased",
                      (OutcomeSpec(OK, "released_in_hole", word("inserted", {"inserted", "released"})),)),
            # push home, leave
            SkillSpec("push_to_seat", "released_in_hole",
                      (OutcomeSpec(OK, "pushed", word({"inserted", "released"}, "inserted", {"inserted", "seated"})),
                       OutcomeSpec("stuck", "released_in_hole", word({"inserted", "released"}, "inserted"), failure=True))),
            SkillSpec("retreat", "pushed", (OutcomeSpec(OK, "done", word({"inserted", "seated"},
                                                                          {"inserted", "seated", "released"})),)),
        ]

    # ========================================================== stress test
    def disturb(self) -> None:
        """(1) shove the standing peg 2-4 cm before the FIRST grasp (inherited trigger: open hand
        hovering above the free peg); (2) grasp-force dropout while the REGRASPED peg is carried
        to the board: the inherited dropout trigger is armed only once the peg has been staged."""
        d = self._pert.get("dropout")
        if self._staged_once and isinstance(d, dict) and d.get("blocked_until_staged"):
            g = self._geometry()
            near_board = (np.linalg.norm(g["tip"][:2] - self.hb.center_xy) < 0.05
                          and g["tip"][2] - self.hb.top_z < 0.05)
            if g["grasp"] and near_board:                     # final approach to the board
                self._pert["dropout"] = None
        super().disturb()

    def perturbation_info(self) -> dict:
        return {k: v for k, v in self._pert.items() if not (isinstance(v, dict) and v.get("blocked_until_staged"))}

    def step_diagnostics(self) -> dict:
        out = super().step_diagnostics()
        out["latch_position_m"] = self.latch_pos()
        return out

    def log_signals(self) -> dict:
        out = super().log_signals()
        out["latch"] = self.latch_pos()
        return out

    def episode_metrics(self) -> dict:
        out = super().episode_metrics()
        g = self._geometry()
        out.update({"final_seated": g["seated"], "final_latch_m": g["latch_q"]})
        return out

    def variant_info(self) -> dict:
        out = super().variant_info()
        out.update({"fixture_xy": FIX_XY, "board_height_m": BOARD_H, "latch_spring_N_per_m": LATCH_K,
                    "latch_preload_N": LATCH_PRELOAD, "latch_friction_N": LATCH_FRICTION,
                    "latch_travel_m": LATCH_TRAVEL, "seat_depth_m": SEAT_DEPTH})
        return out

    def overlay_lines(self) -> list:
        return super().overlay_lines() + [f"latch {1e3 * self.latch_pos():6.1f} mm (seated at "
                                          f"{-1e3 * LATCH_TRAVEL:.0f} mm)   staged once: {self._staged_once}"]
