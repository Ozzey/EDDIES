"""Franka tasks composed ONLY of skillbook Eddies (action tokens, ``eddies.envs.skillbook``).

Each task supplies perception frames (obs -> point / yaw), an object view and its skill
graph; every Eddie is an instantiation of one of the 11 skillbook primitives.  The physics,
perception, labels, cells and disturbances are inherited unchanged from the hand-written
task (registered as ``*_legacy``); only the skills are replaced.

Eddies (primitive:binding), 35 over 10 primitives:
    peg / FMB   reach:peg_above  reach:peg_grasp  grasp:peg  lift:peg  release:peg  retreat:peg
                transport:hole_above  transport:hole_backoff  insert:hole  insert:hole_search
                transport:fixture_above  place:fixture  push:peg_top            (FMB only)
    stack       reach:cube_above  reach:cube_grasp  grasp:cube  lift:cube  transport:cube_above
                place:cube_top  release:cube  retreat:cube
    drawer      reach:handle_pre  reach:handle  grasp:handle  pull:drawer  release:handle
                retreat:handle  reach:box_above  reach:box_grasp  grasp:box  lift:box
                transport:goal_above  place:goal  release:box  retreat:box
"""
from __future__ import annotations

import dataclasses

import numpy as np

from eddies.envs.controllers.franka import MAX_APERTURE
from eddies.envs.framework.perception import wrap_to
from eddies.envs.skillbook import BOOK
from eddies.envs.skillbook.primitives import ObjectView
from eddies.envs.tasks import franka_drawer as D
from eddies.envs.tasks import franka_fmb as F
from eddies.envs.tasks import franka_peg as P
from eddies.envs.tasks import franka_stack as C
from eddies.envs.tasks.franka_base import TABLE_H

# legacy skill name -> skillbook Eddie (the task's LTLf / skill-graph semantics are unchanged)
TOKENS = {
    "reach_above_peg": "reach:peg_above",
    "descend_peg": "reach:peg_grasp",
    "grasp_peg": "grasp:peg",
    "lift_peg": "lift:peg",
    "move_over_hole": "transport:hole_above",
    "insert": "insert:hole",
    "back_off": "transport:hole_backoff",
    "spiral_insert": "insert:hole_search",
    "release": "release:peg",
    "retreat": "retreat:peg",
    "move_over_fixture": "transport:fixture_above",
    "place_in_fixture": "place:fixture",
    "push_to_seat": "push:peg_top",
}
HOME = ([0.474, 0.0, TABLE_H + 0.41], 0, MAX_APERTURE)


def rename_graph(graph, names=None):
    names = names or TOKENS
    return [dataclasses.replace(s, skill=names[s.skill]) for s in graph]


# ----------------------------------------------------------------- shared peg object view
def peg_view(task) -> ObjectView:
    def held(obs):
        return bool(obs["grasp"])

    def lost(obs, st):
        """The peg physically left the fingers (no pad force AND tip away from the hand),
        decided once it has settled (robust to transient force dips)."""
        near = (obs["contact_L"] + obs["contact_R"] > 0.2 or
                np.linalg.norm(obs["tip_true"] - obs["tcp"]) < P.TIP_BELOW_TCP + 0.03)
        if near:
            st.memory["lost"] = None
            return False
        st.memory["lost"] = st.memory.get("lost") or obs["t"]
        return obs["t"] - st.memory["lost"] > 0.1 and (obs["peg_rest_s"] > 0.25 or obs["t"] - st.memory["lost"] > 2.0)

    def drop_outcome(obs, st):
        """Classify a drop from the MEASURED resting state of the peg."""
        if obs.get("in_fixture"):
            return "dropped_in_fixture"
        if obs["inserted"]:
            return "dropped_in_hole"
        if obs["peg_tilt"] > P.STAND_TILT:
            return "toppled"
        return "dropped" if st.memory.get("seen_grasped") else "slipped"

    def toppled(obs):
        t = (obs["peg_tilt"] > P.STAND_TILT and not obs["grasp"]
             and (obs["peg_rest_s"] > 0.25 or obs["peg_tilt"] > np.radians(60)))
        return "toppled" if t else None

    return ObjectView(tip=lambda o: o["tip_est"], top=lambda o: o["tip_est"] + [0, 0, P.PEG_LEN],
                      yaw=lambda o: o["peg_yaw"], held=held, lost=lost, drop_outcome=drop_outcome,
                      inhand=lambda o: o["inhand"], width=P.PEG_SIDE, true_tip=lambda o: o["tip_true"],
                      abort=toppled, symmetry=np.pi / 2)


def aligned_yaw(target_yaw):
    """ABSOLUTE hand yaw that aligns the held peg's faces with ``target_yaw(obs)`` (mod 90 deg),
    nearest the current hand yaw; the in-hand rotation is the latched wrist-camera measurement."""
    def f(obs):
        y = obs["tcp_yaw"] + wrap_to(target_yaw(obs) - obs["inhand_yaw"] - obs["tcp_yaw"], np.pi / 2)
        return float(np.clip(y, -P.YAW_MAX, P.YAW_MAX))
    return f


class _TokenTask:
    """Mixin: build Eddies from the skillbook, anchors / hand-set trust radii as in the legacy tasks."""
    multi_context = ()

    def _tokens(self, spec, nominal, recovery_from):
        S = {}
        for token, (args, timeout, dur) in spec.items():
            prim, binding = token.split(":", 1)
            S[token] = BOOK.eddy(self, prim, binding, anchor=nominal[token], delta=1.0, timeout=timeout,
                                  duration=dur, **args)
        # hand-set trust radii (as in every task): distance to the previous anchor + 0.32
        deltas = {}
        prev = self.cmd(*HOME)
        for token in self.nominal_plan():
            deltas[token] = max(deltas.get(token, 0.0), self.space.norm(nominal[token] - prev) + 0.32)
            prev = nominal[token]
        for token, frm in recovery_from.items():
            deltas[token] = self.space.norm(nominal[token] - nominal[frm]) + 0.32
        for token in self.multi_context:
            deltas[token] = max(deltas[token], 1.2)
        for token, sk in S.items():
            sk.delta = float(deltas[token])
        self._skills = S
        self._nominal_anchors = nominal

    def skills(self) -> dict:
        return self._skills

    def peg_tokens(self, view, hole_top_z):
        """Pick / insert / recover / release Eddies shared by the peg and FMB tasks."""
        top = view.top
        grasp_yaw = getattr(self, "_grasp_yaw", lambda o: wrap_to(o["peg_yaw"], np.pi / 2))
        fix_yaw = getattr(self, "_fix_grasp_yaw", False)
        ins_yaw = aligned_yaw(lambda o: o["hole_yaw"])
        hole = lambda o: o["hole_est"]                                   # noqa: E731
        ins = dict(view=view, hole=hole, yaw=ins_yaw, depth=lambda o: o["depth"], inserted=lambda o: o["inserted"],
                   floor_depth=P.HOLE_DEPTH + 0.003, seat_depth=P.HOLE_DEPTH, min_depth=P.INSERT_MIN,
                   force=P.F_INSERT, search_force=P.F_SEARCH, v_fast=P.V_INSERT, v_search=P.V_SEARCH,
                   v_slow=P.V_SLOW, pitch=P.SPIRAL_PITCH, period=P.SPIRAL_PERIOD, rmax=P.SPIRAL_RMAX)
        return {
            "reach:peg_above": (dict(point=lambda o: top(o) + [0, 0, P.H_ABOVE], yaw=grasp_yaw, view=view,
                                     tol=0.008, fix_yaw=fix_yaw), 10.0, 2.5),
            "reach:peg_grasp": (dict(point=lambda o: top(o) - [0, 0, P.GRASP_BELOW_TOP], yaw=grasp_yaw, view=view,
                                     tol=0.005, open_min=None, fix_yaw=fix_yaw), 6.0, 1.5),
            "grasp:peg": (dict(view=view), 3.0, 1.2),
            "lift:peg": (dict(view=view, height=P.LIFT, surface=TABLE_H), 6.0, 1.5),
            "transport:hole_above": (dict(view=view, point=hole, height=P.PRE_INSERT, yaw=ins_yaw,
                                          align_yaw=lambda o: o["hole_yaw"]), 10.0, 2.5),
            "insert:hole": (dict(**ins), 10.0, 3.5),
            "transport:hole_backoff": (dict(view=view, point=hole, height=P.BACKOFF, yaw=ins_yaw, tol_xy=0.01,
                                            settle=0.3), 4.0, 1.0),
            "insert:hole_search": (dict(**ins, search=True), 14.0, 6.0),
            "release:peg": (dict(view=view), 3.0, 1.0),
            "retreat:peg": (dict(height=0.12, clear=lambda o: o["top_true"][2] + 0.08, open_min=P.PEG_SIDE + 0.01),
                            6.0, 1.5),
        }

    @staticmethod
    def peg_anchors(cmd, peg_nom, hole_nom, hole_top_z, floor_depth, release_depth):
        peg_top = np.r_[peg_nom, TABLE_H + P.PEG_LEN]
        hole_top = np.r_[hole_nom, hole_top_z]
        grasp_p = peg_top - [0, 0, P.GRASP_BELOW_TOP]
        tb = P.TIP_BELOW_TCP
        return {
            "reach:peg_above": cmd(peg_top + [0, 0, P.H_ABOVE], 0, MAX_APERTURE),
            "reach:peg_grasp": cmd(grasp_p, 0, MAX_APERTURE),
            "grasp:peg": cmd(grasp_p, 0, 0.0),
            "lift:peg": cmd(grasp_p + [0, 0, P.LIFT], 0, 0.0),
            "transport:hole_above": cmd(hole_top + [0, 0, P.PRE_INSERT + tb], 0, 0.0),
            "insert:hole": cmd(hole_top + [0, 0, -floor_depth + tb], 0, 0.0, P.F_INSERT),
            "transport:hole_backoff": cmd(hole_top + [0, 0, P.BACKOFF + tb], 0, 0.0, 0.0),
            "insert:hole_search": cmd(hole_top + [0, 0, -floor_depth + tb], 0, 0.0, P.F_SEARCH),
            "release:peg": cmd(hole_top + [0, 0, -release_depth + tb], 0, MAX_APERTURE, 0.0),
            "retreat:peg": cmd(hole_top + [0, 0, -release_depth + tb + 0.12], 0, MAX_APERTURE, 0.0),
        }



# ======================================================================= peg insertion
class FrankaPegTokens(_TokenTask, P.FrankaPegInsertionTask):
    name = "franka_peg_insertion"

    def _build_skills(self):
        view = peg_view(self)
        hz = TABLE_H + P.BLOCK_H
        spec = self.peg_tokens(view, hz)
        nominal = self.peg_anchors(self.cmd, P.PEG_NOM, P.HOLE_NOM, hz, P.HOLE_DEPTH + 0.003, P.HOLE_DEPTH)
        self._tokens(spec, nominal, {"transport:hole_backoff": "insert:hole",
                                     "insert:hole_search": "transport:hole_backoff"})

    def nominal_plan(self) -> list:
        return [TOKENS[n] for n in P.FrankaPegInsertionTask.nominal_plan(self)]

    def skill_graph(self) -> list:
        return rename_graph(P.FrankaPegInsertionTask.skill_graph(self))


# ======================================================================= FMB insertion
class FrankaFMBTokens(_TokenTask, F.FrankaFMBTask):
    name = "franka_fmb"
    # Eddies used in two contexts (table pick and fixture regrasp): the hand-set radius covers both
    multi_context = ("release:peg", "retreat:peg", "reach:peg_above", "reach:peg_grasp", "grasp:peg", "lift:peg")
    _fix_grasp_yaw = True

    def _grasp_yaw(self, obs):
        """Clearance-aware grasp yaw: of the two face-aligned yaws (square peg, 90 deg apart)
        take the one whose open fingers stay farthest from the board and the fixture."""
        def clearance(center_xy, yaw, tip_z):
            axis = np.array([-np.sin(yaw), np.cos(yaw)])
            worst = np.inf
            for sgn in (-1.0, 1.0):
                f = center_xy + sgn * (MAX_APERTURE / 2 + 0.012) * axis
                for blk, half in ((self.hb, F.BOARD_W / 2), (self.fb, 0.035)):
                    if tip_z > blk.top_z + 0.005:
                        continue
                    loc = blk.to_local(np.r_[f, 0.0])[:2]
                    out = np.maximum(np.abs(loc) - half, 0.0)
                    worst = min(worst, float(np.linalg.norm(out)) if out.any() else -float(np.min(half - np.abs(loc))))
            return worst
        y0 = wrap_to(obs["peg_yaw"], np.pi / 2)
        c = obs["tip_est"][:2]
        tip_z = obs["tip_est"][2] + P.PEG_LEN - P.GRASP_BELOW_TOP - 0.012
        if clearance(c, y0, tip_z) > 0.01:
            return float(y0)
        cands = [y for y in (y0 + np.pi / 2, y0 - np.pi / 2) if abs(y) <= P.YAW_MAX] + [y0]
        return float(max(cands, key=lambda y: clearance(c, y, tip_z)))

    def _build_skills(self):
        self._plan_mode = "fmb"
        view = peg_view(self)
        hz = TABLE_H + F.BOARD_H
        fix_top = np.r_[F.FIX_XY, TABLE_H + F.FIX_H]
        fix_floor = TABLE_H + F.FIX_FLOOR
        spec = self.peg_tokens(view, hz)
        spec.update({
            "transport:fixture_above": (dict(view=view, point=lambda o: fix_top, height=0.025,
                                             yaw=aligned_yaw(lambda o: 0.0), tol_xy=0.004, tol_tcp=0.005,
                                             tip=view.true_tip), 10.0, 2.5),
            "place:fixture": (dict(view=view, point=lambda o: np.r_[fix_top[:2], fix_floor],
                                   placed=lambda o: o["in_fixture"] and o["tip_true"][2] < fix_floor + 0.004,
                                   force=8.0, speed=0.03, below=0.004, fail_after=4.0,
                                   fail_key="missed_fixture", fail_if=lambda o: not o["in_fixture"]), 6.0, 1.5),
            "push:peg_top": (dict(top=lambda o: o["top_true"],
                                  done=lambda o: o["seated"] and o["latch_q"] < -F.LATCH_TRAVEL + 0.003,
                                  progress_fn=lambda o: o["latch_q"], force=F.F_PUSH, speed=F.V_PUSH), 8.0, 2.5),
        })
        nominal = self.peg_anchors(self.cmd, F.PEG_NOM, F.HOLE_NOM, hz, F.LATCH_DEPTH0, F.LATCH_DEPTH0)
        hole_top = np.r_[F.HOLE_NOM, hz]
        tb = P.TIP_BELOW_TCP
        nominal.update({
            "transport:fixture_above": self.cmd(fix_top + [0, 0, 0.025 + tb], 0, 0.0),
            "place:fixture": self.cmd(np.r_[F.FIX_XY, fix_floor + tb], 0, 0.0, 8.0),
            "push:peg_top": self.cmd(hole_top + [0, 0, P.PEG_LEN - F.SEAT_DEPTH], 0, 0.0, F.F_PUSH),
            "retreat:peg": self.cmd(hole_top + [0, 0, P.PEG_LEN - F.SEAT_DEPTH + 0.12], 0, MAX_APERTURE, 0.0),
        })
        self._tokens(spec, nominal, {"transport:hole_backoff": "insert:hole",
                                     "insert:hole_search": "transport:hole_backoff"})

    def nominal_plan(self) -> list:
        return [TOKENS[n] for n in F.FrankaFMBTask.nominal_plan(self)]

    def skill_graph(self) -> list:
        return rename_graph(F.FrankaFMBTask.skill_graph(self))


# ======================================================================= cube stacking
STACK_TOKENS = {"reach_above_red": "reach:cube_above", "descend_red": "reach:cube_grasp", "grasp_red": "grasp:cube",
                "lift_red": "lift:cube", "move_over_blue": "transport:cube_above", "lower_onto_blue": "place:cube_top",
                "release": "release:cube", "retreat": "retreat:cube"}


class FrankaStackTokens(_TokenTask, C.FrankaCubeStackTask):
    name = "franka_cube_stack"

    def _build_skills(self):
        H, DZ = C.HALF, C.GRASP_DZ
        down = np.array([0, 0, H])

        def lost(obs, st):
            near = (obs["contact_L"] + obs["contact_R"] > 0.2 or np.linalg.norm(obs["red_true"] - obs["tcp"]) < 0.03)
            if near:
                st.memory["lost"] = None
                return False
            st.memory["lost"] = st.memory.get("lost") or obs["t"]
            return obs["t"] - st.memory["lost"] > 0.1

        # reference point of a cube = its bottom centre (the in-hand offset is measured)
        view = ObjectView(tip=lambda o: o["red_est"] - down, top=lambda o: o["red_est"] + down,
                          yaw=lambda o: o["red_yaw"], held=lambda o: bool(o["grasp"]), lost=lost,
                          drop_outcome=lambda o, st: "dropped", inhand=lambda o: o["red_true"] - down - o["tcp"],
                          width=2 * H, true_tip=lambda o: o["red_true"] - down)

        def grasp_yaw(obs):
            """Of the two finger-closing axes take the one most perpendicular to the neighbour cube."""
            y0 = obs["red_yaw"]
            d = obs["blue_est"][:2] - obs["red_est"][:2]
            d = d / (np.linalg.norm(d) + 1e-9)
            cands = [y0, wrap_to(y0 + np.pi / 2, np.pi)]
            return min(cands, key=lambda y: abs(np.sin(y) * d[0] - np.cos(y) * d[1]) - 1e-3 * abs(y))

        blue_top = lambda o: o["blue_est"] + down                                  # noqa: E731

        def blue_yaw(o):
            """Hand yaw aligning the held cube with the blue cube MODULO 90 deg (cubes are 90-deg
            symmetric), nearest the current hand yaw: |rotation| <= 45 deg.  (The hand-written task
            commanded the raw blue yaw: median 70 deg, up to 116 deg of needless rotation while
            carrying, which saturates the IK step and breaks the linear tracking model.)"""
            ty = self.tcp_yaw()
            inhand = wrap_to(o["red_yaw"] - ty, np.pi / 2)
            return float(np.clip(ty + wrap_to(o["blue_yaw"] - inhand - ty, np.pi / 2), -P.YAW_MAX, P.YAW_MAX))

        def seated(o):
            r, b = o["red_true"], o["blue_true"]
            return np.linalg.norm(r[:2] - b[:2]) < 0.008 and r[2] - b[2] < 2 * H + 0.003
        spec = {
            "reach:cube_above": (dict(point=lambda o: o["red_est"] + [0, 0, 0.12], yaw=grasp_yaw, fix_yaw=False), 10.0, 2.5),
            "reach:cube_grasp": (dict(point=lambda o: o["red_est"] + [0, 0, DZ], yaw=grasp_yaw, fix_yaw=False, tol=0.005,
                                      open_min=None), 6.0, 1.5),
            "grasp:cube": (dict(view=view, miss_margin=0.012), 3.0, 1.2),
            "lift:cube": (dict(view=view, height=0.16, tol=0.06, surface=TABLE_H), 6.0, 1.5),
            "transport:cube_above": (dict(view=view, point=blue_top, height=0.05, yaw=blue_yaw, tip=view.true_tip,
                                          tol_xy=0.006, tol_tcp=0.008, require_held=False), 10.0, 2.5),
            "place:cube_top": (dict(view=view, point=blue_top, placed=seated, force=0.0, speed=None, below=0.003,
                                    fail_after=None, settle=0.3, yaw=blue_yaw, require_held=False), 6.0, 1.5),
            "release:cube": (dict(view=view, margin=0.015), 3.0, 1.0),
            "retreat:cube": (dict(height=0.15, clear=lambda o: o["red_true"][2] + H + 0.08), 6.0, 1.5),
        }
        R, B = C.RED_NOM, C.BLUE_NOM
        nominal = {
            "reach:cube_above": self.cmd(R + [0, 0, 0.12], 0, MAX_APERTURE),
            "reach:cube_grasp": self.cmd(R + [0, 0, DZ], 0, MAX_APERTURE),
            "grasp:cube": self.cmd(R + [0, 0, DZ], 0, 0.0),
            "lift:cube": self.cmd(R + [0, 0, DZ + 0.16], 0, 0.0),
            "transport:cube_above": self.cmd(B + [0, 0, 2 * H + 0.05 + DZ], 0, 0.0),
            "place:cube_top": self.cmd(B + [0, 0, 2 * H + DZ], 0, 0.0),
            "release:cube": self.cmd(B + [0, 0, 2 * H + DZ], 0, MAX_APERTURE),
            "retreat:cube": self.cmd(B + [0, 0, 2 * H + DZ + 0.15], 0, MAX_APERTURE),
        }
        self._tokens(spec, nominal, {})

    def nominal_plan(self) -> list:
        return [STACK_TOKENS[n] for n in C.FrankaCubeStackTask.nominal_plan(self)]

    def skill_graph(self) -> list:
        return rename_graph(C.FrankaCubeStackTask.skill_graph(self), STACK_TOKENS)


# ======================================================================= drawer + box
DRAWER_TOKENS = {"reach_handle": "reach:handle_pre", "approach_handle": "reach:handle", "grasp_handle": "grasp:handle",
                 "pull_drawer": "pull:drawer", "release_handle": "release:handle", "retreat_handle": "retreat:handle",
                 "reach_object": "reach:box_above", "descend_object": "reach:box_grasp", "grasp_object": "grasp:box",
                 "lift_object": "lift:box", "carry_to_goal": "transport:goal_above", "lower_object": "place:goal",
                 "release_object": "release:box", "retreat": "retreat:box"}


class FrankaDrawerTokens(_TokenTask, D.FrankaDrawerTask):
    name = "franka_drawer"
    full_orientation = True                 # pitched pinch on the handle: hand rotation matrices

    def _build_skills(self):
        never = lambda o, st=None: False                                          # noqa: E731
        handle = ObjectView(tip=lambda o: o["handle_est"], top=lambda o: o["handle_est"], yaw=lambda o: o["cab_yaw_est"],
                            held=lambda o: bool(o["hold_handle"]), lost=never, drop_outcome=lambda o, st: "slipped",
                            inhand=lambda o: np.zeros(3), width=2 * D.BAR_R, true_tip=lambda o: o["handle_true"],
                            pads=lambda o: (o["handle_L"], o["handle_R"]))

        def box_lost(obs, st):
            near = (obs["contact_L"] + obs["contact_R"] > 0.2 or np.linalg.norm(obs["obj_true"] - obs["tcp"]) < 0.04)
            if near:
                st.memory["lost"] = None
                return False
            st.memory["lost"] = st.memory.get("lost") or obs["t"]
            return obs["t"] - st.memory["lost"] > 0.1

        # reference point of the box = its centre (the in-hand offset is measured)
        box = ObjectView(tip=lambda o: o["obj_est"], top=lambda o: o["obj_est"] + [0, 0, D.OBJ_HALF[2]],
                         yaw=lambda o: o["obj_yaw"], held=lambda o: bool(o["grasp"]), lost=box_lost,
                         drop_outcome=lambda o, st: "dropped", inhand=lambda o: o["obj_true"] - o["tcp"],
                         width=2 * D.OBJ_HALF[0], true_tip=lambda o: o["obj_true"])
        R_h = lambda o: self.handle_R(o["cab_yaw_est"])                           # noqa: E731
        R_box = lambda o: D.top_down(D.box_grasp_yaw(o["obj_yaw"], self.cab_yaw - np.pi))   # noqa: E731

        def closed_again(o):
            c = (o["opening"] < D.OPEN_MIN and np.linalg.norm(o["obj_true"][:2] - o["tcp"][:2]) < 0.25
                 and o["obj_true"][2] > TABLE_H + 0.05)
            return "closed_again" if c else None
        goal = lambda o: np.r_[self.goal, TABLE_H + o["obj_half_z"]]               # noqa: E731
        spec = {
            "reach:handle_pre": (dict(point=lambda o: o["handle_est"] - D.PRE_HANDLE * R_h(o)[:, 2], yaw=R_h,
                                      aperture=D.AP_HANDLE, tol=0.01, open_min=0.035, rot_tol=0.06, fix_yaw=False), 12.0, 3.0),
            "reach:handle": (dict(point=lambda o: o["handle_est"], yaw=R_h, aperture=D.AP_HANDLE, tol=0.006, open_min=None,
                                  rot_tol=0.05, fix_yaw=False,
                                  abort=lambda o: "already_open" if o["opening"] >= D.OPEN_MIN else None), 6.0, 1.5),
            "grasp:handle": (dict(view=handle, min_force=2.0, settle=0.4, miss_margin=0.004, require_held=False), 3.0, 1.0),
            "pull:drawer": (dict(view=handle, axis=lambda o: np.array([np.cos(o["cab_yaw_est"]), np.sin(o["cab_yaw_est"]), 0.0]),
                                 progress=lambda o: o["opening_est"], goal=D.PULL_CMD,
                                 done=lambda o: o["opening"] >= D.OPEN_TARGET,
                                 slip=lambda o: "slipped_open" if o["opening"] >= D.OPEN_MIN else "slipped",
                                 gone=lambda o: np.linalg.norm(o["tcp"] - o["handle_true"]) > 0.015), 8.0, 2.5),
            "release:handle": (dict(view=handle, aperture=D.AP_HANDLE, margin=0.035 - 2 * D.BAR_R), 3.0, 0.8),
            "retreat:handle": (dict(height=0.05, back=0.07, aperture=D.AP_HANDLE, tol=0.015,
                                    away=lambda o: np.linalg.norm(o["tcp"] - o["handle_true"]) > 0.06), 6.0, 1.2),
            "reach:box_above": (dict(point=lambda o: o["obj_est"] + [0, 0, D.H_ABOVE], yaw=R_box, fix_yaw=False,
                                     abort=closed_again), 12.0, 3.0),
            "reach:box_grasp": (dict(point=lambda o: o["obj_est"] + [0, 0, D.GRASP_DZ], yaw=R_box, fix_yaw=False, tol=0.005,
                                     open_min=None, abort=closed_again), 6.0, 1.5),
            "grasp:box": (dict(view=box, miss_margin=0.012), 3.0, 1.2),
            "lift:box": (dict(view=box, height=D.LIFT, tol=D.LIFT - 0.12, surface=self.floor_z0 + D.OBJ_HALF[2],
                              lifted=lambda o: bool(o["in_air"])), 6.0, 1.5),
            "transport:goal_above": (dict(view=box, point=goal, carry_z=TABLE_H + 0.25, tip=box.true_tip, tol_xy=0.006,
                                          tol_tcp=0.008, require_held=False), 10.0, 2.5),
            "place:goal": (dict(view=box, point=goal, placed=lambda o: bool(o["placed"]), force=0.0, speed=None, below=0.002,
                                fail_after=None, settle=0.3, require_held=False), 6.0, 1.8),
            "release:box": (dict(view=box, margin=0.055 - 2 * D.OBJ_HALF[0], lift=0.001), 3.0, 1.0),
            "retreat:box": (dict(height=0.14, clear=lambda o: o["obj_true"][2] + o["obj_half_z"] + 0.10), 6.0, 1.5),
        }
        R_h_nom = self.handle_R()
        a_nom = R_h_nom[:, 2]
        h_nom = self.cab_center + D.rot_z(self.cab_yaw) @ self.top["handle_closed_local"]
        obj_open = self.obj0 + D.OPEN_TARGET * self.n_out
        pull_end = h_nom + D.PULL_CMD * self.n_out
        retreat_end = pull_end - 0.07 * a_nom + [0, 0, 0.05]
        Rd = D.top_down(self.cab_yaw - np.pi)
        goal_c = np.r_[self.goal, TABLE_H + D.OBJ_HALF[2] + D.GRASP_DZ]
        carry_z = obj_open[2] + D.GRASP_DZ + D.LIFT
        c = self.cmd_R
        nominal = {
            "reach:handle_pre": c(h_nom - D.PRE_HANDLE * a_nom, R_h_nom, D.AP_HANDLE),
            "reach:handle": c(h_nom, R_h_nom, D.AP_HANDLE),
            "grasp:handle": c(h_nom, R_h_nom, 0.0),
            "pull:drawer": c(pull_end, R_h_nom, 0.0),
            "release:handle": c(pull_end, R_h_nom, D.AP_HANDLE),
            "retreat:handle": c(retreat_end, R_h_nom, D.AP_HANDLE),
            "reach:box_above": c(obj_open + [0, 0, D.H_ABOVE], Rd, MAX_APERTURE),
            "reach:box_grasp": c(obj_open + [0, 0, D.GRASP_DZ], Rd, MAX_APERTURE),
            "grasp:box": c(obj_open + [0, 0, D.GRASP_DZ], Rd, 0.0),
            "lift:box": c(obj_open + [0, 0, D.GRASP_DZ + D.LIFT], Rd, 0.0),
            "transport:goal_above": c(np.r_[self.goal, carry_z], Rd, 0.0),
            "place:goal": c(goal_c, Rd, 0.0),
            "release:box": c(goal_c, Rd, MAX_APERTURE),
            "retreat:box": c(goal_c + [0, 0, 0.14], Rd, MAX_APERTURE),
        }
        self._tokens(spec, nominal, {})

    def nominal_plan(self) -> list:
        return [DRAWER_TOKENS[n] for n in D.FrankaDrawerTask.nominal_plan(self)]

    def skill_graph(self) -> list:
        return rename_graph(D.FrankaDrawerTask.skill_graph(self), DRAWER_TOKENS)
