"""Unitree G1 (+ Dex3-1) tasks composed ONLY of Eddies of the G1 skillbook (``skillbook/g1.py``).

Each task supplies perception frames ((obs, st) -> point / pose), object views (:class:`G1View`)
and parameters; every Eddie is an instantiation of one primitive of :data:`G1_BOOK`, whose control
logic is shared by all three tasks.  Physics, perception, labels, cells, disturbances, the
decoder and the embedding (``achieved``) are inherited unchanged from the hand-written tasks
(``g1_peg``, ``g1_drawer``); only the skills are replaced.  Anchors and trust
radii are the legacy tasks' hand-set values (taken from their skill set), timeouts / nominal
durations are the legacy ones.

Eddies (primitive:binding), 45 over 10 primitives:
    peg        reach:peg_above  reach:peg_grasp  grasp:peg  lift:peg  transport:hole_above
               insert:hole  transport:hole_backoff  release:peg  retreat:peg
    drawer     reach:handle_pre  reach:handle_above  reach:handle  grasp:handle  pull:drawer
               release:handle  retreat:handle  reach:caddy_above  reach:caddy_preshape
               reach:caddy_grasp  grasp:caddy  lift:caddy  transport:goal_above  place:goal
               release:caddy  retreat:caddy
    microwave  reach:handle_pre  reach:handle_align  reach:handle  grasp:handle  pull:door
               release:handle  retreat:handle  reach:door_around  reach:door_inner  push:door
               retreat:door  reach:cup_pre  reach:cup_grasp  grasp:cup  lift:cup
               transport:cavity_out  transport:goal_above  place:goal  release:cup  retreat:cup
"""
from __future__ import annotations

import dataclasses

import numpy as np

from eddies.envs.skillbook.g1 import G1_BOOK, G1View
from eddies.envs.tasks import g1_drawer as D
from eddies.envs.tasks import g1_peg as P

I3 = np.eye(3)


def rename_graph(graph, names):
    return [dataclasses.replace(s, skill=names[s.skill]) for s in graph]


def _shifted(pose, d):
    """Frame ``pose`` translated by the world vector d (same hand rotation)."""
    d = np.asarray(d, float)

    def f(obs, st):
        p, R = pose(obs, st)
        return p + d, R
    return f


class _G1Tokens:
    """Mixin: Eddies of G1_BOOK, anchors / trust radii as in the legacy task."""
    TOKENS: dict = {}          # legacy skill name -> Eddie
    LEGACY = None              # legacy task class

    def _tokens(self, spec: dict) -> None:
        self._legacy_names = True                # the legacy builder iterates the legacy plan
        try:
            self.LEGACY._build_skills(self)      # legacy skills: only their anchors and radii are kept
        finally:
            self._legacy_names = False
        legacy = self._skills
        assert set(spec) == set(self.TOKENS.values()), set(spec) ^ set(self.TOKENS.values())
        S = {}
        for old, token in self.TOKENS.items():
            args, timeout, dur = spec[token]
            prim, binding = token.split(":", 1)
            S[token] = G1_BOOK.eddy(self, prim, binding, anchor=legacy[old].anchor, delta=legacy[old].delta,
                                    timeout=timeout, duration=dur, **args)
        self._skills = S

    def skills(self) -> dict:
        return self._skills

    def nominal_plan(self) -> list:
        plan = self.LEGACY.nominal_plan(self)
        return plan if getattr(self, "_legacy_names", False) else [self.TOKENS[n] for n in plan]

    def skill_graph(self) -> list:
        return rename_graph(self.LEGACY.skill_graph(self), self.TOKENS)


# ======================================================================= peg insertion
PEG_TOKENS = {"reach_above_peg": "reach:peg_above", "descend_to_peg": "reach:peg_grasp", "grasp_peg": "grasp:peg",
              "lift_peg": "lift:peg", "move_over_hole": "transport:hole_above", "insert_peg": "insert:hole",
              "realign_search": "transport:hole_backoff", "release_peg": "release:peg", "retreat": "retreat:peg"}


class G1PegTokens(_G1Tokens, P.G1PegInsertionTask):
    TOKENS = PEG_TOKENS
    LEGACY = P.G1PegInsertionTask

    def _next_search(self) -> None:
        self._search_k += 1                       # next spiral search offset for the re-alignment

    def _build_skills(self):
        eye = lambda o, st: I3                                                   # noqa: E731
        view = G1View(held=lambda o: bool(o["held"]), touch=lambda o: o["hand_force"], ref=lambda o: o["peg_true"],
                      near_r=0.04)
        still = lambda o: o["peg_speed"] < 0.01                                  # noqa: E731
        hole = lambda o, st: np.r_[o["hole_est"][:2], self.block_top]           # noqa: E731
        tip = lambda o: o["peg_tip_est"]                                         # noqa: E731
        axis = lambda o: o["peg_axis_est"]                                       # noqa: E731
        carry = dict(view=view, goal=hole, ref=tip, height=P.CARRY_CLEAR, upright=axis,
                     meas=lambda o: np.r_[o["peg_tip_est"][:2], o["peg_bottom"][2]], tol_xy=0.0015, tol_z=0.006,
                     still=lambda o: o["peg_speed"] < 0.02)
        lift_to = self.block_top + P.CARRY_CLEAR
        spec = {
            "reach:peg_above": (dict(pose=lambda o, st: (self._pregrasp(o, o["peg_est"][2] + P.PEG_HH + P.ABOVE), I3),
                                     closure=0.0, tol=0.008, closure_max=0.1, require=still), 10.0, 2.0),
            "reach:peg_grasp": (dict(pose=lambda o, st: (self._pregrasp(o), I3), closure=0.0, tol=0.005, require=still,
                                     abort=lambda o: "knocked" if o["peg_tilt"] > np.radians(12) else None,
                                     capture=(0.006, lambda o: o["peg_speed"] > 0.01)), 6.0, 1.5),
            "grasp:peg": (dict(view=view, orient=eye, ok_closure=0.5, settle=0.7, miss_closure=0.85, miss_after=0.7),
                          4.0, 1.5),
            "lift:peg": (dict(view=view, orient=eye, bottom=lambda o: o["peg_bottom"][2], goal=lambda o: lift_to + 0.004,
                              latch=True, lifted=lambda o: o["peg_bottom"][2] > lift_to - 0.004), 6.0, 1.5),
            "transport:hole_above": (dict(**carry), 10.0, 2.5),
            "insert:hole": (dict(view=view, goal=hole, ref=tip, upright=axis, depth=lambda o: o["depth"],
                                 seat=P.INSERT_GOAL, ok_depth=P.INSERT_OK, search=self._search_offset,
                                 on_jam=self._next_search), 8.0, 2.5),
            "transport:hole_backoff": (dict(**carry, search=self._search_offset), 6.0, 1.5),
            "release:peg": (dict(view=G1View(held=view.held, free=lambda o: o["hand_force"] < 0.2), closure=0.0,
                                 open_tol=0.12, settle=0.3), 4.0, 1.2),
            "retreat:peg": (dict(closure=0.0, shift=(-0.07, -0.03, 0.06), tol=0.02,
                                 away=lambda o: np.linalg.norm(o["grasp_pos"] - o["peg_top"]) > 0.08), 6.0, 1.5),
        }
        self._tokens(spec)


# ======================================================================= drawer + caddy
DRAWER_TOKENS = {
    "reach_handle": "reach:handle_pre", "above_handle": "reach:handle_above", "hook_handle": "reach:handle",
    "grasp_handle": "grasp:handle", "pull_drawer": "pull:drawer", "release_handle": "release:handle",
    "retreat_handle": "retreat:handle", "reach_caddy": "reach:caddy_above", "preshape_caddy": "reach:caddy_preshape",
    "descend_caddy": "reach:caddy_grasp", "grasp_caddy": "grasp:caddy", "lift_caddy": "lift:caddy",
    "carry_caddy": "transport:goal_above", "lower_caddy": "place:goal", "release_caddy": "release:caddy",
    "retreat": "retreat:caddy"}


class G1DrawerTokens(_G1Tokens, D.G1DrawerTask):
    TOKENS = DRAWER_TOKENS
    LEGACY = D.G1DrawerTask

    def _build_skills(self):
        handle_hook = lambda o, st: self._handle_hook(o)                         # noqa: E731
        caddy_hook = lambda o, st: self._caddy_hook(o)                           # noqa: E731
        already_open = lambda o: "already_open" if o["opening"] >= D.OPEN_MIN else None          # noqa: E731
        closed_again = (lambda o: "closed_again" if (o["opening"] < D.OPEN_MIN and o["caddy_in_drawer"])  # noqa: E731
                        else None)
        handle = G1View(held=lambda o: bool(o["hold_handle"]))
        handle_free = G1View(held=handle.held, free=lambda o: o["hand_handle"] < 0.3)
        caddy = G1View(held=lambda o: bool(o["held"]), touch=lambda o: o["hand_caddy"], ref=lambda o: o["caddy_bar"],
                       near_r=0.06)
        caddy_free = G1View(held=caddy.held, free=lambda o: o["hand_caddy_w"] < 0.3)
        calm = lambda o: o["caddy_speed"] < 0.03                                 # noqa: E731
        support = lambda o: self.side_top_z if o["caddy_in_drawer"] else self.table_h + 0.03       # noqa: E731
        goal_xy = lambda o, st: np.r_[self.goal, 0.0]                            # noqa: E731
        n_out = self.n_out.copy()
        spec = {
            "reach:handle_pre": (dict(pose=_shifted(handle_hook, D.PRE_H), closure=D.S_PRE_H, tol=0.008, closure_max=0.5,
                                      open_first=0.15), 10.0, 2.5),
            "reach:handle_above": (dict(pose=_shifted(handle_hook, D.ABOVE_H), closure=D.S_PRE_H, tol=0.008,
                                        abort=already_open), 5.0, 1.2),
            "reach:handle": (dict(pose=handle_hook, closure=D.S_PRE_H, tol=0.005, abort=already_open), 5.0, 1.5),
            "grasp:handle": (dict(view=handle, ok_closure=0.8, settle=0.5, tug=(lambda o: n_out, D.TUG)), 3.0, 1.2),
            "pull:drawer": (dict(engaged=lambda o: bool(o["hold_handle"]), done=lambda o: o["opening"] >= D.OPEN_TARGET,
                                 slip=lambda o: "slipped_open" if o["opening"] >= D.OPEN_MIN else "slipped",
                                 gone=lambda o: (np.linalg.norm(o["handle_true"] - o["grasp_pos"]) > 0.05
                                                 or o["hand_handle"] < 0.3), slip_time=0.15,
                                 axis=lambda o: np.array([np.cos(o["cab_yaw_est"]), np.sin(o["cab_yaw_est"]), 0.0]),
                                 progress=lambda o: o["opening_est"], goal=D.PULL_CMD), 8.0, 2.5),
            "release:handle": (dict(view=handle_free, closure=D.S_REL, lift=(0.0, 0.0, 0.02), open_tol=0.12, settle=0.3),
                               3.0, 0.8),
            "retreat:handle": (dict(closure=D.S_REL, shift=(0.0, 0.0, 0.10), tol=0.015,
                                    away=lambda o: o["hand_handle"] < 0.3), 5.0, 1.2),
            "reach:caddy_above": (dict(pose=_shifted(caddy_hook, (0, 0, 0.11)), closure=1.0, tol=0.010, closure_min=0.6,
                                       abort=closed_again), 8.0, 2.0),
            "reach:caddy_preshape": (dict(pose=_shifted(caddy_hook, (0, 0, 0.08)), closure=D.S_PRE_C, tol=0.008,
                                          closure_max=0.35, abort=closed_again), 4.0, 1.0),
            "reach:caddy_grasp": (dict(pose=caddy_hook, closure=D.S_PRE_C, tol=0.005, abort=closed_again), 5.0, 1.5),
            "grasp:caddy": (dict(view=caddy, held=lambda o: bool(o["held"] and o["in_air"]), ok_closure=0.8, settle=0.5,
                                 tug=(lambda o: np.array([0.0, 0.0, 1.0]), D.TUG), tug_dwell=0.4), 3.0, 1.2),
            "lift:caddy": (dict(view=caddy, bottom=lambda o: o["caddy_bottom"], goal=lambda o: support(o) + D.LIFT_CLEAR,
                                rise=0.03, lifted=lambda o: o["in_air"] and o["caddy_bottom"] > self.side_top_z + D.LIFT_OK),
                           5.0, 1.5),
            "transport:goal_above": (dict(view=caddy, goal=goal_xy, ref=lambda o: o["caddy_pos"], rise=D.CARRY_EXTRA,
                                          smooth=calm, tol_xy=0.008, tol=0.01, still=calm), 10.0, 3.0),
            "place:goal": (dict(view=caddy, goal=lambda o, st: np.r_[self.goal, self.table_h],
                                ref=lambda o: np.r_[o["caddy_pos"][:2], o["caddy_bottom"]], below=0.003, smooth=calm,
                                placed=lambda o: o["placed"] or (o["touchdown"] and not o["in_air"]), settle=0.3,
                                still=lambda o: o["caddy_speed"] < 0.02), 6.0, 2.0),
            "release:caddy": (dict(view=caddy_free, closure=D.S_REL_C, slide=((0.0, -1.0, 0.0), D.SLIDE_OUT),
                                   slide_margin=0.15, slide_delay=0.5, slide_calm=lambda o: o["caddy_speed"] < 0.02,
                                   still=lambda o: o["caddy_speed"] < 0.01, verify=lambda o: bool(o["placed"]),
                                   fail_key="dropped"), 4.0, 1.2),
            "retreat:caddy": (dict(closure=D.S_REL_C, shift=(0.0, 0.0, 0.08), open_first=0.15, tol=0.015,
                                   away=lambda o: o["grasp_pos"][2] > o["caddy_bar"][2] + 0.08), 5.0, 1.2),
        }
        self._tokens(spec)


# ======================================================================= microwave + cup
