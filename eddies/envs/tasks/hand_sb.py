"""UR5e + dexterous-hand tasks (LEAP / Aero) composed ONLY of Eddies of the hand skillbook
(``eddies.envs.skillbook.hand.HAND_BOOK``, vocabulary ur5e-hand-1.0).

Pattern of ``franka_sb.py``: each class subclasses the hand-written task and inherits its physics,
perception, labels, cells, disturbances, ``achieved`` and embedding unchanged; only the skills are
replaced.  A task supplies frames (obs -> tool point / hand frame), the tool point (index tip,
pinch centre or grasp site), synergies, measured predicates and parameters; the control logic
lives in the shared primitives (one implementation for the three tasks and both hands).  Anchors,
trust radii, timeouts and nominal durations are those of the legacy skill each Eddie replaces.

Eddies (primitive:binding), 21 over 6 primitives:
    button   reach:point_synergy  reach:button_above  reach:button  push:button  push:button_hold
             release:button  retreat:button
    wheel    reach:rim_pre  reach:rim  grasp:rim  turn:wheel  release:rim  retreat:rim  release:rim_slip
    tighten  reach:peg_above  reach:peg_grasp  grasp:peg  turn:peg  release:peg  turn:unwind  retreat:peg
"""
from __future__ import annotations

import numpy as np

from eddies.envs.skillbook.hand import HAND_BOOK, HandTool
from eddies.envs.tasks import hand_peg_tighten as K
from eddies.envs.tasks import hand_wheel as W
from eddies.envs.tasks.franka_sb import rename_graph

# legacy skill name -> Eddie (the skill graphs / LTLf semantics are unchanged)
BUTTON_TOKENS = {"form_point": "reach:point_synergy", "reach_above": "reach:button_above", "descend": "reach:button",
                 "press": "push:button", "hold": "push:button_hold", "release": "release:button",
                 "retreat": "retreat:button"}
WHEEL_TOKENS = {"reach_pregrasp": "reach:rim_pre", "approach": "reach:rim", "grasp": "grasp:rim", "turn": "turn:wheel",
                "release": "release:rim", "retreat": "retreat:rim", "let_go": "release:rim_slip"}
TIGHTEN_TOKENS = {"reach_above": "reach:peg_above", "descend": "reach:peg_grasp", "pinch": "grasp:peg",
                  "turn": "turn:peg", "release": "release:peg", "unwind": "turn:unwind", "retreat": "retreat:peg"}

UP = np.array([0.0, 0.0, 1.0])


class _HandTokens:
    """Mixin: instantiate the Eddies of HAND_BOOK with the anchors / trust radii / timeouts of the
    legacy skills they replace (``legacy`` = the hand-written task class, ``tokens`` = name map)."""
    legacy: type
    tokens: dict

    def _eddies(self, spec: dict) -> None:
        self._legacy_names = True                    # legacy anchors and hand-set trust radii
        try:
            self.legacy._build_skills(self)
        finally:
            self._legacy_names = False
        old = {v: self._skills[k] for k, v in self.tokens.items()}
        S = {}
        for token, args in spec.items():
            prim, binding = token.split(":", 1)
            sk = old[token]
            S[token] = HAND_BOOK.eddy(self, prim, binding, anchor=sk.anchor, delta=sk.delta, timeout=sk.timeout,
                                      duration=sk.nominal_duration, **args)
        self._skills = S

    def nominal_plan(self) -> list:
        plan = self.legacy.nominal_plan(self)
        return plan if getattr(self, "_legacy_names", False) else [self.tokens[n] for n in plan]

    def skill_graph(self) -> list:
        return rename_graph(self.legacy.skill_graph(self), self.tokens)


# ======================================================================= steering wheel
class HandWheelTokens(_HandTokens, W.HandWheelTask):
    legacy, tokens = W.HandWheelTask, WHEEL_TOKENS

    def _build_skills(self):
        def grip(o):          # grasp-site pose of the planned grip point (rotated with the wheel), estimated rim
            return self._grip_pose(o["hub_est"], o["W_est"], o["R_est"], o["grip_phi"] + o["theta"])
        f_free = lambda o: o["f_wheel"] < 0.2                                     # noqa: E731
        moved = lambda o: not o["turned"] and abs(o["omega"]) < 0.05              # noqa: E731
        O, PW = W.OPEN, W.POWER
        self._eddies({
            "reach:rim_pre": dict(point=lambda o: grip(o)[0] + self._pre(o["W_est"], grip(o)[1]),
                                  orient=lambda o: grip(o)[1], synergy=O, tol=0.006, rot_tol=0.08, v_tol=0.05,
                                  require=f_free),
            "reach:rim": dict(point=lambda o: grip(o)[0], orient=lambda o: grip(o)[1], synergy=O, tol=0.003,
                              rot_tol=0.08, v_tol=0.03, settle=0.2, fail=(("bumped", lambda o: o["f_wheel"] > 5.0, 0.1),)),
            "grasp:rim": dict(synergy=PW, held=lambda o: o["grasped"], point=lambda o: grip(o)[0],
                              orient=lambda o: grip(o)[1], latch_point=True, hold_time=0.3, settle=0.6,
                              closed=lambda o: o["power"] > 0.85, t_closed=1.0),
            "turn:wheel": dict(axis=lambda o: o["W_est"][:, 2], center=lambda o: o["hub_est"],
                               hand_angle=lambda o: o["psi_hand_est"], synergy=PW, lead=W.LEAD,
                               object_angle=lambda o: o["theta"], object_goal=lambda o: o["theta_target"],
                               object_rate=lambda o: o["omega"], gain=1.5, ki=1.5, i_zone=np.radians(6.0),
                               i_max=np.radians(6.0), held=lambda o: o["grasped"],
                               fail=(("slipped", lambda o: not o["grasped"], 0.2),
                                     ("slipped", lambda o: o["grasped"] and not o["turned"]
                                      and abs(o["slip"]) > W.SLIP_TOL, 0.1)),
                               tol_done=W.TOL_TURN_DONE, slow=0.05, tol_settle=0.6 * W.TOL_TURNED, settle_time=1.5,
                               settle=0.3),
            "release:rim": dict(synergy=O, hold_from="command", away=lambda o, R: -W.H_RELEASE * R[:, 0], delay=0.3,
                                ramp=0.4, settle=0.4, fail=(("moved", lambda o: moved(o) and f_free(o), None),),
                                released=lambda o: o["turned"] and f_free(o) and o["power"] < 0.2),
            "retreat:rim": dict(synergy=O, hold_from="command",
                                away=lambda o, R: -0.05 * R[:, 0] + W.H_RETREAT * o["W_est"][:, 2],
                                fail=(("moved", moved, None),),
                                clear=lambda o: o["turned"] and o["d_rim"] > W.D_AWAY + 0.005 and f_free(o)),
            "release:rim_slip": dict(synergy=O, hold_from="command",
                                     away=lambda o, R: -W.LETGO_OUT * R[:, 0] + W.LETGO_UP * o["W_est"][:, 2],
                                     delay=0.3, ramp=0.5, settle=0.6,
                                     released=lambda o: f_free(o) and o["power"] < 0.2 and o["d_rim"] > 0.045),
        })


# ======================================================================= peg tightening
class HandPegTightenTokens(_HandTokens, K.HandPegTightenTask):
    legacy, tokens = K.HandPegTightenTask, TIGHTEN_TOKENS

    def _build_skills(self):
        pinch = HandTool(offset=lambda o: self.c_pinch, pos=lambda o: o["pinch_center"])   # thumb-index pinch centre
        knob = lambda o: o["knob_est"]                                            # noqa: E731
        start = lambda o: K.turn_frame(self.psi_start)                            # noqa: E731
        here = lambda o: K.turn_frame(o["heading"])                               # noqa: E731
        s_pre, s_close = (0.0, self.s_pre, 0.0), (0.0, self.s_close, 0.0)
        is_pre = lambda o: o["s_pinch"] < self.s_pre + 0.12                       # noqa: E731
        bumped = (("bumped", lambda o: o["f_knob"] > 1.0 or o["f_fingers_constr"] > 1.0, 0.1),)
        # turn frame about the knob axis: zero-heading pose with the pinch centre on the (estimated) axis
        R0 = K.R_PALM_DOWN
        zero = lambda o: (o["knob_est"] - R0 @ self.c_pinch, R0, 0.0)             # noqa: E731
        axis = dict(axis=lambda o: UP, center=knob, hand_angle=lambda o: o["heading"], ref_pose=zero)
        self._eddies({
            "reach:peg_above": dict(point=lambda o: o["knob_est"] + K.H_ABOVE * UP, orient=start, synergy=s_pre,
                                    tool=pinch, tol=0.006, v_tol=0.15,
                                    require=lambda o: (abs(K.wrap(o["heading"] - self.psi_start)) < np.radians(4)
                                                       and abs(o["s_pinch"] - self.s_pre) < 0.2)),
            "reach:peg_grasp": dict(point=knob, orient=start, synergy=s_pre, tool=pinch, tol=0.003, v_tol=0.1,
                                    settle=0.3, fail=bumped),
            "grasp:peg": dict(synergy=s_close, held=lambda o: o["pinched"], point=knob, orient=here, tool=pinch,
                              hold_time=0.25, settle=0.5, closed=lambda o: o["s_pinch"] > 0.95, give_up=2.5),
            "turn:peg": dict(**axis, synergy=s_close, lead=self.lead, goal=self.psi_end, rate=self.w_turn,
                             object_angle=lambda o: o["hinge"], direction=-1.0, stop=lambda o: o["seated"],
                             done=lambda o: o["seated"], done_persist=0.1, backspin=K.BACKSPIN,
                             fail=(("slipped", lambda o: not o["pinched"], 0.15),),
                             end_tol=np.radians(4), end_persist=0.2, end_still=0.1, min_turn=K.TURN_OK),
            "release:peg": dict(synergy=s_pre, hold_from="measured", persist=0.15,
                                released=lambda o: o["f_knob"] < K.F_FREE and is_pre(o)),
            "turn:unwind": dict(**axis, synergy=s_pre, lead=self.lead_unwind, goal=self.psi_start, rate=self.w_unwind,
                                shaped=lambda o: o["s_pinch"] <= self.s_pre + 0.12,
                                free=lambda o: o["f_knob"] < K.F_FREE, fail=bumped, two_sided=True,
                                end_tol=np.radians(4), end_still=0.15,
                                end_require=lambda o: (np.linalg.norm(o["pinch_center"] - o["knob_est"]) < 0.004
                                                       and is_pre(o))),
            "retreat:peg": dict(synergy=(0.0, 0.0, 0.0), point=lambda o: o["knob_est"] + 0.09 * UP, orient=here,
                                tool=pinch, clear=lambda o: o["tips_clear"] and o["f_knob"] < K.F_FREE),
        })
