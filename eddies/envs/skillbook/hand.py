"""UR5e + dexterous-hand skillbook (LEAP Hand and Aero Hand): the skillbook primitives used by the
hand tasks, implemented once for the 9-D hand embedding chart.

Embedding chart (``controllers/hand_arm.py``): grasp-site position (m), rotation vector of the hand
frame about ``base_mat`` (rad), synergies (power, pinch, point) in [0, 1].  The same primitives run
both hands; the hand only changes the controller (IK decoder, synergy basis) and the synergy /
tool parameters a task passes.  Commands are built with ``task.cmd(pos, R, s)``.

Arguments a task supplies (no control logic):

* frames      ``point(obs) -> world point`` for the TOOL point, ``orient(obs) -> 3x3`` hand frame
* tool        :class:`HandTool`: the commanded hand point (index tip, pinch centre; default the
              grasp site) -- ``offset(obs)`` in the hand frame, ``pos(obs)`` measured in the world
* synergy     (power, pinch, point): the hand shape held by the Eddie (hand-shape changes are
              synergy arguments of reach / grasp / release, not separate primitives)
* predicates  measured outcome tests ``obs -> bool`` (held, released, done, ...); failure tests
              are ``(key, predicate, persist_s)`` triples (the condition must hold > persist_s)

Each ``build_*`` returns ``(target, outcome, on_enter)`` for :meth:`SkillBook.eddy`.  Nothing here
is registered in the Franka ``BOOK``: hand Eddies are certified per embodiment over the hand tasks.

Vocabulary ur5e-hand-1.0 (6 of the 11 skillbook primitive names):
    reach, grasp, push, release, turn, retreat
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from eddies.envs.framework.task import OK
from eddies.envs.skillbook.book import Primitive, SkillBook

HAND_BOOK = SkillBook(version="ur5e-hand-1.0")


# --------------------------------------------------------------------------- shared helpers
@dataclass
class HandTool:
    """The hand point a primitive positions: ``offset(obs)`` (hand frame, used for commanding) and
    ``pos(obs)`` (measured world position, used for the outcome)."""
    offset: Callable
    pos: Callable


GRASP_SITE = HandTool(offset=lambda o: np.zeros(3), pos=lambda o: o["grasp"])


def _joint_speed(obs) -> float:
    return float(np.linalg.norm(obs["dq"]))


def _grasp_for(p, R, off) -> np.ndarray:
    """Grasp-site position that puts the tool point (hand-frame offset ``off``) at world point p."""
    return np.asarray(p, float) - np.asarray(R, float) @ np.asarray(off, float)


def _decode(task, e):
    """(grasp position, hand frame, synergy) of a normalized command (e.g. the Eddie's start command)."""
    phys = task.space.denormalize(e)
    return phys[:3].copy(), task.ha.chart.decode_matrix(phys[3:6]), phys[6:9].copy()


def _persist(st, key, cond: bool, t: float, dur) -> bool:
    """True once ``cond`` has held continuously for more than ``dur`` s (``dur`` None / 0: at once)."""
    if not dur:
        return bool(cond)
    if cond:
        st.memory[key] = st.memory.get(key) or t
        return t - st.memory[key] > dur
    st.memory[key] = None
    return False


def _failed(fails, obs, st):
    for i, (key, pred, dur) in enumerate(fails):
        if _persist(st, f"fail{i}", pred(obs), obs["t"], dur):
            return key
    return None


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _rot(axis, angle) -> np.ndarray:
    a = np.asarray(axis, float)
    return Rotation.from_rotvec(a / np.linalg.norm(a) * angle).as_matrix()


def _hold_pose(task, obs, st, hold_from):
    """Pose held by release / retreat: the Eddie's start command or the measured pose at entry."""
    if hold_from == "command":
        g, R, _ = _decode(task, st.start_command)
        return g, R
    return obs["grasp"].copy(), obs["grasp_mat"].copy()


# --------------------------------------------------------------------------- reach
def build_reach(task, *, orient, synergy, point=None, tool=GRASP_SITE, tol=None, rot_tol=None, still=None,
                v_tol=None, settle=None, require=None, fail=()):
    """Move the tool point to ``point(obs)`` with hand frame ``orient(obs)`` while forming the hand
    shape ``synergy`` (``point=None``: stay at the measured entry position, i.e. a pure hand-shape /
    orientation change).  Success: tool within ``tol``, rotation error < ``rot_tol``,
    ``still(obs)`` (default joint speed) < ``v_tol``, elapsed > ``settle`` and ``require(obs)``."""
    still = still or _joint_speed

    def on_enter(obs, st):
        st.memory["hold"] = obs["grasp"].copy()

    def target(obs, st):
        R = orient(obs)
        g = st.memory["hold"] if point is None else _grasp_for(point(obs), R, tool.offset(obs))
        return task.cmd(g, R, synergy)

    def outcome(obs, labels, st):
        f = _failed(fail, obs, st)
        if f:
            return f
        ok = tol is None or point is None or np.linalg.norm(tool.pos(obs) - point(obs)) < tol
        ok = ok and (rot_tol is None or task.rotation_error(orient(obs)) < rot_tol)
        ok = ok and (v_tol is None or still(obs) < v_tol)
        ok = ok and (settle is None or st.elapsed(obs["t"]) > settle)
        ok = ok and (require is None or bool(require(obs)))
        return OK if ok else None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- grasp
def build_grasp(task, *, synergy, held, point, orient, tool=GRASP_SITE, latch_point=False, hold_time=None,
                settle=None, closed=None, t_closed=None, give_up=None):
    """Close the hand to ``synergy`` at ``point(obs)`` (latched at entry if ``latch_point``) with the
    hand frame ``orient(obs)`` latched at entry.  Success is MEASURED: ``held(obs)`` for more than
    ``hold_time`` s and elapsed > ``settle``.  'missed': not held and either the hand has closed
    (``closed(obs)``, after ``t_closed`` s) or ``give_up`` s have passed."""
    def on_enter(obs, st):
        st.memory["R"] = np.asarray(orient(obs), float).copy()
        st.memory["p"] = np.asarray(point(obs), float).copy() if latch_point else None

    def target(obs, st):
        R = st.memory["R"]
        p = st.memory["p"] if latch_point else point(obs)
        return task.cmd(_grasp_for(p, R, tool.offset(obs)), R, synergy)

    def outcome(obs, labels, st):
        t, el = obs["t"], st.elapsed(obs["t"])
        if held(obs):
            st.memory["g"] = st.memory.get("g") or t
            ok = (hold_time is None or t - st.memory["g"] > hold_time) and (settle is None or el > settle)
            return OK if ok else None
        st.memory["g"] = None
        shut = closed is not None and closed(obs) and (t_closed is None or el > t_closed)
        if shut or (give_up is not None and el > give_up):
            return "missed"
        return None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- push
def build_push(task, *, orient, synergy, done, axis=None, point=None, tool=GRASP_SITE, speed=0.0, depth=0.0,
               progress=None, stop_at=None, fail=()):
    """Push the tool point along ``axis`` (world unit vector) from ``point(obs)``: the reference
    advances at ``speed`` m/s (up to ``depth``) while the measured ``progress(obs)`` < ``stop_at``.
    ``point=None``: keep pressing with the entry command (push-and-hold).  The tool offset and the
    hand frame are latched at entry (a compliant finger deflects while it pushes).  Success:
    ``done(obs)``; failures ``fail``."""
    def on_enter(obs, st):
        st.memory["off"] = np.asarray(tool.offset(obs), float).copy()
        st.memory["R"] = np.asarray(orient(obs), float).copy()
        st.memory["d"] = 0.0
        st.memory["hold"] = _decode(task, st.start_command)[:2] if point is None else None

    def target(obs, st):
        if point is None:
            g, R = st.memory["hold"]
            return task.cmd(g, R, synergy)
        if progress is None or stop_at is None or progress(obs) < stop_at:
            st.memory["d"] = min(st.memory["d"] + speed * task.control_dt, depth)
        R = st.memory["R"]
        p = point(obs) + st.memory["d"] * np.asarray(axis, float)
        return task.cmd(_grasp_for(p, R, st.memory["off"]), R, synergy)

    def outcome(obs, labels, st):
        if done(obs):
            return OK
        return _failed(fail, obs, st)
    return target, outcome, on_enter


# --------------------------------------------------------------------------- release / retreat
def _move_off(task, *, synergy, point, orient, tool, hold_from, away, delay, ramp):
    """Shared motion of release and retreat: hold the entry pose (or follow ``point(obs)``) with
    ``synergy`` and back off by ``away(obs, R)`` (world vector), ramped in over ``ramp`` s after ``delay`` s."""
    def on_enter(obs, st):
        g, R = _hold_pose(task, obs, st, hold_from)
        st.memory["g"] = g
        st.memory["R"] = np.asarray(orient(obs), float).copy() if orient is not None else R

    def target(obs, st):
        R = st.memory["R"]
        g = st.memory["g"] if point is None else _grasp_for(point(obs), R, tool.offset(obs))
        if away is not None:
            el = st.elapsed(obs["t"])
            s = float(np.clip((el - delay) / ramp, 0.0, 1.0)) if ramp else float(el >= delay)
            g = g + s * np.asarray(away(obs, R), float)
        return task.cmd(g, R, synergy)
    return target, on_enter


def build_release(task, *, synergy, released, point=None, orient=None, tool=GRASP_SITE, hold_from="measured",
                  away=None, delay=0.0, ramp=0.0, settle=None, persist=None, fail=()):
    """Let go of the object: open to ``synergy`` (or keep the shape and lift the tool off it) at the
    entry pose (``hold_from`` = "command" | "measured") or ``point(obs)``, backing off by ``away``.
    Success: ``released(obs)`` (measured: no hand-object force, hand open) for more than ``persist`` s,
    elapsed > ``settle``."""
    target, on_enter = _move_off(task, synergy=synergy, point=point, orient=orient, tool=tool, hold_from=hold_from,
                                 away=away, delay=delay, ramp=ramp)

    def outcome(obs, labels, st):
        f = _failed(fail, obs, st)
        if f:
            return f
        ok = released(obs) and (settle is None or st.elapsed(obs["t"]) > settle)
        return OK if _persist(st, "rel", ok, obs["t"], persist) else None
    return target, outcome, on_enter


def build_retreat(task, *, synergy, clear, point=None, orient=None, tool=GRASP_SITE, hold_from="command",
                  away=None, fail=()):
    """Move the (open) hand clear of the object: to ``point(obs)`` or the entry pose + ``away``.
    Success: ``clear(obs)`` (measured clearance and no contact)."""
    target, on_enter = _move_off(task, synergy=synergy, point=point, orient=orient, tool=tool, hold_from=hold_from,
                                 away=away, delay=0.0, ramp=0.0)

    def outcome(obs, labels, st):
        f = _failed(fail, obs, st)
        if f:
            return f
        return OK if clear(obs) else None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- turn
def build_turn(task, *, axis, center, hand_angle, synergy, lead, goal=None, rate=np.inf, ref_pose=None,
               object_angle=None, object_goal=None, object_rate=None, direction=1.0, gain=1.5, ki=0.0,
               i_zone=0.0, i_max=0.0, held=None, stop=None, done=None, done_persist=None, backspin=None,
               fail=(), shaped=None, free=None, end_tol=np.radians(4.0), two_sided=False, end_persist=None,
               end_still=None, end_require=None, min_turn=None, tol_done=np.radians(1.5), slow=0.05,
               tol_settle=np.radians(3.0), settle_time=1.5, settle=None):
    """Rotate the hand (and a grasped object) about ``axis(obs)`` through ``center(obs)``.

    The commanded pose is a reference pose rotated by ``ref - a_ref``, where ``ref`` is a carrot on the
    hand angle (measured ``hand_angle(obs)``, unwrapped): it moves toward its goal at most ``rate``
    rad/s and stays within ``lead`` of the measured hand angle.  Reference pose: ``ref_pose(obs) ->
    (grasp position, hand frame, a_ref)`` or (default) the entry command at the measured entry angle.
    Goal: a hand angle ``goal`` (regrasp cycles) or an OBJECT angle ``object_goal`` servoed through the
    measured ``object_angle(obs)`` (hand goal = angle + gain err + integral).  While ``stop(obs)``
    the last measured pose is held; until ``shaped(obs)`` (and, before the first rotation step,
    ``free(obs)``) the hand opens in place at the entry pose.

    Outcomes: ok on ``done(obs)`` (> ``done_persist`` s); 'slipped' if the object turned back by
    ``backspin`` (``direction`` = sign of the object's progress); ``fail`` triples; object goal: ok
    once |err| < ``tol_done`` and still (or within ``tol_settle`` and still for ``settle_time`` s),
    while ``held``; hand goal: once within ``end_tol`` (one-sided: overshoot allowed) and
    ``end_require`` / ``end_still`` hold for > ``end_persist`` s -- ok if the object turned
    >= ``min_turn``, else 'slipped'."""
    dt = task.control_dt

    def angle_now(obs, st):
        a = float(hand_angle(obs))
        st.memory["h"] = st.memory["h"] + _wrap(a - st.memory["h_raw"])
        st.memory["h_raw"] = a
        return st.memory["h"]

    def on_enter(obs, st):
        a = float(hand_angle(obs))
        st.memory.update(h=a, h_raw=a, h0=a, I=0.0, opened=False,
                         last=(obs["grasp"].copy(), obs["grasp_mat"].copy()))
        st.memory["hold"] = st.memory["last"]
        g, R, _ = _decode(task, st.start_command)
        st.memory["ref0"] = (g, R, a)
        st.memory["obj0"] = float(object_angle(obs)) if object_angle is not None else None

    def target(obs, st):
        h = angle_now(obs, st)
        if stop is not None and stop(obs):
            g, R = st.memory["last"]
            return task.cmd(g, R, synergy)
        st.memory["last"] = (obs["grasp"].copy(), obs["grasp_mat"].copy())
        if (shaped is not None and not shaped(obs)) or (free is not None and not free(obs) and not st.memory["opened"]):
            g, R = st.memory["hold"]
            return task.cmd(g, R, synergy)
        st.memory["opened"] = True
        if object_goal is not None:
            err = float(object_goal(obs) - object_angle(obs))
            if abs(err) < i_zone:
                st.memory["I"] = float(np.clip(st.memory["I"] + ki * err * dt, -i_max, i_max))
            g_h = h + gain * err + st.memory["I"]
        else:
            g_h = float(goal)
        ref = st.memory.get("ref", h)
        ref += float(np.clip(g_h - ref, -rate * dt, rate * dt))
        ref = float(np.clip(ref, h - lead, h + lead))
        st.memory["ref"] = ref
        p0, R0, a0 = ref_pose(obs) if ref_pose is not None else st.memory["ref0"]
        Rot = _rot(axis(obs), _wrap(ref - a0))
        c = center(obs)
        return task.cmd(c + Rot @ (np.asarray(p0, float) - c), Rot @ R0, synergy)

    def outcome(obs, labels, st):
        t, el = obs["t"], st.elapsed(obs["t"])
        if done is not None:
            if done(obs):
                return OK if _persist(st, "done", True, t, done_persist) else None
            st.memory["done"] = None
        prog = None
        if object_angle is not None:
            prog = direction * (float(object_angle(obs)) - st.memory["obj0"])
            if backspin is not None and prog < -backspin:
                return "slipped"
        f = _failed(fail, obs, st)
        if f:
            return f
        if object_goal is not None:
            err = abs(float(object_angle(obs) - object_goal(obs)))
            still = abs(float(object_rate(obs))) < slow
            st.memory["slow"] = (st.memory.get("slow") or t) if (still and err < tol_settle) else None
            ok = (err < tol_done and still) or (st.memory["slow"] is not None and t - st.memory["slow"] > settle_time)
            ok = ok and (held is None or held(obs)) and (settle is None or el > settle)
            return OK if ok else None
        h = st.memory["h"]
        if two_sided:
            reached = abs(h - goal) < end_tol
        else:
            sgn = np.sign(goal - st.memory["h0"]) or 1.0
            reached = sgn * (goal - h) < end_tol
        if reached and (end_require is None or end_require(obs)):
            st.memory["end"] = st.memory.get("end") or t
            if (end_persist is None or t - st.memory["end"] > end_persist) and \
                    (end_still is None or _joint_speed(obs) < end_still):
                return OK if (min_turn is None or (prog is not None and prog >= min_turn)) else "slipped"
        return None
    return target, outcome, on_enter


SOURCES = "Bogh et al. 2012; MAPLE (Nasiriany et al. 2021); RAPS (Dalal et al. 2021)"
for _name, _fn, _outs, _desc, _src in [
    ("reach", build_reach, ("ok", "missed", "bumped"),
     "move a hand point to a frame while forming a hand shape (synergy)", "MAPLE reach; RAPS go-to-pose; Bogh move to"),
    ("grasp", build_grasp, ("ok", "missed"), "close the hand to a synergy; measured contact",
     "MAPLE grasp; Bogh pick up"),
    ("push", build_push, ("ok", "missed", "slipped", "lost"), "push a hand point along an axis (and hold)",
     "MAPLE / RAPS push; Bogh press"),
    ("release", build_release, ("ok", "moved"), "let go of the object (open / lift off), measured no contact",
     "MAPLE / RAPS / Bogh release"),
    ("turn", build_turn, ("ok", "slipped", "bumped"), "rotate the hand about an object axis (carrot on the hand angle)",
     "RAPS twist; Bogh turn"),
    ("retreat", build_retreat, ("ok", "moved"), "move the open hand clear of the object", "reach variant"),
]:
    HAND_BOOK.register(Primitive(_name, "ur5e-hand-1.0", _desc, _outs, _fn, (_src,)))
