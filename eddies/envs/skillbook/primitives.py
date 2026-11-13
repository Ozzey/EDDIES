"""The 11 primitives of skillbook v1, implemented once for the Franka embedding chart.

Embedding chart (shared): TCP position (m), hand orientation (rotation-vector chart), aperture
(m), force limit f* (N).  An *orientation* argument is either a yaw (float: top-down hand) or
a 3x3 hand rotation matrix (tasks that set ``full_orientation = True``); ``_cmd`` maps it with
``task.cmd`` or ``task.cmd_R``.  Every primitive reads perception through an *object view*
(:class:`ObjectView`) and frame callables ``obs -> point``; its outcome is decided from
measured state.

Each ``build_*`` returns ``(target, outcome, on_enter)`` for :meth:`SkillBook.eddy`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from eddies.core.online import RecursiveLeastSquares
from eddies.envs.framework.perception import wrap_to
from eddies.envs.framework.task import OK
from eddies.envs.skillbook.book import BOOK, Primitive

MAX_APERTURE = 0.08


# --------------------------------------------------------------------------- object views
def _pads_default(obs):
    return obs["contact_L"], obs["contact_R"]


@dataclass
class ObjectView:
    """What a primitive may know about the manipulated object (supplied by the task).

    tip(obs) / top(obs): estimated reference (bottom) / top point; yaw(obs): estimated yaw;
    held(obs): measured grasp; lost(obs, st): the object has measurably left the hand;
    drop_outcome(obs, st): classification of a drop from the resting state;
    inhand(obs): TCP -> tip offset; width: grasp width [m]; true_tip(obs): measured tip;
    pads(obs): (left, right) pad forces on this object; abort(obs): optional outcome key
    that ends any Eddie (e.g. 'toppled').
    """
    tip: Callable
    top: Callable
    yaw: Callable
    held: Callable
    lost: Callable
    drop_outcome: Callable
    inhand: Callable
    width: float
    true_tip: Callable
    abort: Callable | None = None
    symmetry: float = np.pi / 2
    pads: Callable = _pads_default


def _near(obs, p, tol):
    return float(np.linalg.norm(obs["tcp"] - p)) < tol


def _orient_now(task, obs):
    if getattr(task, "full_orientation", False):
        return np.array(obs["tcp_mat"], float).copy()
    return float(obs["tcp_yaw"]) if "tcp_yaw" in obs else float(task.tcp_yaw())


def _as_orient(x):
    return np.array(x, float) if np.ndim(x) == 2 else float(x)


def _cmd(task, pos, orient, aperture, force=0.0):
    if np.ndim(orient) == 2:
        return task.cmd_R(pos, orient, aperture, force)
    return task.cmd(pos, orient, aperture, force)


def _hold(task):
    def on_enter(obs, st):
        st.memory["hold"] = obs["tcp"].copy()
        st.memory["orient"] = _orient_now(task, obs)
    return on_enter


def _abort(view, abort, obs):
    if abort is not None:
        return abort(obs)
    return view.abort(obs) if (view is not None and view.abort is not None) else None


# --------------------------------------------------------------------------- 1 reach
def build_reach(task, *, point, yaw, view=None, aperture=MAX_APERTURE, tol=0.008, force=0.0,
                open_min=0.07, fix_yaw=True, rot_tol=None, abort=None, fix_point=False):
    """Move the hand (open to ``aperture``) to ``point(obs)`` with orientation ``yaw(obs)``
    (fixed at entry if ``fix_yaw``; the point too if ``fix_point``).  ``rot_tol``: also
    require the hand rotation error."""
    def on_enter(obs, st):
        st.memory["orient"] = _as_orient(yaw(obs))
        if fix_point:
            st.memory["point"] = np.array(point(obs), float)

    def target(obs, st):
        o = st.memory["orient"] if fix_yaw else _as_orient(yaw(obs))
        return _cmd(task, st.memory["point"] if fix_point else point(obs), o, aperture, force)

    def outcome(obs, labels, st):
        a = _abort(view, abort, obs)
        if a:
            return a
        ok = (_near(obs, st.memory["point"] if fix_point else point(obs), tol)
              and (open_min is None or obs["aperture"] > open_min))
        if ok and rot_tol is not None:
            o = st.memory["orient"] if fix_yaw else _as_orient(yaw(obs))
            ok = task.hand_rot_err(o) < rot_tol
        return OK if ok else None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- 2 grasp
def build_grasp(task, *, view, min_force=3.0, settle=0.5, miss_margin=0.008, require_held=True):
    """Close the hand at the current pose; success is MEASURED pad contact (not the command)."""
    def target(obs, st):
        return _cmd(task, st.memory["hold"], st.memory["orient"], 0.0)

    def outcome(obs, labels, st):
        L, R = view.pads(obs)
        if (not require_held or view.held(obs)) and L > min_force and R > min_force \
                and st.elapsed(obs["t"]) > settle:
            return OK
        if obs["aperture"] < view.width - miss_margin and not view.held(obs):
            return "missed"
        return None
    return target, outcome, _hold(task)


# --------------------------------------------------------------------------- 3 lift
def build_lift(task, *, view, height=0.10, tol=0.012, surface=None, lifted=None):
    """Lift the held object until its reference point is ``height - tol`` above ``surface``
    (z, m; default: its height at entry); ``lifted(obs)`` replaces the held test if given."""
    hold = _hold(task)

    def on_enter(obs, st):
        hold(obs, st)
        st.memory["lost"] = None
        st.memory["tip0"] = float(view.true_tip(obs)[2]) if surface is None else float(surface)

    def target(obs, st):
        return _cmd(task, st.memory["hold"] + [0, 0, height], st.memory["orient"], 0.0)

    def outcome(obs, labels, st):
        if "grasped" in labels:
            st.memory["seen_grasped"] = True
        if view.lost(obs, st):
            return view.drop_outcome(obs, st)
        up = lifted(obs) if lifted is not None else view.held(obs)
        ok = up and view.true_tip(obs)[2] > st.memory["tip0"] + height - tol
        return OK if ok else None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- 4 transport
def build_transport(task, *, view, point, height=0.0, yaw=None, align_yaw=None, tol_xy=0.0015,
                    tol_yaw=np.radians(2.0), tol_tcp=0.004, force=0.0, settle=0.0, tip=None,
                    carry_z=None, require_held=True):
    """Carry the held object so that its reference point is ``height`` above ``point(obs)``.

    yaw(obs): hand orientation (None: keep the orientation at entry); align_yaw(obs): also
    require the object's faces aligned with this yaw; tip(obs): the measurement used by the
    alignment test (default: the view's estimate); carry_z: travel at the constant TCP height
    max(entry height, carry_z) instead of descending to the target height."""
    tip = tip or view.tip
    hold = _hold(task)

    def on_enter(obs, st):
        hold(obs, st)
        st.memory["lost"] = None
        if carry_z is not None:
            st.memory["z"] = max(float(obs["tcp"][2]), float(carry_z))

    def tcp_target(obs, st):
        off = view.inhand(obs)
        p = point(obs)
        z = st.memory["z"] if carry_z is not None else p[2] + height - off[2]
        return np.r_[p[:2] - off[:2], z]

    def target(obs, st):
        o = st.memory["orient"] if yaw is None else yaw(obs)
        return _cmd(task, tcp_target(obs, st), o, 0.0, force)

    def outcome(obs, labels, st):
        st.memory["seen_grasped"] = True
        if view.lost(obs, st):
            return view.drop_outcome(obs, st)
        p = point(obs)
        yaw_ok = True
        if align_yaw is not None:
            yaw_ok = abs(wrap_to(view.yaw(obs) - align_yaw(obs), view.symmetry)) < tol_yaw
        ok = ((not require_held or view.held(obs)) and np.linalg.norm(tip(obs)[:2] - p[:2]) < tol_xy
              and yaw_ok and _near(obs, tcp_target(obs, st), tol_tcp) and st.elapsed(obs["t"]) >= settle)
        return OK if ok else None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- 5 place
def build_place(task, *, view, point, placed, force=8.0, speed=0.03, below=0.004, fail_after=4.0,
                fail_key="missed_place", fail_if=None, settle=0.0, yaw=None, require_held=True,
                zone=None, fail_hold=0.0):
    """Lower the held object onto / into ``point(obs)`` (reference point ``below`` under it,
    force-limited, at ``speed`` m/s or directly if None) until ``placed(obs)``.

    zone: guarded touchdown -- descend directly until the measured reference point is within
    ``zone`` of ``point``, then at ``speed`` from the TCP height measured there; fail_hold:
    the failure test must hold continuously for this long."""
    hold = _hold(task)

    def on_enter(obs, st):
        hold(obs, st)
        st.memory["zr"] = float(obs["tcp"][2])
        st.memory["tl"] = float(obs["t"])
        st.memory["lost"] = None
        st.memory["seen_grasped"] = True

    def target(obs, st):
        off = view.inhand(obs)
        p = point(obs)
        z_goal = p[2] - below - off[2]
        if zone is not None:
            if st.memory.get("tz0") is None and view.true_tip(obs)[2] - p[2] < zone:
                st.memory["tz0"], st.memory["tt0"] = float(obs["tcp"][2]), float(obs["t"])
            z = z_goal
            if st.memory.get("tz0") is not None:
                z = max(z_goal, st.memory["tz0"] - speed * (float(obs["t"]) - st.memory["tt0"]))
        elif speed is None:
            z = z_goal
        else:
            t = float(obs["t"])
            st.memory["zr"] = max(z_goal, st.memory["zr"] - speed * (t - st.memory["tl"]))
            st.memory["tl"] = t
            z = st.memory["zr"]
        o = st.memory["orient"] if yaw is None else yaw(obs)
        return _cmd(task, np.r_[p[:2] - off[:2], z], o, 0.0, force)

    def outcome(obs, labels, st):
        if view.lost(obs, st):
            return view.drop_outcome(obs, st)
        held = view.held(obs) or not require_held
        if held and placed(obs) and st.elapsed(obs["t"]) > settle:
            return OK
        if (fail_after is not None and view.held(obs) and st.elapsed(obs["t"]) > fail_after
                and (fail_if is None or fail_if(obs))):
            if fail_hold <= 0.0:
                return fail_key
            st.memory["fail_t"] = st.memory.get("fail_t") or float(obs["t"])
            if obs["t"] - st.memory["fail_t"] > fail_hold:
                return fail_key
        elif fail_hold > 0.0:
            st.memory["fail_t"] = None
        return None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- 6 release
def build_release(task, *, view, margin=0.012, aperture=MAX_APERTURE, lift=0.0, settled=None):
    """Open the hand in place; success = measured opening wider than the object, no pad force
    (and ``settled(obs)`` if given, e.g. the released object at rest)."""
    def target(obs, st):
        return _cmd(task, st.memory["hold"] + [0, 0, lift], st.memory["orient"], aperture, 0.0)

    def outcome(obs, labels, st):
        L, R = view.pads(obs)
        ok = obs["aperture"] > view.width + margin and L + R < 0.3
        if ok and settled is not None:
            ok = bool(settled(obs))
        return OK if ok else None
    return target, outcome, _hold(task)


# --------------------------------------------------------------------------- 7 insert
def build_insert(task, *, view, hole, yaw, depth, inserted, floor_depth, seat_depth, min_depth,
                 force=10.0, search=False, search_force=6.0, v_fast=0.03, v_search=0.02, v_slow=0.012,
                 pitch=0.0025, period=1.0, rmax=0.0075):
    """Force-limited compliant insertion with jam detection (``search=True``: spiral search first).

    hole(obs): hole entrance estimate; depth(obs): measured tip depth below the entrance;
    inserted(obs): task predicate; the descent reference stops ``floor_depth`` below the
    entrance (the force limit stops it physically)."""
    def over_hole_tcp(obs, tip_height):
        off = view.inhand(obs)
        h = hole(obs)
        return np.r_[h[:2] - off[:2], h[2] + tip_height - off[2]]

    def on_enter(obs, st):
        st.memory["zr"] = float(obs["tcp"][2])
        st.memory["tl"] = float(obs["t"])
        st.memory["hist"] = []
        st.memory["seen_grasped"] = True
        st.memory["xy_eng"] = None

    def descend(obs, st, v_main):
        p = over_hole_tcp(obs, -floor_depth)
        d = depth(obs)
        v = v_slow if (-0.006 < d < 0.004 or d > seat_depth - 0.006) else v_main
        t = float(obs["t"])
        st.memory["zr"] = max(p[2], st.memory["zr"] - v * (t - st.memory["tl"]))
        st.memory["tl"] = t
        return np.r_[p[:2], st.memory["zr"]]

    def progress(obs, st, f_lim):
        t, d = obs["t"], depth(obs)
        hist = st.memory["hist"]
        hist.append((t, d))
        while hist and hist[0][0] < t - 0.6:
            hist.pop(0)
        span = hist[-1][0] - hist[0][0] if hist else 0.0
        stalled = span > 0.45 and (max(x for _, x in hist) - hist[0][1]) < 0.001
        if inserted(obs) and (d > seat_depth - 0.003 or (stalled and d > min_depth + 0.004)):
            return "seated"
        if stalled and obs["f_up"] > 0.6 * f_lim and d < min_depth:
            return "jammed"
        return None

    def target(obs, st):
        if not search:
            return _cmd(task, descend(obs, st, v_fast), yaw(obs), 0.0, force)
        el = st.elapsed(obs["t"])
        p = descend(obs, st, v_search if st.memory["xy_eng"] is None else v_fast)
        if st.memory["xy_eng"] is None and depth(obs) > 0.004:
            st.memory["xy_eng"] = obs["tcp"][:2].copy()
        if st.memory["xy_eng"] is not None:
            return _cmd(task, np.r_[st.memory["xy_eng"], p[2]], yaw(obs), 0.0, force)
        th = 2 * np.pi * el / period
        r = min(rmax, pitch * el / period)
        xy = p[:2] + r * np.array([np.cos(th), np.sin(th)])
        return _cmd(task, np.r_[xy, p[2]], yaw(obs), 0.0, search_force)

    def outcome(obs, labels, st):
        if view.lost(obs, st):
            return view.drop_outcome(obs, st)
        if search and st.memory["xy_eng"] is None:
            if st.elapsed(obs["t"]) > period * (rmax / pitch + 1.5) and view.held(obs):
                return "jammed"
            progress(obs, st, search_force)
            return None
        s = progress(obs, st, force)
        if s == "seated" and view.held(obs):
            return OK
        if s == "jammed" and view.held(obs):
            return "jammed"
        return None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- 8 push
def build_push(task, *, top, done, progress_fn, force=25.0, speed=0.015, clear=0.024, press=0.03,
               close_time=1.2, stall_time=0.9):
    """Push down on ``top(obs)`` with the closed fingers, force-limited, until ``done(obs)``.

    Phases: rise with the hand open until it clears the top, close, press.  'stuck' when
    ``progress_fn(obs)`` does not change for ``stall_time`` while the force is at the limit."""
    def on_enter(obs, st):
        st.memory.update(phase="rise", xy=top(obs)[:2].copy(), tl=float(obs["t"]), orient=_orient_now(task, obs),
                         hist=[], zr=float(max(obs["tcp"][2], top(obs)[2] + press)))

    def target(obs, st):
        t = float(obs["t"])
        xy, o = st.memory["xy"], st.memory["orient"]
        if st.memory["phase"] == "rise":
            if obs["tcp"][2] > top(obs)[2] + clear:
                st.memory["phase"], st.memory["t_close"] = "close", t
            st.memory["tl"] = t
            return _cmd(task, np.r_[xy, st.memory["zr"]], o, MAX_APERTURE, 0.0)
        if st.memory["phase"] == "close":
            if obs["aperture"] < 0.004 or t - st.memory["t_close"] > close_time:
                st.memory["phase"] = "press"
            st.memory["tl"] = t
            return _cmd(task, np.r_[xy, st.memory["zr"]], o, 0.0, 0.0)
        st.memory["zr"] = max(top(obs)[2] - press, st.memory["zr"] - speed * (t - st.memory["tl"]))
        st.memory["tl"] = t
        return _cmd(task, np.r_[xy, st.memory["zr"]], o, 0.0, force)

    def outcome(obs, labels, st):
        if st.memory["phase"] == "press" and done(obs):
            return OK
        hist = st.memory["hist"]
        hist.append((obs["t"], progress_fn(obs)))
        while hist and hist[0][0] < obs["t"] - stall_time - 0.1:
            hist.pop(0)
        if (st.memory["phase"] == "press" and hist and hist[-1][0] - hist[0][0] > stall_time
                and abs(hist[-1][1] - hist[0][1]) < 0.0005 and obs["f_up"] > 0.6 * force):
            return "stuck"
        return None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- 9 pull
def build_pull(task, *, view, axis, progress, goal, done, slip, gone, slip_time=0.15, identify=True):
    """Pull the grasped articulation along its motion line until ``done(obs)``.

    axis(obs): unit pull direction estimated at entry; progress(obs): measured joint
    coordinate q; goal: commanded q.  Nominal target: hold + (goal - q0) axis.  With
    ``identify`` and the method's online mode, the line p(q) = p0 + q n is identified by
    recursive least squares from the measured TCP and q while holding.  slip(obs) classifies
    a measured loss of the grasp (held for ``slip_time`` and ``gone(obs)``)."""
    hold = _hold(task)

    def on_enter(obs, st):
        hold(obs, st)
        q0 = progress(obs)
        n = np.asarray(axis(obs), float)
        st.memory.update(q0=q0, n=n, lost=None, rls=None)
        if identify and st.memory.get("online"):
            rls = RecursiveLeastSquares(2, 3, ridge=1.0, forgetting=0.995)
            rls.theta = np.column_stack([obs["tcp"] - q0 * n, n])
            rls.P = np.diag([1e-4, 1e-2])
            st.memory["rls"] = rls

    def target(obs, st):
        rls = st.memory["rls"]
        if rls is not None:
            if view.held(obs):
                rls.update(np.array([1.0, progress(obs)]), obs["tcp"])
            p = rls.predict(np.array([1.0, goal]))
        else:
            p = st.memory["hold"] + (goal - st.memory["q0"]) * st.memory["n"]
        return _cmd(task, p, st.memory["orient"], 0.0)

    def outcome(obs, labels, st):
        if view.held(obs):
            st.memory["lost"] = None
            return OK if done(obs) else None
        st.memory["lost"] = st.memory.get("lost") or obs["t"]
        if obs["t"] - st.memory["lost"] > slip_time and gone(obs):
            return slip(obs)
        return None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- 10 turn
def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _turned(orient, a):
    """Orientation ``orient`` turned by ``a`` about the vertical (yaw or rotation matrix)."""
    if np.ndim(orient) == 2:
        return _rot_z(a) @ orient
    return orient + a


def build_turn(task, *, view, center, done, angle, speed=0.5, track=None, lead=None, identify=False,
               slip=None, gone=None, slip_time=0.15):
    """Rotate the grasp about the vertical axis through ``center(obs)`` by ``angle`` (rad; a
    callable is evaluated at entry), the hand orientation turning with it.

    Default: the angle advances open loop at ``speed`` rad/s on the TCP's circle.  With
    ``track(obs)`` (the measured point of the turning body at the grasp, e.g. a door handle)
    the target leads the measured turned angle of that point by ``lead`` rad on its circle
    (radius re-measured every step); with ``identify`` and the method's online mode the
    circle x^2 + y^2 = 2ax + 2by + c is identified by recursive least squares from the
    tracked points (prior: the nominal circle).  slip(obs): classify a measured loss of the
    grasp (lost for ``slip_time`` and ``gone(obs)``), as in pull; default: view.lost -> 'slipped'."""
    hold = _hold(task)

    def on_enter(obs, st):
        hold(obs, st)
        c = center(obs)
        r = obs["tcp"][:2] - c[:2]
        st.memory.update(c=c, r0=float(np.linalg.norm(r)), th0=float(np.arctan2(r[1], r[0])), a=0.0,
                         tl=float(obs["t"]), lost=None, angle=angle(obs) if callable(angle) else angle)
        if track is not None:
            c2 = np.array(c, float)[:2]
            h = np.asarray(track(obs), float)[:2]
            rh = float(np.linalg.norm(h - c2))
            st.memory.update(c=c2, phi0=float(np.arctan2(h[1] - c2[1], h[0] - c2[0])), rls=None)
            if identify and st.memory.get("online"):
                rls = RecursiveLeastSquares(3, 1, ridge=1.0)
                rls.theta = np.array([[c2[0], c2[1], rh * rh - c2 @ c2]])
                rls.P = np.diag([1e-4, 1e-4, 2.5e-4]) / 1.5e-6
                st.memory["rls"] = rls

    def circle(obs, st, h):
        """Circle of the tracked point: identified (online) or nominal centre, measured radius."""
        rls = st.memory["rls"]
        if rls is not None:
            rls.update(np.array([2 * h[0], 2 * h[1], 1.0]), np.array([h @ h]))
            a, b, cc = rls.theta[0]
            r2 = cc + a * a + b * b
            if r2 > 0.04:
                st.memory["fit"] = np.array([a, b])
                return st.memory["fit"], float(np.sqrt(r2))
        c = st.memory["c"]
        return c, float(np.linalg.norm(h - c))

    def target(obs, st):
        ang = st.memory["angle"]
        if track is not None:
            h = np.asarray(track(obs), float)[:2]
            c, r = circle(obs, st, h)
            sgn = np.sign(ang or 1.0)
            now = wrap_to(float(np.arctan2(h[1] - c[1], h[0] - c[0])) - st.memory["phi0"], 2 * np.pi)
            st.memory["a"] = a = sgn * min(sgn * now + lead, abs(ang))
            th = st.memory["phi0"] + a
            p = np.r_[c[0] + r * np.cos(th), c[1] + r * np.sin(th), st.memory["hold"][2]]
            return _cmd(task, p, _turned(st.memory["orient"], a), 0.0)
        t = float(obs["t"])
        st.memory["a"] = min(abs(ang), abs(st.memory["a"]) + speed * (t - st.memory["tl"])) * np.sign(ang or 1.0)
        st.memory["tl"] = t
        th = st.memory["th0"] + st.memory["a"]
        c = st.memory["c"]
        p = np.r_[c[0] + st.memory["r0"] * np.cos(th), c[1] + st.memory["r0"] * np.sin(th), st.memory["hold"][2]]
        return _cmd(task, p, _turned(st.memory["orient"], st.memory["a"]), 0.0)

    def outcome(obs, labels, st):
        if slip is not None:
            if view.held(obs):
                st.memory["lost"] = None
                return OK if done(obs) else None
            st.memory["lost"] = st.memory.get("lost") or obs["t"]
            if obs["t"] - st.memory["lost"] > slip_time and (gone is None or gone(obs)):
                return slip(obs)
            return None
        if done(obs):
            return OK
        if view.lost(obs, st):
            return "slipped"
        return None
    return target, outcome, on_enter


# --------------------------------------------------------------------------- 11 retreat
def build_retreat(task, *, height=0.12, back=0.0, aperture=MAX_APERTURE, clear=None, open_min=None,
                  tol=None, away=None, horizontal=False, abort=None):
    """Open (to ``aperture``) and move away: up by ``height`` and ``back`` along the negative
    approach axis (full-orientation hands; ``horizontal``: its horizontal direction).  Success:
    every given test holds -- near the target (``tol``), TCP above ``clear(obs)``,
    ``away(obs)``, aperture > ``open_min``; with neither ``tol`` nor ``clear``, 80 % of the
    rise.  abort(obs): optional outcome key that ends the Eddie."""
    def on_enter(obs, st):
        st.memory["hold"] = obs["tcp"].copy()
        st.memory["orient"] = _orient_now(task, obs)
        o = st.memory["orient"]
        a = o[:, 2] if np.ndim(o) == 2 else 0.0
        if horizontal and np.ndim(o) == 2:
            a = np.r_[a[:2], 0.0] / (np.linalg.norm(a[:2]) + 1e-9)
        st.memory["target"] = obs["tcp"] + [0, 0, height] - (back * a if np.ndim(o) == 2 else 0.0)

    def target(obs, st):
        return _cmd(task, st.memory["target"], st.memory["orient"], aperture, 0.0)

    def outcome(obs, labels, st):
        a = abort(obs) if abort is not None else None
        if a:
            return a
        ok = True
        if tol is not None:
            ok &= _near(obs, st.memory["target"], tol)
        if clear is not None:
            ok &= bool(obs["tcp"][2] > clear(obs))
        if tol is None and clear is None:
            ok &= bool(obs["tcp"][2] > st.memory["hold"][2] + 0.8 * height)
        if away is not None:
            ok &= bool(away(obs))
        if open_min is not None:
            ok &= bool(obs["aperture"] > open_min)
        return OK if ok else None
    return target, outcome, on_enter


TAXONOMY = "Bogh et al. 2012; MAPLE (Nasiriany et al. 2021); RAPS (Dalal et al. 2021); FMB (Luo et al. 2024)"
for _name, _fn, _outs, _desc, _src in [
    ("reach", build_reach, ("ok", "toppled", "already_open", "closed_again", "door_closed"), "move the hand to a frame",
     "MAPLE reach; RAPS go-to-pose; Bogh move to"),
    ("grasp", build_grasp, ("ok", "missed"), "close the hand; measured pad contact", "MAPLE grasp; RAPS top grasp; Bogh pick up"),
    ("lift", build_lift, ("ok", "dropped", "slipped", "toppled", "dropped_in_fixture"), "lift the held object", "RAPS lift"),
    ("transport", build_transport, ("ok", "dropped", "dropped_in_fixture", "dropped_in_hole", "toppled"),
     "carry the held object to a frame, faces aligned", "Bogh move to; FMB move to board / rotate"),
    ("place", build_place, ("ok", "missed_place", "missed_fixture", "dropped", "toppled", "touchdown"), "lower the held object onto a support",
     "Bogh place; RAPS drop; FMB place on fixture"),
    ("release", build_release, ("ok",), "open the hand in place", "MAPLE / RAPS / Bogh release"),
    ("insert", build_insert, ("ok", "jammed", "dropped", "dropped_in_hole", "toppled"),
     "force-limited compliant insertion (optional spiral search)", "Bogh align + press; FMB insert"),
    ("push", build_push, ("ok", "stuck"), "force-limited push with closed fingers", "MAPLE / RAPS push; Bogh press"),
    ("pull", build_pull, ("ok", "slipped", "slipped_open"), "pull a grasped articulation along its (identified) line",
     "RAPS pull; Bogh open"),
    ("turn", build_turn, ("ok", "slipped", "slipped_open"), "rotate the grasp about an axis", "RAPS twist; Bogh turn"),
    ("retreat", build_retreat, ("ok", "door_closed"), "open and move away", "reach variant"),
]:
    BOOK.register(Primitive(_name, "1.0", _desc, _outs, _fn, (_src,)))
