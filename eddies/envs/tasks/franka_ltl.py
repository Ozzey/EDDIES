"""Franka scenes driven by an LTLf specification: the Streamer composes the Eddies.

    make_task("franka_drawer_ltl", seed, spec="G !bad & F(on_box_goal)")

A task here supplies only (i) the scene and its Eddies (unchanged from ``franka_sb``),
(ii) each Eddie's symbolic binding (:class:`~eddies.envs.skillbook.contracts.Bind`: object,
site, destination) and (iii) a labelling function ``fluents()`` from MEASURED physical state to
the skillbook fluent vocabulary.  The skill graph, the nominal Eddie sequence, the recovery
transitions, the trust-radius predecessors and the terminal condition are synthesised from the
formula by :mod:`eddies.core.compose` and the robust product game (no hand-written graph or plan).

Termination convention: the task is complete when the formula is accepted AND the hand is
``away`` (empty, open, clear of every object).  Labels: the fluents plus the safety label
``bad`` of the underlying scene (contacts / impacts / a dropped object), monitored every
physics step.  The fluent thresholds are those of the hand-written cell abstractions.
"""
from __future__ import annotations

import numpy as np

from eddies.core.compose import cell_name, compose, parse_cell
from eddies.core.ltlf import Monitor, atoms, consume, parse
from eddies.envs.framework.task import OK, OutcomeSpec, SkillSpec
from eddies.envs.skillbook import BOOK
from eddies.envs.skillbook.contracts import Bind, Scene, schema
from eddies.envs.tasks import franka_drawer as D
from eddies.envs.tasks import franka_fmb as F
from eddies.envs.tasks import franka_peg as P
from eddies.envs.tasks import franka_stack as C
from eddies.envs.tasks.franka_base import TABLE_H
from eddies.envs.tasks.franka_sb import (HOME, FrankaDrawerTokens, FrankaFMBTokens, FrankaPegTokens,
                                          FrankaStackTokens)


class _Stub:
    def __init__(self, anchor, duration):
        self.anchor, self.nominal_duration = np.asarray(anchor, float), float(duration)


class ComposedTask:
    """Mixin (before a ``franka_sb`` token task): composition from ``self.formula``."""
    default_spec = "G !bad"
    BIND: dict = {}
    SCENE: Scene = Scene(objects=())
    INIT: frozenset = frozenset({"away"})         # nominal initial fluents (the run uses the measured ones)
    terminal_fluent = "away"

    def __init__(self, seed: int, spec: str | None = None, **variant):
        self.formula = spec or self.default_spec
        super().__init__(seed, **variant)

    # ------------------------------------------------------------- composition
    def _schemas(self, names):
        out = []
        for n in names:
            if n not in self.BIND:
                continue
            prim = n.split(":", 1)[0]
            out.append(schema(n, prim, self.BIND[n], self.SCENE, BOOK.primitives[prim].outcomes))
        return out

    def _specs(self, transitions):
        specs = []
        for tr in transitions:
            outs = tuple(OutcomeSpec(OK if k == "ok" else k, cell_name(t), w, failure=fl)
                         for k, t, w, fl in tr.outcomes)
            specs.append(SkillSpec(tr.eddie, cell_name(tr.source), outs))
        return specs

    def compose_from(self, cell: str):
        """Graph closure from a measured cell (the supervisor calls this for an unseen cell)."""
        trs, states = compose(self._schemas_all, [parse_cell(cell)])
        terms = [cell_name(s) for s in states if self.terminal_fluent in s]
        self.terminal_cells = tuple(dict.fromkeys(tuple(self.terminal_cells) + tuple(terms)))
        return self._specs(trs), terms

    def _tokens(self, spec, nominal, recovery_from):
        self._schemas_all = self._schemas(list(spec))
        trs, states = compose(self._schemas_all, [self.INIT])
        self._graph = self._specs(trs)
        self.terminal_cells = tuple(cell_name(s) for s in states if self.terminal_fluent in s)
        vocab = set().union(*states) | {"bad"}
        unknown = atoms(parse(self.formula)) - vocab
        if unknown:
            raise ValueError(f"spec atoms {sorted(unknown)} are not fluents of this scene; reachable: {sorted(vocab)}")
        self._plan = self._synthesise_plan({k: _Stub(nominal[k], v[2]) for k, v in spec.items()})
        on_path = set(self._plan)
        sources = {}
        for tr in trs:
            sources.setdefault(tr.eddie, set()).add(tr.source)
        order = list(dict.fromkeys(self._plan + [t.eddie for t in trs]))
        rec = {}
        for n in spec:                       # off-path Eddie: trust radius from its first predecessor,
            if n in on_path:                 # preferring nominal results, then failures, then alternatives
                continue
            best = None
            for rank_p, p in enumerate(order):
                for t in trs:
                    if t.eddie != p or p == n:
                        continue
                    for i, o in enumerate(t.outcomes):
                        if o[1] in sources.get(n, ()):
                            kind = 0 if i == 0 else (1 if o[3] else 2)
                            if best is None or (kind, rank_p) < best[0]:
                                best = ((kind, rank_p), p)
            pred = best[1] if best else None
            if pred is None:                 # never applicable: nearest anchor
                pred = min((m for m in spec if m != n), key=lambda m: self.space.norm(nominal[m] - nominal[n]))
            rec[n] = pred
        self._recovery = rec
        super()._tokens(spec, nominal, rec)

    def _synthesise_plan(self, stubs):
        """Nominal Eddie sequence: play the product game from the nominal initial state and
        follow the winning policy along nominal outcomes (the supervisor's own choices)."""
        from eddies.envs.framework.supervisor import Supervisor
        sup = Supervisor(self._graph, stubs, self.space.metric, self.rho_game, self.terminal_cells,
                         self.failure_budget, use_online_costs=False)
        cell = cell_name(self.INIT)
        residual = consume(parse(self.formula), [self.INIT])
        e = self.cmd(*HOME)                     # the arm's home command (reset pose)
        plan = []
        for _ in range(200):
            dec = sup.decide(cell, residual, e)
            if dec.contract is None:
                if dec.reason != "terminal_accepting":
                    raise ValueError(f"spec {self.formula!r} is not realisable from {cell}: {dec.reason}")
                return plan
            sup.commit(dec)
            sp = sup.meta[dec.contract]["spec"]
            o = next(x for x in sp.outcomes if not x.failure)
            plan.append(dec.skill)
            cell, residual, e = o.target, consume(residual, o.word), stubs[dec.skill].anchor
        raise RuntimeError("plan synthesis did not terminate")

    def skill_graph(self) -> list:
        return list(self._graph)

    def nominal_plan(self) -> list:
        return list(self._plan) if hasattr(self, "_plan") else super().nominal_plan()

    # ---------------------------------------------------------------- labels
    def reset(self) -> None:
        self._mon = None
        super().reset()
        self._mon = Monitor(self.formula)
        self._fl = frozenset()

    def labels(self) -> set:
        safety = super().labels()                 # the scene's 'bad' (and its side effects)
        out = set(self.fluents())
        if "bad" in safety:
            out.add("bad")
        self._fl = frozenset(out)
        if getattr(self, "_mon", None) is not None:
            self._mon.step(self._fl)
        return out

    def cell(self, obs: dict, labels: frozenset) -> str:
        return cell_name(frozenset(labels) - {"bad"})

    def task_success(self) -> bool:
        return bool(self._mon is not None and self._mon.accepts() and self.terminal_fluent in self._fl)

    def episode_metrics(self) -> dict:
        out = super().episode_metrics()
        out.update({"spec": self.formula, "composed_plan": list(self._plan),
                    "scene_task_success": bool(super().task_success())})
        return out

    @staticmethod
    def _engaged(out, near_obj, released, clear):
        """Hand fluents: 'away' if empty, open and clear; else close_<nearest object>."""
        if released and clear:
            out.add("away")
        else:
            out.add(f"close_{near_obj}")


# ============================================================================ stack
class FrankaStackLTL(ComposedTask, FrankaStackTokens):
    name = "franka_cube_stack_ltl"
    default_spec = "G !bad & F(on_red_blue)"
    SCENE = Scene(objects=("red",), drops={"red": {"dropped": ("on_red_table",)}}, high_sites={"red": ("above",)})
    INIT = frozenset({"on_red_table", "away"})
    BIND = {
        "reach:cube_above": Bind("red", site="above"),
        "reach:cube_grasp": Bind("red", site="grasp", after="above"),
        "grasp:cube": Bind("red"),
        "lift:cube": Bind("red"),
        "transport:cube_above": Bind("red", dest="blue"),
        "place:cube_top": Bind("red", dest="blue", rel="on"),
        "release:cube": Bind("red"),
        "retreat:cube": Bind("red"),
    }

    def fluents(self) -> set:
        g = self._geometry()
        f = g["f"]
        tcp, ap = self.arm.tcp_pos(), self.arm.aperture()
        red, blue = g["red"], g["blue"]
        if red[2] < TABLE_H - 0.02:
            return {"lost_red"}
        out = set()
        if g["on_blue"]:
            out.add("on_red_blue")
        elif red[2] < TABLE_H + C.HALF + 0.06:          # resting until lifted 6 cm (hand-written 'lifted')
            out.add("on_red_table")
        if g["grasp"]:
            out.add("holding_red")
            if np.linalg.norm(red[:2] - blue[:2]) < 0.012 and red[2] > blue[2] + 2 * C.HALF + 0.02:
                out.add("over_red_blue")
            return out
        dxy = float(np.linalg.norm(tcp[:2] - red[:2]))
        if not g["on_blue"] and ap > 0.065:
            if dxy < 0.015 and abs(tcp[2] - (red[2] + C.GRASP_DZ)) < 0.012:
                out.add("at_red_grasp")
            elif dxy < 0.02 and 0.06 < tcp[2] - red[2] < 0.16:
                out.add("at_red_above")
        released = f["L_red"] + f["R_red"] < 0.3 and ap > 0.055
        self._engaged(out, "red", released, dxy > 0.05 or tcp[2] > red[2] + C.HALF + 0.05)
        return out


# ============================================================================ peg
PEG_BIND = {
    "reach:peg_above": Bind("peg", site="above"),
    "reach:peg_grasp": Bind("peg", site="grasp", after="above"),
    "grasp:peg": Bind("peg"),
    "lift:peg": Bind("peg"),
    "transport:hole_above": Bind("peg", dest="hole"),
    "insert:hole": Bind("peg", dest="hole"),
    "transport:hole_backoff": Bind("peg", dest="hole", fluent="near_peg_hole", when=("jammed_peg_hole",)),
    "insert:hole_search": Bind("peg", dest="hole", when=("near_peg_hole",)),
    "release:peg": Bind("peg"),
    "retreat:peg": Bind("peg"),
}


class _PegFluents:
    def _peg_fluents(self, fixture: bool) -> set:
        g = self._geometry()
        f = g["f"]
        tcp, ap = self.arm.tcp_pos(), self.arm.aperture()
        tip, top = g["tip"], g["top"]
        if tip[2] < TABLE_H - 0.02 and top[2] < TABLE_H - 0.02:
            return {"lost_peg"}
        out = set()
        ins = g["inserted"]
        in_fix = fixture and g["in_fixture"]
        if ins:
            out.add("in_peg_hole")
        if in_fix:
            out.add("in_peg_fixture")
        if fixture and g["seated"]:
            out.add("latched_peg")
        low = not ins and not in_fix and tip[2] < TABLE_H + 0.05   # resting until lifted 5 cm (hand-written 'lifted')
        if g["grasp"]:
            out.add("holding_peg")
            if not ins and not in_fix:
                site = None
                if fixture:
                    fl = self.fb.to_local(tip)
                    dxy_f = np.linalg.norm(tip[:2] - self.fb.center_xy)
                    if dxy_f < 0.006 and 0.0 < tip[2] - self.fb.top_z <= 0.035:
                        site = "over_peg_fixture"
                    elif max(abs(fl[0]), abs(fl[1])) < 0.035 and tip[2] - self.fb.top_z <= 0.0:
                        site = "jammed_peg_fixture"
                if site is None:
                    hole = self.hb.entrance
                    dz, dxy = tip[2] - hole[2], np.linalg.norm(tip[:2] - hole[:2])
                    loc = self.hb.to_local(tip)
                    if max(abs(loc[0]), abs(loc[1])) < P.BLOCK_W / 2 and -P.HOLE_DEPTH - 0.005 < dz <= 0.003:
                        site = "jammed_peg_hole"
                    elif dxy < 0.006 and dz <= 0.020:
                        site = "near_peg_hole"
                    elif dxy < 0.006 and dz <= 0.080:
                        site = "over_peg_hole"
                if site:
                    out.add(site)
                elif low:
                    out.add("on_peg_table")
            return out
        if low:
            out.add("on_peg_table")
        dxy = float(np.linalg.norm(tcp[:2] - tip[:2]))
        if not ins:
            if g["tilt"] > P.STAND_TILT:
                out.add("fallen_peg")
            elif ap > 0.065:
                if dxy < 0.015 and abs(tcp[2] - (top[2] - P.GRASP_BELOW_TOP)) < 0.012:
                    out.add("at_peg_grasp")
                elif dxy < 0.02 and 0.06 < tcp[2] - top[2] < 0.16:
                    out.add("at_peg_above")
        released = self._released(f["L_peg"] + f["R_peg"], ap)
        self._engaged(out, "peg", released, dxy > 0.05 or tcp[2] > top[2] + 0.05)
        return out


class FrankaPegLTL(ComposedTask, _PegFluents, FrankaPegTokens):
    name = "franka_peg_insertion_ltl"
    default_spec = "G !bad & F(in_peg_hole)"
    SCENE = Scene(objects=("peg",), drops={"peg": {"dropped": ("on_peg_table",), "slipped": ("on_peg_table",),
                                                   "dropped_in_hole": ("in_peg_hole",)}}, high_sites={"peg": ("above",)},
                  receptacles={"peg": ("hole",)})
    INIT = frozenset({"on_peg_table", "away"})
    BIND = PEG_BIND

    def fluents(self) -> set:
        return self._peg_fluents(fixture=False)


class FrankaFMBLTL(ComposedTask, _PegFluents, FrankaFMBTokens):
    name = "franka_fmb_ltl"
    # stage the peg in the fixture and let go of it (regrasp), then latch it in the board
    default_spec = "G !bad & F(in_peg_fixture & !holding_peg & F(latched_peg))"
    SCENE = Scene(objects=("peg",), drops={"peg": {"dropped": ("on_peg_table",), "slipped": ("on_peg_table",),
                                                   "dropped_in_hole": ("in_peg_hole",),
                                                   "dropped_in_fixture": ("in_peg_fixture",)}}, high_sites={"peg": ("above",)},
                  receptacles={"peg": ("hole", "fixture")})
    INIT = frozenset({"on_peg_table", "away"})
    BIND = {**PEG_BIND,
            "transport:fixture_above": Bind("peg", dest="fixture"),
            "place:fixture": Bind("peg", dest="fixture", rel="in",
                                  aborts=(("missed_fixture", ("jammed_peg_fixture",), ("over_peg_fixture",)),)),
            "push:peg_top": Bind("peg", dest="hole")}

    def fluents(self) -> set:
        return self._peg_fluents(fixture=True)


# ============================================================================ drawer
class FrankaDrawerLTL(ComposedTask, FrankaDrawerTokens):
    name = "franka_drawer_ltl"
    default_spec = "G !bad & F(on_box_goal)"
    SCENE = Scene(objects=("handle", "box"), articulated={"handle": "drawer"},
                  drops={"box": {"dropped": ("on_box_table", "in_box_drawer")}},
                  access={"box": (("!in_box_drawer", "open_drawer"),)},      # a box in the drawer needs it open
                  confined={"box": (("in_box_drawer", "open_drawer"),)},    # in the open drawer: opening the hand
                                                                            # hits its walls, rising hits the cabinet
                  high_sites={"box": ("above",), "handle": ("pre",)})
    INIT = frozenset({"in_box_drawer", "away"})
    BIND = {
        "reach:handle_pre": Bind("handle", site="pre"),
        "reach:handle": Bind("handle", site="grasp", after="pre",
                             aborts=(("already_open", ("open_drawer", "close_handle"), ()),)),
        "grasp:handle": Bind("handle"),
        "pull:drawer": Bind("handle", articulation="drawer"),
        "release:handle": Bind("handle"),
        "retreat:handle": Bind("handle"),
        "reach:box_above": Bind("box", site="above", aborts=(("closed_again", ("close_box",), ("open_drawer",)),)),
        "reach:box_grasp": Bind("box", site="grasp", after="above",
                                aborts=(("closed_again", ("close_box",), ("open_drawer",)),)),
        "grasp:box": Bind("box"),
        "lift:box": Bind("box"),
        "transport:goal_above": Bind("box", dest="goal"),
        "place:goal": Bind("box", dest="goal", rel="on"),
        "release:box": Bind("box"),
        "retreat:box": Bind("box"),
    }

    def fluents(self) -> set:
        g = self._geometry()
        f = g["f"]
        tcp, ap = self.arm.tcp_pos(), self.arm.aperture()
        obj = g["obj"]
        if obj[2] < TABLE_H - 0.05:
            return {"lost_box"}
        out = set()
        opened = g["opening"] >= D.OPEN_MIN
        if opened:
            out.add("open_drawer")
        if g["placed"]:
            out.add("on_box_goal")
        elif g["on_table"]:
            out.add("on_box_table")
        elif self.support_z(obj) == self.floor_z0 and not (g["in_air"] and obj[2] > self.floor_z0 + D.OBJ_HALF[2] + 0.10):
            out.add("in_box_drawer")          # over the drawer floor and not lifted 10 cm (hand-written 'lifted')
        if g["hold_obj"]:
            out.add("holding_box")
            if g["in_air"] and np.linalg.norm(obj[:2] - self.goal) < 0.012 and obj[2] > TABLE_H + 2 * D.OBJ_HALF[2] + 0.02:
                out.add("over_box_goal")
        if g["hold_handle"]:
            out.add("holding_handle")
        if g["hold_obj"] or g["hold_handle"]:
            return out
        h = self.handle_pos()
        dh = float(np.linalg.norm(tcp - h))
        if ap > 0.03:
            if dh < (0.05 if opened else 0.012):
                out.add("at_handle_grasp")
            elif not opened and np.linalg.norm(tcp - (h - D.PRE_HANDLE * self.approach_axis())) < 0.02:
                out.add("at_handle_pre")
        dxy = float(np.linalg.norm(tcp[:2] - obj[:2]))
        if not g["placed"] and ap > 0.065 and not any(x.startswith("at_") for x in out):
            if dxy < 0.015 and abs(tcp[2] - (obj[2] + D.GRASP_DZ)) < 0.012:
                out.add("at_box_grasp")
            elif dxy < 0.02 and 0.06 < tcp[2] - obj[2] < 0.18:
                out.add("at_box_above")
        released = f["L_obj"] + f["R_obj"] < 0.3 and f["L_h"] + f["R_h"] < 0.3
        clear = dh > 0.06 and (dxy > 0.05 or tcp[2] > obj[2] + g["half_z"] + 0.08)
        near = "handle" if dh < float(np.linalg.norm(tcp - obj)) else "box"
        self._engaged(out, near, released, clear)
        return out
