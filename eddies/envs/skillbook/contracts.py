"""Symbolic contracts of the skillbook primitives (for composition from an LTLf specification).

Every primitive has ONE generic contract (preconditions, nominal effect, failure effects) over
a fixed fluent vocabulary; an Eddie only declares its binding (:class:`Bind`: which object,
which site or destination).  No Eddie declares what comes before or after it: the order is
synthesised by the product game (:mod:`eddies.core.compose`).

Fluent vocabulary (o: an object, d: a destination, a: an articulation):
    holding_o        the hand holds o (measured pad contact)
    at_o_site        the open hand is at a named site of o ('above', 'grasp', 'pre')
    close_o          the empty hand is engaged near o (not clear); o = the nearest object
    away             the empty, open hand is clear of every object (retreated / at rest); a hand
                     at a high site (above / pre-grasp) may be clear or not, so those reaches and
                     retreat list both measured results
    on_o_d, in_o_d   o rests on / in d (d = 'table', 'blue', 'hole', 'fixture', 'drawer', 'goal')
    over_o_d         the held o is positioned over d;  near_o_d: just above d (search start)
    jammed_o_d       the held o is stuck on the rim of d
    open_a           articulation a is open;   latched_o: o is pushed home (latch engaged)
    fallen_o         o lies on its side (no top-down grasp exists)
An object with no on_/in_ fluent is unsupported (carried).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from eddies.core.compose import Effect, Schema, lit


@dataclass(frozen=True)
class Bind:
    obj: str                         # manipulated object / handle
    site: str = ""                   # reach: target site
    after: str = ""                  # reach: the hand must already be at this site of obj
    dest: str = ""                   # transport / place / insert / push: destination
    rel: str = "on"                  # place: resulting relation ('on' | 'in')
    fluent: str = ""                 # transport: added fluent (default over_<obj>_<dest>)
    when: tuple = ()                 # extra precondition literals (e.g. a recovery context)
    aborts: tuple = ()               # ((outcome key, add tuple, delete tuple), ...)
    articulation: str = ""           # pull / release of a handle: the articulation it opens


@dataclass(frozen=True)
class Scene:
    """Symbolic facts of a scene: object kinds, where a dropped object can end up, and
    which fluents an object needs before the hand may reach it."""
    objects: tuple                                       # objects the hand interacts with
    articulated: dict = field(default_factory=dict)      # handle -> articulation
    drops: dict = field(default_factory=dict)            # object -> {outcome key: (sink fluents...)}
    access: dict = field(default_factory=dict)           # object -> (clause, ...), clause = (literal str, ...)
    confined: dict = field(default_factory=dict)         # object -> conjunctions of fluents under which the hand may
                                                         # neither open (release) nor retreat straight up (collision)
    high_sites: dict = field(default_factory=dict)       # object -> sites a retreat can end in ('above')
    receptacles: dict = field(default_factory=dict)      # object -> destinations it can be IN (lift may end over them)


def _hand_empty(s):
    return not any(f.startswith("holding_") for f in s)


def _none_other(prefix, o):
    """No fluent prefix_<x>... for an object x != o."""
    p = f"{prefix}_"

    def ok(s):
        return not any(f.startswith(p) and f.split("_")[1] != o for f in s)
    return ok


def _supported(o):
    def ok(s):
        return any(f.startswith((f"on_{o}_", f"in_{o}_")) for f in s)
    return ok


def _unsupported(o):
    sup = _supported(o)
    return lambda s: not sup(s)


def _forbid(conjunctions):
    """CNF clauses excluding every listed conjunction of fluents."""
    return tuple(tuple(lit("!" + x) for x in conj) for conj in conjunctions)


def _clauses(lits):
    return tuple((lit(x),) for x in lits)


ENGAGED = ("at_", "close_")


def _drop_effects(scene, o, keys):
    """A dropped object lands in one of the scene's sinks; the empty hand is then either
    still engaged near it or already clear (both are listed)."""
    out = []
    for key in keys:
        for sink in scene.drops.get(o, {}).get(key, ()):
            for hand in ("close_" + o, "away"):
                out.append((key, Effect(add=frozenset({sink, hand}), delete=frozenset({f"holding_{o}"}),
                                        delete_prefix=(f"over_{o}_", f"near_{o}_", f"jammed_{o}_", f"on_{o}_", f"in_{o}_"))))
    return out


def _aborts(b):
    """An abort leaves the empty hand engaged or (aborted early, still high) already clear."""
    out = []
    for k, a, d in b.aborts:
        out.append((k, Effect(add=frozenset(a), delete=frozenset(d), delete_prefix=ENGAGED)))
        if any(x.startswith("close_") for x in a):
            clear = {x for x in a if not x.startswith("close_")} | {"away"}
            out.append((k, Effect(add=frozenset(clear), delete=frozenset(d), delete_prefix=ENGAGED)))
    return out


def schema(eddie: str, primitive: str, b: Bind, scene: Scene, outcomes=()) -> Schema:
    o = b.obj
    hold = f"holding_{o}"
    drops = _drop_effects(scene, o, [k for k in outcomes if k.startswith(("dropped", "slipped")) and k in
                                      scene.drops.get(o, {})])
    extra = _clauses(b.when)
    if primitive == "reach":
        other_at, other_close = _none_other("at", o), _none_other("close", o)
        free = lambda s: "away" in s or (other_at(s) and other_close(s))   # noqa: E731  (clear, or engaged with o only)
        pre = (_hand_empty, free, (lit(f"!at_{o}_{b.site}"),),
               (lit(f"!fallen_{o}"),)) + tuple(tuple(lit(x) for x in c) for c in scene.access.get(o, ())) + extra
        if b.after:
            pre += ((lit(f"at_{o}_{b.after}"),),)
        ok = Effect(add=frozenset({f"at_{o}_{b.site}", f"close_{o}"}), delete=frozenset({"away"}), delete_prefix=ENGAGED)
        alt = () if b.site == "grasp" else (Effect(add=frozenset({f"at_{o}_{b.site}", "away"}), delete_prefix=ENGAGED),)
        return Schema(eddie, pre, ok, tuple(_aborts(b)), alt_ok=alt)
    if primitive == "grasp":
        pre = ((lit(f"at_{o}_grasp"),),) + extra
        ok = Effect(add=frozenset({hold}), delete=frozenset({"away"}), delete_prefix=ENGAGED)
        missed = Effect(add=frozenset({f"close_{o}"}), delete_prefix=("at_",))
        return Schema(eddie, pre, ok, (("missed", missed),))
    if primitive == "lift":
        pre = ((lit(hold),), _supported(o)) + extra
        ok = Effect(delete_prefix=(f"on_{o}_", f"in_{o}_"))
        # lifted out of a receptacle d, the object may still be measured over / just above d
        alt = tuple(Effect(delete_prefix=(f"on_{o}_", f"in_{o}_"),
                           cond_add=tuple((f"in_{o}_{d}", f"{rel}_{o}_{d}") for d in scene.receptacles.get(o, ())))
                    for rel in ("over", "near"))
        return Schema(eddie, pre, ok, tuple(drops), alt_ok=alt if scene.receptacles.get(o) else ())
    if primitive == "transport":
        pre = ((lit(hold),), _unsupported(o)) + extra
        add = b.fluent or f"over_{o}_{b.dest}"
        ok = Effect(add=frozenset({add}), delete_prefix=(f"over_{o}_", f"near_{o}_", f"jammed_{o}_"))
        return Schema(eddie, pre, ok, tuple(drops))
    if primitive == "place":
        pre = ((lit(hold),), (lit(f"over_{o}_{b.dest}"),)) + extra
        ok = Effect(add=frozenset({f"{b.rel}_{o}_{b.dest}"}), delete=frozenset({f"over_{o}_{b.dest}"}))
        fails = [(k, Effect(add=frozenset(a), delete=frozenset(d))) for k, a, d in b.aborts]
        return Schema(eddie, pre, ok, tuple(fails + drops))
    if primitive == "insert":
        pre = ((lit(hold),),) + (extra or ((lit(f"over_{o}_{b.dest}"),),))
        moved = (f"over_{o}_", f"near_{o}_", f"jammed_{o}_")
        ok = Effect(add=frozenset({f"in_{o}_{b.dest}"}), delete_prefix=moved)
        jam = Effect(add=frozenset({f"jammed_{o}_{b.dest}"}), delete_prefix=moved)
        return Schema(eddie, pre, ok, (("jammed", jam),) + tuple(drops))
    if primitive == "release":
        art = scene.articulated.get(o)
        pre = ((lit(hold),),) + (() if art else (_supported(o),)) + extra + \
            _forbid(scene.confined.get(o, ()))
        add = {f"close_{o}"} | ({f"at_{o}_grasp"} if art else set())
        return Schema(eddie, pre, Effect(add=frozenset(add), delete=frozenset({hold})))
    if primitive == "push":
        pre = (_hand_empty, (lit(f"in_{o}_{b.dest}"),), (lit(f"!latched_{o}"),)) + extra
        ok = Effect(add=frozenset({f"latched_{o}", f"close_{o}"}), delete=frozenset({"away"}), delete_prefix=ENGAGED)
        return Schema(eddie, pre, ok, (("stuck", Effect()),))
    if primitive == "pull":
        a = b.articulation or scene.articulated[o]
        pre = ((lit(hold),), (lit(f"!open_{a}"),)) + extra
        ok = Effect(add=frozenset({f"open_{a}"}))
        slip = Effect(add=frozenset({f"close_{o}"}), delete=frozenset({hold}))
        slip_open = Effect(add=frozenset({f"open_{a}", f"close_{o}"}), delete=frozenset({hold}))
        return Schema(eddie, pre, ok, (("slipped", slip), ("slipped_open", slip_open)))
    if primitive == "retreat":
        pre = ((lit(f"close_{o}"),),) + extra + _forbid(scene.confined.get(o, ()))
        high = tuple(Effect(add=frozenset({"away", f"at_{o}_{site}"}), delete_prefix=ENGAGED) for site in scene.high_sites.get(o, ()))
        return Schema(eddie, pre, Effect(add=frozenset({"away"}), delete_prefix=ENGAGED), alt_ok=high)
    raise KeyError(f"no symbolic contract for primitive {primitive!r}")
