"""Symbolic composition of Eddies from an LTLf specification.

The user gives an LTLf formula over *fluents* (ground facts such as ``holding_red`` or
``on_red_blue``).  Every Eddie carries a symbolic contract (:class:`Schema`): when it may
start (a precondition in conjunctive normal form over fluent literals), what its nominal
outcome changes (add / delete lists, STRIPS style) and what each of its failure outcomes
changes.  From the measured initial fluents, :func:`compose` enumerates the reachable
abstract states (a state = the set of fluents that hold) and emits one transition per
(Eddie, state) where the Eddie is applicable, with all of its outcomes.  That transition
system is exactly the skill graph the robust LTLf product game (``eddies.core.contracts``) is
played on, so the ORDER of Eddies, the recovery after a failure and the end of the task
are all synthesised by the game; nothing about the sequence is written by hand.

Label words.  An outcome from state S to S' emits the two letters (S & S', S'): the fluents
that persist while the Eddie runs, then the new state.  Atoms of the formula must be fluent
names (or ``bad``, which no outcome emits and the runtime monitor observes).  Use
stutter-insensitive formulas (no ``X``): the Eddie's true intermediate trace is only
over-approximated by these two letters; the runtime supervisor re-solves from the MEASURED
state and residual at every hand-over, so a mismatch is a re-plan, never a silent error.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

Literal = tuple          # (fluent, positive: bool)


def lit(name: str) -> Literal:
    """'holding_red' -> positive literal, '!holding_red' -> negative literal."""
    return (name[1:], False) if name.startswith("!") else (name, True)


@dataclass(frozen=True)
class Effect:
    add: frozenset = frozenset()
    delete: frozenset = frozenset()
    delete_prefix: tuple = ()          # delete every fluent starting with one of these prefixes
    cond_add: tuple = ()               # ((if fluent in the source state, add fluent), ...)

    def apply(self, state: frozenset) -> frozenset:
        s = {f for f in state if f not in self.delete and not any(f.startswith(p) for p in self.delete_prefix)}
        return frozenset(s | set(self.add) | {a for c, a in self.cond_add if c in state})


@dataclass(frozen=True)
class Schema:
    """Symbolic contract of one Eddie.

    pre: tuple of clauses; a clause is a tuple of literals and holds when ANY literal holds
    (CNF).  A clause may also be a callable ``state -> bool`` for quantified conditions
    ('the hand holds nothing').  ok: the nominal effect.  failures: (outcome key, Effect)
    pairs; a key may appear several times (several possible measured results)."""
    eddie: str
    pre: tuple
    ok: Effect
    failures: tuple = ()
    cost: float = 1.0
    alt_ok: tuple = ()                 # further nominal results (the measured state may be either)

    def applicable(self, state: frozenset) -> bool:
        for clause in self.pre:
            if callable(clause):
                if not clause(state):
                    return False
            elif not any((f in state) == pos for f, pos in clause):
                return False
        return True


def cell_name(state) -> str:
    """Canonical name of an abstract state (used as a game cell)."""
    s = sorted(state)
    return "+".join(s) if s else "none"


def parse_cell(name: str) -> frozenset:
    return frozenset() if name == "none" else frozenset(name.split("+"))


def outcome_word(src: frozenset, dst: frozenset) -> tuple:
    return (frozenset(src & dst), frozenset(dst))


@dataclass
class Transition:
    eddie: str
    source: frozenset
    outcomes: list = field(default_factory=list)     # [(key, target state, word, failure)]


def compose(schemas, initial, max_states: int = 20000):
    """Forward closure from ``initial`` (an iterable of fluent sets): every reachable state
    and every applicable (Eddie, state) transition with all of its outcomes."""
    starts = [frozenset(s) for s in initial]
    seen, queue, out = set(starts), deque(starts), []
    while queue:
        s = queue.popleft()
        for sc in schemas:
            if not sc.applicable(s):
                continue
            tr = Transition(sc.eddie, s)
            for key, eff, failure in ([("ok", sc.ok, False)] + [("ok", e, False) for e in sc.alt_ok]
                                      + [(k, e, True) for k, e in sc.failures]):
                t = eff.apply(s)
                if t == s and not failure:
                    tr = None                      # an Eddie with no effect never progresses
                    break
                tr.outcomes.append((key, t, outcome_word(s, t), failure))
                if t not in seen:
                    seen.add(t)
                    queue.append(t)
                    if len(seen) > max_states:
                        raise RuntimeError("abstract state cap exceeded")
            if tr is not None:
                out.append(tr)
    return out, seen
