"""Skillbook registry: a versioned vocabulary of primitives, instantiated as Eddies.

EDDIE = Embedding-space Discrete Dynamical Instruction with Envelope: a primitive bound to one
task frame (``primitive:binding``), executed in the shared embedding chart, with an identified
contracting affine model (the "eddy": a local attractor around its equilibrium z*) and a
certified tube (the envelope)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from eddies.common.config import load
from eddies.envs.framework.task import Skill

ALPHA_SHARE = float(load().skillbook.alpha_share)


@dataclass(frozen=True)
class Primitive:
    name: str
    since: str                      # vocabulary version that introduced it
    description: str
    outcomes: tuple                 # outcome keys the primitive can report (ok + failures)
    build: Callable                 # (task, **args) -> (target, outcome, on_enter)
    sources: tuple = ()             # literature the primitive follows
    alpha_share: float = ALPHA_SHARE   # fixed share of the vocabulary's miscoverage budget alpha (config: skillbook.alpha_share)


@dataclass
class SkillBook:
    version: str = "1.0"
    primitives: dict = field(default_factory=dict)

    def register(self, p: Primitive) -> Primitive:
        """Add a word.  Its share of the miscoverage budget is fixed at registration and the shares
        never sum above 1, so certifying a new primitive (calibrated at alpha * share) leaves every
        existing certificate unchanged and episode coverage stays >= 1 - alpha (union bound)."""
        if p.name in self.primitives:
            raise ValueError(f"primitive {p.name!r} already in the book")
        if not 0 < p.alpha_share <= self.reserve() + 1e-12:
            raise ValueError(f"primitive {p.name!r}: alpha share {p.alpha_share} exceeds the reserve {self.reserve()}")
        self.primitives[p.name] = p
        return p

    def reserve(self) -> float:
        """Unallocated share of the miscoverage budget (for future primitives)."""
        return 1.0 - sum(q.alpha_share for q in self.primitives.values())

    def vocabulary(self) -> list:
        return list(self.primitives)

    def eddy(self, task, primitive: str, binding: str, *, anchor, delta: float,
             timeout: float, duration: float, **args) -> Skill:
        """Instantiate ``primitive`` with arguments for one task: the Eddie ``primitive:binding``."""
        if primitive not in self.primitives:
            raise KeyError(f"{primitive!r} is not in the skillbook vocabulary {self.vocabulary()}")
        target, outcome, on_enter = self.primitives[primitive].build(task, **args)
        name = f"{primitive}:{binding}" if binding else primitive
        sk = Skill(name, anchor, delta, target, outcome, timeout=timeout, nominal_duration=duration,
                   on_enter=on_enter, description=self.primitives[primitive].description)
        sk.primitive = primitive
        sk.binding = binding
        return sk


def primitive_of(token_name: str | None) -> str | None:
    """Eddie name -> primitive: 'reach:peg_above' -> 'reach' (certificates are pooled per primitive)."""
    if token_name is None:
        return None
    return str(token_name).split(":", 1)[0]


BOOK = SkillBook()
from eddies.envs.skillbook import primitives as _primitives  # noqa: E402,F401  (registers the 11 primitives)
