"""Robust LTLf product-game supervisor for physical skill tokens.

The ``eddies.core.contracts.ContractGame`` solves the AND-OR reachability
game over product states s = (cell, previous token, LTLf residual) with the
token-level slew condition ||ē_i - ē_p||_M <= ρ_game (eq. (8) with δ = 0 codes).

Physical skills have genuine failure outcomes (a missed grasp, a slipped
handle).  With an unbounded adversary a failure self-loop has no finite robust
rank, so — as the paper allows ("clocks or budgets are
included when needed") — each cell is augmented with an *environment failure
budget* f: a failure outcome is only in the cover while f > 0 and consumes one
unit.  Theorem 3 then gives a conditional guarantee: *if at most F failures
occur*, the rank-decreasing policy completes φ within r0·Hmax primitive steps.
A failure beyond the budget is logged as an assumption violation, never hidden.

At run time the supervisor re-solves the game from the MEASURED product state
(observed cell, executed previous token, residual produced by monitoring every
physics-step label) — receding-horizon robust synthesis — and records whether
each observed successor was inside the contract's predicted cover.
Online learning (``eddies.core.online.OnlineCost``) only re-orders actions that are
already rank-decreasing; it can never select a stall or an unsafe action.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eddies.core.contracts import Contract, ContractGame, Outcome, State
from eddies.core.ltlf import FALSE, Formula, dead
from eddies.core.online import OnlineCost

INITIAL = "@initial"


def budget_cell(cell: str, f: int) -> str:
    return f"{cell}|f{f}"


def contract_name(skill: str, source: str, f: int) -> str:
    return f"{skill}:{source}@f{f}"


def expand_contracts(specs, skills, budget: int, cost_scale: dict | None = None):
    """SkillSpecs x failure budget -> bistl Contracts (uncertified engineering contracts)."""
    contracts, meta = [], {}
    for spec in specs:
        if spec.skill not in skills:
            raise KeyError(f"skill graph references unknown skill {spec.skill!r}")
        anchor = tuple(float(x) for x in skills[spec.skill].anchor)
        for f in range(budget + 1):
            outs = []
            for o in spec.outcomes:
                if o.failure:
                    if f > 0:
                        outs.append(Outcome(budget_cell(o.target, f - 1), o.word))
                else:
                    outs.append(Outcome(budget_cell(o.target, f), o.word))
            name = contract_name(spec.skill, spec.source, f)
            base_cost = skills[spec.skill].nominal_duration * spec.cost
            contracts.append(Contract(name, budget_cell(spec.source, f), anchor, tuple(outs),
                                      float(base_cost), certified=False))
            meta[name] = {"skill": spec.skill, "spec": spec, "budget": f}
    return contracts, meta


class ResumableGame(ContractGame):
    """ContractGame started from an arbitrary product state (not only the initial word)."""

    def __init__(self, contracts, initial: State, initial_embedding, rho, terminal_cells,
                 metric, max_states: int = 50_000):
        self.contracts = {c.name: c for c in contracts}
        if len(self.contracts) != len(contracts):
            raise ValueError("Duplicate token names")
        self.initial_embedding = np.asarray(initial_embedding, float)
        self.metric = np.asarray(metric, float)
        if not np.isfinite(rho) or rho < 0:
            raise ValueError("Invalid slew radius")
        n = len(self.initial_embedding)
        if any(len(c.embedding) != n for c in contracts):
            raise ValueError("Embedding dimension mismatch")
        self.rho = float(rho)
        self.terminals = set(terminal_cells)
        self.initial = initial
        self.edges, self.rank, self.max_states = {}, {}, max_states
        self._expand()
        self._solve()


@dataclass
class Decision:
    contract: str | None
    skill: str | None
    rank: int | None
    reason: str = ""
    successors: tuple = ()
    game_states: int = 0
    allowed: tuple = ()
    solve_ms: float = 0.0


@dataclass
class Supervisor:
    specs: list
    skills: dict
    metric: np.ndarray
    rho_game: float
    terminal_cells: tuple
    budget: int
    use_online_costs: bool = True
    extend: object = None          # optional cell -> (specs, terminal cells): graph closure from an unseen cell
    failures_left: int = field(init=False)
    previous: str = field(init=False, default=INITIAL)
    log: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.specs = list(self.specs)
        self._build()
        self.extensions = 0
        self.failures_left = self.budget
        self.costs = OnlineCost(alpha=0.5)
        self.assumption_violations = 0
        self.uncovered_outcomes = 0
        self.cover_violations = 0
        self.initial_rank = None
        self.ranks = []

    def _build(self):
        self.contracts, self.meta = expand_contracts(self.specs, self.skills, self.budget)
        self.terminals = {budget_cell(t, f) for t in self.terminal_cells for f in range(self.budget + 1)}
        self._sources = {s.source for s in self.specs}

    def _maybe_extend(self, cell: str) -> None:
        """Composed tasks: a measured cell the graph does not contain gets its closure added."""
        if self.extend is None or cell in self._sources or cell in self.terminal_cells:
            return
        specs, terms = self.extend(cell)
        known = {(s.skill, s.source) for s in self.specs}
        new = [s for s in specs if (s.skill, s.source) not in known]
        self.terminal_cells = tuple(dict.fromkeys(tuple(self.terminal_cells) + tuple(terms)))
        self.specs += new
        self._sources.add(cell)
        if new or terms:
            self.extensions += 1
            self._build()

    # ----------------------------------------------------------------- decide
    def product_state(self, cell: str, residual: Formula) -> State:
        return State(budget_cell(cell, max(self.failures_left, 0)), self.previous, residual)

    def decide(self, cell: str, residual: Formula, current_embedding) -> Decision:
        import time
        t0 = time.perf_counter()
        self._maybe_extend(cell)
        state = self.product_state(cell, residual)
        if residual == FALSE or dead(residual):
            return Decision(None, None, None, "property_violated")
        game = ResumableGame(self.contracts, state, current_embedding, self.rho_game,
                             self.terminals, self.metric)
        rank = game.rank.get(state)
        ms = (time.perf_counter() - t0) * 1e3
        if self.initial_rank is None:
            self.initial_rank = rank
        self.ranks.append(rank)
        if rank is None:
            return Decision(None, None, None, "not_in_winning_set", game_states=len(game.edges), solve_ms=ms)
        if rank == 0:
            return Decision(None, None, 0, "terminal_accepting", game_states=len(game.edges), solve_ms=ms)
        allowed = tuple(game.allowed(state))
        costs = {}
        for name in allowed:
            skill = self.meta[name]["skill"]
            base = game.contracts[name].cost
            learned = self.costs.values.get(skill) if self.use_online_costs else None
            costs[name] = base if learned is None else learned * self.meta[name]["spec"].cost
        name = game.choose(state, costs)
        successors = next(ch for n, ch in game.edges[state] if n == name)
        return Decision(name, self.meta[name]["skill"], rank, "rank_decreasing", tuple(successors),
                        len(game.edges), allowed, ms)

    def commit(self, decision: Decision) -> None:
        self.previous = decision.contract

    # ---------------------------------------------------------------- observe
    def observe_outcome(self, decision: Decision, outcome_key: str, duration: float,
                        new_cell: str, new_residual: Formula, t: float) -> dict:
        """Classify the executed outcome from the MEASURED successor, not the callback key.

        * measured state equals a predicted non-failure successor  -> covered, no charge;
        * equals a predicted failure successor (budget f-1)         -> covered failure, one unit;
        * matches nothing in Post(s, a)                            -> cover violation: an
          unmodelled outcome, charged to the failure budget like a failure and counted.
        Exhausting the budget is an assumption violation of Theorem 3's premise; it is
        counted and exported (never hidden) and the supervisor keeps re-solving.
        """
        spec = self.meta[decision.contract]["spec"]
        keys = {o.key: o for o in spec.outcomes}
        f = self.failures_left
        as_success = State(budget_cell(new_cell, f), decision.contract, new_residual)
        as_failure = State(budget_cell(new_cell, f - 1), decision.contract, new_residual) if f > 0 else None
        if as_success in decision.successors:
            kind = "covered"
        elif as_failure is not None and as_failure in decision.successors:
            kind = "covered_failure"
        else:
            kind = "cover_violation"
        charged = kind != "covered"
        if charged:
            self.failures_left -= 1
        if kind == "cover_violation":
            self.cover_violations += 1
            self.uncovered_outcomes += 1
        violated = self.failures_left < 0
        if violated:
            self.assumption_violations += 1
            self.failures_left = 0
        if self.use_online_costs and outcome_key == "ok" and kind == "covered":
            self.costs.update(decision.skill, max(duration, 1e-3))
        entry = {"t": round(float(t), 4), "skill": decision.skill, "contract": decision.contract,
                 "rank": decision.rank, "outcome": outcome_key,
                 "failure": kind == "covered_failure", "classification": kind,
                 "unlisted_outcome": outcome_key not in keys, "budget_assumption_violated": bool(violated),
                 "observed_cell": new_cell, "successor_covered": kind != "cover_violation",
                 "duration": round(float(duration), 4), "allowed": list(decision.allowed),
                 "game_states": decision.game_states, "solve_ms": round(decision.solve_ms, 3)}
        self.log.append(entry)
        return entry

    def summary(self) -> dict:
        return {"initial_rank": self.initial_rank, "failure_budget": self.budget,
                "failures_left": self.failures_left,
                "assumption_violations": self.assumption_violations,
                "uncovered_outcomes": self.uncovered_outcomes,
                "cover_violations": self.cover_violations,
                "rank_strictly_decreasing": all(a > b for a, b in zip(
                    [r for r in self.ranks if r is not None], [r for r in self.ranks if r is not None][1:])),
                "rank_sequence": self.ranks, "graph_extensions": getattr(self, "extensions", 0), "learned_costs": dict(self.costs.values),
                "decisions": self.log, "contracts": len(self.contracts),
                "certified": False,
                "note": "engineering contracts: outcome covers are designed, not proven tubes"}

