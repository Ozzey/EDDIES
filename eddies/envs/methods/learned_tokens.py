"""The Streamer with a LEARNED Eddie proposer (eddies.cert.proposer.token_proposer) in place of the symbolic token choice.

At every hand-over the MLP proposes the next Eddie (or <done>) from measured features; the symbolic supervisor
no longer chooses.  Everything inside an Eddie is unchanged (the Eddie's target, Streamer projection onto
box ∩ trust ball ∩ slew ball, termination from measured labels), so the certificate machinery is untouched.

guard=True  (bistl_learned, bistl_learned_state): the proposal is checked against the contracts: it must be a
            rank-decreasing contract of the measured product state (cell, previous token, LTLf residual,
            failure budget), or <done> exactly when the product state is accepting.  A proposal that fails the
            check is REJECTED and replaced by the proposer's most probable token that passes it (counted).  When no
            contract is winning the robot stops, as with the symbolic supervisor.
guard=False (bistl_learned_raw, bistl_learned_state_raw): the proposal is executed as is (diagnostic: what the
            learned proposer does without the contract check); <done> idles the robot for the rest of the
            episode.  Proposals the check would have rejected are still counted.  The LTLf monitor's
            property_violated stop stays active in both modes.
The symbolic supervisor's own choice is computed at every hand-over for the agreement statistic only.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from eddies.core.ltlf import FALSE, dead
from eddies.envs.framework.supervisor import Decision, ResumableGame, Supervisor
from eddies.envs.methods.streamer import StreamerMethod

DONE = "<done>"
UNVETTED = "unvetted:"


class LearnedSupervisor(Supervisor):
    """Supervisor whose decide() takes the token from a learned proposer (set up by LearnedTokenMethod)."""

    def setup(self, proposer, guard):
        self.proposer = proposer                    # () -> (probabilities over vocab, vocab)
        self.guard = guard
        self.plog = []
        self.pstats = {"proposals": 0, "rejected": 0, "agree_symbolic": 0, "masked_choice_not_top": 0,
                       "unvetted_executed": 0, "done_proposed": 0, "done_premature": 0, "guard_stops": 0}
        self._last_done_cell = None

    def decide(self, cell, residual, current_embedding):
        t0 = time.perf_counter()
        self._maybe_extend(cell)
        state = self.product_state(cell, residual)
        if residual == FALSE or dead(residual):
            return Decision(None, None, None, "property_violated")
        game = ResumableGame(self.contracts, state, current_embedding, self.rho_game, self.terminals, self.metric)
        rank = game.rank.get(state)
        if self.initial_rank is None:
            self.initial_rank = rank
        self.ranks.append(rank)
        allowed = tuple(game.allowed(state)) if rank else ()
        costs = {}
        for name in allowed:
            skill = self.meta[name]["skill"]
            learned = self.costs.values.get(skill) if self.use_online_costs else None
            costs[name] = game.contracts[name].cost if learned is None else learned * self.meta[name]["spec"].cost
        symbolic = (self.meta[game.choose(state, costs)]["skill"] if allowed
                    else (DONE if rank == 0 else None))
        P, vocab = self.proposer()
        order = list(np.argsort(-P))
        raw = vocab[order[0]]
        by_skill = {self.meta[n]["skill"]: n for n in allowed}
        passes = (raw == DONE) if rank == 0 else (raw in by_skill)
        ms = (time.perf_counter() - t0) * 1e3
        # in the unguarded mode a repeated <done> while idling is one proposal, not one per control step
        repeat_done = (not self.guard and raw == DONE and self._last_done_cell == (cell, self.previous))
        if not repeat_done:
            self.pstats["proposals"] += 1
            self.pstats["rejected"] += int(not passes and rank is not None)
            self.pstats["agree_symbolic"] += int(raw == symbolic)
            self.pstats["done_proposed"] += int(raw == DONE)
            self.pstats["done_premature"] += int(raw == DONE and rank != 0)
        entry = {"cell": cell, "rank": rank, "raw": raw, "p_raw": round(float(P[order[0]]), 4),
                 "symbolic": symbolic, "allowed": sorted(by_skill), "passes": bool(passes)}
        if self.guard:
            if rank is None:
                self.pstats["guard_stops"] += 1
                entry["chosen"] = "STOP"
                self.plog.append(entry)
                return Decision(None, None, None, "not_in_winning_set", game_states=len(game.edges), solve_ms=ms)
            if rank == 0:
                entry["chosen"] = DONE
                self.plog.append(entry)
                return Decision(None, None, 0, "terminal_accepting", game_states=len(game.edges), solve_ms=ms)
            pick = next(vocab[i] for i in order if vocab[i] in by_skill)
            self.pstats["masked_choice_not_top"] += int(pick != raw)
            entry["chosen"] = pick
            self.plog.append(entry)
            name = by_skill[pick]
            successors = next(ch for n, ch in game.edges[state] if n == name)
            return Decision(name, pick, rank, "learned" if pick == raw else "learned_masked", tuple(successors),
                            len(game.edges), allowed, ms)
        # ------------------------------------------------------------------ unguarded (diagnostic)
        entry["chosen"] = raw
        if not repeat_done:
            self.plog.append(entry)
        if raw == DONE:
            self._last_done_cell = (cell, self.previous)
            return Decision(None, None, rank, "terminal_accepting", game_states=len(game.edges), solve_ms=ms)
        self._last_done_cell = None
        if raw in by_skill:
            name = by_skill[raw]
            successors = next(ch for n, ch in game.edges[state] if n == name)
            return Decision(name, raw, rank, "learned", tuple(successors), len(game.edges), allowed, ms)
        self.pstats["unvetted_executed"] += 1
        return Decision(UNVETTED + raw, raw, rank, "unvetted", (), len(game.edges), allowed, ms)

    def commit(self, decision):
        if not str(decision.contract).startswith(UNVETTED):
            self.previous = decision.contract

    def observe_outcome(self, decision, outcome_key, duration, new_cell, new_residual, t):
        if str(decision.contract).startswith(UNVETTED):
            entry = {"t": round(float(t), 4), "skill": decision.skill, "contract": None, "rank": decision.rank,
                     "outcome": outcome_key, "classification": "unvetted", "observed_cell": new_cell,
                     "duration": round(float(duration), 4)}
            self.log.append(entry)
            return entry
        return super().observe_outcome(decision, outcome_key, duration, new_cell, new_residual, t)

    def summary(self):
        out = super().summary()
        out["proposer"] = {"guard": self.guard, **self.pstats, "log": self.plog}
        return out


class LearnedTokenMethod(StreamerMethod):
    def __init__(self, task, guard: bool = True, features: str = "full", model: str | None = None, **kw):
        super().__init__(task, **kw)
        from eddies.cert.proposer.token_proposer import MLP, model_dir
        path = Path(model or os.environ.get("EDDIES_TOKPROP_MODEL", "") or model_dir() / f"{task.name}_{features}.npz")
        self.model, self.fspec = MLP.load(path)
        self.model_path = str(path)
        self.guard, self.features = guard, features
        self.name = "bistl_learned" + ("_state" if features == "state" else "") + ("" if guard else "_raw")
        self.display = f"Streamer, learned proposer ({features}{', contract-guarded' if guard else ', unguarded'})"
        self._obs = None

    def reset(self, obs):
        super().reset(obs)
        self.sup.__class__ = LearnedSupervisor
        self.sup.setup(self._propose, self.guard)

    def _propose(self):
        from eddies.cert.proposer.token_proposer import runtime_features
        x = runtime_features(self.fspec, self.task, self._obs, self.tracker.history)
        self.sup._features = x
        return self.model.proba(x)[0], self.fspec.vocab

    def act(self, obs):
        self._obs = obs
        return super().act(obs)

    def info(self):
        out = super().info()
        out["proposer_model"] = self.model_path
        return out


# ------------------------------------------------------------------------------ learned commands inside an Eddie
class _LearnedCommands:
    """Replaces the Eddie's proposal by e_{k-1} + (BC increment); the projection (_project) is unchanged."""

    def _init_cmd(self, cmd_model=None):
        from eddies.cert.proposer.command_proposer import MLPReg
        from eddies.cert.proposer.token_proposer import model_dir
        path = Path(cmd_model or os.environ.get("EDDIES_CMDPROP_MODEL", "") or model_dir() / f"{self.task.name}_cmd.npz")
        self.cmd_model, self.cspec = MLPReg.load(path)
        self.cmd_model_path = str(path)
        self.cmd_stats = {"steps": 0, "proposal_outside_trust": 0}

    def _rank(self, obs, target):
        sk, st = self.tracker.active, self.tracker.state
        steps = int(round((float(obs["t"]) - st.start_time) / self.task.control_dt))
        bits = sum(1 << i for i, p in enumerate(self.cspec.props) if p in obs["labels"])
        x = self.cspec(sk.name, self.task.achieved(obs), self.e_prev, self.task.log_signals(), bits, steps)
        prop = self.e_prev + self.cmd_model.predict(x)[0]
        self.cmd_stats["steps"] += 1
        if np.isfinite(sk.delta) and self.space.norm(prop - sk.anchor) > sk.delta:
            self.cmd_stats["proposal_outside_trust"] += 1
        return prop

    def info(self):
        out = super().info()
        out["command_proposer"] = {"model": self.cmd_model_path, **self.cmd_stats}
        return out


class LearnedTokenCmdMethod(_LearnedCommands, LearnedTokenMethod):
    """Learned Eddie proposer (contract-guarded) AND learned commands inside every Eddie."""

    def __init__(self, task, cmd_model: str | None = None, **kw):
        super().__init__(task, **kw)
        self._init_cmd(cmd_model)
        self.name = "bistl_learned_cmd"
        self.display = "Streamer, learned Eddies + learned commands (projected)"


class SymbolicCmdMethod(_LearnedCommands, StreamerMethod):
    """Symbolic supervisor, learned commands inside every Eddie (isolates the command proposer)."""

    def __init__(self, task, cmd_model: str | None = None, **kw):
        super().__init__(task, **kw)
        self._init_cmd(cmd_model)
        self.name = "bistl_cmd"
        self.display = "Streamer, learned commands (projected)"
