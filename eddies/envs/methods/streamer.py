"""The Streamer (ours): bounded skill embeddings + LTLf robust supervisor + online adaptation.

Per control step:

1. Skill termination is decided from measured labels/observations (a close
   command is not a grasp).  The observed outcome, the abstraction cell and the
   physics-rate LTLf residual are fed back to the supervisor.
2. The supervisor re-solves the robust product game from the measured product
   state and selects a rank-decreasing token (online costs only re-order
   rank-decreasing choices).  No winning action -> stop, never a guess.
3. The active token proposes a target command (adaptation inside the frozen
   vocabulary).  Nominal online RLS ranks speed-scaled proposals.
4. The committed command is the exact M-metric projection onto
   box ∩ B_M(ē_i, δ_i) ∩ B_M(e_{k-1}, ρ)  (eddies.core.embedding.EmbeddingProjector).
   When the previous command is farther than ρ from the new token's domain, an
   explicit *bridge* step moves toward the domain under box ∩ slew only (the
   paper's "intermediate bridge skill"); bridge steps are counted.

Ablations: use_slew=False (fixed domain only, ρ=∞), use_supervisor=False
(fixed nominal skill order, no recovery), use_online=False, use_trust=False.
"""
from __future__ import annotations

import numpy as np

from eddies.core.embedding import InfeasibleEmbedding
from eddies.core.ltlf import FALSE
from eddies.core.online import RecursiveLeastSquares
from eddies.envs.framework.supervisor import Supervisor
from eddies.envs.methods.base import OK, Method, SkillTracker


class StreamerMethod(Method):
    name = "bistl"
    display = "Streamer (ours)"

    def __init__(self, task, use_slew: bool = True, use_trust: bool = True,
                 use_supervisor: bool = True, use_online: bool = True,
                 skill_rho: dict | None = None, skill_delta: dict | None = None,
                 skill_timeout: dict | None = None, **kw):
        super().__init__(task, **kw)
        # certified design (eddies.cert): per-skill slew bound rho_i and trust radius delta_i.
        # Defaults (None) keep the task's global rho_step and hand-set radii unchanged.
        self.skill_rho = dict(skill_rho or {})
        self.skill_delta = dict(skill_delta or {})
        self.skill_timeout = dict(skill_timeout or {})
        self.use_slew, self.use_trust = use_slew, use_trust
        self.use_supervisor, self.use_online = use_supervisor, use_online
        if not use_slew:
            self.name, self.display = "bistl_noslew", "Streamer w/o slew (domain only)"
        elif not use_supervisor:
            self.name, self.display = "bistl_nosup", "Streamer w/o supervisor"
        elif not use_online:
            self.name, self.display = "bistl_noonline", "Streamer w/o online"

    # ----------------------------------------------------------------- reset
    def reset(self, obs):
        super().reset(obs)
        task = self.task
        self.tracker = SkillTracker(task, online=self.use_online)
        for name, d in self.skill_delta.items():
            if name in self.tracker.skills and d is not None and d > 0:
                self.tracker.skills[name].delta = float(d)
        for name, tmo in self.skill_timeout.items():
            if name in self.tracker.skills and tmo is not None and tmo > 0:
                self.tracker.skills[name].timeout = float(tmo)
        self.sup = (Supervisor(task.skill_graph(), self.tracker.skills, self.space.metric,
                               task.rho_game, tuple(task.terminal_cells), task.failure_budget,
                               use_online_costs=self.use_online, extend=getattr(task, "compose_from", None))
                    if self.use_supervisor else None)
        self.plan = list(task.nominal_plan())
        self.plan_idx = 0
        self.decision = None
        self.terminal = False
        self.stats = {"control_steps": 0, "projections": 0, "slew_active": 0, "trust_active": 0,
                      "bridge_steps": 0, "projection_failures": 0, "max_jump": 0.0,
                      "max_trust_excess": 0.0, "skills_started": 0}
        has_achieved = hasattr(task, "achieved")
        self.rls = (RecursiveLeastSquares(self.space.dim + 1, self.space.dim, ridge=0.01)
                    if (self.use_online and has_achieved) else None)
        self._rls_prev = None
        self.rls_choices = {"full": 0, "scaled": 0}

    # ------------------------------------------------------------------- act
    def act(self, obs):
        task = self.task
        self.stats["control_steps"] += 1
        residual = obs["ltl_residual"]
        if residual == FALSE or obs.get("ltl_status") == "violated":   # incl. dead residuals
            self.stop("property_violated")
            return self.e_prev
        self._rls_observe(obs)
        if self.tracker.active is not None:
            out = self.tracker.check(obs)
            if out is not None:
                decision = self.decision
                duration = self.tracker.finish(obs, out)
                if self.sup is not None:
                    self.sup.observe_outcome(decision, out, duration, task.cell(obs, obs["labels"]),
                                             residual, obs["t"])
                elif out == OK:
                    self.plan_idx += 1
                else:
                    self.stop(f"no_recovery:{decision or self.plan[self.plan_idx]}:{out}")
                    return self.e_prev
        if self.terminal and self.sup is not None:
            if task.cell(obs, obs["labels"]) not in task.terminal_cells:
                self.terminal = False           # disturbed after completion: re-plan
        if self.tracker.active is None and not self.terminal:
            if self.sup is not None:
                cell = task.cell(obs, obs["labels"])
                dec = self.sup.decide(cell, residual, self.e_prev)
                if dec.contract is None:
                    if dec.reason == "terminal_accepting":
                        self.terminal = True
                    else:
                        self.stop(dec.reason)
                        return self.e_prev
                else:
                    self.sup.commit(dec)
                    self.decision = dec
                    self.tracker.start(dec.skill, obs, self.e_prev)
                    self.stats["skills_started"] += 1
            else:
                if self.plan_idx >= len(self.plan):
                    self.terminal = True
                else:
                    self.decision = self.plan[self.plan_idx]
                    self.tracker.start(self.plan[self.plan_idx], obs, self.e_prev)
                    self.stats["skills_started"] += 1
        if self.terminal or self.tracker.active is None:
            return self.e_prev
        target = self.tracker.target(obs)
        proposal = self._rank(obs, target)
        e = self._project(proposal)
        self.stats["max_jump"] = max(self.stats["max_jump"], self.space.norm(e - self.e_prev))
        self._rls_prev = (obs, e - self.e_prev)
        self.e_prev = e
        return e

    # ------------------------------------------------------------ projection
    def _project(self, proposal):
        sk = self.tracker.active
        sp = self.space
        rho = (self.skill_rho.get(sk.name, self.task.rho_step) if self.use_slew else 1e3)
        use_ball = self.use_trust and np.isfinite(sk.delta)
        self.stats["projections"] += 1
        try:
            if use_ball:
                dist = sp.norm(self.e_prev - sk.anchor)
                if dist > sk.delta + rho - 1e-9:
                    # Bridge: domains are disjoint from the slew ball; head for the
                    # nearest point of the token domain under box ∩ slew only.
                    self.stats["bridge_steps"] += 1
                    nearest = sk.anchor + (self.e_prev - sk.anchor) * (sk.delta / dist)
                    e = sp.projector.project(nearest, self.e_prev, rho)
                else:
                    e = sp.projector.project(proposal, self.e_prev, rho, anchor=sk.anchor,
                                             delta=sk.delta)
                    excess = sp.norm(e - sk.anchor) - sk.delta
                    self.stats["max_trust_excess"] = max(self.stats["max_trust_excess"], excess)
                    if sp.norm(proposal - sk.anchor) > sk.delta + 1e-9:
                        self.stats["trust_active"] += 1
            else:
                e = sp.projector.project(proposal, self.e_prev, rho)
        except InfeasibleEmbedding:
            self.stats["projection_failures"] += 1
            e = self.e_prev.copy()
        if sp.norm(e - self.e_prev) >= rho - 1e-6:
            self.stats["slew_active"] += 1
        return e

    # ------------------------------------------------- nominal online learning
    def _rank(self, obs, target):
        """RLS ranks two speed-scaled proposals (never alters hard constraints)."""
        if self.rls is None or self._rls_samples < 20:
            return target
        achieved = np.asarray(self.task.achieved(obs), float)
        best, best_cost = target, np.inf
        for key, scale in (("full", 1.0), ("scaled", 0.6)):
            cand = self.e_prev + scale * (target - self.e_prev)
            d = cand - self.e_prev
            pred = achieved + self.rls.predict(np.r_[d, 1.0])
            cost = self.space.norm(target - pred) ** 2 + 1e-3 * self.space.norm(d) ** 2
            if cost < best_cost:
                best, best_cost, choice = cand, cost, key
        self.rls_choices[choice] += 1
        return best

    @property
    def _rls_samples(self):
        return getattr(self, "_rls_n", 0)

    def _rls_observe(self, obs):
        if self.rls is None or self._rls_prev is None:
            return
        prev_obs, d = self._rls_prev
        try:
            y = np.asarray(self.task.achieved(obs), float) - np.asarray(self.task.achieved(prev_obs), float)
        except Exception:
            return
        if np.isfinite(y).all():
            self.rls.update(np.r_[d, 1.0], y)
            self._rls_n = self._rls_samples + 1

    # ------------------------------------------------------------- reporting
    def status(self):
        if self.done:
            return f"STOP ({self.done_reason})"
        if self.terminal:
            return "terminal (accepting)"
        if self.tracker.active is None:
            return "-"
        extra = ""
        if self.sup is not None and self.decision is not None:
            extra = f"  rank {self.decision.rank}  fail-budget {self.sup.failures_left}"
        return f"{self.tracker.name}{extra}"

    def info(self):
        out = {"stats": self.stats, "skills": self.tracker.history,
               "rho_step": self.task.rho_step if self.use_slew else None,
               "skill_rho": self.skill_rho or None, "skill_delta": self.skill_delta or None,
               "use_trust": self.use_trust, "use_supervisor": self.use_supervisor,
               "use_online": self.use_online, "rls_choices": self.rls_choices,
               "rls_samples": self._rls_samples}
        if self.sup is not None:
            out["supervisor"] = self.sup.summary()
        return out

