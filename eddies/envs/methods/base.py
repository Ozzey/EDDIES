"""Shared method machinery: skill tracking and the common command interface."""
from __future__ import annotations

import numpy as np

from eddies.envs.framework.task import OK, TIMEOUT, SkillState, Task


class Method:
    name = "base"
    display = "base"

    def __init__(self, task: Task, **kwargs):
        self.task = task
        self.space = task.space
        self.done = False
        self.done_reason = ""
        self.e_prev: np.ndarray | None = None
        self.kwargs = kwargs

    def reset(self, obs: dict) -> None:
        self.e_prev = np.asarray(self.task.initial_command(), float).copy()

    def act(self, obs: dict) -> np.ndarray:
        raise NotImplementedError

    def status(self) -> str:
        return ""

    def label(self) -> str:
        return self.display

    def info(self) -> dict:
        return {}

    def stop(self, reason: str) -> None:
        self.done = True
        self.done_reason = reason


class SkillTracker:
    """Runs one skill token at a time and records the executed skill word."""

    def __init__(self, task: Task, online: bool = False):
        self.task = task
        self.skills = task.skills()
        self.active = None
        self.state: SkillState | None = None
        self.history: list[dict] = []
        self.online = online

    def start(self, name: str, obs: dict, e_now: np.ndarray, **memory) -> None:
        self.active = self.skills[name]
        self.state = SkillState(name, float(obs["t"]), np.asarray(e_now, float).copy(),
                                {"online": self.online, **memory})
        if self.active.on_enter is not None:
            self.active.on_enter(obs, self.state)

    def check(self, obs: dict) -> str | None:
        if self.active is None:
            return None
        out = self.active.outcome(obs, obs["labels"], self.state)
        if out is None and self.state.elapsed(obs["t"]) > self.active.timeout:
            out = TIMEOUT
        return out

    def target(self, obs: dict) -> np.ndarray:
        return np.asarray(self.active.target(obs, self.state), float)

    def finish(self, obs: dict, outcome: str) -> float:
        duration = self.state.elapsed(obs["t"])
        self.history.append({"skill": self.active.name, "start": round(self.state.start_time, 4),
                             "end": round(float(obs["t"]), 4), "outcome": outcome})
        self.active, self.state = None, None
        return duration

    @property
    def name(self) -> str:
        return self.active.name if self.active is not None else "-"


def rate_limited_step(space, e_prev: np.ndarray, target: np.ndarray, rho: float,
                      gain: float = 1.0) -> np.ndarray:
    """Proportional step toward ``target`` saturated at metric step size ``rho``."""
    d = gain * (np.asarray(target, float) - e_prev)
    n = space.norm(d)
    if n > rho > 0:
        d *= rho / n
    return space.clip(e_prev + d)


def min_jerk(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return s ** 3 * (10 - 15 * s + 6 * s * s)


__all__ = ["Method", "SkillTracker", "rate_limited_step", "min_jerk", "OK", "TIMEOUT"]
