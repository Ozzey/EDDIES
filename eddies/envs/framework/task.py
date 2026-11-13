"""Task / skill-token interfaces shared by every robot, task and method.

Vocabulary:

* A **Skill** is a frozen token i with anchor ē_i and trust radius δ_i in the
  normalized command embedding (``Task.space``).  At run time a skill proposes
  a *target* command adapted to the current observation; the Streamer then commits the
  projection of that proposal onto box ∩ B_M(ē_i, δ_i) ∩ B_M(e_{k-1}, ρ).
* A **SkillSpec** is the abstract contract of a token: source cell and a finite
  cover of outcomes, each with a target cell and a COMPLETE abstract label word
  (all intermediate letters, not only the endpoint).  Failure outcomes are
  explicit and are budgeted by the supervisor (see ``supervisor.py``).
* ``Task.cell`` is the observation-to-cell abstraction map; ``Task.labels``
  grounds atomic propositions in measured physical state (contacts, joint
  encoders, object poses), never in issued commands.

Every method (the Streamer and its variants) talks to a task only
through this interface and command the robot only through ``apply_command``
(the shared decoder), so comparisons hold perception, decoder, physics and
skill targets fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

from eddies.common.config import load

SIM = load().sim

from eddies.envs.framework.embedding import EmbeddingSpace

OK = "ok"
TIMEOUT = "timeout"


@dataclass
class SkillState:
    name: str
    start_time: float
    start_command: np.ndarray
    memory: dict = field(default_factory=dict)

    def elapsed(self, t: float) -> float:
        return float(t - self.start_time)


@dataclass
class Skill:
    name: str
    anchor: np.ndarray                     # normalized anchor ē_i
    delta: float                           # trust radius δ_i in metric M (np.inf: box only)
    target: Callable[[dict, SkillState], np.ndarray]
    outcome: Callable[[dict, frozenset, SkillState], str | None]
    timeout: float = 20.0
    nominal_duration: float = 3.0          # nominal duration of the primitive [s]
    on_enter: Callable[[dict, SkillState], None] | None = None
    description: str = ""

    def __post_init__(self):
        self.anchor = np.asarray(self.anchor, float)
        if not np.isfinite(self.anchor).all():
            raise ValueError(f"skill {self.name}: non-finite anchor")
        if not (self.delta > 0):
            raise ValueError(f"skill {self.name}: delta must be positive")


@dataclass(frozen=True)
class OutcomeSpec:
    key: str                               # value returned by Skill.outcome
    target: str                            # abstract target cell
    word: tuple                            # tuple[frozenset[str], ...], nonempty
    failure: bool = False

    def __post_init__(self):
        if not self.word:
            raise ValueError("an outcome needs a nonempty label word")
        object.__setattr__(self, "word", tuple(frozenset(x) for x in self.word))


@dataclass(frozen=True)
class SkillSpec:
    skill: str
    source: str
    outcomes: tuple
    cost: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        if not any(not o.failure for o in self.outcomes):
            raise ValueError(f"{self.skill}@{self.source}: needs at least one non-failure outcome")


def word(*letters) -> tuple:
    """Helper: word({'a'}, {'a','b'}) -> tuple of frozensets; '' or None -> empty letter."""
    out = []
    for x in letters:
        if x is None or x == "":
            out.append(frozenset())
        elif isinstance(x, str):
            out.append(frozenset([x]))
        else:
            out.append(frozenset(x))
    return tuple(out)


class Task:
    """Base class.  Subclasses build a randomized physical scene in ``__init__``."""

    name = "base"
    robot = ""
    family = ""
    description = ""
    # timing
    physics_dt = float(SIM.physics_dt)
    control_dt = float(SIM.control_dt)
    horizon = float(SIM.horizon_s)
    success_hold = float(SIM.success_hold_s)
    label_decimation = 1
    # logic / Streamer parameters
    formula = str(SIM.formula)
    terminal_cells: tuple = ()
    failure_budget = int(SIM.failure_budget)
    rho_step = float(SIM.rho_step)
    rho_game = float(SIM.rho_game)
    # video
    main_camera = "main"
    detail_camera: str | None = None
    video_fps = int(SIM.video.fps)

    def __init__(self, seed: int, **variant):
        self.seed = int(seed)
        self.rng = np.random.default_rng(10_000 + self.seed)
        self.variant = dict(variant)
        # Stress-test condition (paper protocol: dropped grasps, pushes, gusts).
        # Disturbances are EXTERNAL FORCES (data.xfrc_applied) triggered by measured
        # task events and drawn from the task RNG, so every method faces the same
        # disturbance on a seed.  Never teleport, weld or edit object poses.
        self.perturb = bool(self.variant.pop("perturb", False))
        self.model: mujoco.MjModel
        self.data: mujoco.MjData
        self.space: EmbeddingSpace

    # ------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        raise NotImplementedError

    @property
    def time(self) -> float:
        return float(self.data.time)

    # --------------------------------------------------------------- sensing
    def observe(self) -> dict:
        raise NotImplementedError

    def labels(self) -> set:
        raise NotImplementedError

    def task_success(self) -> bool:
        raise NotImplementedError

    def cell(self, obs: dict, labels: frozenset) -> str:
        raise NotImplementedError

    # ---------------------------------------------------------------- skills
    def skills(self) -> dict:
        raise NotImplementedError

    def skill_graph(self) -> list:
        raise NotImplementedError

    def nominal_plan(self) -> list:
        raise NotImplementedError

    def initial_command(self) -> np.ndarray:
        raise NotImplementedError

    def apply_command(self, e: np.ndarray) -> None:
        """Shared decoder: normalized command -> actuator controls (control rate)."""
        raise NotImplementedError

    def inner_loop(self) -> None:
        """Optional physics-rate low-level controller (e.g. attitude loop, PD torques)."""

    def disturb(self) -> None:
        """Physics-rate perturbation hook, active only when ``self.perturb`` is True.

        Set/clear ``self.data.xfrc_applied`` for an external push/impulse (e.g. knock
        a held object out of the grasp, push a door back, a wind gust on a drone).
        Must be triggered by measured state/time only, never by the method's identity.
        """

    def perturbation_info(self) -> dict:
        """Description of the disturbance actually applied (for the manifest)."""
        return {}

    # ----------------------------------------------------------- diagnostics
    def step_diagnostics(self) -> dict:
        """Scalars evaluated at label rate; the runner records their maxima."""
        return {}

    def episode_metrics(self) -> dict:
        return {}

    def log_signals(self) -> dict:
        """Small dict of floats/arrays logged at control rate into trajectory.npz."""
        return {}

    def variant_info(self) -> dict:
        return {}

    def overlay_lines(self) -> list:
        return []
