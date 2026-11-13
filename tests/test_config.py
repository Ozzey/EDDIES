"""Every YAML loads, the defaults reproduce the paper's protocol, and the task / robot files agree with the code."""
from pathlib import Path

import pytest
import yaml

from eddies.common.config import CONFIGS, GLOBAL, Config, available, blocks, load, mu_list, robots, seeds, span

ALL = sorted(p.relative_to(CONFIGS).with_suffix("").as_posix() for p in CONFIGS.rglob("*.yaml"))


def test_global_loads():
    cfg = load()
    assert isinstance(cfg, Config) and cfg.conformal.alpha == 0.1 and cfg.conformal.kappa_quantile == 0.9
    assert seeds(cfg.splits.train) == list(range(1000, 1200)) and seeds(cfg.splits.calib) == list(range(1200, 1400))
    assert seeds(cfg.splits.test) == list(range(2000, 2200)) and seeds(cfg.splits.gate_test) == list(range(2000, 2100))
    assert cfg.identification.shrink_grid == [0.0, 0.03, 0.3, 3.0] and cfg.identification.reference.min_transitions == 5000
    assert cfg.adaptive_gate.eta == 0.5 and cfg.adaptive_gate.lam0.eddie == 200.0 and cfg.adaptive_gate.h_max_steps == 600
    assert len(cfg.paper_tasks) == 10 and cfg.robots == ["franka", "g1", "leap", "aero"]


@pytest.mark.parametrize("name", ALL)
def test_every_yaml_loads(name):
    cfg = load(name)
    assert isinstance(cfg, Config)
    raw = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    for k in raw:
        if k != "defaults":
            assert k in cfg


def test_overrides_and_interpolation(monkeypatch):
    cfg = load("experiments/gate_fixed", "conformal.alpha=0.05", "mu_cm=[5,10]", "paths.results=/x/y")
    assert cfg.conformal.alpha == 0.05 and cfg.mu_cm == [5, 10] and cfg.paths.results == "/x/y"
    monkeypatch.setenv("EDDIES_RESULTS", "/tmp/eddies_results_test")
    assert load().paths.results == "/tmp/eddies_results_test"
    monkeypatch.delenv("EDDIES_RESULTS")
    assert load().paths.results == "results"


def test_helpers():
    assert seeds("1000-1002,2000") == [1000, 1001, 1002, 2000] and seeds([[1, 2], [5, 5]]) == [1, 2, 5]
    assert span([2000, 2099]) == "2000-2099" and mu_list([5, 7.5]) == [0.05, 0.075]
    b = blocks()
    assert set(b) == {"franka", "g1", "hand"} and b["franka"] == ([0, 1, 2, 3, 4, 5], [0, 1, 2])


def test_experiment_defaults_match_paper():
    assert load("experiments/gate_fixed").mu_cm == [2, 3, 4, 5, 7.5, 10, 15]
    assert load("experiments/embodiments").mu_cm == [2, 3, 5, 7.5, 10, 15]
    assert load("experiments/gate_adaptive").mu_cm == [1, 2, 3, 5, 7.5, 10, 15]
    assert load("experiments/gate_margin").mu_cm == [2, 2.5, 3, 3.5, 4, 4.5, 5]
    assert load("experiments/table1_eddies").mu_cm == [5, 7.5, 10] and load("experiments/table1_ci").B == 2000
    assert load("experiments/transfer").fewshot == [5, 10, 20] and load("experiments/transfer").cal_h == 50
    assert load("experiments/physics").refit == [10, 25, 50] and load("experiments/stack_compare").speeds == [2, 3, 4]
    assert len(load("experiments/ltl").runs) == 13 and load("experiments/token_proposer").mlp.hidden == [64, 64]


def test_robot_files():
    R = robots()
    assert [r.name for r in R.values()] == ["franka", "g1", "leap", "aero"]
    assert R["franka"].n_eddies == 35 and R["g1"].n_eddies == 25 and R["leap"].n_eddies == R["aero"].n_eddies == 13
    assert R["franka"].rho_step == 0.012 and R["g1"].rho_step == 0.0075 and R["leap"].rho_step == 0.009
    assert len(R["franka"].primitives) == 11 and len(R["g1"].primitives) == 10 and len(R["leap"].primitives) == 6
    assert sum(len(r.tasks) for r in R.values()) == 10


def test_task_files_agree_with_classes():
    """configs/tasks/<t>.yaml mirrors the class attributes of eddies.envs (horizon, budget, rho_step, formula)."""
    mujoco = pytest.importorskip("mujoco")  # noqa: F841
    import importlib
    from eddies.envs.tasks import TASKS, PAPER_TASKS
    for t in PAPER_TASKS:
        c = load(f"tasks/{t}")
        mod, cls, _ = TASKS[t]
        K = getattr(importlib.import_module(mod), cls)
        assert c.registry == f"{mod}:{cls}"
        assert c.horizon_s == float(K.horizon) and c.failure_budget == int(K.failure_budget)
        assert c.rho_step == float(K.rho_step) and c.formula == str(K.formula) and c.control_dt == float(K.control_dt)
        assert c.rho_step == load(f"robots/{c.robot}").rho_step


def test_skillbooks_match_robot_files():
    pytest.importorskip("mujoco")
    from eddies.envs.skillbook import BOOK
    from eddies.envs.skillbook.g1 import G1_BOOK
    from eddies.envs.skillbook.hand import HAND_BOOK
    R = robots()
    assert BOOK.vocabulary() == R["franka"].primitives
    assert G1_BOOK.vocabulary() == R["g1"].primitives
    assert HAND_BOOK.vocabulary() == R["leap"].primitives == R["aero"].primitives
