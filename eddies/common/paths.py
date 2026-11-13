"""Filesystem layout of the repository, resolved from ``paths:`` of eddies/config.yaml.

Environment variables (all optional):
    EDDIES_RESULTS        analysis JSON / NPZ written by the experiments        (default results/)
    EDDIES_EPISODES       MuJoCo episode logs                                   (default datasets/episodes/)
    EDDIES_RESULTS_ISAAC  Isaac Sim (PhysX) episode logs                        (default datasets/episodes_isaac/)
    EDDIES_RESULTS_REAL   real-robot episode logs                               (default datasets/episodes_real/)
    EDDIES_MODELS         learned proposer models                               (default results/tokprop/)
    EDDIES_MENAGERIE      mujoco_menagerie checkout                             (default external/mujoco_menagerie/)
    EDDIES_SCRATCH        scratch output of try_task / stills                   (default /tmp/eddies)

Episode logs keep the layout  <episodes>/<tag>/<condition>/<task>/<method>/seed_XXXX/{manifest.json, trajectory.npz}.
For compatibility with a results tree that holds its own ``episodes/`` folder (an alternative
layout), a directory that contains ``episodes/`` is accepted wherever an episode root is expected,
and EDDIES_EPISODES defaults to ``$EDDIES_RESULTS/episodes`` when that folder exists.  Without the variables,
the layout of datasets/download.sh (datasets/episodes/{mujoco,isaac}/episodes/<tag>/...) is found as well.
"""
from __future__ import annotations

import os
from pathlib import Path

from eddies.common.config import REPO, load

_cfg = load()


def resolve(p) -> Path:
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else REPO / p


def _episodes(p: Path) -> Path:
    return p / "episodes" if (p / "episodes").is_dir() else p


_DOWNLOAD_TOPS = ("mujoco", "isaac", "derived", "stills", ".cache")


def _downloaded(sim: str) -> Path | None:
    """<sim>/episodes of the Hugging Face dataset as datasets/download.sh lays it out, if downloaded."""
    p = REPO / "datasets" / "episodes" / sim / "episodes"
    return p if p.is_dir() else None


ROOT = resolve(_cfg.paths.results)                                  # analysis outputs (results/)
CACHE = ROOT / "cache"
if os.environ.get("EDDIES_EPISODES"):
    EPISODES = _episodes(resolve(os.environ["EDDIES_EPISODES"]))
elif (ROOT / "episodes").is_dir():
    EPISODES = ROOT / "episodes"
else:
    EPISODES = resolve(_cfg.paths.episodes)
    if _downloaded("mujoco") and EPISODES == REPO / "datasets" / "episodes" and not any(
            c.is_dir() and c.name not in _DOWNLOAD_TOPS for c in EPISODES.iterdir()):   # no own collection here
        EPISODES = _downloaded("mujoco")
EPISODES_ISAAC = _episodes(resolve(_cfg.paths.episodes_isaac))
if not os.environ.get("EDDIES_RESULTS_ISAAC") and not EPISODES_ISAAC.is_dir() and _downloaded("isaac"):
    EPISODES_ISAAC = _downloaded("isaac")
EPISODES_REAL = _episodes(resolve(_cfg.paths.episodes_real))
MODELS = resolve(_cfg.paths.models)
FIGURES = resolve(_cfg.paths.figures)
MENAGERIE = resolve(_cfg.paths.menagerie)
MENAGERIE_COMMIT = str(_cfg.paths.menagerie_commit)
SCRATCH = resolve(_cfg.paths.scratch)
LAYOUT = str(_cfg.episode_layout)


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def episode_dir(tag, condition, task, method, seed, root: Path | None = None) -> Path:
    return (root or EPISODES) / LAYOUT.format(tag=tag, condition=condition, task=task, method=method, seed=int(seed))


def menagerie(relative: str) -> str:
    path = MENAGERIE / relative
    if not path.exists():
        raise FileNotFoundError(f"menagerie asset missing: {path}  (run scripts/download_menagerie.sh)")
    return str(path)
