"""Load instrumented episodes: trajectories (z, e, skill), manifests and per-seed skill anchors."""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from eddies.common.paths import CACHE, EPISODES

NON_SKILL = ("STOP", "terminal", "-", "")


def skill_of(status: str) -> str | None:
    tok = str(status).split()
    if not tok or tok[0] in NON_SKILL or tok[0].startswith("STOP") or tok[0] == "terminal":
        return None
    return tok[0]


def episode_dirs(tag, condition, task, method="bistl", seeds=None):
    base = EPISODES / tag / condition / task / method
    if not base.exists():
        return []
    out = []
    for d in sorted(base.glob("seed_*")):
        s = int(d.name.split("_")[1])
        if seeds is not None and s not in seeds:
            continue
        if (d / "manifest.json").exists():
            out.append((s, d))
    return out


def load_episode(d: Path, need_traj: bool = True) -> dict | None:
    man = json.loads((d / "manifest.json").read_text())
    if man.get("status") != "completed":
        return None
    ep = {"manifest": man, "seed": man["seed"]}
    if need_traj:
        f = d / "trajectory.npz"
        if not f.exists():
            return None
        npz = np.load(f, allow_pickle=True)
        if "z" not in npz:
            return None
        ep.update(t=npz["t"], e=npz["e"], z=npz["z"],
                  skill=np.array([skill_of(s) for s in npz["skill"]], dtype=object))
    return ep


def load_split(tag, condition, task, seeds, method="bistl", need_traj=True) -> list[dict]:
    eps = []
    for s, d in episode_dirs(tag, condition, task, method, set(seeds)):
        ep = load_episode(d, need_traj)
        if ep is not None:
            eps.append(ep)
    return eps


# ------------------------------------------------------------- skill anchors
def _anchors_job(args):
    task, seed, perturbed = args
    import warnings
    warnings.filterwarnings("ignore")
    from eddies.envs.tasks import make_task
    t = make_task(task, seed, perturb=perturbed)
    sk = t.skills()
    return seed, {n: (np.asarray(s.anchor, float), float(s.delta)) for n, s in sk.items()}, \
        {"rho_step": float(t.rho_step), "weights": np.asarray(t.space.weights, float),
         "scale": np.asarray(t.space._scale, float), "names": list(t.space.names)}


def anchors(task: str, seeds, perturbed: bool, workers: int = 2) -> tuple[dict, dict]:
    """{seed: {skill: (anchor, delta)}}, space info.  Cached on disk."""
    cache = CACHE / f"anchors_{task}_{'p' if perturbed else 'n'}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    have = {}
    info = None
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        have = z["anchors"].item()
        info = z["info"].item()
    todo = [s for s in seeds if s not in have]
    if todo:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
            for seed, a, inf in ex.map(_anchors_job, [(task, s, perturbed) for s in todo]):
                have[seed] = a
                info = inf
        np.savez(cache, anchors=np.array(have, dtype=object), info=np.array(info, dtype=object))
    return {s: have[s] for s in seeds if s in have}, info
