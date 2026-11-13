#!/usr/bin/env python3
"""eddies.cert data collection in Isaac Sim (PhysX): many episodes of ONE task in one process.

    python -m eddies.isaac.run_cert_task --task franka_fmb --seeds 1000-1399 \
        --conditions nominal,perturbed --tag base [--label bistl] [--method bistl] \
        [--rho-scale 2] [--budget 1] [--params cert_params/franka_fmb.json]

Episodes: $EDDIES_RESULTS_ISAAC/<tag>/<condition>/<task>/<label>/seed_XXXX/{manifest.json,
trajectory.npz} (no video).  Same layout as eddies.cert.collect, so every eddies.cert analysis
runs unchanged with EDDIES_EPISODES pointing at the Isaac episodes.  Resumable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # isaac/


def parse_seeds(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def done(path: Path) -> bool:
    m = path / "manifest.json"
    if not m.exists():
        return False
    try:
        return json.loads(m.read_text()).get("status") == "completed"
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--conditions", default="nominal,perturbed")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", default="bistl")
    ap.add_argument("--label", default=None)
    ap.add_argument("--rho-scale", type=float, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--params", default=None, help="JSON with skill_rho / skill_delta / failure_budget")
    a = ap.parse_args()
    from eddies.common.paths import EPISODES_ISAAC
    root = EPISODES_ISAAC
    label = a.label or a.method
    jobs = []
    for cond in a.conditions.split(","):
        for s in parse_seeds(a.seeds):
            out = root / a.tag / cond / a.task / label / f"seed_{s:04d}"
            if not done(out):
                jobs.append((cond, s, out))
    print(f"[cert] {a.task}: {len(jobs)} episodes to run -> {root / a.tag}", flush=True)
    if not jobs:
        return
    from eddies.isaac.app import close, launch
    launch(640, 360)
    from eddies.isaac.runner import run_episode
    from eddies.envs.tasks import TASKS
    import importlib
    mod, cls, _ = TASKS[a.task]
    klass = getattr(importlib.import_module(mod), cls)
    overrides = {}
    if a.rho_scale is not None:
        overrides["rho_step"] = float(klass.rho_step) * a.rho_scale
    if a.budget is not None:
        overrides["failure_budget"] = int(a.budget)
    mkw = {}
    if a.params:
        P = json.loads(Path(a.params).read_text())
        mkw = {k: P[k] for k in ("skill_rho", "skill_delta", "skill_timeout") if k in P}
        if P.get("failure_budget") is not None and a.budget is None:
            overrides["failure_budget"] = int(P["failure_budget"])
    t0, n_ok = time.time(), 0
    try:
        for i, (cond, s, out) in enumerate(jobs, 1):
            man = run_episode(a.task, a.method, s, out, perturb=cond == "perturbed", record=False,
                              save_trajectory=True, method_kwargs=mkw, task_overrides=overrides)
            man["cert_overrides"] = {"tag": a.tag, "label": label, "rho_scale": a.rho_scale,
                                     "budget": overrides.get("failure_budget"), "params": a.params}
            (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
            n_ok += bool(man.get("success"))
            if i % 10 == 0 or i == len(jobs):
                print(f"[cert] {a.task} [{i}/{len(jobs)}] {time.time() - t0:7.0f}s  last {cond}/{s} "
                      f"{man.get('status')} success={man.get('success')}  successes {n_ok}", flush=True)
    finally:
        close()


if __name__ == "__main__":
    main()
