#!/usr/bin/env python3
"""Run and log Streamer episodes in MuJoCo (manifest.json + trajectory.npz per episode, no video).

    python -m eddies.collect --config experiments/collect_sb                 # the paper's episodes, every task
    python -m eddies.collect --config experiments/collect_sb --tasks franka_cube_stack --workers 8
    python -m eddies.collect --config experiments/collect_gate               # Franka test seeds at 1.5-4x speed
    python -m eddies.collect --tasks franka_fmb --seeds 2000-2009 --tag demo --conditions nominal --workers 4

Episodes go to $EDDIES_EPISODES/<tag>/<condition>/<task>/<method>/seed_XXXX/ (default datasets/episodes/);
finished episodes are skipped, so a run can be resumed.  Other flags (--rho-scale, --budget, --method,
--label, --spec, --params) are those of eddies.cert.collect.  Heavy: hours of CPU per task;
scripts/reproduce.sh collect-sb runs the paper's collection.
"""
from __future__ import annotations

import argparse
import sys

from eddies.common.config import load


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="experiments/collect_sb | experiments/collect_gate | ...")
    ap.add_argument("--tasks", default=None, help="comma list (default: the config's tasks)")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=None)
    a, rest = ap.parse_known_args(argv)
    from eddies.cert import collect as mod
    cfg = load(a.config) if a.config else load()
    tasks = a.tasks or ",".join(cfg.get("tasks", []))
    seeds = a.seeds or cfg.get("seeds")
    conds = a.conditions or ",".join(cfg.get("conditions", cfg.collect.conditions))
    tag = a.tag or cfg.get("tag")
    if not (tasks and seeds and tag):
        ap.error("need --tasks, --seeds and --tag (or a --config that provides them)")
    base = ["--tasks", tasks, "--seeds", str(seeds), "--conditions", conds, "--tag", tag,
            "--workers", str(a.workers or cfg.collect.workers), *rest]
    scales = cfg.get("rho_scales")
    if scales:                       # one pass per commanded speed, labelled bistl_rho<s>
        for s in scales:
            label = str(cfg.get("label", "bistl_rho{scale:g}")).format(scale=float(s))
            sys.argv = ["eddies.collect", *base, "--rho-scale", f"{float(s):g}", "--label", label]
            mod.main()
        return 0
    sys.argv = ["eddies.collect", *base]
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
