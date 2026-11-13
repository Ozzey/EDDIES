#!/usr/bin/env python3
"""Identify, calibrate and test Eddie certificates from logged episodes (no simulator needed).

    python -m eddies.certify --robot franka            # one embodiment: its tasks' certificates, fixed gate,
                                                        #   shared Eddies (results/embodiments/embodiments_franka.json)
    python -m eddies.certify --all                      # every robot + LEAP<->Aero cross certificates
                                                        #   (the paper's results/embodiments/embodiments.json)
    python -m eddies.certify --task franka_fmb          # one task, tube certificates per condition
                                                        #   (results/cert/<task>.json; eddies.cert.experiments.certify_task)
    python -m eddies.certify --robot g1 --mu 5,10       # extra flags are passed to the experiment

Episodes are read from $EDDIES_EPISODES (datasets/episodes/sb/nominal/<task>/bistl/seed_XXXX); the
splits, alpha, kappa quantile and identification settings come from eddies/config.yaml and
eddies/configs/experiments/{embodiments,certify_task}.yaml.
"""
from __future__ import annotations

import argparse
import sys

from eddies.common.config import load


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--robot", help="franka | g1 | leap | aero (configs/robots)")
    g.add_argument("--all", action="store_true", help="every robot (results/embodiments/embodiments.json)")
    g.add_argument("--task", help="one task through certify_task (tag/conditions from certify_task.yaml)")
    a, rest = ap.parse_known_args(argv)
    cfg = load()
    if a.task:
        from eddies.cert.experiments import certify_task as mod
        sys.argv = ["eddies.certify", "--task", a.task, *rest]
    else:
        from eddies.cert.experiments import embodiments as mod
        groups = list(cfg.robots) if a.all else [a.robot]
        if not a.all and a.robot not in cfg.robots:
            ap.error(f"unknown robot {a.robot}; one of {', '.join(cfg.robots)}")
        name = "embodiments" if a.all else f"embodiments_{a.robot}"
        sys.argv = ["eddies.certify", "--groups", ",".join(groups), "--name", name, *rest]
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
