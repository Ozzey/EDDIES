#!/usr/bin/env python3
"""Replay the certify-or-stop gates on logged test episodes (no simulator needed).

    python -m eddies.gate fixed [--pool eddy|primitive] [--mu 2,3,...] [--recal] [--name NAME]
        fixed-calibration gate on the Franka tasks at nominal and faster commanded speeds
        (Table D.5; results/gate/gate_<pool>.json)                      -> eddies.cert.experiments.gate_fixed
    python -m eddies.gate self --task franka_fmb [--known] [--group franka] [--mu ...]
        self-calibrating gate: a task held out of training (no calibration) or started from its own
        calibration (--known); --group <robot> for every task of a robot (Fig. 5, Fig. 7, Tables D.1, D.2;
        results/adaptive/adaptive_<task>[_known][_<group>].json)        -> eddies.cert.experiments.gate_adaptive
    python -m eddies.gate margin [--tasks ...] [--workers N]
        one-step check with a fixed calibration vs threshold / velocity monitors on all ten tasks
        (Appendix D; results/gate_margin/)                                -> eddies.cert.experiments.gate_margin
    python -m eddies.gate margin-table
        the pooled table of the above                                     -> eddies.cert.tables.gate_margin
    python -m eddies.gate ltl             LTLf-specified tasks (results/gate/ltl.json)   -> gate_ltl
    python -m eddies.gate containment     funnel containment vs recursion (Table D.4)   -> containment
    python -m eddies.gate physics         MuJoCo -> PhysX certificates (Table 2)         -> physics
    python -m eddies.gate stack [--rows mujoco,...]  cube stacking MuJoCo / PhysX / robot (Table 3) -> stack_compare
    python -m eddies.gate transfer --task franka_fmb  held-out task, few-shot (Sec. 5.4)  -> transfer
    python -m eddies.gate table1 --variant eddie|eddie_refonly [--robot g1|leap|aero] [--episodes]
        Table 1 rows "Eddies" and "Eddies, one reference model" (results/table1/<variant>.json,
        table1_robots/<robot>/)                                           -> table1_eddies
    python -m eddies.gate calibration [--robot ...]  Table 1 row "Eddies, per-step bound"
        (results/table1/calibration.json)                                 -> calibration_ablation
    python -m eddies.gate pool ...        per-Eddie vs per-primitive certificates (Table D.3)  -> pool
    python -m eddies.gate trace           per-step traces for Fig. 4                           -> trace_figs

Every flag after the sub-command is passed to that experiment; defaults come from
eddies/configs/experiments/<experiment>.yaml.
"""
from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "fixed": "eddies.cert.experiments.gate_fixed",
    "self": "eddies.cert.experiments.gate_adaptive",
    "adaptive": "eddies.cert.experiments.gate_adaptive",
    "margin": "eddies.cert.experiments.gate_margin",
    "margin-table": "eddies.cert.tables.gate_margin",
    "ltl": "eddies.cert.experiments.gate_ltl",
    "containment": "eddies.cert.experiments.containment",
    "physics": "eddies.cert.experiments.physics",
    "stack": "eddies.cert.experiments.stack_compare",
    "transfer": "eddies.cert.experiments.transfer",
    "table1": "eddies.cert.experiments.table1_eddies",
    "calibration": "eddies.cert.experiments.calibration_ablation",
    "pool": "eddies.cert.experiments.pool",
    "trace": "eddies.cert.experiments.trace_figs",
}


def main(argv=None):
    # The dispatcher owns only the command; everything after it (--help included) goes to the sub-module,
    # so `python -m eddies.gate pool --help` prints the experiment's own flags.
    ap = argparse.ArgumentParser(prog="eddies.gate", description=__doc__, add_help=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=sorted(COMMANDS))
    ap.add_argument("-h", "--help", action="store_true")
    argv = list(sys.argv[1:] if argv is None else argv)
    a, rest = ap.parse_known_args(argv)
    if a.command is None:
        ap.print_help()
        return 0 if a.help else 2
    mod = importlib.import_module(COMMANDS[a.command])
    sys.argv = [f"eddies.gate {a.command}", *(["--help"] if a.help else []), *rest]
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
