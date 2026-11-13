"""``eddies <command> ...``: one console script for the entry points.

    eddies collect  ...   run and log episodes                 (eddies.collect)
    eddies certify  ...   identify + calibrate + test           (eddies.certify)
    eddies gate     ...   replay the gates on logged episodes   (eddies.gate)
    eddies evaluate ...   tables and figures from results/      (eddies.evaluate)
    eddies render   ...   video of one episode                  (eddies.render)
    eddies try      ...   smoke-run a task                      (eddies.envs.scripts.try_task)
    eddies config   [NAME] [key=value ...]   print a resolved configuration
"""
from __future__ import annotations

import importlib
import sys

COMMANDS = {"collect": "eddies.collect", "certify": "eddies.certify", "gate": "eddies.gate", "evaluate": "eddies.evaluate",
            "render": "eddies.render", "try": "eddies.envs.scripts.try_task"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "config":
        import yaml
        from eddies.common.config import load
        name = rest[0] if rest and "=" not in rest[0] else None
        cfg = load(name, *[x for x in rest if "=" in x])
        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        return 0
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}\n{__doc__}")
        return 2
    mod = importlib.import_module(COMMANDS[cmd])
    sys.argv = [f"eddies {cmd}", *rest]
    return mod.main(rest) if cmd != "try" else mod.main()


if __name__ == "__main__":
    sys.exit(main())
