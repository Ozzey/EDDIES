"""Print the skillbook vocabulary and the Eddies each task composes from it.

    python -m eddies.envs.skillbook [--tasks franka_fmb,franka_peg_insertion,...]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from eddies.envs.skillbook import BOOK, primitive_of

DEFAULT = "franka_peg_insertion,franka_cube_stack,franka_drawer,franka_fmb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=DEFAULT)
    a = ap.parse_args()
    from eddies.envs.tasks import make_task
    uses = defaultdict(dict)                       # primitive -> task -> tokens
    plans = {}
    for name in a.tasks.split(","):
        t = make_task(name, 1000)
        plans[name] = t.nominal_plan()
        for tok in t.skills():
            uses[primitive_of(tok)].setdefault(name, []).append(tok)
    print(f"skillbook v{BOOK.version}: {len(BOOK.vocabulary())} primitives\n")
    print(f"{'primitive':10s} {'since':5s} {'eddies':>6s}  {'outcomes':44s} sources")
    for p in BOOK.vocabulary():
        P = BOOK.primitives[p]
        n = sum(len(v) for v in uses[p].values())
        print(f"{p:10s} {P.since:5s} {n:6d}  {','.join(P.outcomes):44s} {'; '.join(P.sources)}")
    print()
    toks = {tok for v in uses.values() for ts in v.values() for tok in ts}
    print(f"{len(toks)} distinct Eddies over {sum(1 for p in uses if uses[p])} primitives in {len(plans)} tasks\n")
    for name, plan in plans.items():
        c = Counter(primitive_of(x) for x in plan)
        print(f"{name}: {len(plan)} Eddie executions, {len(set(plan))} Eddies, primitives "
              + ", ".join(f"{p}x{k}" for p, k in c.items()))
        print("   " + " > ".join(plan))


if __name__ == "__main__":
    main()
