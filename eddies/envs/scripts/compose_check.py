#!/usr/bin/env python3
"""Composition from LTLf specifications: static synthesis report (+ optional episodes).

    python -m eddies.envs.scripts.compose_check [--seed 2000] [--run N] [--out DIR]

For every (scene, spec) below: the synthesised Eddie sequence, the size of the composed
graph, and (for the specs of the hand-written tasks) whether the sequence equals the
hand-written nominal plan.  Unrealisable specs must be refused before execution.
With --run N, N episodes per spec (the Streamer, nominal) are executed and summarised.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SPECS = [
    # (scene task, spec, hand-written task with the same goal or None)
    ("franka_cube_stack_ltl", "G !bad & F(on_red_blue)", "franka_cube_stack"),
    ("franka_peg_insertion_ltl", "G !bad & F(in_peg_hole)", "franka_peg_insertion"),
    ("franka_drawer_ltl", "G !bad & F(on_box_goal)", "franka_drawer"),
    ("franka_fmb_ltl", "G !bad & F(in_peg_fixture & !holding_peg & F(latched_peg))", "franka_fmb"),
    # new goals in the same scenes (no hand-written graph or plan exists)
    ("franka_drawer_ltl", "G !bad & F(open_drawer)", None),
    ("franka_drawer_ltl", "G !bad & F(open_drawer & F(holding_box))", None),
    ("franka_fmb_ltl", "G !bad & F(in_peg_fixture)", None),
    ("franka_fmb_ltl", "G !bad & F(latched_peg)", None),
    ("franka_fmb_ltl", "G !bad & F(in_peg_hole & F(in_peg_fixture))", None),
    ("franka_peg_insertion_ltl", "G !bad & F(in_peg_hole & F(on_peg_table))", None),
    # unrealisable: must be refused before any motion
    ("franka_drawer_ltl", "G !bad & G !open_drawer & F(on_box_goal)", None),
    ("franka_cube_stack_ltl", "G !bad & G !holding_red & F(on_red_blue)", None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--out", default="results_ltl")
    a = ap.parse_args()
    from eddies.envs.tasks import make_task
    report = []
    for task, spec, hw in SPECS:
        row = {"task": task, "spec": spec, "hand_written": hw}
        try:
            t = make_task(task, a.seed, spec=spec)
            row.update(plan=t.nominal_plan(), cells=len({s.source for s in t.skill_graph()}),
                       transitions=len(t.skill_graph()), realisable=True)
            if hw:
                row["equals_hand_written"] = t.nominal_plan() == make_task(hw, a.seed).nominal_plan()
        except ValueError as ex:
            row.update(realisable=False, reason=str(ex))
        report.append(row)
        print(f"\n[{task}]  {spec}")
        if row["realisable"]:
            print(f"  {len(row['plan'])} Eddies, graph {row['cells']} cells / {row['transitions']} transitions"
                  + (f", equals hand-written: {row['equals_hand_written']}" if hw else ""))
            print("  " + " -> ".join(row["plan"]))
        else:
            print("  REFUSED:", row["reason"])
    if a.run:
        from eddies.envs.framework.runner import run_episode
        for row in report:
            if not row["realisable"]:
                continue
            res = []
            for k in range(a.run):
                seed = a.seed + k
                tag = abs(hash(row["spec"])) % 10 ** 6
                m = run_episode(row["task"], "bistl", seed, Path(a.out) / f"{row['task']}_{tag}" / str(seed),
                                record=False, task_kwargs={"spec": row["spec"]})
                sup = m.get("method_info", {}).get("supervisor", {})
                res.append({"seed": seed, "success": m["success"], "bad": m.get("bad_steps"),
                            "stop": m.get("stop_reason"), "cover_violations": sup.get("cover_violations"),
                            "executed": [d["skill"] for d in sup.get("decisions", [])]})
            row["episodes"] = res
            ok = sum(r["success"] for r in res)
            cv = sum(r["cover_violations"] or 0 for r in res)
            print(f"[{row['task']}] {row['spec']}: success {ok}/{len(res)}, cover violations {cv}, "
                  f"bad {sum(bool(r['bad']) for r in res)}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"compose_check_{a.seed}.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
