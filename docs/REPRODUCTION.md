# Reproduction

Three levels. **A** regenerates every table and figure from the committed `results/` in seconds. **B** recomputes the
result files from the episode logs of the dataset with `scripts/reproduce.sh` (minutes to hours per experiment on a
multi-core machine). **C** recollects the episodes themselves in MuJoCo / Isaac Sim. Expected outputs are the paper's
numbers; `tests/test_reproduce_smoke.py` checks the headline ones.

```bash
pip install -e ".[test]"
scripts/download_menagerie.sh                                  # robot models (~130 MB); every step that builds a scene needs them
datasets/download.sh --subset sb,gate,ltl,tokprop,isaac        # episode logs (B only; see the "episodes" column below)
scripts/reproduce.sh --list                                    # the experiments
scripts/reproduce.sh certify table1                            # B: rewrite their result files in results/
python -m eddies.evaluate all                                  # A: tables and figures from results/
pytest -q
```

No scheduler is needed: `scripts/reproduce.sh` runs one experiment at a time in the foreground and sets only
`PYTHONPATH`, `MUJOCO_GL=egl`, one BLAS thread per process and, when unset, `EDDIES_EPISODES` /
`EDDIES_RESULTS_ISAAC` to the download (`datasets/episodes/{mujoco,isaac}`). A cluster is optional: wrap one
`scripts/reproduce.sh <experiment>` per job in your scheduler, with the cores and memory of the "compute" column.
`--dry-run` prints the commands, `--workers N` sets the process pool of `gate-margin` and the collection steps.

## A. Tables and figures from `results/`

```bash
python -m eddies.evaluate all            # or a subset: table1 table2 table3 tableD1 ... fig4 fig5 fig7 supp pooled
```

| paper item | step | inputs (`results/`) | expected |
| --- | --- | --- | --- |
| Table 1 (Eddies rows) | `table1` (bootstrap `B: 2000` from `evaluate.yaml`; `--B 0` for no CI) | `table1/{eddie,eddie_refonly,calibration}.json`, `table1_robots/{leap,aero,g1}/*.json`, `table1/table1_ci_final.json` | Franka Eddies 4.3 cm, 275 / 359; LEAP 11.9, 174 / 174; Aero 6.5, 164 / 340; G1 7.9, 172 / 362 (see README) |
| Table 2 (moved certificates) | `table2` | `embodiments/embodiments.json` (LEAP ↔ Aero), `physics/physics.json` (MuJoCo → PhysX, `refit10`) | LEAP→Aero wheel 20 / 78 borrowed vs 0 / 176 re-identified; … ; PhysX staged insertion 10 / 22 vs 0 / 45, peg 5 / 19 vs 0 / 46 |
| Table 3 (stacking) | `table3` | `compare/stack_{mujoco,physx_sim,physx_own}.json`, `real/table3_robot_rows.json` | MuJoCo 89 / 11, 298 / 298, 96 / 4, 287 / 298; Isaac 31 / 19, 41 / 9; robot 30 / 20, 43 / 7 |
| Table D.1 (per task) | `tableD1` ("Eddies" = the paper's count, 96; "distinct" = 91, see PAPER_RESULTS.md) | `embodiments.json`, `adaptive/adaptive_<task>_known_<robot>.json` ×10, `gate_margin/summary.json` (the one-step column) | All row: 5 viol., 890 fixed, 1968 one-step, 1955 self-calibrating, 1581 / 1627 caught at 2 cm |
| Table D.2 (held out) | `tableD2` | `adaptive/adaptive_<franka task>.json`, `_known.json` | staged insertion 99 / 100 caught, 2 false stops, 98 finished (no calibration) vs 99 / 100, 1, 99; drawer 100 / 100, 5, 95 vs 100 / 100, 0, 100 |
| Table D.3 (per Eddie vs per primitive) | `tableD3` | `pool/final_indist_{eddy,primitive}.json` | coverage 0.940 / 0.925 / 0.900 / 0.835, bounds 5.2 / 4.5 / 6.9 / 8.6 cm vs 10.8 / 10.3 / 11.1 / 15.7 |
| Table D.4 (containment vs recursion) | `tableD4` | `gate/containment.json` | at 5 cm hand-overs 79 / 93, 86 / 100, 69 / 86, 69 / 77 % |
| Table D.5 (faster commands) | `tableD5` | `gate/gate_eddy.json`, `gate/gate_eddy_recal.json` | 1.5× at 5 cm: 35 % leave μ, 100 % refused; recalibrated 1.5× at 15 cm: 41 % complete |
| Table D.6 (shared Eddies) | `tableD6` | `embodiments.json` | pull drawer 3.0 (Franka) vs 8.3 cm (G1); reach peg 2.9–4.9 on all four |
| App. D pooled gates (1,968) | `pooled` (`eddies.gate margin-table`) | `gate_margin/summary.json` (or the per-step logs `<task>.npz`, `<task>_a.json` of the dataset's `derived/gate_margin/`, 68 MB, when present) | 1,968 of 2,000 finish at 5 cm; at 3 cm 387 of 395 stopped, 1,328 finish |
| Fig. 4 | `fig4` | `figs_data/trace_fmb.json` | `res_cert.pdf`: 95 % of the first 100 test episodes inside the envelope (94 % of 200) |
| Fig. 5 | `fig5` | `embodiments.json`, `adaptive_*_known_*.json` | `res_gate.pdf`: Franka 69 % (fixed) vs 98.6 % (self-calibrating) at 5 cm |
| Fig. 7 | `fig7` | `adaptive/adaptive_<task>.json`, `_known.json` | `res_heldout.pdf`: miss rate 10–12 % on the held-out tasks |
| Figs. B.1, D.1, E.1, per-task gate figure | `supp` | `facts/gain_all.json`, `dynamics_map/*.json`, `adaptive/` | `supp_gains`, `supp_dynmap`, `supp_qtrace`, `supp_gate_tasks` |
| Fig. 6 / Fig. C.1 | `eddies.cert.plots.env_strip` | stills (dataset `stills/`) | the two-row task strip |
| Fig. 8 | `eddies.cert.plots.real_setup` | `results/real/calib/`, `assets/results/real_setup.png`, `assets/results/real_stop/` | the real-robot figure |
| text: equilibrium offset (0.20 / 1.53 / 1.92 cm), gains (81 of 85 `a_i < 1`), shrinkage (193 → 2.7 cm) | `scripts/facts/{eq_offset,gain_check,shrink_probe}.py` | `eqoff/eqoff.json`, `facts/gain_all.json`, `facts/shrink_probe.json` | as in Sec. 4.1, App. B |

Figures are written to `assets/results/generated/` (PDF + PNG). Figs. 1–3 are schematic; Fig. 1 is redrawn in
`assets/architectures/`.

## B. Result files from the episode logs

Download the subsets named in the "episodes" column (`datasets/download.sh --subset …`,
[DATA_FORMAT.md](DATA_FORMAT.md)), then `scripts/reproduce.sh <experiment>`; each writes the files that level A reads.
`scripts/reproduce.sh all` runs every experiment below in this order and then `python -m eddies.evaluate all`. The
experiments of the first block need only `sb` and do not read each other's files; the second block needs other
subsets. "Compute" is the budget of the authors' runs (cores, memory, wall-time limit), not a measured runtime.

| experiment | command(s) it runs | writes (`results/`) | episodes | compute | paper |
| --- | --- | --- | --- | --- | --- |
| `certify` | `python -m eddies.certify --all` (`eddies.cert.experiments.embodiments`) | `embodiments/embodiments.json` | `sb` (all ten tasks) | 8 cores, 48 GB, < 8 h | Tab. 2 LEAP ↔ Aero, D.1, D.6, C.2; Fig. 5 (fixed); App. B |
| `adaptive-groups` | `python -m eddies.gate self --group <r> --task <t> --known --mu 2,3,5,7.5,10,15`, ten (robot, task) pairs | `adaptive/adaptive_<t>_known_<r>.json` | `sb` | 10 runs of 4 cores, 32 GB, < 4 h | Fig. 5 (self-calibrating), Tab. D.1, Sec. 5.1 |
| `adaptive` | `python -m eddies.gate self --task <t> --mu 1,2,3,5,7.5,10,15 [--known]`, four Franka tasks | `adaptive/adaptive_<t>[_known].json` | `sb` Franka | 8 runs of 4 cores, 32 GB, < 4 h | Fig. 7, Fig. E.1, Tab. D.2, Sec. 5.4 |
| `gate-margin` | `python -m eddies.gate margin --workers N`; `python -m eddies.gate margin-table --out results/gate_margin/summary.json` | `gate_margin/summary.json` (+ the per-step `<task>.npz`, `<task>_a.json`, not committed) | `sb` | about 16 GB per worker, < 5 h with 10 workers | App. D pooled, Tab. D.1 one-step column, Sec. 5.1 (1,968) |
| `table1` | `python -m eddies.gate table1 --variant {eddie,eddie_refonly} --episodes`; `python -m eddies.gate calibration --episodes`; `python -m eddies.cert.tables.table1_ci --dir table1 --B 2000 --json results/table1/table1_ci_final.json` | `table1/*.json` | `sb` Franka | 16 cores, 50 GB, < 8 h | Tab. 1 Franka, Sec. 5.2, App. D |
| `table1-robots` | the same with `--robot {g1,leap,aero} --out table1_robots/<r>` | `table1_robots/<r>/*.json` | `sb` | 16 cores, 64 GB, < 6 h | Tab. 1 G1 / LEAP / Aero |
| `containment` | `python -m eddies.gate containment` | `gate/containment.json` | `sb` Franka | minutes | Tab. D.4, Sec. 5.1 |
| `pool` | `python -m eddies.gate pool --train F4 --test F4 --pool {eddy,primitive} --calib joint --select --mondrian --name final_indist_<pool>`; the same without `--mondrian`, `--name mixture_<pool>`; vocabulary growth `--train <3 tasks> --test franka_fmb --pool primitive --calib per_label --select --mondrian --n-calib 200 [--extra franka_fmb:push] --name final_grow_v{0,1}`; `python -m eddies.cert.plots.coverage_bb` | `pool/*.json` | `sb` Franka | minutes; 64 GB for the mixture runs | Tab. D.3, App. D mixture text and Beta(181, 20), Prop. E.4 |
| `transfer` | `python -m eddies.gate transfer --task <t> --no-indist` per Franka task; `python -m eddies.gate transfer --only-indist` | `transfer/*.json` | `sb` Franka | 5 runs of 8 cores, 32 GB, < 10 h | Sec. 5.4 (~55 calibration episodes) |
| `trace` | `python -m eddies.gate trace` | `figs_data/trace_fmb.json` | `sb` Franka | 4 cores, 32 GB, < 2 h | Fig. 4 |
| `facts` | `python scripts/facts/eq_offset.py --robots franka,g1,leap,aero --out results/eqoff/eqoff.json`; `scripts/facts/gain_check.py`; `scripts/facts/shrink_probe.py` | `eqoff/eqoff.json`, `facts/gain_all.json`, `facts/shrink_probe.json` | `sb` | 2 cores, 16 GB, < 1 h | Sec. 4.1, Thm. 1 (81 of 85), Fig. B.1, App. B, App. D offsets |
| `fixed` | `python -m eddies.gate fixed`; `python -m eddies.gate fixed --recal` | `gate/gate_eddy{,_recal}.json` | `sb` + `gate` Franka | minutes | Tab. D.5, Sec. 5.1 |
| `ltl` | `python -m eddies.gate ltl`; `python -m eddies.envs.scripts.compose_check --seed 2000 --out results/ltl` (static: the product game only, three of the twelve goals refused) | `gate/ltl.json`, `ltl/compose_check_2000.json` | `sb` + `ltl` | minutes | Sec. 5.4, App. A |
| `physics` | `python -m eddies.gate physics` | `physics/physics.json` | `sb` Franka + Isaac `sb` (the episodes, not Isaac Sim) | 8 cores, 32 GB, < 6 h | Tab. 2 MuJoCo → Isaac Sim, Sec. 5.3 |
| `dynmap` | `python -m eddies.cert.experiments.dynamics_map`; `python -m eddies.cert.plots.dynamics_map --model own` | `dynamics_map/*.json` | `sb` + Isaac `sb` | 8 cores, 48 GB, < 4 h | Fig. D.1 and its App. D statistics |
| `stack` | `python -m eddies.gate stack --rows <r>` for `mujoco`, `physx_sim`, `physx_own`; `python -m eddies.gate stack --table` | `compare/stack_*.json` | `sb` + `gate` + Isaac `sb` + `gate` | 3 runs of 4 cores, 48 GB, < 6 h | Tab. 3 MuJoCo / Isaac Sim rows, Sec. 5.5 |
| `tokprop` (= `tokprop-train`, `tokprop-eval`) | `python -m eddies.cert.proposer.token_proposer --task <t> --train sb:nominal:1000-1199,tokprop:perturbed:1000-1199 --val sb:nominal:1200-1399,tokprop:perturbed:1200-1399 --test tokprop:nominal:2000-2099,tokprop:perturbed:2000-2099` for stacking, staged insertion and peg insertion; `python -m eddies.cert.proposer.command_proposer --task franka_cube_stack` (same `--train`, `--val`); `python -m eddies.cert.experiments.token_proposer_eval --task <t> --gate` (stacking with `--methods` including `bistl_cmd,bistl_learned_cmd`) | `tokprop/*.json` (models `.npz`, not committed) | `sb` + `tokprop` | 8 cores, 32 GB, < 5 h per task | Sec. 5.4, App. D learned proposer |

F4 = `franka_fmb,franka_peg_insertion,franka_cube_stack,franka_drawer`. Most steps build the MuJoCo scene of a task to
read its nominal plan, so the menagerie checkout is needed even though nothing is simulated. `gate-margin` writes 68 MB
of per-step logs next to `summary.json`; they are gitignored and are in the dataset under `derived/gate_margin/`
(`datasets/download.sh --subset gate_margin` copies them into `results/gate_margin/`, where `evaluate pooled` then
reads them instead of the summary). The App. D sentence on the mixture calibration (per primitive the drawer's
coverage falls to 0.68; per Eddie the stack's and the drawer's to 0.87 and 0.83) is printed after Table D.3 by
`python -m eddies.evaluate tableD3` from `pool/mixture_{eddy,primitive}.json`.

## C. The episodes

Instead of `datasets/download.sh`, the episodes can be recollected; finished episodes are skipped, so every step
resumes. Run them before the analyses that read them; for `tokprop` the order is `collect-tokprop`, `tokprop-train`,
`collect-tokprop-learned`, `tokprop-eval`.

| experiment | command(s) it runs | dataset subset | compute |
| --- | --- | --- | --- |
| `collect-sb` | `python -m eddies.collect --config experiments/collect_sb --workers N` (tag `sb`, seeds 1000-1399 and 2000-2199, the ten tasks) | `mujoco/sb` (10 tasks × 600) | 8 cores, 32 GB, < 5 h per task |
| `collect-gate` | `python -m eddies.collect --config experiments/collect_gate --workers N` (tag `gate`, Franka seeds 2000-2099 at 1.5 / 2 / 3 / 4× the slew bound, labels `bistl_rho<s>`) | `mujoco/gate` | 12 cores, 32 GB, < 3 h |
| `collect-ltl` | `python -m eddies.cert.collect --seeds 2000-2099 --conditions nominal,perturbed --tag ltl` on the four hand-written tasks, the four `*_ltl` scenes, and five new goals with `--spec "…" --label {drawer_open,drawer_hold,fmb_stage,fmb_direct,fmb_hole_fixture}` | `mujoco/ltl` (13 scene/spec pairs) | 12 cores, 32 GB, < 4 h |
| `collect-tokprop` | `python -m eddies.cert.collect --tag tokprop --tasks <t> --seeds 1000-1399 --conditions perturbed`, then `--seeds 2000-2099 --conditions nominal,perturbed` | `mujoco/tokprop` (supervisor) | 8 cores, 32 GB, hours per task |
| `collect-tokprop-learned` | the same with `--method bistl_learned{,_raw,_state,_state_raw}` on seeds 2000-2099, and `bistl_cmd`, `bistl_learned_cmd` on stacking | `mujoco/tokprop` (learned proposers) | 8 cores, 32 GB, hours per task |
| `collect-isaac` | `$ISAAC_PY -m eddies.isaac.run_cert_task --task <t> --seeds {1000-1049,1200-1249,2000-2049} --conditions nominal --tag sb` for staged insertion, peg insertion and stacking; stacking also `--seeds 2000-2049 --tag gate --rho-scale <s> --label bistl_rho<s>` | `isaac/sb`, `isaac/gate` | 1 GPU, 8 cores, 64 GB, hours |

Seeds fix the scene variants, so a recollected `sb` episode reproduces the paper's one up to the simulator version
(MuJoCo 3.12, menagerie `0059d43`, recorded in each manifest). Isaac Sim needs its own Python (`ISAAC_PY`,
[external/README.md](../external/README.md)).

Not paper results, so not in `scripts/reproduce.sh`: the stills behind Fig. 6 / Fig. C.1 (in the dataset under
`stills/`; `python -m eddies.envs.scripts.stills --task <t> --seed 2000 --steps <n>` in MuJoCo, `$ISAAC_PY -m
eddies.isaac.stills --seed 2000 --out DIR --tasks <task>:<steps>,…` in Isaac Sim, then `eddies.cert.plots.env_strip`)
and the videos (`MUJOCO_GL=egl python -m eddies.render --task <t> --seed 2000`, about a minute per episode).

## Values taken from the paper

Not recomputed by this release: the other token rows of Table 1 (one global model, one model per primitive, BeT,
FAST + k-means, VQ-BeT) with their App. D results, the reference-model one-step error and the second Franka split
of App. D, and the robot rows of Table 3 and Sec. 5.5 (printed from `results/real/table3_robot_rows.json`; the
robot logs are not released).
The robot protocol is in [REAL_ROBOT.md](REAL_ROBOT.md).
