# Usage

Five entry points under `eddies/` (also reachable through the console script `eddies <command>`), each a thin
wrapper around the experiment modules of `eddies/cert/experiments/`. Every flag after a sub-command is passed to
that experiment; its defaults come from `eddies/config.yaml` and `eddies/configs/experiments/<name>.yaml`
([CONFIGS.md](CONFIGS.md)).

| command | what | needs |
| --- | --- | --- |
| `python -m eddies.evaluate` | the paper's tables (markdown) and figures from `results/` | `results/` only |
| `python -m eddies.certify` | identify the Eddies, calibrate the envelopes, test the tubes | episode logs |
| `python -m eddies.gate` | replay the fixed-calibration and the self-calibrating gate; Tables 1–3, D.3–D.5, LTL_f | episode logs |
| `python -m eddies.collect` | run and log Streamer episodes in MuJoCo | simulator |
| `python -m eddies.render` | video of one episode | simulator + EGL |

## `eddies.evaluate` — tables and figures from `results/`

```bash
python -m eddies.evaluate --list                  # steps and whether their inputs are present
python -m eddies.evaluate all                     # everything
python -m eddies.evaluate table1 --B 2000         # Table 1 with the paired bootstrap intervals (B draws; 0 = none)
python -m eddies.evaluate tableD1 --strict        # fail instead of skipping when an input is missing
```

| step | paper | inputs in `results/` |
| --- | --- | --- |
| `table1` | Table 1, Eddies rows (Eddies and the two ablations on each robot) | `table1/`, `table1_robots/<robot>/` |
| `table2` | Table 2 (certificates moved between systems) | `embodiments/embodiments.json`, `physics/physics.json` |
| `table3` | Table 3 (cube stacking: MuJoCo / Isaac Sim / robot) | `compare/stack_*.json`, `real/table3_robot_rows.json` |
| `tableD1` … `tableD6` | Tables D.1–D.6 | `embodiments`, `adaptive/`, `pool/`, `gate/` |
| `pooled` | App. D "1,968" pooled gates | `gate_margin/summary.json` (or the per-step logs from the dataset) |
| `fig4`, `fig5`, `fig7` | Figs. 4, 5, 7 | `figs_data/trace_fmb.json`, `embodiments`, `adaptive/` |
| `supp` | Figs. B.1, D.1, E.1 and the per-task gate figure | `facts/gain_all.json`, `dynamics_map/`, `adaptive/` |

Tables print as markdown; figures go to `assets/results/generated/` (`paths.figures`). The expected output of the
paper is listed step by step in [REPRODUCTION.md](REPRODUCTION.md).

## `eddies.certify` — identification, calibration, tubes

```bash
python -m eddies.certify --robot franka            # the Franka's four tasks: Eddies, violations, tube radii, fixed-calibration gate,
                                                   #   shared Eddies -> results/embodiments/embodiments_franka.json
python -m eddies.certify --all                     # every robot + LEAP <-> Aero cross certificates (the paper's embodiments.json)
python -m eddies.certify --task franka_fmb         # one task through certify_task: tubes per condition -> results/cert/<task>.json
python -m eddies.certify --robot g1 --mu 5,10      # flags are passed on (here the tolerances of the gate)
```

Reads `$EDDIES_EPISODES/sb/nominal/<task>/bistl/seed_XXXX/` with the splits of `config.yaml` (train 1000–1199,
calibration 1200–1399, test 2000–2199), `conformal.alpha = 0.1`, `conformal.kappa_quantile = 0.9` and the
identification settings (shrinkage grid, reference rule, stability cuts).

## `eddies.gate` — the gates and the remaining experiments

```bash
python -m eddies.gate fixed [--pool eddy|primitive] [--mu 2,3,...] [--recal] [--name NAME]
      # fixed-calibration gate on the Franka tasks at nominal and 1.5-4x speed (Table D.5) -> results/gate/gate_<pool>.json
python -m eddies.gate self --task franka_fmb                # task held out of training, no calibration (Fig. 7, Table D.2)
python -m eddies.gate self --task franka_fmb --known        # started from the task's own calibration
python -m eddies.gate self --group franka --task franka_cube_stack --known --mu 2,3,5,7.5,10,15   # Fig. 5 / Table D.1 row
python -m eddies.gate margin [--tasks a,b] [--workers N]    # one-step check with a fixed calibration vs monitors, ten tasks
python -m eddies.gate margin-table                          # the pooled table of the above (App. D: 1,968 / 387 of 395 / 1,328)
python -m eddies.gate ltl                                   # LTLf-specified tasks (Sec. 5.4, App. A) -> results/gate/ltl.json
python -m eddies.gate containment                           # funnel containment vs recursion (Table D.4)
python -m eddies.gate physics                               # MuJoCo -> PhysX certificates (Table 2)
python -m eddies.gate stack [--rows mujoco,physx_sim,...]   # cube stacking MuJoCo / PhysX / robot (Table 3)
python -m eddies.gate transfer --task franka_fmb            # held-out task, few-shot re-identification (Sec. 5.4)
python -m eddies.gate table1 --variant eddie_refonly [--robot leap] [--episodes]   # Table 1: Eddies / one reference model
python -m eddies.gate calibration [--robot leap]            # Table 1: per-step bound + union bound (last row)
python -m eddies.gate pool --train <tasks> --test <tasks> --pool eddy|primitive --calib joint --select --mondrian --name N  # Table D.3
python -m eddies.gate trace                                 # per-step traces for Fig. 4 -> results/figs_data/trace_fmb.json
```

`self` and `adaptive` are the same command. `--smoke` runs any of the heavy ones on a slice (`smoke.per_split`,
`smoke.n_test`). The subcommand → module map is in `eddies/gate.py` (`COMMANDS`).

## `eddies.collect` — episodes (simulator, heavy)

```bash
python -m eddies.collect --config experiments/collect_sb                       # the paper's episodes, every task
python -m eddies.collect --config experiments/collect_sb --tasks franka_cube_stack --workers 8
python -m eddies.collect --config experiments/collect_gate                     # Franka test seeds at 1.5-4x speed
python -m eddies.collect --tasks franka_fmb --seeds 2000-2009 --tag demo --conditions nominal --workers 4
```

Episodes go to `$EDDIES_EPISODES/<tag>/<condition>/<task>/<method>/seed_XXXX/{manifest.json, trajectory.npz}`;
finished episodes are skipped, so a run resumes. Other flags (`--rho-scale`, `--budget`, `--method`, `--label`,
`--spec`, `--params`) are those of `eddies.cert.collect`. A task's 600 episodes take hours of CPU: run one task per process or machine
(`--tasks <t> --workers N`; `scripts/reproduce.sh collect-sb` runs them all). Isaac Sim episodes:
`python -m eddies.isaac.run_cert_task --task franka_fmb --seeds 1000-1049,1200-1249,2000-2049 --tag sb`.

## `eddies.render` and the developer scripts

```bash
MUJOCO_GL=egl python -m eddies.render --task franka_cube_stack --seed 2000 [--rho-scale 2] [--perturb] [--horizon 20] [--out DIR]
python -m eddies.envs.scripts.try_task --task g1_drawer_sb --methods bistl --seeds 0-9 --workers 8 [--video]
python -m eddies.envs.scripts.stills --task franka_fmb --seed 2000 --steps 250 --cams main,detail --out /tmp/stills   # Fig. 6 inputs
python -m eddies.envs.scripts.compose_check --seed 2000 [--run N]      # LTLf synthesis report: plans recovered, goals refused
```

`render` writes `<out>/<task>_s<seed>[_<label>]/{video.mp4, manifest.json, trajectory.npz}`; the hand-overs of the
Eddie stream are in `manifest["method_info"]["skills"]` (name, outcome, start / end), which the video overlays use.

## Configuration overrides

Any key of the resolved configuration can be overridden on the command line as `key=value` (dotted keys, YAML
values); `eddies config` prints the result:

```bash
eddies config experiments/gate_fixed conformal.alpha=0.05 mu_cm=[5,10]
python -m eddies.gate fixed --alpha 0.05 --mu 5,10            # the experiments' own flags are equivalent
EDDIES_RESULTS=/data/eddies_results python -m eddies.evaluate tableD1
```

Resolution order: `eddies/config.yaml` < the `defaults:` list of the named file < `eddies/configs/<name>.yaml` <
`key=value` overrides < `${ENV:-default}` interpolation. In Python:

```python
from eddies.common.config import load, seeds, mu_list
cfg = load("experiments/gate_adaptive", "mu_cm=[5]")
seeds(cfg.splits.test), mu_list(cfg.mu_cm), cfg.adaptive_gate.eta
```

## Tests

```bash
pytest -q                   # tests/test_core.py (certificates, projection), test_compose.py (product game),
                            # test_config.py (every YAML loads; YAML vs task-class attributes; skillbook vs robot),
                            # test_reproduce_smoke.py (eddies.evaluate reproduces Tables 1-3, D.1-D.5 and writes Figs. 4, 5, 7)
```

Tests whose inputs are missing (`results/`, the menagerie checkout) are skipped. With the released `results/`,
`test_tableD1_totals_match_paper` checks 5 / 890 / 1955 / 1581 of 1627 against the paper.
