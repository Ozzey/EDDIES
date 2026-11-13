# Configuration reference

Every constant of the paper lives in YAML and is read through `eddies.common.config`. The defaults reproduce the
paper.

```python
from eddies.common.config import load, seeds, mu_list, robots, tasks, blocks
cfg = load("experiments/gate_fixed", "conformal.alpha=0.05", "mu_cm=[5,10]")   # defaults + overrides
cfg.splits.gate_test, seeds(cfg.splits.gate_test), mu_list(cfg.mu_cm)           # [2000, 2099], [2000, ..., 2099], [0.05, 0.10]
```

Resolution order: `eddies/config.yaml` (global defaults) < the `defaults:` list of the named file (other named
files merged first) < `eddies/configs/<name>.yaml` < dotted `key=value` overrides (YAML-parsed values) <
`${ENV:-default}` interpolation. `eddies config <name> [key=value ...]` prints a resolved configuration;
`available("experiments")` lists the files. `Config` is a dict with attribute access and dotted `get` / `set`.

## 1. `eddies/config.yaml` (global)

| key | default | meaning |
| --- | --- | --- |
| `seed` | 0 | seed of the analyses (bootstrap, k-means) |
| `paths.results` | `${EDDIES_RESULTS:-results}` | result JSON / NPZ; `eddies.common.paths.ROOT` |
| `paths.episodes` | `${EDDIES_EPISODES:-datasets/episodes}` | MuJoCo episode logs; defaults to `$EDDIES_RESULTS/episodes` when that folder exists, else to `datasets/episodes/mujoco` as `datasets/download.sh` lays it out |
| `paths.episodes_isaac`, `paths.episodes_real` | `datasets/episodes_isaac`, `datasets/episodes_real` (`EDDIES_RESULTS_ISAAC`, `EDDIES_RESULTS_REAL`) | PhysX / robot episode roots (`physics`, `stack_compare`, `dynamics_map`) |
| `paths.models` | `${EDDIES_MODELS:-results/tokprop}` | learned proposer weights |
| `paths.figures` | `assets/results/generated` | where `eddies.evaluate` and the plots write |
| `paths.menagerie`, `paths.menagerie_commit` | `external/mujoco_menagerie`, `0059d43…` | robot models; the commit is recorded in every manifest |
| `paths.scratch` | `${EDDIES_SCRATCH:-/tmp/eddies}` | `try_task`, `render` output |
| `episode_layout` | `{tag}/{condition}/{task}/{method}/seed_{seed:04d}` | episode directory layout |
| `robots` | `[franka, g1, leap, aero]` | the four robots, paper order (`configs/robots/<name>.yaml`) |
| `paper_tasks` | the ten tasks | order of the tables |
| `franka_tasks` | the four Franka tasks | held-out and gate experiments |
| `method` | `bistl` | directory name of the Streamer's episodes |
| `splits.train` / `calib` / `test` / `test_extra` | 1000–1199 / 1200–1399 / 2000–2199 / 2200–2599 | identification / conformal calibration / test; `test_extra` only in `certify_task` when present |
| `splits.gate_test` | 2000–2099 | "first 100 test episodes": Franka gate, Table 1 (Franka), Table 3, D.2, D.4, D.5 (D.3 uses all 200) |
| `splits.recal_split` | 2050 | `gate fixed --recal`: seeds < 2050 calibrate at the commanded speed, ≥ 2050 test |
| `splits.isaac` | train 1000–1049, calib 1200–1249, test 2000–2049 | PhysX episodes |
| `splits.real` | train 0–49, calib 50–99, test 100–149 | default seed ranges for your own robot logs (`stack_compare --real-*`) |
| `conformal.alpha` | 0.1 | miscoverage of the episode-level envelope (every `--alpha` default) |
| `conformal.kappa_quantile` | 0.9 | per-Eddie normalizer `κ_i`: quantile of the training episodes' max residual |
| `identification.order` | 1 | model order (first-order affine) |
| `identification.ridge` | 1e-6 | `λ = ridge · trace(G) / n` for unshrunk fits |
| `identification.shrink_grid` | `[0, 0.03, 0.3, 3]` | `λ = shrink · n_i` toward the reference model (Eq. B.1) |
| `identification.select` | `min_train_tube` | per Eddie: the grid value with a stable fit and the smallest training tube |
| `identification.reference.min_transitions`, `fallback_fraction` | 5000, 0.5 | reference Eddie: ≥ 5,000 transitions and a stable own fit (fallback: half the largest count) |
| `identification.min_train_transitions` | 50 | fewer → the Eddie is left uncertified |
| `identification.min_std` | 0.01 | coordinates whose pooled std is below this are dropped from `z` |
| `identification.max_spectral_radius` | 0.999 | at or above → uncertified |
| `identification.horizon` | 4000 | steps of the propagated tube |
| `gate.mu_cm` | `[2, 3, 5, 7.5, 10, 15]` | tolerances of the fixed-calibration gate |
| `gate.speed_scales` | `[1, 1.5, 2, 3, 4]` | commanded speeds (slew-bound multiples) of Table D.5 |
| `adaptive_gate.eta` | 0.5 | step size γ of Eq. 6 |
| `adaptive_gate.lam0` | eddie 200, primitive 50, global 10 | prior weight in pseudo-transitions by prior level |
| `adaptive_gate.kappa_min_episodes` | 5 | runs before `κ_i` is taken from the Eddie's own runs |
| `adaptive_gate.h_max_steps` | 600 | tube horizon of the online models |
| `adaptive_gate.max_spectral_radius` | 0.995 | a refit is kept only below this |
| `adaptive_gate.lookahead` | 0 | extra steps of the one-step check (0 = Eq. 5) |
| `smoke.per_split`, `smoke.n_test` | 40, 20 | `--smoke` slices |
| `sim.physics_dt`, `sim.control_dt` | 0.002, 0.02 | 500 Hz physics, 50 Hz control |
| `sim.horizon_s`, `sim.success_hold_s`, `sim.failure_budget` | 60, 1, 2 | task defaults (task YAMLs override the horizon) |
| `sim.rho_step`, `sim.rho_game` | 0.012, 10 | default slew bound; the game's step bound |
| `sim.formula` | `G !bad` | default safety formula |
| `sim.physics.*` | implicitfast, 100 iterations, 50 ls, 0 noslip, elliptic cone, impratio 10, air | MuJoCo options (`envs/framework/physics.py`) |
| `sim.video` | 30 fps, 1920×1080 | `render`, `run_episode(video=True)` |
| `collect.conditions`, `workers`, `max_tasks_per_child` | `[nominal, perturbed]`, 20, 10 | `eddies.collect` |
| `skillbook.alpha_share` | 1/12 | share of α per registered primitive (Prop. E.4) |
| `plots.colors`, `plots.font` | the paper's scheme; DejaVu Sans 7.5 pt | `cert/plots/style.py` |

## 2. `eddies/configs/robots/<robot>.yaml`

| key | meaning |
| --- | --- |
| `name`, `display`, `short`, `family`, `color` | identifiers (`family` = key of the certified block: `franka`, `g1`, `hand`), plot colour key |
| `menagerie_xml`, `hand_xml` | robot (and hand) model inside the menagerie checkout |
| `skillbook` | `module:BOOK` that holds the robot's primitive implementations |
| `primitives` | the robot's subset of the eleven primitives (Table C.1) |
| `tasks` | its paper tasks (groups of `embodiments`, `gate self --group`, the plots) |
| `n_eddies` | distinct Eddies over its tasks (35 / 25 / 13 / 13), the "#" column of Table 1 |
| `rho_step`, `control_dt`, `physics_dt` | slew bound per step (0.012 / 0.0075 / 0.009), control and physics periods |
| `home` | Franka home joint configuration |
| `embedding.names`, `embedding.weights` | the chart's coordinates and the diagonal of `M` (Table B.1) |
| `embedding.max_aperture`, `force_max`, `ranges` | Franka aperture and force normalization; G1 workspace box |
| `block.pose`, `block.position` | indices of the certified block `J` and of the position rows `S` |

Readers: `envs/tasks/{franka,g1,hand}_base.py`, `envs/controllers/{franka,g1,hand_arm}.py`, `config.robots()`,
`config.blocks()`.

## 3. `eddies/configs/tasks/<task>.yaml`

Generated from the task classes; `tests/test_config.py` fails if a class attribute and its YAML disagree.

| key | meaning |
| --- | --- |
| `name`, `display`, `robot` | identifiers |
| `registry`, `kwargs`, `legacy`, `ltl_variant` | `module:Class` of the skillbook task, its constructor arguments, the task class it extends (`legacy`), the `*_ltl` scene |
| `horizon_s`, `failure_budget`, `rho_step`, `rho_game`, `control_dt` | class-level parameters (Table C.2 horizons 45–90 s) |
| `formula` | the LTL_f task specification (Table C.2) |
| `tag`, `conditions` | where its paper episodes live (`sb`, nominal) |
| `isaac` | whether the task is also run in Isaac Sim (staged insertion, peg insertion, cube stacking) |
| `n_eddies`, `n_distinct_eddies`, `nominal_plan` | the hand-written Eddie stream; `n_eddies` is the paper's count of Table D.1 (plan steps, with the peg-tightening loop counted once: 7, although `nominal_plan` lists its 26 executed steps), `n_distinct_eddies` the distinct Eddies in the plan |

Task-internal geometry and thresholds (grasp offsets, per-Eddie tolerances and timeouts, friction tables) stay in
the task classes and are logged in every manifest.

## 4. `eddies/configs/experiments/<experiment>.yaml`

One file per experiment of the paper; `mu_cm` lists the tolerances in centimetres.

| file | module | keys |
| --- | --- | --- |
| `embodiments` | `cert.experiments.embodiments` (`eddies.certify`) | `mu_cm` |
| `certify_task` | `certify_task` (`eddies.certify --task`) | `tag`, `conditions`, `rho_star_mu_m` |
| `gate_fixed` | `gate_fixed` (`eddies.gate fixed`) | `pool` (eddy / primitive), `mu_cm`, `recal` |
| `gate_adaptive` | `gate_adaptive` (`eddies.gate self`) | `mu_cm` (held-out), `mu_cm_group`, `n_test` 100, `n_test_group` 200 |
| `gate_margin` | `gate_margin`, `tables.gate_margin` | `mu_cm`, `margins_m`, `velocities`, `n_test`, `workers` |
| `containment` | `containment` | `mu_cm` |
| `transfer` | `transfer` | `mu_cm`, `fewshot` [5, 10, 20], `cal_h` 50, `levels` |
| `physics` | `physics` | `tasks`, `mu_cm`, `refit` [10, 25, 50] |
| `stack_compare` | `stack_compare` (`eddies.gate stack`) | `task`, `rows`, `mu_cm`, `speeds` [2, 3, 4], `physx_ref_tasks` |
| `ltl` | `gate_ltl`, `envs.scripts.compose_check` | `tag`, `mu_cm`, `runs` (scene, directory, label, spec) ×13 |
| `pool` | `pool` | `tag` (flags on the command line) |
| `table1_eddies` | `table1_eddies` (`eddies.gate table1`) | `mu_cm`, `variants` [eddie, eddie_refonly], `smoke` |
| `calibration_ablation` | `calibration_ablation` (`eddies.gate calibration`) | `mu_cm` |
| `table1_ci` | `tables.table1_ci` | `dirs`, `robot_dirs`, `B` 2000, `seed`, `mu_cm`, `rows` |
| `token_proposer` | `proposer.token_proposer`, `command_proposer`, `token_proposer_eval` | `tag`, `tasks`, `features`, `mlp` (64×64, 600 epochs, lr 3e-3, …), `cmd_mlp` (128×128), `max_decisions`, `sets`, `methods`, `mu_cm` |
| `dynamics_map` | `dynamics_map`, `plots.dynamics_map` | `physx_tasks`, `rise` |
| `trace_figs` | `trace_figs` (`eddies.gate trace`) | `task` |
| `collect_sb`, `collect_gate` | `eddies.collect` | `tag`, `conditions`, `seeds`, `tasks`, `rho_scales`, `label` |
| `evaluate` | `eddies.evaluate` | per-table presentation: `table1.dirs/finished_cm/B`, `table2.mu_cm` 15, `table3.mu_cm`, `tableD1.finished_cm` 5 / `caught_cm` 2, `tableD2.mu_cm`, `tableD4.mu_cm`, `tableD5.mu_cm/files` |

Task-internal geometry and thresholds, the `ident.choose_gamma` grid, the seed 2000 used to read a nominal plan,
the `bistl_rho<s>` label format (only in `collect_gate.yaml`) and plot geometry are not config-driven.
