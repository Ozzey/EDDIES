# Data format

The analyses read episode logs; `results/` holds what the analyses wrote. The episode logs are the Hugging Face
dataset `Ozzey/eddies` (`datasets/README.md` is its dataset card and lists every subset, split and size;
`datasets/manifest.json` every episode directory with counts, bytes, seed range and the paper artefacts it feeds;
`datasets/download.sh` fetches subsets into `datasets/episodes/`). Nothing below is committed to the repository
except `results/`.

## Episode directory

```
<root>/<tag>/<condition>/<task>/<method>/seed_XXXX/
    manifest.json      the run record
    trajectory.npz     per-control-step arrays at 50 Hz (control_dt = 0.02 s)
```

`<root>` is `$EDDIES_EPISODES` (MuJoCo), `$EDDIES_RESULTS_ISAAC` (PhysX) or `$EDDIES_RESULTS_REAL` (robot); in
the dataset the roots are `mujoco/episodes/` and `isaac/episodes/`. The layout string is
`episode_layout` in `eddies/config.yaml`; `eddies.common.paths.episode_dir(tag, condition, task, method, seed)`
builds a path and `eddies.cert.data.episode_dirs(tag, condition, task, method, seeds)` lists them.

| field | values |
| --- | --- |
| `tag` | `sb` (the paper's skillbook episodes), `gate` (Franka test seeds at 1.5–4× speed), `ltl` (LTL_f runs), `tokprop` (learned proposer) |
| `condition` | `nominal`, `perturbed` (pushes, drops, friction and mass changes; `manifest["perturbation"]` records what happened) |
| `task` | the ten paper tasks (`eddies/config.yaml: paper_tasks`), plus the `*_ltl` scenes of the `ltl` tag |
| `method` | `bistl` = the Streamer (the registry key is kept because every released episode directory uses it); `bistl_rho1.5`, `bistl_rho2`, `bistl_rho3`, `bistl_rho4` = every Eddie commanded that many times faster; `bistl_learned*`, `bistl_cmd*` = learned proposers; `drawer_open`, `drawer_hold`, `fmb_stage`, `fmb_direct`, `fmb_hole_fixture` = plans synthesised for new goals |
| `seed` | fixes the scene variant; the splits are by seed (below) |

## Seeds and splits (`eddies/config.yaml: splits`)

| seeds | role |
| --- | --- |
| 1000–1199 | training: identification of the models and the normalizers `κ_i` (200 per task) |
| 1200–1399 | calibration: the coverage level `q` (200 per task) |
| 2000–2199 | test: every certificate, tube and gate number (200 per task); `gate_test` 2000–2099 is "the first 100" of Table 1 (Franka), Table 3, D.2, D.4, D.5 (D.3 uses all 200) |
| 2050 | `recal_split`: in `gate --recal`, seeds < 2050 calibrate at the commanded speed and ≥ 2050 test |
| Isaac Sim | 1000–1049 / 1200–1249 / 2000–2049 |
| robot | 0–49 / 50–99 / 100–149 (protocol in [REAL_ROBOT.md](REAL_ROBOT.md)) |

## `manifest.json`

| key | meaning |
| --- | --- |
| `task`, `method`, `seed`, `condition` | identity of the run |
| `status`, `success`, `task_success`, `property_success`, `ltl_status`, `stop_reason` | outcome: `status` is `completed` for every shipped episode; `stop_reason` is `task_success`, `horizon`, `bad` or a Streamer stop reason (`no_recovery:…`, `property_violated`) |
| `simulator`, `menagerie_commit`, `python`, `physics` | engine (`mujoco` 3.12, `isaac` = Isaac Sim 6.0.1, PhysX 5; `real`), timestep (0.002 s), gravity, solver settings, model sizes |
| `robot`, `family`, `description`, `formula`, `propositions` | robot, task family, the LTL_f task formula over the listed propositions |
| `control_dt`, `horizon_s` | 0.02 s; 45–90 s per task (Table C.2) |
| `embedding` | the chart: `names`, `units`, `phys_lower/upper`, `norm_lower/upper`, `metric_diag` (the weights of `M`; position 1, orientation 0.3, gripper / hand 0.05–0.15) |
| `variant` | the seeded scene: object poses and yaws, masses, friction, perception biases, drawer / wheel parameters |
| `method_info.stats` | control steps, projections, slew-active steps, max command jump, Eddies started |
| `method_info.skills` | **the Eddie stream**: one record per Eddie run (`skill`, e.g. `reach:cube_above`, `start`, `end`, `outcome`) |
| `method_info.rho_step`, `use_trust`, `use_online`, `rls_*` | slew bound, trust radius on, online RLS settings |
| `method_info.supervisor` | the contract supervisor: rank sequence, failure budget, every hand-over decision (`decisions[*]` with the observed cell, allowed contracts, game size, solve time) |
| `task_metrics`, `diagnostics_max`, `bad_steps`, `first_bad_time_s` | task-specific end metrics, peak contact forces, steps in a `bad` state |
| `completion_time_s`, `sim_time_s`, `wall_time_s`, `compute_ms_per_control_step` | timing |
| `cert_overrides` | collection tag and the run's overrides (`rho_scale`, `budget`, `params`, `spec`) |
| `perturbation` | what was perturbed and when (perturbed condition only) |

## `trajectory.npz`

Arrays indexed by control step `k` (N steps). `e` is the command sent at step `k`; `z` the measured state in the
same chart, observed before `e_k` is applied, so Eddies are identified on `z_{k+1} = A z_k + B e_k + d + w_k`.
Both are in the normalized chart. `m = 8` on the Franka (x, y, z, rx, ry, rz, aperture, force), 9 on the hands
(pose, power, pinch, point synergies), 7 on the G1 (pose, grasp); the certified state block `J` is the pose part
(Table B.1).

| array | shape | dtype | meaning |
| --- | --- | --- | --- |
| `t` | (N,) | float64 | simulation time [s] |
| `e` | (N, m) | float64 | command `e_k`, normalized chart |
| `z` | (N, m) | float64 | measured embedding `z_k`, normalized chart |
| `skill` | (N,) | str | Streamer status: `"<eddie>  rank R  fail-budget B"`, `STOP …` or `terminal`; `eddies.cert.data.skill_of` extracts the Eddie name |
| `labels` | (N,) | int64 | bit mask of the true propositions (bit *i* = `propositions[i]`) |
| `residual_false` | (N,) | bool | the LTL_f monitor is already false |
| `sig_*` | (N,) or (N, 3) | float64 | task signals in physical units (e.g. `sig_red`, `sig_blue`, `sig_tcp` [m], `sig_aperture` [m] on stacking; `sig_theta`, `sig_omega`, `sig_wheel_force` on the wheel; `sig_drawer_opening` on the drawers) |

Loader: `eddies.cert.data.load_episode(dir)` returns a dict with these arrays plus the manifest;
`load_split(tag, condition, task, seeds)` a list of them. A robot run must be logged in exactly this form
(`z` and `e` in the same chart, `skill` naming the active Eddie) for the gates to replay on it.

## `results/` (committed)

`results/README.md` and `results/MANIFEST.json` list every file with its size, sha256, the paper artefacts that
use it and where it came from. The main producers and consumers:

| file | written by | read by |
| --- | --- | --- |
| `embodiments/embodiments.json` | `eddies.certify --all` | `evaluate table2 tableD1 tableD6 fig5` |
| `adaptive/adaptive_<task>[_known][_<robot>].json` | `eddies.gate self` | `evaluate tableD1 tableD2 fig5 fig7 supp` |
| `table1/{eddie,eddie_refonly,calibration}.json`, `table1_robots/<robot>/*.json`, `table1/table1_ci_final.json` | `eddies.gate table1 --episodes`, `eddies.gate calibration`, `eddies.cert.tables.table1_ci` | `evaluate table1` |
| `physics/physics.json`, `compare/stack_*.json` | `eddies.gate physics`, `eddies.gate stack` | `evaluate table2 table3` |
| `gate/{gate_eddy,gate_eddy_recal,containment,ltl}.json` | `eddies.gate fixed [--recal]`, `containment`, `ltl` | `evaluate tableD4 tableD5` |
| `gate_margin/summary.json` (per-step `<task>.npz`, `<task>_a.json` in the dataset, `derived/gate_margin/`) | `eddies.gate margin`, then `eddies.gate margin-table --out` | `eddies.gate margin-table` (`evaluate pooled tableD1`) |
| `pool/final_*.json`, `transfer/*.json`, `tokprop/*.json`, `dynamics_map/*.json`, `figs_data/trace_fmb.json`, `eqoff/`, `facts/` | the matching experiments and `scripts/facts/` | `evaluate tableD3 supp fig4`, the text numbers |
| `real/` | the robot session (camera calibration, protocol, the Table 3 robot rows) | `evaluate table3` |

Per-task result JSONs share the shape `{"mu_m": [...], "groups"/"tasks": {<task>: {"gate": [{"mu_m", "n",
"completed", "stopped", ...}], "radius": {"median_cm", "p90_cm"}, "violations": ...}}}`; the self-calibrating
gate's files are `{"rows": {"<mu_m>": {"summary": {"n", "completed", "stopped", "caught", "would_exceed", ...},
"q": [...], "err": [...]}}}`. `eddies/evaluate.py` is the reference reader for each.
