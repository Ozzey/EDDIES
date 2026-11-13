---
license: cc-by-4.0
pretty_name: Eddies episode logs
language:
- en
tags:
- robotics
- manipulation
- mujoco
- isaac-sim
- conformal-prediction
- action-tokens
size_categories:
- 10K<n<100K
---

# Eddies: episode logs of the paper

Episode logs behind *Eddies: Dynamical Action Tokens with Certified Composition* (A. Narendra,
G. Tadevosyan, A. Swikir; MBZUAI; L4DC 2027). Code: https://github.com/Ozzey/EDDIES.
Website: https://ozzey.github.io/papers/eddies/. Every table and figure of the paper that is computed from
episodes is computed from the episodes in this dataset; the JSON results derived from them are in
the code repository under `results/`.

**15,450 episodes, 2,437 MB** (93 episode directories),
plus 68 MB of derived per-step margin arrays and 15 MB of rendered stills (2,520 MB in all).

## What an episode is

One run of one task by the Streamer (registry key `bistl`, the method that streams Eddies) or one of its
ablations, from a seeded initial condition to `task_success`, a certified stop or the horizon. Each
episode is a directory

```
<simulator>/episodes/<tag>/<condition>/<task>/<method>/seed_XXXX/
    manifest.json      run record: task, seed, outcome, physics, embedding chart, scene variant, Eddie stream
    trajectory.npz     per-control-step arrays at 50 Hz (control_dt = 0.02 s)
```

`simulator` is `mujoco` (MuJoCo 3.12, menagerie commit `0059d43`) or `isaac` (episodes recorded with Isaac Sim 6.0.1, pip build, PhysX 5).
`condition` is `nominal` or `perturbed` (pushes, drops, friction and mass changes; the manifest's
`perturbation` field records what happened). `method` is `bistl` for the Streamer; `bistl_rhoX`
commands every Eddie X times faster (slew bound x1.5/2/3/4), `bistl_learned*` and `bistl_cmd*` are the
learned-proposer variants of App. D, and `drawer_open`, `drawer_hold`, `fmb_stage`, `fmb_direct`,
`fmb_hole_fixture` are hand-written plans in the `ltl` tag.

### `manifest.json`

| key | meaning |
|---|---|
| `task`, `method`, `seed`, `condition` | identity of the run (`seed` also fixes the scene variant below) |
| `status`, `success`, `task_success`, `property_success`, `ltl_status`, `stop_reason` | outcome: `status` is `completed` for every shipped episode; `stop_reason` is `task_success`, `horizon`, `bad` or a gate stop |
| `simulator`, `menagerie_commit`, `python`, `physics` | physics engine, timestep (0.002 s), gravity, solver settings, model sizes `nq/nv/nu/nbody/ngeom` |
| `robot`, `family`, `description`, `formula`, `propositions` | robot, task family, LTLf task formula over the listed propositions |
| `control_dt`, `horizon_s` | control period (0.02 s) and horizon, 45–90 s per task (Table C.2) |
| `embedding` | the chart the Eddies live in: `names`, `units`, physical bounds `phys_lower/upper`, normalised bounds `norm_lower/upper`, and `metric_diag`, the diagonal weights of the norm in which tubes and tolerances are measured (position 1, orientation 0.3, gripper/hand 0.05-0.15) |
| `variant` | the seeded scene: object poses and yaws, masses, friction, perception biases, drawer/wheel parameters |
| `method_info.stats` | control steps, projections onto the trust radius, slew-active steps, max command jump, Eddies started |
| `method_info.skills` | the Eddie stream: each Eddie (`skill`, e.g. `reach:cube_above`) with `start`/`end` time and `outcome` |
| `method_info.rho_step`, `use_trust`, `use_online`, `rls_*` | slew bound per step, trust radius on, online recursive least squares choices |
| `method_info.supervisor` | the contract supervisor: rank sequence, failure budget, every hand-over decision (`decisions[*]` with the observed cell, the allowed contracts, game size and solve time) |
| `task_metrics`, `diagnostics_max`, `bad_steps`, `first_bad_time_s` | task-specific end metrics (e.g. final cube offset), peak contact forces, steps in a `bad` state |
| `completion_time_s`, `sim_time_s`, `wall_time_s`, `compute_ms_per_control_step` | timing |
| `cert_overrides` | collection tag and the overrides of the run (`rho_scale`, `budget`, `params`, `spec`) |
| `perturbation` | what was perturbed and when (perturbed condition only) |

### `trajectory.npz`

Arrays indexed by control step k (N steps). `e` is the command e_k sent to the robot at step k and `z`
the measured state in the same chart, observed before e_k is applied; Eddies are identified on
z_(k+1) = A z_k + B e_k + d + w. Both are in the normalised chart (`embedding.norm_lower/upper`);
the embedding has m = 8 coordinates on the Franka (x, y, z, rx, ry, rz, aperture, force), 9 on the hands
(x, y, z, rx, ry, rz, power, pinch, point; the last three are synergy weights) and 7 on the G1 (pose + grasp).

| array | example | shape | dtype | meaning |
|---|---|---|---|---|
| `t` | all | (N,) | float64 | simulation time [s] of step k |
| `e` | all | (N, m) | float64 | command e_k, normalised chart |
| `z` | all | (N, m) | float64 | measured embedding z_k, normalised chart |
| `skill` | all | (N,) | str | Streamer status at step k: `"<eddie>  rank R  fail-budget B"`, `STOP ...` or `terminal` (the Eddie name is the first token) |
| `labels` | all | (N,) | int64 | bit mask of the true propositions, bit i = `propositions[i]` |
| `residual_false` | all | (N,) | bool | the LTLf monitor is already false |
| `sig_*` | per task | (N,) or (N, 3) | float64 | task signals in physical units: e.g. Franka stacking `sig_red`, `sig_blue`, `sig_tcp` [m], `sig_aperture` [m]; LEAP wheel `sig_grasp` [m], `sig_theta` [rad], `sig_omega` [rad/s], `sig_wheel_force` [N], `sig_synergy`; G1 drawer `sig_drawer_opening` [m], `sig_com`, `sig_pelvis` [m] |

Examples: `mujoco/episodes/sb/nominal/franka_cube_stack/bistl/seed_1000` has N = 315 steps,
m = 8, signals sig_red, sig_blue, sig_tcp, sig_aperture; `mujoco/episodes/sb/nominal/leap_wheel_sb/bistl/seed_1000`
has N = 448, m = 9, signals sig_grasp, sig_theta, sig_omega, sig_wheel_force, sig_synergy. Isaac Sim episodes
carry the same keys as their MuJoCo counterparts. `schema_samples.json` lists the keys, shapes
and dtypes of the first episode of every directory.

## Splits, tags and sizes

Seeds define the splits, the same on every task:

| tag | seeds | role |
|---|---|---|
| `sb` (MuJoCo) | 1000-1199 | training: identification of the Eddies (200 episodes per task) |
| `sb` (MuJoCo) | 1200-1399 | calibration: conformal envelopes at coverage level q (200 per task) |
| `sb` (MuJoCo) | 2000-2199 | test: all certificate, gate and tube numbers (200 per task; Table 3 uses the first 100 of stacking) |
| `gate` (MuJoCo) | 2000-2099 | Franka test seeds with every Eddie commanded 1.5/2/3/4x faster (Tab. D.5, Tab. 3 "caught") |
| `ltl` (MuJoCo) | 2000-2099 | LTLf composition on nominal and perturbed scenes (Sec. 5.4, App. A) |
| `tokprop` (MuJoCo) | 1000-1399 / 2000-2099 | proposer training-validation / held-out test (App. D learned proposer) |
| `sb` (Isaac Sim) | 1000-1049 / 1200-1249 / 2000-2049 | PhysX train / calibration / test (Tab. 2, Tab. 3, Fig. D.1) |
| `gate` (Isaac Sim) | 2000-2049 | PhysX stacking at 1.5-4x (Tab. 3) |

| subset (`simulator/tag`) | directories | episodes | size |
|---|---|---|---|
| `mujoco/sb` | 10 | 6,000 | 1,025 MB |
| `mujoco/gate` | 16 | 1,600 | 147 MB |
| `mujoco/ltl` | 26 | 2,600 | 288 MB |
| `mujoco/tokprop` | 34 | 4,600 | 914 MB |
| `isaac/sb` | 3 | 450 | 54 MB |
| `isaac/gate` | 4 | 200 | 8.0 MB |

The ten paper tasks in the `sb` tag (train + calibration + test = 600 episodes each in MuJoCo):

| simulator | robot | task | episodes | successful | size |
|---|---|---|---|---|---|
| mujoco | Franka Panda | `franka_fmb` (staged insertion (FMB)) | 600 | 600 | 108 MB |
| mujoco | Franka Panda | `franka_peg_insertion` (peg insertion) | 600 | 600 | 55 MB |
| mujoco | Franka Panda | `franka_cube_stack` (cube stacking) | 600 | 596 | 41 MB |
| mujoco | Franka Panda | `franka_drawer` (drawer) | 600 | 600 | 68 MB |
| mujoco | Unitree G1 | `g1_peg_insertion_sb` (peg insertion) | 600 | 600 | 55 MB |
| mujoco | Unitree G1 | `g1_drawer_sb` (drawer) | 600 | 593 | 89 MB |
| mujoco | UR5e + LEAP hand | `leap_wheel_sb` (wheel) | 600 | 599 | 43 MB |
| mujoco | UR5e + LEAP hand | `leap_peg_tighten_sb` (peg tighten) | 600 | 600 | 209 MB |
| mujoco | UR5e + Aero hand | `aero_wheel_sb` (wheel) | 600 | 600 | 50 MB |
| mujoco | UR5e + Aero hand | `aero_peg_tighten_sb` (peg tighten) | 600 | 578 | 307 MB |
| isaac | Franka Panda | `franka_fmb` (staged insertion (FMB)) | 150 | 125 | 28 MB |
| isaac | Franka Panda | `franka_peg_insertion` (peg insertion) | 150 | 128 | 17 MB |
| isaac | Franka Panda | `franka_cube_stack` (cube stacking) | 150 | 150 | 8.2 MB |

`manifest.json` at the dataset root lists every episode directory with its episode count, bytes, seed
range, success count and stop reasons, and the paper artefacts it feeds; `checksums.sha256` covers every file.

Also in the dataset:

- `derived/gate_margin/<task>.npz`, `<task>_a.json` — per-step gate margins of the 200 test episodes of each task (the pooled 1,968 / 387-395 / 1,328 headline of Sec. 5.1). The code repository keeps only their counts (`results/gate_margin/summary.json`, from which `eddies.evaluate pooled` and the Table D.1 one-step column are printed); `datasets/download.sh --subset gate_margin` also copies the per-step files into `results/gate_margin/`.
- `stills/{mujoco,isaac}/` — the rendered stills behind Fig. 6 and Fig. C.1.

## How the code consumes the episodes

`datasets/download.sh [--subset sb,gate,...]` places the files under `datasets/episodes/` in the code
repository (`scripts/reproduce.sh` points `EDDIES_EPISODES` / `EDDIES_RESULTS_ISAAC` at `datasets/episodes/{mujoco,isaac}`). `eddies/certify.py` reads the `sb` training seeds of a task to identify its Eddies (one
linear model per Eddie on `z`, `e`, segmented by `skill`), the calibration seeds to set the conformal
envelope at coverage level q, and the test seeds to report violations, tube widths and finished runs.
`eddies/gate.py` replays the fixed-calibration and the self-calibrating gate on the test episodes
(`sb`, and `gate` for the faster runs) and on the LTLf runs (`ltl`). `eddies/evaluate.py` turns the
results in `results/` into the paper's tables and figures; `scripts/reproduce.sh <experiment>` recomputes those
results from the episodes. Robot episode logs are not part of the dataset; the robot rows of Table 3 are the values
reported in the paper.

## License and citation

CC-BY-4.0. Please cite the paper:

```
@inproceedings{narendra2027eddies,
  title     = {Eddies: Dynamical Action Tokens with Certified Composition},
  author    = {Narendra, Aditya and Tadevosyan, Grik and Swikir, Abdalla},
  booktitle = {Learning for Dynamics and Control (L4DC)},
  year      = {2027}
}
```
