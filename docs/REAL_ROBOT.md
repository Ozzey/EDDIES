# Real robot: cube stacking on a Franka Panda

Sec. 5.5, Fig. 8, Table 3 and Appendix C of the paper. Cube stacking is the task that runs on all three systems,
MuJoCo, Isaac Sim (PhysX) and the robot, with the same eight Eddies, the same Streamer and the same gates.

## Setup

| | |
| --- | --- |
| robot | Franka Emika Panda, parallel gripper |
| low-level control | the 1 kHz compliant Cartesian controller of CRISP (San José et al., 2025) tracks the 50 Hz Eddie commands; gains fixed for every episode (the certificate is of this closed loop) |
| command `e` at 50 Hz | TCP position [m], hand rotation vector [rad], gripper aperture [m], force limit [N], normalized as in simulation (`eddies/envs/tasks/franka_base.py`, `configs/robots/franka.yaml: embedding`); slew bound ρ = 0.012 per 20 ms step in the metric M = diag(1, 1, 1, 0.3, 0.3, 0.3, 0.15, 0.05) on `e` (W is its pose part, Table B.1) |
| measured `z` | the TCP pose from the robot state and the aperture, logged each tick before the new command is applied; only the six pose coordinates are certified |
| perception | a static Intel RealSense camera, calibrated hand-to-eye (`results/real/calib/T_base_cam.yaml`: `T_base_cam`, intrinsics `K`, 1280×720, reprojection 0.91 px over 12 samples); cube poses in the robot base frame from two sources, boxes in the RGB image (FoundationPose) and heights above the table from the aligned depth (Fig. 8b, c) |
| cubes | two 5 cm cubes: a grey one to pick, a black one to stack on (`red_cube` and `blue_cube` in the code); poses randomized per episode as in simulation (nominal grey at (0.50, −0.12) m, black at (0.50, +0.12) m, ±6 cm in x, ±5 cm in y, ±0.6 rad yaw) |
| the Eddie stream (8 Eddies, `FrankaStackTokens.nominal_plan`) | `reach:cube_above` → `reach:cube_grasp` → `grasp:cube` → `lift:cube` → `transport:cube_above` → `place:cube_top` → `release:cube` → `retreat:cube` |
| success | after release the grey cube rests on the black one: centres within 20 mm in xy, 44–56 mm height difference, black cube not lifted, grey cube at rest, gripper retreated ≥ 5 cm (the same `task_success` as in MuJoCo) |

The safe set is the set of cube poses for which the `reach` Eddie is certified. A cube pushed out of it makes the
Eddie refuse its next command; the arm holds its last command, a **certified stop**, until the cube is back
(App. C).

## Protocol (App. C)

- Two certificates on the same 50 seeds: the **MuJoCo certificate** unchanged, and a **re-identified** one refit on
  the MuJoCo data plus 50 robot episodes and calibrated on 50 more.
- **Nominal speed**: 50 test runs, half of them perturbed (a stick pushes a cube during the approach). Both gates
  run **online** on the same measurements: the fixed-calibration gate acts on the arm; the self-calibrating gate
  runs alongside and its stops are logged. A stopped run holds until the check passes again, so it can still end
  with the cube stacked.
- **Fast runs**: 50 each at 2×, 3× and 4× the nominal speed (the slew bound scaled), executed with no gate acting
  and both gates replayed on the logs, as in simulation; so whether each run *would* have left μ is known.
- Tolerance μ = 5 cm throughout.

## Numbers (Table 3, robot rows; Sec. 5.5)

| certificate | runs | success | viol. | tube p90 | fixed-calibration gate fin. / stop. | caught (2–4×) | self-calibrating fin. / stop. | caught |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MuJoCo | 50 | 50 | 0 | 4.3 cm | 30 / 20 | 150 / 150 | 46 / 4 | 145 / 150 |
| re-identified | 50 | 50 | 0 | 4.1 cm | 43 / 7 | 150 / 150 | 45 / 5 | 143 / 150 |

- The MuJoCo certificate holds on the robot (and in Isaac Sim): no run leaves a tube and every run stacks the cube.
- Its envelopes are missed more often than in MuJoCo, so the fixed-calibration gate stops 20 of 50 runs (7 once
  re-identified); the self-calibrating gate stops 4–5.
- At 2–4× speed the fixed-calibration gate catches every run that would leave μ (150 / 150) and the self-calibrating gate
  95–99 % (145 and 143 of 150).
- A cube pushed during the approach is followed by the arm; pushed out of the safe set, the reach Eddie refuses its
  next command and the arm holds for 15 s until the cube is back, then grasps and stacks it (Fig. 8d–i; the
  supplementary video and `assets/videos/real/perturbation.mp4`).

For comparison, MuJoCo (first 100 test episodes): 99 succeed, 0 violations, tube p90 4.3 cm, fixed-calibration gate 89 / 11
with 298 / 298 caught, self-calibrating 96 / 4 with 287 / 298; Isaac Sim: 31 / 19 and 41 / 9 (fixed), 48 / 2
(self-calibrating).

## Files

| in `results/real/` | |
| --- | --- |
| `table3_robot_rows.json` | the two robot rows of Table 3 (the values reported in the paper, as above) |
| `calib/T_base_cam.yaml` | camera calibration (hand-to-eye transform, intrinsics) |

| in `assets/videos/real/` | |
| --- | --- |
| `stacking.mp4` | a nominal run, side view with the robot-camera feed, overlaid with the Eddie stream |
| `perturbation.mp4` | the perturbed run of Fig. 8d–i: pushes, certified stop (15 s hold), resume, grasp, stack |

The per-tick robot logs and the robot driver (CRISP controller, RealSense / FoundationPose perception) are not part
of this release; `eddies.evaluate table3` prints the robot rows from `table3_robot_rows.json` and recomputes the
simulation rows from `compare/stack_{mujoco,physx_sim,physx_own}.json`.

## Running the analysis on your own robot logs

Log every run in the episode format of [DATA_FORMAT.md](DATA_FORMAT.md) under
`$EDDIES_RESULTS_REAL/sb/nominal/franka_cube_stack/bistl/seed_XXXX/` (fast runs under `gate/…/bistl_rho<s>/`), with
`manifest["simulator"] = "real"`, the `embedding` block copied from a simulation manifest, and
`method_info.rho_step = 0.012`. Then

```bash
EDDIES_RESULTS_REAL=/path/to/real_root python -m eddies.gate stack --rows real_sim,real_own \
    --real-train 0-49 --real-calib 50-99 --real-test 100-149
python -m eddies.evaluate table3
```

replays both gates on the logged runs from both certificates (`eddies/cert/experiments/stack_compare.py`) and
writes the robot rows. The online gate on the robot is `eddies.cert.gate.RuntimeGate` (`plan()` before motion,
`enter()` at every hand-over, `step()` every tick) and `eddies.cert.experiments.gate_adaptive` for the
self-calibrating variant.
