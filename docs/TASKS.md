# Tasks, robots and primitives

Ten tasks on four robots (Fig. 6, Appendix C of the supplement). Each robot implements its own subset of the
eleven primitives once, and every one of its tasks is composed from that subset: a task binds primitives to its
frames, and a skill with its identified model and envelope is an Eddie.

## The four robots (`eddies/configs/robots/*.yaml`)

| key | robot (MuJoCo Menagerie model) | primitives | tasks | distinct Eddies certified | ρ per 20 ms step | command `e` (m) | state `z` (n) |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| `franka` | Franka Emika Panda (`franka_emika_panda/panda.xml`) | all eleven: reach, grasp, lift, transport, place, release, insert, push, pull, turn, retreat (turn is unused by its tasks) | 4 | 35 (34 certified; "backing off from the hole" has 16 transitions) | 0.012 | 8: x y z rx ry rz aperture force | 6 |
| `g1` | Unitree G1 + Dex3-1 hands (`unitree_g1/g1_with_hands.xml`) | all but turn | 2 | 25 | 0.0075 | 7: pose + hand closure | 6 |
| `leap` | UR5e + LEAP hand (`universal_robots_ur5e`, `leap_hand/right_hand.xml`) | reach, grasp, push, release, turn, retreat | 2 | 13 | 0.009 | 9: pose + power, pinch, point synergies | 5 |
| `aero` | UR5e + Tetheria Aero Hand Open (`tetheria_aero_hand_open/right_hand.xml`) | the same six | 2 | 13 | 0.009 | 9 | 5 |

Weights of `M` (Table B.1): pose 1, 1, 1, 0.3, 0.3, 0.3; Franka aperture 0.15 and force 0.05; G1 closure 0.05;
hand synergies 0.05 each. `S` (metres per normalized unit): 0.35, 0.5, 0.35 (Franka, hands), 0.4, 0.4, 0.4 (G1).
Reference Eddie toward which every model of the robot is shrunk: releasing the box (Franka), grasping the peg
(G1), releasing the peg (hands). On the hands the rotation about *y* barely moves and is dropped from `z`
(`identification.min_std = 0.01`).

## The eleven primitives (Table C.1, `eddies/envs/skillbook/primitives.py`, `g1.py`, `hand.py`)

| primitive | behaviour | listed by |
| --- | --- | --- |
| reach | move the hand to a frame | MAPLE (reach), RAPS (go to pose) |
| grasp | close the hand; success is measured pad contact | Bøgh et al. (pick up), MAPLE, RAPS, FMB |
| lift | lift the held object | RAPS |
| transport | carry the held object to a frame, faces aligned | cf. Bøgh et al. (move to), FMB (move to board) |
| place | lower the held object onto a support, force-limited | Bøgh et al., RAPS (drop), FMB (place on fixture) |
| release | open the hand in place | Bøgh et al., MAPLE, RAPS |
| insert | force-limited compliant insertion, optional spiral search | cf. Bøgh et al. (align), FMB |
| push | force-limited push with closed fingers | Bøgh et al. (press), MAPLE, RAPS |
| pull | pull a grasped articulation along its identified line | RAPS |
| turn | rotate the grasp about an axis | Bøgh et al., RAPS (twist), FMB (rotate) |
| retreat | open and move clear of the object | (post-release clearance) |

Each primitive carries a symbolic contract (precondition and possible outcomes,
`eddies/envs/skillbook/contracts.py`) and a fixed share of the miscoverage budget (`skillbook.alpha_share` = 1/12,
Proposition E.4): adding a primitive, as `push` was added for staged insertion, leaves every existing certificate
unchanged.

## The ten tasks (Table C.2, `eddies/configs/tasks/*.yaml`, `eddies/envs/tasks/`)

Eddies = length of the nominal plan (distinct Eddies in parentheses when the plan repeats some). `bad` collects
collisions, excessive contact forces, drops and unrecoverable states. Every episode is judged on its complete
measured label trace.

| task key | robot | task | Eddies | LTL_f specification | horizon | Isaac Sim |
| --- | --- | --- | ---: | --- | ---: | :-: |
| `franka_fmb` | Franka | Staged insertion (FMB): pick the peg, stage it on a fixture, regrasp, insert at 1–2 mm clearance, push home | 16 (11) | `G !bad & F(grasped & F(staged & F(grasped & F(inserted & F(seated & released)))))` | 90 s | yes |
| `franka_peg_insertion` | Franka | Peg insertion | 8 | `G !bad & F(grasped & F(inserted & released))` | 60 s | yes |
| `franka_cube_stack` | Franka | Cube stacking | 8 | `G !bad & F(grasped & F(stacked & released))` | 45 s | yes |
| `franka_drawer` | Franka | Drawer + box: open the drawer, transfer the box | 14 | `G !bad & F(opened & F(grasped & F(placed & released)))` | 80 s | |
| `g1_peg_insertion_sb` | G1 + Dex3 | Peg insertion | 8 | `G !bad & F(grasped & F(inserted & released))` | 60 s | |
| `g1_drawer_sb` | G1 + Dex3 | Drawer + caddy | 16 | `G !bad & F(opened & F(grasped & F(placed & released)))` | 75 s | |
| `leap_wheel_sb` | UR5e + LEAP | Steering wheel | 6 | `G !bad & F(grasped_rim & F(turned & F released))` | 45 s | |
| `leap_peg_tighten_sb` | UR5e + LEAP | Peg tightening: six turn / release / unwind / regrasp cycles | 26 (7) | `G !bad & F(seated & F released)` | 90 s | |
| `aero_wheel_sb` | UR5e + Aero | Steering wheel | 6 | as LEAP | 45 s | |
| `aero_peg_tighten_sb` | UR5e + Aero | Peg tightening | 26 (7) | as LEAP | 90 s | |

Table C.2 and Table D.1 of the paper count peg tightening as 7 Eddies (distinct) and staged insertion as 16 (plan
length); `eddies.evaluate tableD1` prints both conventions.

### Nominal plans (the Eddie streams of the hand-written tasks)

```
franka_fmb            reach:peg_above reach:peg_grasp grasp:peg lift:peg transport:fixture_above place:fixture release:peg
                      reach:peg_above reach:peg_grasp grasp:peg lift:peg transport:hole_above insert:hole release:peg push:peg_top retreat:peg
franka_peg_insertion  reach:peg_above reach:peg_grasp grasp:peg lift:peg transport:hole_above insert:hole release:peg retreat:peg
franka_cube_stack     reach:cube_above reach:cube_grasp grasp:cube lift:cube transport:cube_above place:cube_top release:cube retreat:cube
franka_drawer         reach:handle_pre reach:handle grasp:handle pull:drawer release:handle retreat:handle
                      reach:box_above reach:box_grasp grasp:box lift:box transport:goal_above place:goal release:box retreat:box
g1_peg_insertion_sb   reach:peg_above reach:peg_grasp grasp:peg lift:peg transport:hole_above insert:hole release:peg retreat:peg
g1_drawer_sb          reach:handle_pre reach:handle_above reach:handle grasp:handle pull:drawer release:handle retreat:handle
                      reach:caddy_above reach:caddy_preshape reach:caddy_grasp grasp:caddy lift:caddy transport:goal_above place:goal release:caddy retreat:caddy
leap/aero_wheel_sb    reach:rim_pre reach:rim grasp:rim turn:wheel release:rim retreat:rim
leap/aero_peg_tighten reach:peg_above reach:peg_grasp grasp:peg [turn:peg release:peg turn:unwind grasp:peg] x5 turn:peg release:peg retreat:peg
```

The product game recovers all four Franka plans from the formulas above (`results/ltl/compose_check_2000.json`),
runs five new goals in the same scenes with 99–100 % success and refuses three impossible ones before motion
(Sec. 5.4, App. A; `eddies/configs/experiments/ltl.yaml` lists the specs).

## Simulation

MuJoCo 3 with the physics at 500 Hz (`sim.physics_dt = 0.002`), elliptic friction cones, joint damping, friction
loss and actuator force limits; control at 50 Hz through damped least-squares inverse kinematics on the hand point
into joint position servos (`eddies/envs/controllers/ik.py`). The Franka's insertion tasks add a force-limiting
admittance loop on a filtered, noisy wrist force sensor; the G1 keeps its balance by leg inverse kinematics and
centre-of-mass regulation (`controllers/g1.py`); the LEAP hand has joint position servos and the Aero hand
pull-only tendon servos (`controllers/hand_arm.py`). Scenes (peg, hole, fixture, cabinet, wheel, cubes) are built
procedurally (`eddies/envs/assets/`); every seed draws object poses, masses, friction and perception biases
(`manifest["variant"]`).

Staged insertion, peg insertion and cube stacking also run in Isaac Sim (PhysX) from the same task code: the
compiled MuJoCo scene is converted to USD, PhysX integrates with a 2 ms step and the states are written back into
the MuJoCo data so the task logic reads the same sites (`eddies/isaac/`, Fig. C.1).

## Per-task certificates (Table D.1, μ = 5 cm, 200 test episodes each)

| robot | task | Eddies | viol. | tube median / p90 [cm] | finished, fixed | finished, self-calibrating | caught at 2 cm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Franka | Staged insertion | 16 | 1 | 2.9 / 3.6 | 188 | 199 | 193 / 200 |
| | Peg insertion | 8 | 1 | 3.1 / 3.5 | 184 | 199 | 194 / 200 |
| | Cube stack | 8 | 0 | 3.4 / 4.3 | 180 | 191 | 175 / 200 |
| | Drawer + box | 14 | 0 | 3.0 / 4.2 | 0 | 200 | 200 / 200 |
| G1 + Dex3 | Peg insertion | 8 | 0 | 4.8 / 6.6 | 0 | 197 | 22 / 26 |
| | Drawer + caddy | 16 | 2 | 4.8 / 7.9 | 0 | 197 | 200 / 200 |
| LEAP hand | Wheel | 6 | 1 | 7.4 / 11.9 | 0 | 172 | 196 / 200 |
| | Peg tightening | 7 | 0 | 3.9 / 4.6 | 174 | 200 | 200 / 200 |
| Aero hand | Wheel | 6 | 0 | 6.3 / 6.5 | 0 | 200 | 200 / 200 |
| | Peg tightening | 7 | 0 | 2.9 / 4.1 | 164 | 200 | 1 / 1 |
| All (85 certified) | | 96 | 5 | | 890 | 1955 | 1581 / 1627 |
