# Third-party assets

Nothing in this folder is committed; everything is fetched or installed separately.

| asset | used by | how |
|---|---|---|
| [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) at commit `0059d4335f8156206f63a35662313385f7ad6d74` (`eddies/config.yaml: paths.menagerie_commit`): `franka_emika_panda`, `unitree_g1`, `universal_robots_ur5e`, `leap_hand`, `tetheria_aero_hand_open` (~130 MB) | every MuJoCo task (`eddies/envs`) | `scripts/download_menagerie.sh` -> `external/mujoco_menagerie/` (or set `EDDIES_MENAGERIE`) |
| Isaac Sim 6.0.1 (pip build, PhysX 5) | `eddies/isaac` only (Table 2 / Table 3 PhysX rows, Fig. 6 bottom row) | install Isaac Sim, set `ISAAC_INSTALL_ROOT`; run with its `python.sh` (`ISAAC_PY=<isaac>/python.sh scripts/reproduce.sh collect-isaac`) |

The peg, hole, fixture, drawer cabinet, steering wheel and cubes are built procedurally with `MjSpec`
(`eddies/envs/assets`); no mesh files are needed.  The episode runner records the menagerie commit in
every `manifest.json`.
