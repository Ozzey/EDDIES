# Installation

Everything that reproduces the paper's tables and figures from `results/` runs on a laptop with the core
dependencies. The simulator needs MuJoCo and the robot models; the data collection and the heavier analyses are
long multi-core runs (`scripts/reproduce.sh`, see [REPRODUCTION.md](REPRODUCTION.md)). Isaac Sim is optional.

## 1. Core (analysis from logs, tables, figures)

Python 3.10 or newer.

```bash
git clone https://github.com/Ozzey/EDDIES && cd EDDIES
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python -m eddies.evaluate --list          # which steps have their inputs in results/
pytest -q                                 # unit tests + table reproduction from results/
```

Dependencies (`pyproject.toml`): numpy, scipy, pyyaml, matplotlib, pillow, imageio(+ffmpeg), mujoco >= 3.3, < 4
(the paper's episodes were collected with 3.12). The console script `eddies` wraps the entry points (`eddies evaluate all`, see
[USAGE.md](USAGE.md)).

## 2. Simulation (the ten MuJoCo tasks)

The scenes are built procedurally with `MjSpec`; only the robot models are external. They come from MuJoCo
Menagerie at the commit pinned in `eddies/config.yaml` (`paths.menagerie_commit`,
`0059d4335f8156206f63a35662313385f7ad6d74`): `franka_emika_panda`, `unitree_g1`, `universal_robots_ur5e`,
`leap_hand`, `tetheria_aero_hand_open`.

```bash
scripts/download_menagerie.sh             # sparse checkout into external/mujoco_menagerie (~130 MB, gitignored)
# or point at an existing checkout:
export EDDIES_MENAGERIE=/path/to/mujoco_menagerie
python -m eddies.envs.scripts.try_task --task franka_cube_stack --seeds 0 --methods bistl --workers 1
```

Rendering (videos, stills) uses off-screen EGL: `export MUJOCO_GL=egl` (or `osmesa` without a GPU). Every episode
manifest records the menagerie commit.

## 3. Episode logs (dataset)

The analyses of `eddies.certify` and `eddies.gate` read the logged episodes of the paper (15,450 episodes,
2.4 GB) from the Hugging Face dataset `Ozzey/eddies`:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s     # Hugging Face CLI, or: pip install -U huggingface_hub
datasets/download.sh --subset sb                      # MuJoCo train/calibration/test episodes of the ten tasks (1.0 GB)
datasets/download.sh --subset franka,gate,isaac       # subsets by robot or tag; see datasets/README.md
```

The files land in `datasets/episodes/` (gitignored). `EDDIES_EPISODES` overrides the location;
`EDDIES_RESULTS` the results root (default `results/`). See [DATA_FORMAT.md](DATA_FORMAT.md).

## 4. Optional

| what | needed for | install |
| --- | --- | --- |
| Isaac Sim 6.0.1 (pip build, PhysX 5) | `eddies.isaac`: PhysX episodes of Table 2 / Table 3, bottom row of Fig. 6 | install Isaac Sim, set `ISAAC_INSTALL_ROOT`, run with its `python.sh` (`ISAAC_PY=<isaac>/python.sh scripts/reproduce.sh collect-isaac`) |

`external/README.md` lists the third-party assets; they are fetched or installed separately.

## 5. Long runs

`scripts/reproduce.sh <experiment>` runs one paper experiment with the settings of the paper and exports what it
needs: `PYTHONPATH`, `MUJOCO_GL=egl`, one BLAS thread per process and the episode roots of the download. No
scheduler is needed; on a cluster, wrap one experiment per job. [REPRODUCTION.md](REPRODUCTION.md) lists the cores,
memory and hours of each. No GPU is needed except for rendering and Isaac Sim (`OMNI_KIT_ACCEPT_EULA=YES`).

## Environment variables

| variable | default | meaning |
| --- | --- | --- |
| `EDDIES_RESULTS` | `results/` | result JSON / NPZ (read by `eddies.evaluate`, written by the experiments) |
| `EDDIES_EPISODES` | `datasets/episodes/` (or `$EDDIES_RESULTS/episodes` if it exists) | MuJoCo episode logs |
| `EDDIES_RESULTS_ISAAC`, `EDDIES_RESULTS_REAL` | `datasets/episodes_isaac`, `datasets/episodes_real` | PhysX and robot episode roots |
| `EDDIES_MENAGERIE` | `external/mujoco_menagerie` | robot models |
| `EDDIES_MODELS` | `results/tokprop` | learned proposer weights |
| `EDDIES_SCRATCH` | `/tmp/eddies` | scratch output of `try_task`, `render` |
| `MUJOCO_GL` | — | `egl` / `osmesa` for off-screen rendering |
| `ISAAC_INSTALL_ROOT` | `~/isaacsim` | Isaac Sim installation (optional) |
