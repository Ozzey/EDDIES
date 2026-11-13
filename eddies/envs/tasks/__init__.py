"""Task registry: name -> (module, class, constructor kwargs).

The ten paper tasks, the hand-written base classes they refine (``*_legacy`` / the G1 and hand
originals) and the four LTLf-specified Franka variants.  Episode directories are named by these keys.
"""
from __future__ import annotations

TASKS = {
    # Franka Emika Panda: skillbook (Eddie) tasks; the hand-written originals are *_legacy
    "franka_fmb": ("eddies.envs.tasks.franka_sb", "FrankaFMBTokens", {}),
    "franka_fmb_legacy": ("eddies.envs.tasks.franka_fmb", "FrankaFMBTask", {}),
    "franka_peg_insertion": ("eddies.envs.tasks.franka_sb", "FrankaPegTokens", {}),
    "franka_peg_insertion_legacy": ("eddies.envs.tasks.franka_peg", "FrankaPegInsertionTask", {}),
    "franka_cube_stack": ("eddies.envs.tasks.franka_sb", "FrankaStackTokens", {}),
    "franka_cube_stack_legacy": ("eddies.envs.tasks.franka_stack", "FrankaCubeStackTask", {}),
    "franka_drawer": ("eddies.envs.tasks.franka_sb", "FrankaDrawerTokens", {}),
    "franka_drawer_legacy": ("eddies.envs.tasks.franka_drawer", "FrankaDrawerTask", {}),
    # LTLf-specified: the Streamer composes the Eddie stream from the formula (task kwarg spec=...)
    "franka_cube_stack_ltl": ("eddies.envs.tasks.franka_ltl", "FrankaStackLTL", {}),
    "franka_peg_insertion_ltl": ("eddies.envs.tasks.franka_ltl", "FrankaPegLTL", {}),
    "franka_fmb_ltl": ("eddies.envs.tasks.franka_ltl", "FrankaFMBLTL", {}),
    "franka_drawer_ltl": ("eddies.envs.tasks.franka_ltl", "FrankaDrawerLTL", {}),
    # Unitree G1 humanoid with Dex3 hands
    "g1_peg_insertion_sb": ("eddies.envs.tasks.g1_sb", "G1PegTokens", {}),
    "g1_drawer_sb": ("eddies.envs.tasks.g1_sb", "G1DrawerTokens", {}),
    "g1_peg_insertion": ("eddies.envs.tasks.g1_peg", "G1PegInsertionTask", {}),
    "g1_drawer": ("eddies.envs.tasks.g1_drawer", "G1DrawerTask", {}),
    # UR5e + LEAP / Aero hand (one primitive set, both hands)
    "leap_wheel_sb": ("eddies.envs.tasks.hand_sb", "HandWheelTokens", {"hand": "leap"}),
    "aero_wheel_sb": ("eddies.envs.tasks.hand_sb", "HandWheelTokens", {"hand": "aero"}),
    "leap_peg_tighten_sb": ("eddies.envs.tasks.hand_sb", "HandPegTightenTokens", {"hand": "leap"}),
    "aero_peg_tighten_sb": ("eddies.envs.tasks.hand_sb", "HandPegTightenTokens", {"hand": "aero"}),
    "leap_wheel": ("eddies.envs.tasks.hand_wheel", "HandWheelTask", {"hand": "leap"}),
    "aero_wheel": ("eddies.envs.tasks.hand_wheel", "HandWheelTask", {"hand": "aero"}),
    "leap_peg_tighten": ("eddies.envs.tasks.hand_peg_tighten", "HandPegTightenTask", {"hand": "leap"}),
    "aero_peg_tighten": ("eddies.envs.tasks.hand_peg_tighten", "HandPegTightenTask", {"hand": "aero"}),
}

PAPER_TASKS = ["franka_fmb", "franka_peg_insertion", "franka_cube_stack", "franka_drawer",
               "g1_peg_insertion_sb", "g1_drawer_sb", "leap_wheel_sb", "leap_peg_tighten_sb",
               "aero_wheel_sb", "aero_peg_tighten_sb"]
TASK_ORDER = list(TASKS)


def make_task(name: str, seed: int, **kwargs):
    import importlib
    module, cls, preset = TASKS[name]
    klass = getattr(importlib.import_module(module), cls)
    return klass(seed, **{**preset, **kwargs})
