"""Method registry.  Every method outputs commands in the task's embedding space.

The registry key ``bistl`` is the Streamer (the paper's method): it is kept because every episode
directory and manifest of the released logs is named after it (``.../<task>/bistl/seed_XXXX``).
"""
from __future__ import annotations

METHODS = {
    # the Streamer (ours)
    "bistl": ("eddies.envs.methods.streamer", "StreamerMethod", {}),
    "streamer": ("eddies.envs.methods.streamer", "StreamerMethod", {}),
    # Streamer variants: fixed domain only (no slew bound) / no temporal supervisor
    "bistl_noslew": ("eddies.envs.methods.streamer", "StreamerMethod", {"use_slew": False}),
    "bistl_nosup": ("eddies.envs.methods.streamer", "StreamerMethod", {"use_supervisor": False}),
    # learned Eddie proposer (eddies.cert.proposer) in place of the symbolic token choice (App. D)
    "bistl_learned": ("eddies.envs.methods.learned_tokens", "LearnedTokenMethod", {}),
    "bistl_learned_raw": ("eddies.envs.methods.learned_tokens", "LearnedTokenMethod", {"guard": False}),
    "bistl_learned_state": ("eddies.envs.methods.learned_tokens", "LearnedTokenMethod", {"features": "state"}),
    "bistl_learned_state_raw": ("eddies.envs.methods.learned_tokens", "LearnedTokenMethod",
                                {"features": "state", "guard": False}),
    "bistl_learned_cmd": ("eddies.envs.methods.learned_tokens", "LearnedTokenCmdMethod", {}),
    "bistl_cmd": ("eddies.envs.methods.learned_tokens", "SymbolicCmdMethod", {}),
}

DISPLAY = {
    "bistl": "Streamer (ours)",
    "streamer": "Streamer (ours)",
    "bistl_noslew": "Streamer w/o slew",
    "bistl_nosup": "Streamer w/o supervisor",
}


def make_method(name: str, task, **kwargs):
    import importlib
    module, cls, preset = METHODS[name]
    klass = getattr(importlib.import_module(module), cls)
    return klass(task, **{**preset, **kwargs})
