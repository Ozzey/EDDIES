"""MuJoCo environments of the paper.

Scenes are built from google-deepmind/mujoco_menagerie robots (external/mujoco_menagerie, pinned commit in
eddies/config.yaml); every task is composed from its embodiment's primitive set (``skillbook``); the Streamer
(``methods/streamer.py``) chains Eddies under the LTLf supervisor; ``framework/runner.py`` logs
manifest.json + trajectory.npz (+ video) per episode.
"""
