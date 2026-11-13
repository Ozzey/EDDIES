"""One Isaac episode = one (task, method, seed, condition); same contract as the MuJoCo runner.

Writes ``<out>/manifest.json`` (identical schema to eddies.envs's runner, plus
``"simulator": "Isaac Sim ..."`` and bridge diagnostics), ``trajectory.npz`` and a
1920x1080 ``video.mp4``.  Success = physical task predicate held for ``success_hold``
seconds AND the complete measured trace satisfies the LTLf formula.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def run_episode(task_name: str, method_name: str, seed: int, out_dir, *, perturb: bool = False,
                record: bool = True, width: int = 1920, height: int = 1080, horizon: float | None = None,
                save_trajectory: bool = True,
                method_kwargs: dict | None = None, task_overrides: dict | None = None) -> dict:
    from eddies.core.ltlf import FALSE, Monitor, atoms
    from eddies.isaac.bridge import PhysXBridge
    from eddies.isaac.export import make_task_and_usd
    from eddies.envs.methods import make_method
    from eddies.isaac.recorder import IsaacRecorder
    from eddies.envs.framework.physics import physics_summary
    from eddies.envs.paths import MENAGERIE_COMMIT
    try:
        from isaacsim.core.version import get_version
        isaac_version = ".".join(str(v) for v in get_version()[:3])
    except Exception:
        isaac_version = "6.0"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"task": task_name, "method": method_name, "seed": int(seed), "status": "initializing",
                "success": False, "task_success": False, "property_success": False,
                "simulator": f"Isaac Sim {isaac_version} (PhysX 5, RTX)", "menagerie_commit": MENAGERIE_COMMIT,
                "condition": "perturbed" if perturb else "nominal", "video": None,
                "task_source_sha256": None}
    wall0 = time.perf_counter()
    recorder = None
    try:
        task, usd = make_task_and_usd(task_name, seed, perturb)
        for k_, v_ in (task_overrides or {}).items():      # eddies.cert: rho_step, failure_budget
            setattr(task, k_, v_)
        task.reset()                                     # pre-episode configuration (MuJoCo reset)
        bridge = PhysXBridge(task, usd)
        bridge.write_state_from_mirror()
        bridge.world.step(render=False)
        bridge.sync_from_physx()
        bridge.d.time = 0.0
        method = make_method(method_name, task, **(method_kwargs or {}))
        monitor = Monitor(task.formula)
        props = sorted(atoms(monitor.initial))
        horizon = float(horizon or task.horizon)
        manifest.update({"robot": task.robot, "family": task.family, "formula": task.formula,
                         "physics": {**physics_summary(task.model), "engine": "PhysX 5 (TGS, CPU)"},
                         "bridge": bridge.diagnostics(), "initial_pose_error_m": bridge.pose_error(),
                         "control_dt": task.control_dt, "horizon_s": horizon, "embedding": task.space.describe(),
                         "variant": _jsonable(task.variant_info()), "description": task.description})
        bridge.read_contacts()
        labels = frozenset(task.labels())
        monitor.step(labels)
        obs = task.observe()
        obs.update(labels=labels, ltl_residual=monitor.state, ltl_status=monitor.status, t=task.time)
        method.reset(obs)
        if record:
            recorder = IsaacRecorder(bridge, out / "video.mp4", task.main_camera, task.detail_camera,
                                     width=width, height=height, fps=task.video_fps)
        dt = bridge.dt
        dec = max(1, int(round(task.control_dt / dt)))
        ldec = max(1, int(task.label_decimation))
        n_steps = int(round(horizon / dt))
        frame_period, next_frame = 1.0 / task.video_fps, 0.0
        e = np.asarray(method.e_prev if getattr(method, "e_prev", None) is not None else task.initial_command(), float)
        act_ms, render_s, physics_wall = [], 0.0, 0.0
        rows = {"t": [], "e": [], "skill": [], "labels": [], "residual_false": []}
        signal_rows: dict = {}
        diag_max: dict = {}
        bad_steps, first_bad, success_since, completion = 0, None, None, None
        task_success, stop_reason, nonfinite_cmd, max_jump = False, "horizon", 0, 0.0
        manifest["status"] = "running"
        label = getattr(method, "label", lambda: method_name)
        for k in range(n_steps):
            if k % dec == 0:
                obs = task.observe()
                obs.update(labels=labels, ltl_residual=monitor.state, ltl_status=monitor.status, t=task.time)
                t0 = time.perf_counter()
                e_new = np.asarray(method.act(obs), float)
                act_ms.append((time.perf_counter() - t0) * 1e3)
                if e_new.shape != e.shape or not np.isfinite(e_new).all():
                    nonfinite_cmd += 1
                    e_new = e.copy()
                max_jump = max(max_jump, task.space.norm(e_new - e))
                e = e_new
                task.apply_command(e)
                rows["t"].append(task.time); rows["e"].append(e.copy()); rows["skill"].append(method.status())
                if hasattr(task, "achieved"):          # measured embedding z_k (eddies.cert)
                    try:
                        rows.setdefault("z", []).append(np.asarray(task.achieved(obs), float).copy())
                    except Exception:
                        pass
                rows["labels"].append(sum(1 << i for i, p in enumerate(props) if p in labels))
                rows["residual_false"].append(monitor.state == FALSE)
                for key, val in task.log_signals().items():
                    signal_rows.setdefault(key, []).append(np.asarray(val, float))
                if method.done:
                    stop_reason = "method_stop:" + str(method.done_reason)
                    break
            tp = time.perf_counter()
            if task.perturb:
                task.disturb()
            task.inner_loop()
            bridge.step()                                 # PhysX physics step
            physics_wall += time.perf_counter() - tp
            if (k + 1) % ldec == 0:
                labels = frozenset(task.labels())
                monitor.step(labels)
                if "bad" in labels:
                    bad_steps += 1
                    first_bad = first_bad if first_bad is not None else float(task.time)
                for key, val in task.step_diagnostics().items():
                    if val is not None and np.isfinite(val):
                        diag_max[key] = max(diag_max.get(key, -np.inf), float(val))
                if task.task_success():
                    success_since = success_since if success_since is not None else float(task.time)
                    if task.time - success_since >= task.success_hold - 1e-9:
                        task_success, completion, stop_reason = True, success_since, "task_success"
                        break
                else:
                    success_since = None
            if not np.isfinite(bridge.d.qpos).all():
                stop_reason = "physics_divergence"
                break
            if recorder is not None and task.time + 1e-9 >= next_frame:
                tr = time.perf_counter()
                lines = [f"method: {label()}   seed: {seed}   t = {task.time:6.2f} s   [Isaac Sim / PhysX]",
                         f"skill: {method.status()}",
                         f"LTLf: {monitor.status}   labels: {', '.join(sorted(labels)) or '-'}"]
                lines += task.overlay_lines()
                recorder.capture(f"{task.name}  ({task.robot})", lines)
                render_s += time.perf_counter() - tr
                next_frame += frame_period
        if recorder is not None:
            for _ in range(int(task.video_fps * 0.5)):
                recorder.capture(f"{task.name}  ({task.robot})",
                                 [f"method: {label()}   seed: {seed}   t = {task.time:6.2f} s   [Isaac Sim / PhysX]",
                                  f"END: {stop_reason}   task success: {task_success}", f"LTLf: {monitor.status}"])
        property_ok = bool(monitor.accepts())
        success = bool(task_success and property_ok)
        act = np.asarray(act_ms) if act_ms else np.zeros(1)
        manifest.update({
            "status": "completed", "success": success, "task_success": bool(task_success),
            "property_success": property_ok, "ltl_status": monitor.status, "stop_reason": stop_reason,
            "completion_time_s": completion if success else None, "task_completion_time_s": completion,
            "sim_time_s": float(task.time), "bad_steps": int(bad_steps), "first_bad_time_s": first_bad,
            "max_command_jump": float(max_jump), "nonfinite_commands": int(nonfinite_cmd),
            "diagnostics_max": diag_max,
            "compute_ms_per_control_step": {"mean": float(act.mean()), "p50": float(np.median(act)),
                                            "p95": float(np.percentile(act, 95)), "max": float(act.max()),
                                            "n": int(len(act_ms))},
            "render_time_s": render_s, "physics_wall_time_s": physics_wall,
            "method_info": _jsonable(method.info()), "task_metrics": _jsonable(task.episode_metrics()),
            "perturbation": _jsonable(task.perturbation_info()) if task.perturb else None,
            "propositions": props})
        if save_trajectory and rows["t"]:
            arrays = {"t": np.asarray(rows["t"]), "e": np.asarray(rows["e"]),
                      "labels": np.asarray(rows["labels"], np.int64),
                      "residual_false": np.asarray(rows["residual_false"]),
                      "skill": np.asarray(rows["skill"], dtype=object).astype(str)}
            if rows.get("z") and len(rows["z"]) == len(rows["t"]):
                arrays["z"] = np.asarray(rows["z"])
            for key, vals in signal_rows.items():
                try:
                    arrays["sig_" + key] = np.asarray(vals)
                except ValueError:
                    pass
            np.savez_compressed(out / "trajectory.npz", **arrays)
        if recorder is not None:
            manifest.update({"video": "video.mp4", "video_frames": recorder.frames,
                             "video_resolution": [width, height]})
    except Exception as ex:
        manifest.update({"status": "error", "success": False, "error": repr(ex),
                         "traceback": traceback.format_exc()})
    finally:
        if recorder is not None:
            recorder.close()
        manifest["wall_time_s"] = time.perf_counter() - wall0
        (out / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=1))
    return manifest
