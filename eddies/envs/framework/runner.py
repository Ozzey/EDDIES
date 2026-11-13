"""Episode runner: physics loop, LTLf monitoring, metrics, manifest, video.

One episode = one (task, method, seed).  The runner

* steps MuJoCo at the task's physics rate (no kinematic object writes after
  ``task.reset()``); method.act() runs at the control rate and its output is
  decoded by the task's shared decoder,
* evaluates measured labels and the LTLf monitor at every ``label_decimation``
  physics steps (default: every step),
* declares success only when the physical task predicate has held for
  ``task.success_hold`` seconds AND the full finite trace satisfies the formula
  (task-and-property success; task-only success is also stored),
* writes manifest.json, trajectory.npz and (optionally) a Full-HD video.
"""
from __future__ import annotations

import json
import platform
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np

from eddies.core.ltlf import FALSE, Monitor, atoms
from eddies.core.metrics import wilson  # noqa: F401  (re-exported for scripts)
from eddies.envs.framework.physics import physics_summary
from eddies.envs.paths import MENAGERIE_COMMIT
from eddies.common.config import load

SIM = load().sim


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
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def run_episode(task_name: str, method_name: str, seed: int, out_dir: str | Path, *,
                record: bool = True, width: int = SIM.video.width, height: int = SIM.video.height,
                horizon: float | None = None, method_kwargs: dict | None = None,
                task_kwargs: dict | None = None, save_trajectory: bool = True,
                verbose: bool = False) -> dict:
    from eddies.envs.methods import make_method
    from eddies.envs.tasks import make_task

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"task": task_name, "method": method_name, "seed": int(seed), "status": "initializing",
                "success": False, "task_success": False, "property_success": False,
                "simulator": f"MuJoCo {mujoco.__version__}", "menagerie_commit": MENAGERIE_COMMIT,
                "python": platform.python_version(), "video": None}
    wall0 = time.perf_counter()
    recorder = None
    task = None
    try:
        task = make_task(task_name, seed, **(task_kwargs or {}))
        import hashlib, inspect, sys as _sys
        manifest["task_source_sha256"] = hashlib.sha256(
            open(inspect.getfile(_sys.modules[type(task).__module__]), "rb").read()).hexdigest()
        task.reset()
        model, data = task.model, task.data
        method = make_method(method_name, task, **(method_kwargs or {}))
        monitor = Monitor(task.formula)
        props = sorted(atoms(monitor.initial))
        horizon = float(horizon or task.horizon)
        manifest.update({"robot": task.robot, "family": task.family, "formula": task.formula,
                         "physics": physics_summary(model), "control_dt": task.control_dt,
                         "horizon_s": horizon, "embedding": task.space.describe(),
                         "variant": _jsonable(task.variant_info()),
                         "description": task.description})
        labels = frozenset(task.labels())
        monitor.step(labels)                      # the initial letter is observed, not assumed
        obs = task.observe()
        obs.update(labels=labels, ltl_residual=monitor.state, ltl_status=monitor.status, t=data.time)
        method.reset(obs)
        if record:
            recorder = __import__("eddies.envs.framework.recorder", fromlist=["VideoRecorder"]) \
                .VideoRecorder(model, out / "video.mp4", task.main_camera, task.detail_camera,
                               width=width, height=height, fps=task.video_fps)
        dec = max(1, int(round(task.control_dt / model.opt.timestep)))
        ldec = max(1, int(task.label_decimation))
        n_steps = int(round(horizon / model.opt.timestep))
        frame_period = 1.0 / task.video_fps
        next_frame = 0.0
        e = np.asarray(method.e_prev if getattr(method, "e_prev", None) is not None
                       else task.initial_command(), float)
        act_ms, render_s = [], 0.0
        rows = {"t": [], "e": [], "skill": [], "labels": [], "residual_false": []}
        signal_rows: dict[str, list] = {}
        diag_max: dict[str, float] = {}
        bad_steps, first_bad, success_since, completion = 0, None, None, None
        task_success, stop_reason, nonfinite_cmd = False, "horizon", 0
        max_jump = 0.0
        manifest["status"] = "running"
        physics_wall = 0.0
        for k in range(n_steps):
            if k % dec == 0:
                obs = task.observe()
                obs.update(labels=labels, ltl_residual=monitor.state, ltl_status=monitor.status,
                           t=data.time)
                t0 = time.perf_counter()
                e_new = np.asarray(method.act(obs), float)
                act_ms.append((time.perf_counter() - t0) * 1e3)
                if e_new.shape != e.shape or not np.isfinite(e_new).all():
                    nonfinite_cmd += 1
                    e_new = e.copy()
                max_jump = max(max_jump, task.space.norm(e_new - e))
                e = e_new
                task.apply_command(e)
                rows["t"].append(data.time)
                rows["e"].append(e.copy())
                if hasattr(task, "achieved"):
                    # measured embedding z_k (same chart as e), observed BEFORE e_k is applied;
                    # used by eddies.cert to identify z_{k+1} = A z_k + B e_k + d + w
                    try:
                        rows.setdefault("z", []).append(np.asarray(task.achieved(obs), float).copy())
                    except Exception:
                        pass
                rows["skill"].append(method.status())
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
            mujoco.mj_step(model, data)
            physics_wall += time.perf_counter() - tp
            if (k + 1) % ldec == 0:
                labels = frozenset(task.labels())
                monitor.step(labels)
                if "bad" in labels:
                    bad_steps += 1
                    if first_bad is None:
                        first_bad = float(data.time)
                for key, val in task.step_diagnostics().items():
                    if val is not None and np.isfinite(val):
                        diag_max[key] = max(diag_max.get(key, -np.inf), float(val))
                if task.task_success():
                    if success_since is None:
                        success_since = float(data.time)
                    if data.time - success_since >= task.success_hold - 1e-9:
                        task_success = True
                        completion = success_since
                        stop_reason = "task_success"
                        break
                else:
                    success_since = None
            if not np.isfinite(data.qpos).all() or data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
                stop_reason = "physics_divergence"
                break
            if recorder is not None and data.time + 1e-9 >= next_frame:
                tr = time.perf_counter()
                status = method.status()
                lines = [f"method: {method.label()}   seed: {seed}   t = {data.time:6.2f} s",
                         f"skill: {status}",
                         f"LTLf: {monitor.status}   labels: {', '.join(sorted(labels)) or '-'}"]
                lines += task.overlay_lines()
                recorder.capture(data, f"{task.name}  ({task.robot})", lines)
                render_s += time.perf_counter() - tr
                next_frame += frame_period
        # hold the final frame for half a second so the end state is visible
        if recorder is not None:
            for _ in range(int(task.video_fps * 0.5)):
                recorder.capture(data, f"{task.name}  ({task.robot})",
                                 [f"method: {method.label()}   seed: {seed}   t = {data.time:6.2f} s",
                                  f"END: {stop_reason}   task success: {task_success}",
                                  f"LTLf: {monitor.status}"])
        property_ok = bool(monitor.accepts())
        success = bool(task_success and property_ok)
        act = np.asarray(act_ms) if act_ms else np.zeros(1)
        manifest.update({
            "status": "completed",
            "success": success, "task_success": bool(task_success), "property_success": property_ok,
            "ltl_status": monitor.status, "stop_reason": stop_reason,
            "completion_time_s": completion if success else None,
            "task_completion_time_s": completion,
            "sim_time_s": float(data.time), "bad_steps": int(bad_steps), "first_bad_time_s": first_bad,
            "max_command_jump": float(max_jump), "nonfinite_commands": int(nonfinite_cmd),
            "diagnostics_max": diag_max,
            "compute_ms_per_control_step": {"mean": float(act.mean()), "p50": float(np.median(act)),
                                            "p95": float(np.percentile(act, 95)),
                                            "max": float(act.max()), "n": int(len(act_ms))},
            "wall_time_s": time.perf_counter() - wall0, "render_time_s": render_s,
            "physics_wall_time_s": physics_wall,
            "method_info": _jsonable(method.info()),
            "task_metrics": _jsonable(task.episode_metrics()),
            "condition": "perturbed" if task.perturb else "nominal",
            "perturbation": _jsonable(task.perturbation_info()) if task.perturb else None,
            "propositions": props,
        })
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
            manifest["video"] = "video.mp4"
            manifest["video_frames"] = recorder.frames
            manifest["video_resolution"] = [width, height]
    except Exception as ex:  # recorded, never reported as success
        manifest.update({"status": "error", "success": False, "error": repr(ex),
                         "traceback": traceback.format_exc()})
        if verbose:
            traceback.print_exc()
    finally:
        if recorder is not None:
            recorder.close()
        manifest["wall_time_s"] = time.perf_counter() - wall0
        (out / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=1))
    return manifest
