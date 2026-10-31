"""Evaluate fixed FastWAM inference configurations on LIBERO tasks.

Example:
    ./.conda/fastwam/bin/python eval_fixed_baselines.py \
        --model-path checkpoints/fastwam_release/libero_optional_idm_2cam224.pt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collect_counterfactual_data import WorkerContext
from core.sim_state_manager import SimStateManager
from oracle.physics import (
    classify_physical_phase,
    collect_object_positions,
    measure_physics,
)
from real_fastwam_adapter import FASTWAM_ROOT, LIBERO_ROOT


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "fastwam_release"
    / "libero_optional_idm_2cam224.pt"
)
DEFAULT_HARDWARE_COST_TABLE = PROJECT_ROOT / "configs" / "hardware_cost_table.json"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "results" / "fixed_baselines_summary.csv"
DEFAULT_EPISODES_CSV = PROJECT_ROOT / "results" / "fixed_baselines_episodes.csv"
DEFAULT_TASK_IDS = "0"
DEFAULT_TRAJECTORY_DIR = PROJECT_ROOT / "outputs" / "trajectories"
TRAJECTORY_SCHEMA_VERSION = 1
_ARRAY_REF_KEY = "__trajectory_array_ref__"
_ARRAY_BLOB_THRESHOLD = 1024


@dataclass(frozen=True)
class FixedConfig:
    name: str
    H: int
    D: int
    W: int
    E: int

    def as_dict(self) -> dict[str, int]:
        return {"H": self.H, "D": self.D, "W": self.W, "E": self.E}


FIXED_CONFIGS: tuple[FixedConfig, ...] = (
    FixedConfig("No-Imagination", H=0, D=0, W=1, E=10),
    FixedConfig("Fixed-Small", H=2, D=2, W=1, E=4),
    FixedConfig("Fixed-Full", H=8, D=16, W=4, E=2),
)


def parse_methods(value: str) -> tuple[FixedConfig, ...]:
    """Resolve a comma-separated method list while preserving grid order."""
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise ValueError("--methods must select at least one method")
    by_name = {config.name: config for config in FIXED_CONFIGS}
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(
            "Unknown fixed method(s): " + ", ".join(unknown)
            + "; choose from " + ", ".join(by_name)
        )
    if len(requested) != len(set(requested)):
        raise ValueError("--methods contains duplicate method names")
    requested_names = set(requested)
    return tuple(config for config in FIXED_CONFIGS if config.name in requested_names)


def _episode_key(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    try:
        return (
            str(row["task_suite"]),
            int(row["task_id"]),
            int(row["seed"]),
            str(row["method"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed episode record: {row}") from exc


def read_episode_csv(path: Path) -> list[dict[str, Any]]:
    """Load an existing checkpoint CSV and reject duplicate episode keys."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        rows = [dict(row) for row in csv.DictReader(source)]
    seen: set[tuple[str, int, int, str]] = set()
    for row in rows:
        key = _episode_key(row)
        if key in seen:
            raise ValueError(f"Duplicate episode key in {path}: {key}")
        seen.add(key)
    return rows


def merge_episode_records(
    existing: Sequence[Mapping[str, Any]],
    new: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge records by suite/task/seed/method, rejecting conflicting reruns."""
    merged = [dict(row) for row in existing]
    by_key = {_episode_key(row): row for row in merged}
    if len(by_key) != len(merged):
        raise ValueError("Existing episode records contain duplicate keys")
    for source_row in new:
        row = dict(source_row)
        key = _episode_key(row)
        previous = by_key.get(key)
        if previous is not None:
            if previous != row:
                raise ValueError(f"Conflicting episode record for key {key}")
            continue
        merged.append(row)
        by_key[key] = row
    return merged


class EpisodeAdapter(Protocol):
    env: Any
    parameterized_wam: Callable[..., Tensor]
    cuda_devices: Sequence[str]

    def start_task(self, task_id: int, seed: int) -> Any: ...

    def step(self, action: np.ndarray) -> Any: ...

    def is_success(self, observation: Any, info: dict[str, Any]) -> bool: ...

    def close(self) -> None: ...


AdapterFactory = Callable[[WorkerContext], EpisodeAdapter]


def parse_task_ids(value: str) -> list[int]:
    """Parse comma-separated task IDs and inclusive ranges, e.g. '0-4,7'."""
    task_ids: set[int] = set()
    for item in value.split(","):
        token = item.strip()
        if not token:
            raise ValueError("--task-ids contains an empty item")
        if "-" in token:
            lower_text, separator, upper_text = token.partition("-")
            if not separator or not lower_text.isdigit() or not upper_text.isdigit():
                raise ValueError(f"Invalid task ID range {token!r}")
            lower, upper = int(lower_text), int(upper_text)
            if upper < lower:
                raise ValueError(f"Task ID range has descending bounds: {token!r}")
            task_ids.update(range(lower, upper + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"Invalid task ID {token!r}")
            task_ids.add(int(token))
    if not task_ids:
        raise ValueError("--task-ids must select at least one task")
    return sorted(task_ids)


def parse_seeds(value: str) -> list[int]:
    """Parse explicit seed IDs using the same comma/range syntax as tasks."""
    try:
        seeds = parse_task_ids(value)
    except ValueError as exc:
        raise ValueError(str(exc).replace("--task-ids", "--seeds")) from exc
    return seeds


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.") or "item"


def trajectory_path_for_episode(
    trajectory_dir: Path,
    *,
    task_suite: str,
    task_id: int,
    seed: int,
    method: str,
) -> Path:
    """Return the deterministic path for one baseline-method trajectory."""
    return (
        trajectory_dir.expanduser().resolve()
        / _safe_path_component(task_suite)
        / f"task_{task_id}"
        / f"seed_{seed}"
        / f"{_safe_path_component(method)}_ep0.pt"
    )


def _array_digest(array: np.ndarray, kind: str) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(kind.encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _is_uint8_image(array: np.ndarray) -> bool:
    return bool(
        array.dtype == np.uint8
        and array.ndim in (2, 3)
        and array.shape[0] >= 16
        and array.shape[1] >= 16
        and (array.ndim == 2 or array.shape[2] in (1, 3, 4))
    )


def _pack_array(
    value: np.ndarray | Tensor,
    blobs: dict[str, dict[str, Any]],
) -> Any:
    is_tensor = isinstance(value, Tensor)
    array = (
        value.detach().cpu().numpy()
        if is_tensor
        else np.asarray(value)
    )
    array = np.ascontiguousarray(array)
    if array.nbytes < _ARRAY_BLOB_THRESHOLD:
        return value.detach().cpu().clone() if is_tensor else array.copy()

    kind = "torch" if is_tensor else "numpy"
    digest = _array_digest(array, kind)
    if digest not in blobs:
        if _is_uint8_image(array):
            image_array = array[..., 0] if array.ndim == 3 and array.shape[2] == 1 else array
            encoded = io.BytesIO()
            Image.fromarray(image_array).save(encoded, format="PNG", compress_level=1)
            encoding = "png"
            data = encoded.getvalue()
        else:
            encoding = "raw"
            data = array.tobytes(order="C")
        blobs[digest] = {
            "kind": kind,
            "dtype": array.dtype.str,
            "shape": tuple(int(item) for item in array.shape),
            "encoding": encoding,
            "data": data,
        }
    return {_ARRAY_REF_KEY: digest}


def _pack_trajectory_value(
    value: Any,
    blobs: dict[str, dict[str, Any]],
) -> Any:
    """Losslessly deduplicate arrays and PNG-compress uint8 camera frames."""
    if isinstance(value, (np.ndarray, Tensor)):
        return _pack_array(value, blobs)
    if isinstance(value, Mapping):
        return {key: _pack_trajectory_value(item, blobs) for key, item in value.items()}
    if isinstance(value, list):
        return [_pack_trajectory_value(item, blobs) for item in value]
    if isinstance(value, tuple):
        return tuple(_pack_trajectory_value(item, blobs) for item in value)
    return value


def _unpack_trajectory_value(
    value: Any,
    blobs: Mapping[str, Mapping[str, Any]],
) -> Any:
    if isinstance(value, Mapping) and set(value) == {_ARRAY_REF_KEY}:
        blob = blobs[str(value[_ARRAY_REF_KEY])]
        shape = tuple(int(item) for item in blob["shape"])
        dtype = np.dtype(str(blob["dtype"]))
        if blob["encoding"] == "png":
            array = np.asarray(Image.open(io.BytesIO(blob["data"])), dtype=dtype)
            if len(shape) == 3 and shape[2] == 1 and array.ndim == 2:
                array = array[..., None]
            array = np.ascontiguousarray(array.reshape(shape))
        elif blob["encoding"] == "raw":
            array = np.frombuffer(blob["data"], dtype=dtype).reshape(shape).copy()
        else:
            raise ValueError(f"Unsupported trajectory blob encoding {blob['encoding']!r}")
        return torch.from_numpy(array.copy()) if blob["kind"] == "torch" else array
    if isinstance(value, Mapping):
        return {key: _unpack_trajectory_value(item, blobs) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack_trajectory_value(item, blobs) for item in value]
    if isinstance(value, tuple):
        return tuple(_unpack_trajectory_value(item, blobs) for item in value)
    return value


def load_trajectory(path: Path, *, materialize: bool = True) -> dict[str, Any]:
    """Load a saved trajectory, optionally restoring compact array references."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported trajectory payload in {path}")
    if not materialize:
        return payload
    blobs = payload.get("array_blobs", {})
    result = dict(payload)
    result["steps"] = _unpack_trajectory_value(payload["steps"], blobs)
    return result


def _write_trajectory(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _trajectory_is_valid(path_text: Any) -> bool:
    if not path_text:
        return False
    path = Path(str(path_text)).expanduser()
    if not path.is_file():
        return False
    try:
        payload = load_trajectory(path, materialize=False)
    except (OSError, RuntimeError, ValueError, EOFError):
        return False
    return bool(payload.get("steps"))


def _uint8_observation(observation: Any) -> Any:
    copied = _copy_observation(observation)
    if not isinstance(copied, dict):
        return copied
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        if key not in copied:
            continue
        image = np.asarray(copied[key])
        if image.dtype != np.uint8:
            if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
                raise ValueError(f"Trajectory image {key} is not finite numeric data")
            if image.size and image.min() >= 0 and image.max() <= 1:
                image = image * 255.0
            image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
        copied[key] = np.ascontiguousarray(image)
    return copied


def _contact_value(info: Mapping[str, Any], contact_force_n: float | None) -> Any:
    for key in ("contact", "contact_force", "contact_force_n", "is_contact"):
        if key in info:
            return _copy_observation(info[key])
    return contact_force_n


def load_hardware_costs(path: Path) -> dict[tuple[int, int, int, int], dict[str, float]]:
    """Read latency and relative-compute metadata keyed by H/D/W/E."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read hardware cost table {path}: {exc}") from exc
    rows = payload.get("configurations")
    if not isinstance(rows, list):
        raise ValueError(f"Hardware cost table {path} has no configurations list")
    costs: dict[tuple[int, int, int, int], dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Hardware cost table configuration entries must be objects")
        try:
            key = (
                int(row["H"]),
                int(row["D"]),
                int(row["W"]),
                int(row.get("E", 0)),
            )
            latency = float(
                row.get("latency_ms_mean", row.get("latency_ms", float("nan")))
            )
            relative = float(row.get("relative_compute", float("nan")))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed configuration in hardware cost table: {row}") from exc
        if np.isfinite(latency) and latency >= 0 and np.isfinite(relative) and relative >= 0:
            costs[key] = {
                "latency_ms_mean": latency,
                "relative_compute": relative,
            }
    return costs


def _cost_for_config(
    config: FixedConfig,
    table: Mapping[tuple[int, int, int, int], dict[str, float]],
) -> dict[str, float] | None:
    exact_key = (config.H, config.D, config.W, config.E)
    if exact_key in table:
        return table[exact_key]
    wildcard_key = (config.H, config.D, config.W, 0)
    return table.get(wildcard_key)


def _synchronize_devices(device_names: Sequence[str]) -> None:
    if not torch.cuda.is_available():
        return
    for device_name in device_names:
        torch.cuda.synchronize(torch.device(device_name))


def _copy_observation(observation: Any) -> Any:
    if isinstance(observation, dict):
        return {
            key: _copy_observation(value)
            for key, value in observation.items()
        }
    if isinstance(observation, np.ndarray):
        return observation.copy()
    if isinstance(observation, Tensor):
        return observation.detach().cpu().clone()
    return observation


def _make_virtual_observation(observation: Any) -> Any:
    """Copy a simulator observation and zero image inputs for model warm-up."""
    virtual_observation = _copy_observation(observation)
    if not isinstance(virtual_observation, dict):
        return virtual_observation
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        image = virtual_observation.get(key)
        if isinstance(image, np.ndarray):
            virtual_observation[key] = np.zeros_like(image)
        elif isinstance(image, Tensor):
            virtual_observation[key] = torch.zeros_like(image)
    return virtual_observation


def evaluate_episode(
    adapter: EpisodeAdapter,
    *,
    task_suite: str,
    task_id: int,
    seed: int,
    max_steps: int,
    configs: Sequence[FixedConfig] = FIXED_CONFIGS,
    state_manager: type[SimStateManager] = SimStateManager,
    warmup_iterations: int = 5,
    warmed_configs: set[tuple[int, int, int, int]] | None = None,
    save_trajectories: bool = False,
    trajectory_dir: Path = DEFAULT_TRAJECTORY_DIR,
    trajectory_stride: int = 1,
) -> list[dict[str, Any]]:
    """Evaluate every fixed method from the same saved LIBERO initial state."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not configs:
        raise ValueError("At least one fixed configuration is required")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if trajectory_stride <= 0:
        raise ValueError("trajectory_stride must be positive")

    initial_observation = adapter.start_task(task_id, seed)
    initial_state = state_manager.save_state(adapter.env)
    device_names = tuple(adapter.cuda_devices)
    records: list[dict[str, Any]] = []
    warmed_config_keys = warmed_configs if warmed_configs is not None else set()

    for config in configs:
        observation = state_manager.restore_state(adapter.env, initial_state)
        if observation is None:
            observation = _copy_observation(initial_observation)
        if hasattr(adapter, "observation"):
            adapter.observation = observation
        config_key = (config.H, config.D, config.W, config.E)
        if config_key not in warmed_config_keys:
            virtual_observation = _make_virtual_observation(observation)
            for _ in range(warmup_iterations):
                warmup_output = adapter.parameterized_wam(
                    virtual_observation,
                    config.H,
                    config.D,
                    config.W,
                    config.E,
                )
                del warmup_output
            _synchronize_devices(device_names)
            del virtual_observation
            warmed_config_keys.add(config_key)

        observation = state_manager.restore_state(adapter.env, initial_state)
        if observation is None:
            observation = _copy_observation(initial_observation)
        if hasattr(adapter, "observation"):
            adapter.observation = observation

        episode_return = 0.0
        steps = 0
        wam_calls = 0
        call_latencies_ms: list[float] = []
        success = bool(adapter.is_success(observation, {}))
        terminated = success
        trajectory_steps: list[dict[str, Any]] = []
        trajectory_blobs: dict[str, dict[str, Any]] = {}
        initial_object_positions = (
            collect_object_positions(adapter.env) if save_trajectories else {}
        )

        while steps < max_steps and not success and not terminated:
            _synchronize_devices(device_names)
            started = time.perf_counter()
            actions = adapter.parameterized_wam(
                observation,
                config.H,
                config.D,
                config.W,
                config.E,
            )
            _synchronize_devices(device_names)
            call_latencies_ms.append((time.perf_counter() - started) * 1000.0)
            wam_calls += 1

            action_chunk = (
                actions.detach().cpu().numpy()
                if isinstance(actions, Tensor)
                else np.asarray(actions)
            )
            if action_chunk.ndim != 2 or action_chunk.shape[0] < config.E:
                raise ValueError(
                    f"{config.name} returned {action_chunk.shape}; expected "
                    f"at least [{config.E}, action_dim]"
                )

            for action in action_chunk[: min(config.E, max_steps - steps)]:
                executed_action = np.asarray(action, dtype=np.float32).copy()
                transition = adapter.step(executed_action)
                observation = transition.observation
                episode_return += float(transition.reward)
                steps += 1
                success = bool(adapter.is_success(observation, transition.info))
                terminated = bool(transition.terminated or transition.truncated)
                if save_trajectories and (
                    steps % trajectory_stride == 0
                    or success
                    or terminated
                    or steps >= max_steps
                ):
                    metrics = measure_physics(
                        adapter.env,
                        initial_object_positions=initial_object_positions,
                        observation=observation,
                    )
                    step_payload = {
                        "step": int(steps),
                        "action": executed_action,
                        "reward": float(transition.reward),
                        "done": bool(terminated),
                        "success": bool(success),
                        "sim_state": state_manager.save_state(adapter.env),
                        "obs": _uint8_observation(observation),
                        "contact": _contact_value(
                            transition.info,
                            metrics.contact_force_n,
                        ),
                        "phase": classify_physical_phase(metrics),
                        "physical_metrics": metrics.as_dict(),
                    }
                    trajectory_steps.append(
                        _pack_trajectory_value(step_payload, trajectory_blobs)
                    )
                if success or terminated or steps >= max_steps:
                    break

        trajectory_path = ""
        if save_trajectories:
            if not trajectory_steps:
                raise RuntimeError(
                    f"No trajectory states captured for task={task_id}, seed={seed}, "
                    f"method={config.name}"
                )
            target = trajectory_path_for_episode(
                trajectory_dir,
                task_suite=task_suite,
                task_id=task_id,
                seed=seed,
                method=config.name,
            )
            trajectory_payload = {
                "schema_version": TRAJECTORY_SCHEMA_VERSION,
                "storage": "content_addressed_arrays_v1",
                "task_suite": task_suite,
                "task_id": task_id,
                "seed": seed,
                "method": config.name,
                "config": config.as_dict(),
                "success": bool(success),
                "episode_steps": int(steps),
                "episode_return": float(episode_return),
                "trajectory_stride": int(trajectory_stride),
                "steps": trajectory_steps,
                "array_blobs": trajectory_blobs,
            }
            _write_trajectory(target, trajectory_payload)
            trajectory_path = str(target)

        records.append(
            {
                "task_suite": task_suite,
                "task_id": task_id,
                "seed": seed,
                "method": config.name,
                "config": config.as_dict(),
                "success": int(success),
                "steps": steps,
                "episode_return": episode_return,
                "wam_calls": wam_calls,
                "local_latency_ms_mean": (
                    statistics.fmean(call_latencies_ms) if call_latencies_ms else None
                ),
                "local_latency_ms_total": float(sum(call_latencies_ms)),
                "trajectory_path": trajectory_path,
            }
        )
    return records


def aggregate_results(
    episode_records: Sequence[Mapping[str, Any]],
    hardware_costs: Mapping[tuple[int, int, int, int], dict[str, float]],
) -> list[dict[str, Any]]:
    """Aggregate episode metrics using total inference cost per episode."""
    summary_rows: list[dict[str, Any]] = []
    for config in FIXED_CONFIGS:
        rows = [row for row in episode_records if row["method"] == config.name]
        if not rows:
            continue
        hardware_cost = _cost_for_config(config, hardware_costs)
        episode_compute_ms: list[float] = []
        cost_sources: set[str] = set()
        for row in rows:
            if hardware_cost is not None:
                calls = int(row["wam_calls"])
                total_ms = calls * float(hardware_cost["latency_ms_mean"])
                cost_sources.add("hardware_cost_table")
            else:
                local_total = row.get("local_latency_ms_total")
                if local_total is not None:
                    total_ms = float(local_total)
                else:
                    per_call = row.get("local_latency_ms_mean")
                    total_ms = (
                        int(row["wam_calls"]) * float(per_call)
                        if per_call is not None
                        else 0.0
                    )
                cost_sources.add("measured_local_wall_time")
            episode_compute_ms.append(total_ms)

        summary_rows.append(
            {
                "method": config.name,
                "H": config.H,
                "D": config.D,
                "W": config.W,
                "E": config.E,
                "episodes": len(rows),
                "success_rate": statistics.fmean(float(row["success"]) for row in rows),
                "avg_steps": statistics.fmean(float(row["steps"]) for row in rows),
                "avg_return": statistics.fmean(
                    float(row["episode_return"]) for row in rows
                ),
                "avg_wam_calls": statistics.fmean(
                    float(row["wam_calls"]) for row in rows
                ),
                "avg_episode_compute_ms": statistics.fmean(episode_compute_ms),
                "episode_latency_s": statistics.fmean(episode_compute_ms) / 1000.0,
                "relative_compute_percent": None,
                "cost_source": next(iter(cost_sources))
                if len(cost_sources) == 1
                else "mixed",
            }
        )

    full_row = next(
        (row for row in summary_rows if row["method"] == "Fixed-Full"), None
    )
    full_compute = (
        float(full_row["avg_episode_compute_ms"])
        if full_row is not None
        else None
    )
    if full_compute is not None and full_compute > 0:
        for row in summary_rows:
            row["relative_compute_percent"] = (
                float(row["avg_episode_compute_ms"]) / full_compute * 100.0
            )
    return summary_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a non-empty, rectangular list of records to CSV."""
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def render_markdown_table(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the requested comparison metrics."""
    lines = [
        "| Method | Success Rate | Episode Latency (s) | Relative Compute (%) | Avg WAM Calls |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        relative_compute = row["relative_compute_percent"]
        relative_text = (
            f"{float(relative_compute):.1f}" if relative_compute is not None else "N/A"
        )
        lines.append(
            f"| {row['method']} | {float(row['success_rate']):.1%} | "
            f"{float(row['episode_latency_s']):.3f} | {relative_text} | "
            f"{float(row['avg_wam_calls']):.2f} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument(
        "--task-ids",
        "--task-id",
        dest="task_ids",
        default=DEFAULT_TASK_IDS,
        help="Comma-separated task IDs or inclusive ranges (e.g. 0-4); smoke default is task 0.",
    )
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Explicit comma-separated seed IDs or ranges; overrides --seed-start/--num-seeds.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(config.name for config in FIXED_CONFIGS),
        help="Comma-separated fixed methods to evaluate.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from --episodes-csv, skip completed suite/task/seed/method "
            "keys, and checkpoint atomically after every episode."
        ),
    )
    parser.add_argument(
        "--merge-episodes-csv",
        action="append",
        type=Path,
        default=[],
        help=(
            "Merge a prior episode CSV into --episodes-csv before resuming. "
            "May be repeated; conflicting duplicate keys are rejected."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--fastwam-root", type=Path, default=FASTWAM_ROOT)
    parser.add_argument("--libero-root", type=Path, default=LIBERO_ROOT)
    parser.add_argument("--task-config", default="libero_optional_idm_2cam224_1e-4")
    parser.add_argument("--config-name", default="sim_libero")
    parser.add_argument("--dataset-stats-path", type=Path, default=None)
    parser.add_argument("--direct-depth", type=int, default=10)
    parser.add_argument(
        "--hardware-cost-table", type=Path, default=DEFAULT_HARDWARE_COST_TABLE
    )
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--episodes-csv", type=Path, default=DEFAULT_EPISODES_CSV)
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Persist replayable simulator state and observation records during every rollout.",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=DEFAULT_TRAJECTORY_DIR,
        help="Root directory for per-method trajectory .pt files.",
    )
    parser.add_argument(
        "--trajectory-stride",
        type=int,
        default=1,
        help="Save one post-transition state every N physical environment steps.",
    )
    parser.add_argument(
        "--adapter", default="real_fastwam_adapter:create_adapter"
    )
    return parser


def _load_adapter_factory(specification: str) -> AdapterFactory:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("--adapter must use module.path:factory syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError(f"Adapter factory {specification!r} is not callable")
    return factory


def _parse_gpu_ids(value: str) -> list[int]:
    try:
        gpu_ids = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("--gpu-ids must be comma-separated CUDA indices") from exc
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4 or min(gpu_ids) < 0:
        raise ValueError("--gpu-ids must contain exactly four distinct non-negative IDs")
    return gpu_ids


def main() -> None:
    args = build_parser().parse_args()
    task_ids = parse_task_ids(args.task_ids)
    seeds = (
        parse_seeds(args.seeds)
        if args.seeds is not None
        else list(range(args.seed_start, args.seed_start + args.num_seeds))
    )
    selected_configs = parse_methods(args.methods)
    if args.num_seeds <= 0 or args.seed_start < 0 or args.max_steps <= 0:
        raise ValueError("--num-seeds/max-steps must be positive and --seed-start non-negative")
    if min(seeds) < 0:
        raise ValueError("Seeds must be non-negative")
    if args.trajectory_stride <= 0:
        raise ValueError("--trajectory-stride must be positive")
    if args.direct_depth <= 0:
        raise ValueError("--direct-depth must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Real FastWAM baseline evaluation requires CUDA")

    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"FastWAM checkpoint does not exist: {model_path}")
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if any(gpu_id >= torch.cuda.device_count() for gpu_id in gpu_ids):
        raise ValueError(f"GPU IDs {gpu_ids} exceed visible device count")
    devices = [f"cuda:{gpu_id}" for gpu_id in gpu_ids]
    random.seed(args.seed_start)
    np.random.seed(args.seed_start)
    torch.manual_seed(args.seed_start)
    torch.cuda.manual_seed_all(args.seed_start)

    options: dict[str, Any] = {
        "model_path": str(model_path),
        "fastwam_root": str(args.fastwam_root.expanduser().resolve()),
        "libero_root": str(args.libero_root.expanduser().resolve()),
        "task_config": args.task_config,
        "config_name": args.config_name,
        "task_suite": args.task_suite,
        "direct_depth": args.direct_depth,
        "max_steps": args.max_steps,
        "cuda_devices": devices,
    }
    if args.dataset_stats_path is not None:
        options["dataset_stats_path"] = str(
            args.dataset_stats_path.expanduser().resolve()
        )
    factory = _load_adapter_factory(args.adapter)
    adapter = factory(
        WorkerContext(
            device=devices[0],
            task_suite=args.task_suite,
            options=options,
        )
    )
    episodes_path = args.episodes_csv.expanduser().resolve()
    episode_records: list[dict[str, Any]] = (
        read_episode_csv(episodes_path) if args.resume else []
    )
    for source_path in args.merge_episodes_csv:
        source_path = source_path.expanduser().resolve()
        if source_path == episodes_path:
            continue
        episode_records = merge_episode_records(
            episode_records,
            read_episode_csv(source_path),
        )
    if args.save_trajectories:
        stale_keys = {
            _episode_key(row)
            for row in episode_records
            if not _trajectory_is_valid(row.get("trajectory_path"))
        }
        if stale_keys:
            print(
                f"Re-running {len(stale_keys)} episode records with missing/invalid trajectories",
                flush=True,
            )
            episode_records = [
                row for row in episode_records if _episode_key(row) not in stale_keys
            ]
    for row in episode_records:
        row.setdefault("trajectory_path", "")
    if episode_records:
        write_csv(episodes_path, episode_records)
    completed_keys = {_episode_key(row) for row in episode_records}
    warmed_configs: set[tuple[int, int, int, int]] = set()
    try:
        for task_id in task_ids:
            for seed in seeds:
                missing_configs = tuple(
                    config
                    for config in selected_configs
                    if (args.task_suite, task_id, seed, config.name)
                    not in completed_keys
                )
                if not missing_configs:
                    print(
                        f"Skipping completed {args.task_suite} task={task_id} seed={seed}",
                        flush=True,
                    )
                    continue
                print(
                    f"Evaluating {args.task_suite} task={task_id} seed={seed}: "
                    + ", ".join(config.name for config in missing_configs),
                    flush=True,
                )
                for config in missing_configs:
                    new_records = evaluate_episode(
                        adapter,
                        task_suite=args.task_suite,
                        task_id=task_id,
                        seed=seed,
                        max_steps=args.max_steps,
                        configs=(config,),
                        warmed_configs=warmed_configs,
                        save_trajectories=args.save_trajectories,
                        trajectory_dir=args.trajectory_dir,
                        trajectory_stride=args.trajectory_stride,
                    )
                    episode_records = merge_episode_records(episode_records, new_records)
                    completed_keys.update(_episode_key(row) for row in new_records)
                    write_csv(episodes_path, episode_records)
    finally:
        adapter.close()

    hardware_costs = load_hardware_costs(
        args.hardware_cost_table.expanduser().resolve()
    )
    summary_rows = aggregate_results(episode_records, hardware_costs)
    write_csv(episodes_path, episode_records)
    write_csv(args.summary_csv.expanduser().resolve(), summary_rows)
    print()
    print(render_markdown_table(summary_rows))
    print(f"\nEpisode records: {args.episodes_csv.expanduser().resolve()}")
    print(f"Summary CSV: {args.summary_csv.expanduser().resolve()}")


if __name__ == "__main__":
    main()
