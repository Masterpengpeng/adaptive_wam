"""Evaluate inference configurations from saved LIBERO simulation states."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collect_counterfactual_data import InferenceConfig, WorkerContext
from core.sim_state_manager import SimStateManager
from oracle.physics import measure_physics


DEFAULT_COST_TABLE = PROJECT_ROOT / "configs" / "hardware_cost_table.json"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "oracle_counterfactual_results.csv"
CONTINUATION_CONFIG = InferenceConfig(H=0, D=0, W=0, E=10)


@dataclass(frozen=True)
class OracleConfig:
    name: str
    H: int
    D: int
    W: int
    E: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self) | {"name": self.name}

    def policy_config(self) -> InferenceConfig:
        return InferenceConfig(
            H=self.H,
            D=self.D,
            W=0 if self.H == 0 else self.W,
            E=self.E,
        )


CONFIG_GRID: tuple[OracleConfig, ...] = (
    OracleConfig("B_action", 0, 0, 1, 10),
    OracleConfig("B_small", 2, 2, 1, 4),
    OracleConfig("B_deep", 2, 16, 1, 2),
    OracleConfig("B_wide", 4, 4, 4, 4),
    OracleConfig("B_far", 8, 4, 1, 4),
    OracleConfig("B_full", 8, 16, 4, 2),
)


def load_config_grid(path: Path) -> tuple[OracleConfig, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read configuration grid {path}: {exc}") from exc
    rows = payload.get("configurations") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Configuration grid {path} has no configurations list")
    configs: list[OracleConfig] = []
    for row in rows:
        try:
            config = OracleConfig(
                str(row["name"]),
                int(row["H"]),
                int(row["D"]),
                int(row["W"]),
                int(row["E"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed configuration-grid row: {row}") from exc
        if not config.name or config.E <= 0 or min(config.H, config.D, config.W) < 0:
            raise ValueError(f"Invalid configuration-grid row: {row}")
        configs.append(config)
    names = [config.name for config in configs]
    keys = [(config.H, config.D, config.W, config.E) for config in configs]
    if len(names) != len(set(names)):
        raise ValueError("Configuration grid contains duplicate names")
    if len(keys) != len(set(keys)):
        raise ValueError("Configuration grid contains duplicate H/D/W/E settings")
    return tuple(configs)


def load_snapshots(
    path: Path,
    *,
    required_sampling_protocol: str | None = None,
) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.rglob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl snapshots found in {path}")
    snapshots: list[dict[str, Any]] = []
    for file_path in files:
        with file_path.open("rb") as source:
            payload = pickle.load(source)
        if not isinstance(payload, dict) or "state" not in payload:
            raise ValueError(f"Invalid oracle snapshot: {file_path}")
        if (
            required_sampling_protocol is not None
            and payload.get("sampling_protocol") != required_sampling_protocol
        ):
            continue
        payload["snapshot_file"] = str(file_path)
        snapshots.append(payload)
    if not snapshots:
        if required_sampling_protocol is not None:
            raise FileNotFoundError(
                f"No snapshots with sampling_protocol={required_sampling_protocol!r} in {path}"
            )
        raise FileNotFoundError(f"No usable .pkl snapshots found in {path}")
    return snapshots


def load_relative_costs(path: Path) -> dict[tuple[int, int, int], dict[str, float | int]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read hardware cost table {path}: {exc}") from exc
    rows = payload.get("configurations")
    if not isinstance(rows, list):
        raise ValueError(f"Hardware cost table {path} has no configurations list")
    costs: dict[tuple[int, int, int], dict[str, float | int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            key = (int(row["H"]), int(row["D"]), int(row["W"]))
            relative = float(row["relative_compute"])
            table_e = int(row.get("E", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(relative) and relative >= 0:
            if key[0] == 0 and key[1] == 0:
                key = (0, 0, 0)
            costs[key] = {"relative_compute": relative, "table_E": table_e}
    return costs


def cost_for_config(
    config: OracleConfig,
    costs: Mapping[tuple[int, int, int], Mapping[str, float | int]],
) -> Mapping[str, float | int]:
    lookup_w = 0 if config.H == 0 else config.W
    key = (config.H, config.D, lookup_w)
    if key not in costs:
        raise KeyError(f"No relative_compute cost for {config.name} H/D/W={key}")
    return costs[key]


def _normalize_actions(actions: Any) -> np.ndarray:
    if hasattr(actions, "detach"):
        actions = actions.detach().cpu().numpy()
    result = np.asarray(actions)
    if result.ndim == 1:
        result = result[None, :]
    if result.ndim != 2 or result.shape[0] == 0:
        raise ValueError(f"Policy returned invalid action chunk with shape {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Policy returned NaN or infinite actions")
    return result


def _run_branch(
    adapter: Any,
    observation: Any,
    first_config: OracleConfig,
    *,
    max_steps: int,
    eval_mode: str = "sustained",
    distance_reward_scale: float = 1.0,
) -> dict[str, Any]:
    if eval_mode not in {"sustained", "one_shot"}:
        raise ValueError("eval_mode must be 'sustained' or 'one_shot'")
    if max_steps <= 0 or adapter.is_success(observation, {}):
        return {
            "success": int(bool(adapter.is_success(observation, {}))),
            "remaining_steps": 0,
            "reward": 0.0,
            "dense_reward": 0.0,
            "distance_reward": 0.0,
            "inference_calls": 0,
            "actual_elapsed_seconds": 0.0,
        }
    cuda_devices = tuple(getattr(adapter, "cuda_devices", ()))
    if torch.cuda.is_available():
        for device in cuda_devices:
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
    started = time.perf_counter()
    previous_metrics = measure_physics(adapter.env, observation=observation)
    total_reward = 0.0
    dense_reward = 0.0
    distance_reward = 0.0
    steps = 0
    inference_calls = 0
    success = bool(adapter.is_success(observation, {}))
    terminated = success
    action_chunk: np.ndarray | None = None
    action_index = 0
    active_config = first_config.policy_config()
    while steps < max_steps and not terminated:
        if action_chunk is None or action_index >= len(action_chunk):
            if eval_mode == "one_shot" and inference_calls >= 1:
                active_config = CONTINUATION_CONFIG
            action_chunk = _normalize_actions(adapter.infer_actions(observation, active_config))
            if action_chunk.shape[0] < active_config.E:
                raise ValueError(
                    f"{first_config.name} returned {action_chunk.shape[0]} actions, "
                    f"fewer than E={active_config.E}"
                )
            action_chunk = action_chunk[: active_config.E]
            inference_calls += 1
            action_index = 0
        result = adapter.step(action_chunk[action_index])
        action_index += 1
        observation = result.observation
        reward = float(result.reward)
        total_reward += reward
        current_metrics = measure_physics(adapter.env, observation=observation)
        if (
            previous_metrics.goal_distance_m is not None
            and current_metrics.goal_distance_m is not None
        ):
            progress = previous_metrics.goal_distance_m - current_metrics.goal_distance_m
            distance_reward += float(progress)
            dense_reward += reward + distance_reward_scale * float(progress)
        else:
            dense_reward += reward
        previous_metrics = current_metrics
        steps += 1
        success = bool(adapter.is_success(observation, result.info))
        terminated = success or bool(result.terminated) or bool(result.truncated)
    if torch.cuda.is_available():
        for device in cuda_devices:
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
    elapsed_seconds = time.perf_counter() - started
    return {
        "success": int(success),
        "remaining_steps": steps,
        "reward": total_reward,
        "dense_reward": dense_reward,
        "distance_reward": distance_reward,
        "inference_calls": inference_calls,
        "actual_elapsed_seconds": elapsed_seconds,
    }


def evaluate_snapshot(
    snapshot: Mapping[str, Any],
    adapter: Any,
    costs: Mapping[tuple[int, int, int], Mapping[str, float | int]],
    *,
    max_episode_steps: int = 400,
    repeat_index: int = 0,
    eval_mode: str = "sustained",
    state_manager: Any = SimStateManager,
    configs: Sequence[OracleConfig] = CONFIG_GRID,
) -> list[dict[str, Any]]:
    task_id = int(snapshot["task_id"])
    seed = int(snapshot["seed"])
    observation = adapter.start_task(task_id, seed)
    observation = state_manager.restore_state(adapter.env, snapshot["state"])
    initial_step = int(snapshot.get("step", 0))
    if hasattr(adapter, "observation"):
        adapter.observation = observation
    max_remaining = max(0, max_episode_steps - initial_step)
    records: list[dict[str, Any]] = []

    for config in configs:
        branch_started = time.perf_counter()
        observation = state_manager.restore_state(adapter.env, snapshot["state"])
        if observation is None:
            observation = getattr(adapter, "observation", None)
        if hasattr(adapter, "observation"):
            adapter.observation = observation
        branch_seed = (seed + repeat_index * 1_000_003) % (2**32)
        random.seed(branch_seed)
        np.random.seed(branch_seed)
        torch.manual_seed(branch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(branch_seed)
        result = _run_branch(
            adapter,
            observation,
            config,
            max_steps=max_remaining,
            eval_mode=eval_mode,
        )
        result["actual_elapsed_seconds"] = time.perf_counter() - branch_started
        measured_cost = cost_for_config(config, costs)
        continuation_key = (0, 0, 0)
        continuation_cost = costs.get(continuation_key)
        if continuation_cost is None:
            raise KeyError("No relative_compute cost for No-Imagination continuation policy")
        if eval_mode == "sustained":
            total_relative_compute = (
                float(measured_cost["relative_compute"])
                / config.E
                * int(result["remaining_steps"])
            )
        else:
            candidate_steps = min(config.E, int(result["remaining_steps"]))
            continuation_steps = max(0, int(result["remaining_steps"]) - candidate_steps)
            continuation_horizon = int(continuation_cost["table_E"])
            if continuation_horizon <= 0:
                continuation_horizon = CONTINUATION_CONFIG.E
            total_relative_compute = (
                float(measured_cost["relative_compute"]) / config.E * candidate_steps
                + float(continuation_cost["relative_compute"])
                / continuation_horizon
                * continuation_steps
            )
        records.append(
            {
                "snapshot_file": snapshot.get("snapshot_file", ""),
                "task_suite": snapshot["task_suite"],
                "task_id": task_id,
                "seed": seed,
                "repeat": repeat_index,
                "step": initial_step,
                "phase": snapshot.get("phase", ""),
                "configuration": config.name,
                "H": config.H,
                "D": config.D,
                "W": config.W,
                "E": config.E,
                "success": result["success"],
                "remaining_steps": result["remaining_steps"],
                "reward": result["reward"],
                "dense_reward": result["dense_reward"],
                "distance_reward": result["distance_reward"],
                "actual_elapsed_seconds": result["actual_elapsed_seconds"],
                "eval_mode": eval_mode,
                "inference_calls": result["inference_calls"],
                "relative_compute": measured_cost["relative_compute"],
                "cost_table_E": measured_cost["table_E"],
                "continuation_relative_compute": continuation_cost["relative_compute"],
                "estimated_total_relative_compute": total_relative_compute,
            }
        )
    return records


def write_results(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError("No counterfactual results to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(path)


def _result_key(row: Mapping[str, Any]) -> tuple[str, int, str, str]:
    try:
        return (
            str(row["snapshot_file"]),
            int(row["repeat"]),
            str(row["configuration"]),
            str(row["eval_mode"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed oracle result row: {row}") from exc


def read_results(path: Path) -> list[dict[str, Any]]:
    """Read a resumable result checkpoint and reject duplicate branch keys."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        rows = [dict(row) for row in csv.DictReader(source)]
    seen: set[tuple[str, int, str, str]] = set()
    for row in rows:
        key = _result_key(row)
        if key in seen:
            raise ValueError(f"Duplicate oracle branch key in {path}: {key}")
        seen.add(key)
    return rows


def _load_factory(specification: str) -> Callable[[WorkerContext], Any]:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--adapter must have the form python.module:factory")
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise TypeError(f"Adapter factory {specification!r} is not callable")
    return factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_path", nargs="?", type=Path)
    parser.add_argument("--snapshots-dir", type=Path)
    parser.add_argument("--adapter", default="real_fastwam_adapter:create_adapter")
    parser.add_argument("--cost-table", type=Path, default=DEFAULT_COST_TABLE)
    parser.add_argument(
        "--config-grid",
        type=Path,
        default=None,
        help="Optional JSON configuration grid; defaults to the six Oracle branches.",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "checkpoints" / "fastwam_release" / "libero_optional_idm_2cam224.pt")
    parser.add_argument("--task-config", default="libero_optional_idm_2cam224_1e-4")
    parser.add_argument("--config-name", default="sim_libero")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--eval-mode", choices=("sustained", "one_shot"), default="sustained")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from --output-csv and atomically checkpoint after every "
            "snapshot/repeat group."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_episode_steps <= 0:
        raise ValueError("--max-episode-steps must be positive")
    if args.num_repeats <= 0:
        raise ValueError("--num-repeats must be positive")
    if args.snapshot_path is not None and args.snapshots_dir is not None:
        raise ValueError("Specify either snapshot_path or --snapshots-dir, not both")
    snapshot_path = args.snapshots_dir or args.snapshot_path
    if snapshot_path is None:
        raise ValueError("Specify a snapshot file or --snapshots-dir")
    snapshots = load_snapshots(
        snapshot_path,
        required_sampling_protocol="physical_events_v2",
    )
    costs = load_relative_costs(args.cost_table)
    configs = (
        load_config_grid(args.config_grid.expanduser().resolve())
        if args.config_grid is not None
        else CONFIG_GRID
    )
    for config in configs:
        cost_for_config(config, costs)
    options = {
        "model_path": str(args.model_path.expanduser().resolve()),
        "task_config": args.task_config,
        "config_name": args.config_name,
    }
    output_csv = args.output_csv.expanduser().resolve()
    all_records: list[dict[str, Any]] = read_results(output_csv) if args.resume else []
    completed_keys = {_result_key(row) for row in all_records}
    factory = _load_factory(args.adapter)
    try:
        device_ids = [part.strip() for part in args.gpu_ids.split(",") if part.strip()]
        devices = [item if item == "cpu" else f"cuda:{int(item)}" for item in device_ids]
    except ValueError as exc:
        raise ValueError("--gpu-ids must be comma-separated CUDA indices") from exc
    for task_suite in sorted({str(item["task_suite"]) for item in snapshots}):
        suite_snapshots = [item for item in snapshots if item["task_suite"] == task_suite]
        suite_has_pending_work = any(
            (
                str(snapshot["snapshot_file"]),
                repeat_index,
                config.name,
                args.eval_mode,
            )
            not in completed_keys
            for snapshot in suite_snapshots
            for repeat_index in range(args.num_repeats)
            for config in configs
        )
        if not suite_has_pending_work:
            print(f"Skipping completed task suite {task_suite}", flush=True)
            continue
        context = WorkerContext(
            device=devices[0] if devices else "cpu",
            task_suite=task_suite,
            options={**options, "cuda_devices": devices},
        )
        adapter = factory(context)
        try:
            for snapshot in suite_snapshots:
                for repeat_index in range(args.num_repeats):
                    missing_configs = tuple(
                        config
                        for config in configs
                        if (
                            str(snapshot["snapshot_file"]),
                            repeat_index,
                            config.name,
                            args.eval_mode,
                        )
                        not in completed_keys
                    )
                    if not missing_configs:
                        continue
                    new_records = evaluate_snapshot(
                        snapshot,
                        adapter,
                        costs,
                        max_episode_steps=args.max_episode_steps,
                        repeat_index=repeat_index,
                        eval_mode=args.eval_mode,
                        configs=missing_configs,
                    )
                    all_records.extend(new_records)
                    completed_keys.update(_result_key(row) for row in new_records)
                    write_results(output_csv, all_records)
        finally:
            adapter.close()
    write_results(output_csv, all_records)
    print(f"Wrote {len(all_records)} results to {output_csv}")


if __name__ == "__main__":
    main()
