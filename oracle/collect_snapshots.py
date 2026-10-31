"""Replay Phase 1 episodes and persist task-stage simulation snapshots."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collect_counterfactual_data import InferenceConfig, WorkerContext
from core.sim_state_manager import SimStateManager
from eval_fixed_baselines import load_trajectory, trajectory_path_for_episode
from oracle.physics import (
    PhysicalMetrics,
    classify_physical_phase,
    collect_object_positions,
    measure_physics,
)


DEFAULT_EPISODES_CSV = PROJECT_ROOT / "results" / "fixed_baselines_episodes.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "snapshots"
NO_IMAGINATION_CONFIG = InferenceConfig(H=0, D=0, W=0, E=10)
FIXED_FULL_CONFIG = InferenceConfig(H=8, D=16, W=4, E=2)
FIXED_SMALL_CONFIG = InferenceConfig(H=2, D=2, W=1, E=4)
SNAPSHOT_PROTOCOL = "physical_events_v2"
PHYSICAL_PHASES = ("reaching", "precontact", "manipulation", "placement")
WATERSHED_OFFSETS = (1, 2, 3, 4, 5)
TRAJECTORY_SUFFIXES = {".hdf5", ".h5", ".pkl", ".pt"}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.") or "task"


def _trajectory_file_counts(search_root: Path) -> dict[Path, int]:
    counts: dict[Path, int] = defaultdict(int)
    if not search_root.is_dir():
        return counts
    for item in search_root.rglob("*"):
        if item.is_file() and item.suffix.lower() in TRAJECTORY_SUFFIXES:
            counts[item.parent] += 1
    return counts


def resolve_trajectories_dir(
    requested: Path | None,
    configured_dataset_dir: Path,
    *,
    search_root: Path = PROJECT_ROOT / "outputs",
) -> tuple[Path, int, str]:
    """Resolve an optional recorded-data root or the LIBERO dataset placeholder."""
    if requested is not None:
        requested = requested.expanduser().resolve()
        if requested.is_dir():
            files = sum(_trajectory_file_counts(requested).values())
            return requested, files, "explicit"

    candidates = _trajectory_file_counts(search_root)
    if candidates:
        selected_dir, file_count = max(
            candidates.items(), key=lambda item: (item[1], len(item[0].parts))
        )
        return selected_dir.resolve(), file_count, "auto-discovered"

    configured = configured_dataset_dir.expanduser().resolve()
    if configured.is_dir():
        return configured, 0, "configured-empty-or-unindexed"
    if configured.name == "no_demonstrations":
        configured.mkdir(parents=True, exist_ok=True)
        return configured, 0, "created-no-demonstrations-placeholder"
    if requested is not None:
        raise FileNotFoundError(
            f"--trajectories-dir {requested} does not exist, no .hdf5/.h5/.pkl "
            f"files were found under {search_root}, and configured LIBERO dataset "
            f"path {configured} is unavailable"
        )
    raise FileNotFoundError(f"Configured LIBERO datasets path does not exist: {configured}")


def prepare_libero_config(
    source_config_dir: Path,
    runtime_config_dir: Path,
    trajectories_dir: Path | None,
    *,
    search_root: Path = PROJECT_ROOT / "outputs",
) -> tuple[Path, Path, int, str]:
    source_file = source_config_dir.expanduser().resolve() / "config.yaml"
    if not source_file.is_file():
        raise FileNotFoundError(f"LIBERO config file does not exist: {source_file}")
    payload = yaml.safe_load(source_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("datasets"):
        raise ValueError(f"LIBERO config has no datasets path: {source_file}")
    dataset_dir, file_count, resolution = resolve_trajectories_dir(
        trajectories_dir,
        Path(str(payload["datasets"])),
        search_root=search_root,
    )
    payload["datasets"] = str(dataset_dir)
    runtime_config_dir.mkdir(parents=True, exist_ok=True)
    (runtime_config_dir / "config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    print(
        f"LIBERO datasets path: {dataset_dir} ({resolution}; "
        f"trajectory files discovered={file_count})"
    )
    if file_count == 0:
        print(
            "Episode source: CSV task_suite/task_id/seed metadata; "
            "rollouts are regenerated from LIBERO initial states"
        )
    return runtime_config_dir, dataset_dir, file_count, resolution


def read_episode_records(path: Path) -> list[dict[str, Any]]:
    """Read and validate all methods from the fixed-baseline episode CSV."""
    with path.open(newline="", encoding="utf-8") as source:
        records = [dict(row) for row in csv.DictReader(source)]
    if not records:
        raise ValueError(f"No episode records found in {path}")
    for row in records:
        for key in ("task_id", "seed", "steps", "success"):
            try:
                row[key] = int(row[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {key!r} in episode record: {row}") from exc
        row["task_suite"] = str(row.get("task_suite", "")).strip()
        if not row["task_suite"]:
            raise ValueError(f"Episode record has no task_suite: {row}")
        row["method"] = str(row.get("method", "")).strip()
        if not row["method"]:
            raise ValueError(f"Episode record has no method: {row}")
    return records


def select_episode_records(
    records: Sequence[Mapping[str, Any]],
    successful_episode_limit: int | None = None,
    phase_success_method: str = "Fixed-Full",
) -> list[tuple[Mapping[str, Any], str]]:
    """Select paired watershed failures and successful Fixed-Full episodes."""
    if successful_episode_limit is not None and successful_episode_limit < 0:
        raise ValueError("successful_episode_limit must be non-negative")
    by_episode: dict[tuple[str, int, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in records:
        key = (str(row["task_suite"]), int(row["task_id"]), int(row["seed"]))
        by_episode[key][str(row["method"])] = row
    selected: list[tuple[Mapping[str, Any], str]] = []
    for episode in by_episode.values():
        no_imagination = episode.get("No-Imagination")
        fixed_full = episode.get("Fixed-Full")
        if (
            no_imagination is not None
            and fixed_full is not None
            and int(no_imagination["success"]) == 0
            and int(fixed_full["success"]) == 1
        ):
            selected.append((no_imagination, "watershed"))
    successful_records = [
        episode[phase_success_method]
        for episode in by_episode.values()
        if phase_success_method in episode
        and int(episode[phase_success_method]["success"]) == 1
    ]
    if successful_episode_limit is not None:
        successful_records = successful_records[:successful_episode_limit]
    success_kind = "successful_small" if phase_success_method == "Fixed-Small" else "successful_full"
    selected.extend((row, success_kind) for row in successful_records)
    if not selected:
        raise ValueError("No paired watershed failures or Fixed-Full success episodes found")
    return selected


def _environment_sim(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        sim = getattr(current, "sim", None)
        if sim is not None:
            return sim
        current = getattr(current, "env", None)
    raise TypeError("Adapter environment does not expose a MuJoCo sim")


def _camera_images(observation: Any) -> dict[str, np.ndarray]:
    if not isinstance(observation, Mapping):
        raise TypeError("LIBERO observations must be mappings containing both camera images")
    keys = {
        "agentview": "agentview_image",
        "wrist": "robot0_eye_in_hand_image",
    }
    images: dict[str, np.ndarray] = {}
    for camera, key in keys.items():
        value = observation.get(key)
        if value is None:
            raise ValueError(f"Observation is missing required camera image {key!r}")
        image = np.asarray(value)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image for {key}, got {image.shape}")
        images[camera] = image.copy()
    return images


def _physical_metadata(
    env: Any,
    state: Mapping[str, Any],
    metrics: PhysicalMetrics | None = None,
) -> dict[str, Any]:
    sim = _environment_sim(env)
    data = sim.data
    return {
        "time": float(getattr(data, "time", 0.0)),
        "qpos": np.asarray(data.qpos).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "ctrl": np.asarray(data.ctrl).copy() if hasattr(data, "ctrl") else None,
        "object_poses": state.get("object_poses", {}),
        "events": metrics.as_dict() if metrics is not None else {},
    }


def _snapshot_payload(
    *,
    task_suite: str,
    task_id: int,
    seed: int,
    step: int,
    phase: str,
    env: Any,
    observation: Any,
    metrics: PhysicalMetrics | None = None,
    state_manager: Any = SimStateManager,
) -> dict[str, Any]:
    task_name = f"{_safe_name(task_suite)}_task{task_id}"
    state = state_manager.save_state(env)
    return {
        "schema_version": 2,
        "sampling_protocol": SNAPSHOT_PROTOCOL,
        "task_suite": task_suite,
        "task_id": task_id,
        "task_name": task_name,
        "seed": seed,
        "step": step,
        "phase": phase,
        "state": state,
        "observation_images": _camera_images(observation),
        "physical_metadata": _physical_metadata(env, state, metrics),
    }


def _write_snapshot(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    filename = (
        f"{payload['task_name']}_seed{payload['seed']}_step{payload['step']}_"
        f"{payload['phase']}.pkl"
    )
    target = output_dir / filename
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot {target}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as destination:
        pickle.dump(dict(payload), destination, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def _trajectory_snapshot_payload(
    trajectory: Mapping[str, Any],
    step: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """Convert one persisted post-transition trajectory state to Oracle format."""
    observation = step.get("obs")
    if not isinstance(observation, Mapping):
        raise ValueError("Trajectory step has no observation mapping")
    state = step.get("sim_state")
    if not isinstance(state, Mapping):
        raise ValueError("Trajectory step has no complete sim_state mapping")
    metrics = step.get("physical_metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    task_suite = str(trajectory["task_suite"])
    task_id = int(trajectory["task_id"])
    return {
        "schema_version": 2,
        "sampling_protocol": SNAPSHOT_PROTOCOL,
        "task_suite": task_suite,
        "task_id": task_id,
        "task_name": f"{_safe_name(task_suite)}_task{task_id}",
        "seed": int(trajectory["seed"]),
        "step": int(step["step"]),
        "phase": phase,
        "state": dict(state),
        "observation_images": _camera_images(observation),
        "physical_metadata": {
            "object_poses": state.get("object_poses", {}),
            "events": dict(metrics),
            "source_method": str(trajectory.get("method", "")),
        },
    }


def _spaced_phase_steps(
    steps: Sequence[Mapping[str, Any]],
    phase: str,
    *,
    maximum: int = 2,
    minimum_gap: int = 5,
) -> list[Mapping[str, Any]]:
    candidates = [step for step in steps if str(step.get("phase")) == phase]
    if not candidates or maximum <= 0:
        return []
    selected = [candidates[0]]
    if maximum > 1:
        for candidate in reversed(candidates[1:]):
            if int(candidate["step"]) - int(selected[0]["step"]) >= minimum_gap:
                selected.append(candidate)
                break
    return selected[:maximum]


def _metric(step: Mapping[str, Any], name: str) -> float | None:
    metrics = step.get("physical_metrics")
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(name)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _failure_onset_index(
    low_steps: Sequence[Mapping[str, Any]],
    full_steps: Sequence[Mapping[str, Any]],
) -> int:
    """Locate the first sustained physical divergence between paired policies."""
    if not low_steps:
        raise ValueError("No-Imagination failure trajectory has no recorded steps")
    phase_rank = {"reaching": 0, "precontact": 1, "manipulation": 2, "placement": 3}
    comparable = min(len(low_steps), len(full_steps))
    divergence_run = 0
    for index in range(comparable):
        low = low_steps[index]
        full = full_steps[index]
        low_rank = phase_rank.get(str(low.get("phase")), 0)
        full_rank = phase_rank.get(str(full.get("phase")), 0)
        low_goal = _metric(low, "goal_distance_m")
        full_goal = _metric(full, "goal_distance_m")
        low_displacement = _metric(low, "target_object_displacement_m")
        full_displacement = _metric(full, "target_object_displacement_m")
        diverged = (
            (full_rank >= 2 and low_rank <= 1)
            or (
                low_goal is not None
                and full_goal is not None
                and low_goal - full_goal >= 0.04
            )
            or (
                full_rank >= 1
                and low_displacement is not None
                and full_displacement is not None
                and full_displacement - low_displacement >= 0.04
            )
        )
        divergence_run = divergence_run + 1 if diverged else 0
        if divergence_run >= 3:
            return max(0, index - divergence_run + 1)

    best_goal: float | None = None
    for index, step in enumerate(low_steps):
        goal = _metric(step, "goal_distance_m")
        if goal is None:
            continue
        best_goal = goal if best_goal is None else min(best_goal, goal)
        if index >= 2 and goal - best_goal >= 0.04:
            return max(0, index - 2)
    return max(0, len(low_steps) - len(WATERSHED_OFFSETS))


def _read_watershed_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No watershed pairs found in {path}")
    required = {"task_suite", "task_id", "seed"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Watershed CSV must contain {sorted(required)}")
    return rows


def _resolve_pair_path(
    pair: Mapping[str, Any],
    episode: Mapping[str, Any],
    *,
    method: str,
    trajectories_dir: Path | None,
) -> Path:
    column = (
        "no_imagination_trajectory" if method == "No-Imagination"
        else "fixed_full_trajectory"
    )
    candidates = [pair.get(column), episode.get("trajectory_path")]
    for value in candidates:
        if value:
            path = Path(str(value)).expanduser()
            if path.is_file():
                return path.resolve()
    if trajectories_dir is not None:
        path = trajectory_path_for_episode(
            trajectories_dir,
            task_suite=str(pair["task_suite"]),
            task_id=int(pair["task_id"]),
            seed=int(pair["seed"]),
            method=method,
        )
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No readable {method} trajectory for task={pair['task_id']} seed={pair['seed']}"
    )


def collect_targeted_from_trajectories(
    episodes_csv: Path,
    watershed_pairs: Path,
    output_dir: Path,
    *,
    trajectories_dir: Path | None = None,
    snapshot_limit: int = 60,
    minimum_snapshots: int = 40,
    max_per_episode_per_phase: int = 2,
    phase_sample_gap: int = 5,
) -> list[Path]:
    """Extract balanced physical/failure snapshots without replaying the model."""
    if snapshot_limit <= 0 or minimum_snapshots < 0 or minimum_snapshots > snapshot_limit:
        raise ValueError("Targeted snapshot limits must satisfy 0 <= minimum <= limit")
    if max_per_episode_per_phase not in (1, 2) or phase_sample_gap <= 0:
        raise ValueError("Invalid per-phase sampling settings")
    records = read_episode_records(episodes_csv)
    by_method = {
        (str(row["task_suite"]), int(row["task_id"]), int(row["seed"]), str(row["method"])): row
        for row in records
    }
    pairs = _read_watershed_pairs(watershed_pairs)
    candidates: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "precontact": [],
        "manipulation": [],
        "failure_recovery": [],
    }
    for pair in pairs:
        suite = str(pair["task_suite"])
        task_id = int(pair["task_id"])
        seed = int(pair["seed"])
        episode_key = f"{suite}:{task_id}:{seed}"
        low_row = by_method.get((suite, task_id, seed, "No-Imagination"))
        full_row = by_method.get((suite, task_id, seed, "Fixed-Full"))
        if low_row is None or full_row is None:
            raise ValueError(f"Pair {episode_key} is missing a baseline method row")
        if int(low_row["success"]) != 0 or int(full_row["success"]) != 1:
            raise ValueError(f"Pair {episode_key} is not low-fail / full-success")
        low_path = _resolve_pair_path(
            pair, low_row, method="No-Imagination", trajectories_dir=trajectories_dir
        )
        full_path = _resolve_pair_path(
            pair, full_row, method="Fixed-Full", trajectories_dir=trajectories_dir
        )
        low = load_trajectory(low_path)
        full = load_trajectory(full_path)
        if bool(low.get("success")) or not bool(full.get("success")):
            raise ValueError(f"Trajectory outcomes disagree with CSV for {episode_key}")
        low_steps = list(low["steps"])
        full_steps = list(full["steps"])
        for phase in ("precontact", "manipulation"):
            for step in _spaced_phase_steps(
                full_steps,
                phase,
                maximum=max_per_episode_per_phase,
                minimum_gap=phase_sample_gap,
            ):
                payload = _trajectory_snapshot_payload(full, step, phase=phase)
                candidates[phase].append((episode_key, payload))
        onset = _failure_onset_index(low_steps, full_steps)
        for step in low_steps[onset : onset + len(WATERSHED_OFFSETS)]:
            payload = _trajectory_snapshot_payload(low, step, phase="failure_recovery")
            candidates["failure_recovery"].append((episode_key, payload))

    sampled = {
        phase: _sample_candidates(
            rows,
            snapshot_limit,
            max_per_episode=(
                len(WATERSHED_OFFSETS)
                if phase == "failure_recovery"
                else max_per_episode_per_phase
            ),
        )
        for phase, rows in candidates.items()
    }
    selected: list[dict[str, Any]] = []
    offsets = {phase: 0 for phase in sampled}
    while len(selected) < snapshot_limit:
        added = False
        for phase in ("precontact", "manipulation", "failure_recovery"):
            index = offsets[phase]
            if index < len(sampled[phase]) and len(selected) < snapshot_limit:
                selected.append(sampled[phase][index])
                offsets[phase] += 1
                added = True
        if not added:
            break
    if len(selected) < minimum_snapshots:
        counts = {phase: len(rows) for phase, rows in sampled.items()}
        raise RuntimeError(
            f"Only {len(selected)} targeted snapshots available; minimum is "
            f"{minimum_snapshots}; candidates={counts}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    written = [_write_snapshot(output_dir, payload) for payload in selected]
    phase_counts = {
        phase: sum(str(payload["phase"]) == phase for payload in selected)
        for phase in sampled
    }
    print(
        f"Saved {len(written)} targeted snapshots from {len(pairs)} watershed pairs: "
        + json.dumps(phase_counts, sort_keys=True)
    )
    return written


def save_snapshot(
    output_dir: Path,
    *,
    task_suite: str,
    task_id: int,
    seed: int,
    step: int,
    phase: str,
    env: Any,
    observation: Any,
    metrics: PhysicalMetrics | None = None,
    state_manager: Any = SimStateManager,
) -> Path:
    payload = _snapshot_payload(
        task_suite=task_suite,
        task_id=task_id,
        seed=seed,
        step=step,
        phase=phase,
        env=env,
        observation=observation,
        metrics=metrics,
        state_manager=state_manager,
    )
    return _write_snapshot(output_dir, payload)


def replay_episode(
    adapter: Any,
    record: Mapping[str, Any],
    *,
    episode_kind: str,
    candidate_pool: dict[str, list[tuple[str, dict[str, Any]]]],
    max_steps: int,
    max_per_episode_per_phase: int = 2,
    phase_sample_gap: int = 5,
    state_manager: Any = SimStateManager,
) -> None:
    """Collect event snapshots while replaying one recorded baseline policy."""
    expected_steps = int(record["steps"])
    if expected_steps <= 0 or max_steps <= 0:
        raise ValueError("recorded steps and max_steps must be positive")
    task_suite = str(record["task_suite"])
    task_id = int(record["task_id"])
    seed = int(record["seed"])
    observation = adapter.start_task(task_id, seed)
    initial_positions = collect_object_positions(adapter.env)
    task_key = f"{task_suite}:{task_id}:{seed}"
    phase_capture_steps: dict[str, list[int]] = defaultdict(list)

    if episode_kind == "watershed":
        config = NO_IMAGINATION_CONFIG
        target_steps = {
            expected_steps - offset: offset
            for offset in WATERSHED_OFFSETS
            if expected_steps - offset >= 0
        }
    elif episode_kind == "successful_full":
        config = FIXED_FULL_CONFIG
        target_steps = {}
    elif episode_kind == "successful_small":
        config = FIXED_SMALL_CONFIG
        target_steps = {}
    else:
        raise ValueError(f"Unknown episode kind {episode_kind!r}")

    action_chunk: np.ndarray | None = None
    action_index = 0
    step_index = 0
    captured_watershed_steps: set[int] = set()
    finished = bool(adapter.is_success(observation, {}))
    while step_index < min(expected_steps, max_steps) and not finished:
        metrics = measure_physics(
            adapter.env,
            initial_object_positions=initial_positions,
            observation=observation,
        )
        if episode_kind == "watershed" and step_index in target_steps:
            offset = target_steps[step_index]
            phase = f"failure_recovery_pre_{offset}"
            payload = _snapshot_payload(
                task_suite=task_suite,
                task_id=task_id,
                seed=seed,
                step=step_index,
                phase=phase,
                env=adapter.env,
                observation=observation,
                metrics=metrics,
                state_manager=state_manager,
            )
            candidate_pool["failure_recovery"].append((task_key, payload))
            captured_watershed_steps.add(step_index)
        elif episode_kind in {"successful_full", "successful_small"}:
            phase = classify_physical_phase(metrics)
            captured_steps = phase_capture_steps[phase] if phase in PHYSICAL_PHASES else []
            if (
                phase in PHYSICAL_PHASES
                and len(captured_steps) < max_per_episode_per_phase
                and (
                    not captured_steps
                    or step_index - captured_steps[-1] >= phase_sample_gap
                )
            ):
                payload = _snapshot_payload(
                    task_suite=task_suite,
                    task_id=task_id,
                    seed=seed,
                    step=step_index,
                    phase=phase,
                    env=adapter.env,
                    observation=observation,
                    metrics=metrics,
                    state_manager=state_manager,
                )
                candidate_pool[phase].append((task_key, payload))
                captured_steps.append(step_index)

        if action_chunk is None or action_index >= len(action_chunk):
            actions = adapter.infer_actions(observation, config)
            action_chunk = np.asarray(actions)
            if action_chunk.ndim == 1:
                action_chunk = action_chunk[None, :]
            if action_chunk.ndim != 2 or action_chunk.shape[0] < config.E:
                raise ValueError(
                    f"{episode_kind} policy returned {action_chunk.shape}; "
                    f"expected at least {config.E} action rows"
                )
            action_chunk = action_chunk[: config.E]
            action_index = 0
        result = adapter.step(action_chunk[action_index])
        observation = result.observation
        action_index += 1
        step_index += 1
        finished = bool(
            adapter.is_success(observation, result.info)
            or result.terminated
            or result.truncated
        )

    if episode_kind in {"successful_full", "successful_small"} and adapter.is_success(observation, {}):
        terminal_metrics = measure_physics(
            adapter.env,
            initial_object_positions=initial_positions,
            observation=observation,
        )
        phase = classify_physical_phase(terminal_metrics)
        captured_steps = phase_capture_steps[phase] if phase in PHYSICAL_PHASES else []
        if (
            phase in PHYSICAL_PHASES
            and len(captured_steps) < max_per_episode_per_phase
            and (
                not captured_steps
                or step_index - captured_steps[-1] >= phase_sample_gap
            )
        ):
            payload = _snapshot_payload(
                task_suite=task_suite,
                task_id=task_id,
                seed=seed,
                step=step_index,
                phase=phase,
                env=adapter.env,
                observation=observation,
                metrics=terminal_metrics,
                state_manager=state_manager,
            )
            candidate_pool[phase].append((task_key, payload))
            captured_steps.append(step_index)

    if episode_kind == "watershed" and set(target_steps) != captured_watershed_steps:
        missing = sorted(set(target_steps) - captured_watershed_steps)
        raise RuntimeError(
            f"Replay ended at step {step_index}; missed watershed states {missing} "
            f"for task={task_id}, seed={seed}"
        )


def _sample_candidates(
    candidates: Sequence[tuple[str, dict[str, Any]]],
    maximum: int,
    max_per_episode: int = 2,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode_key, payload in candidates:
        grouped[episode_key].append(payload)
    episode_candidates = {
        episode_key: sorted(rows, key=lambda row: int(row["step"]))[:max_per_episode]
        for episode_key, rows in sorted(grouped.items())
    }
    selected: list[dict[str, Any]] = []
    for sample_index in range(max_per_episode):
        for rows in episode_candidates.values():
            if sample_index < len(rows) and len(selected) < maximum:
                selected.append(rows[sample_index])
        if len(selected) >= maximum:
            break
    return selected


def _load_factory(specification: str) -> Callable[[WorkerContext], Any]:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--adapter must have the form python.module:factory")
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise TypeError(f"Adapter factory {specification!r} is not callable")
    return factory


def collect_from_csv(
    episodes_csv: Path,
    output_dir: Path,
    adapter_factory: Callable[[WorkerContext], Any],
    *,
    options: Mapping[str, Any],
    device: str,
    successful_episode_limit: int | None = None,
    phase_success_method: str = "Fixed-Full",
    snapshots_per_category: int = 50,
    minimum_per_category: int = 40,
    max_per_episode_per_phase: int = 2,
    phase_sample_gap: int = 5,
    max_steps: int = 400,
    state_manager: Any = SimStateManager,
) -> list[Path]:
    if snapshots_per_category <= 0 or minimum_per_category < 0:
        raise ValueError("snapshot quotas must be positive/non-negative")
    if max_per_episode_per_phase not in (1, 2) or phase_sample_gap < 1:
        raise ValueError("max_per_episode_per_phase must be 1 or 2; phase_sample_gap must be positive")
    records = read_episode_records(episodes_csv)
    selected = select_episode_records(
        records,
        successful_episode_limit,
        phase_success_method=phase_success_method,
    )
    unique_episode_count = len(
        {(str(row["task_suite"]), int(row["task_id"]), int(row["seed"])) for row in records}
    )
    success_rollouts = sum(kind != "watershed" for _, kind in selected)
    watershed_rollouts = sum(kind == "watershed" for _, kind in selected)
    capacity = {
        phase: success_rollouts * max_per_episode_per_phase
        for phase in PHYSICAL_PHASES
    }
    capacity["failure_recovery"] = watershed_rollouts * len(WATERSHED_OFFSETS)
    print(
        f"Episode CSV: {len(records)} method rows, {unique_episode_count} unique episodes; "
        f"selected rollouts={len(selected)} (success={success_rollouts}, "
        f"watershed={watershed_rollouts})"
    )
    print("Maximum candidate capacity by category: " + json.dumps(capacity, sort_keys=True))
    print(f"Maximum total snapshot capacity from this selection: {sum(capacity.values())}")
    if any(value < minimum_per_category for value in capacity.values()):
        print(
            f"Warning: requested minimum {minimum_per_category} cannot be reached "
            "from this CSV without additional independent episodes"
        )
    groups: dict[str, list[tuple[Mapping[str, Any], str]]] = defaultdict(list)
    for record, kind in selected:
        groups[str(record["task_suite"])].append((record, kind))

    candidate_pool: dict[str, list[tuple[str, dict[str, Any]]]] = {
        **{phase: [] for phase in PHYSICAL_PHASES},
        "failure_recovery": [],
    }
    for task_suite, suite_records in groups.items():
        context = WorkerContext(device=device, task_suite=task_suite, options=dict(options))
        adapter = adapter_factory(context)
        try:
            for record, kind in suite_records:
                replay_episode(
                    adapter,
                    record,
                    episode_kind=kind,
                    candidate_pool=candidate_pool,
                    max_steps=max_steps,
                    max_per_episode_per_phase=max_per_episode_per_phase,
                    phase_sample_gap=phase_sample_gap,
                    state_manager=state_manager,
                )
        finally:
            adapter.close()

    snapshots: list[Path] = []
    counts: dict[str, int] = {}
    for phase, candidates in candidate_pool.items():
        quota = min(snapshots_per_category, len(candidates))
        selected_payloads = _sample_candidates(
            candidates,
            quota,
            max_per_episode=5 if phase == "failure_recovery" else max_per_episode_per_phase,
        )
        counts[phase] = len(selected_payloads)
        for payload in selected_payloads:
            snapshots.append(_write_snapshot(output_dir, payload))
    print("Snapshot counts by physical category: " + json.dumps(counts, sort_keys=True))
    if any(count < minimum_per_category for count in counts.values()):
        print(
            f"Warning: snapshot pool is below minimum_per_category={minimum_per_category}; "
            "collect additional independent episodes/seeds rather than duplicating states"
        )
    return snapshots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-csv", type=Path, default=DEFAULT_EPISODES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        default=None,
        help="Recorded data root; if missing, search outputs/ then fall back to LIBERO config",
    )
    parser.add_argument(
        "--watershed-pairs",
        type=Path,
        default=None,
        help=(
            "CSV of No-Imagination-fail/Fixed-Full-success episode keys. "
            "When set, read the saved .pt trajectories directly instead of replaying."
        ),
    )
    parser.add_argument("--targeted-snapshot-limit", type=int, default=60)
    parser.add_argument("--minimum-targeted-snapshots", type=int, default=40)
    parser.add_argument("--adapter", default="real_fastwam_adapter:create_adapter")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "fastwam_release" / "libero_optional_idm_2cam224.pt",
    )
    parser.add_argument("--task-config", default="libero_optional_idm_2cam224_1e-4")
    parser.add_argument("--config-name", default="sim_libero")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--successful-episodes", type=int, default=None)
    parser.add_argument(
        "--phase-success-method",
        choices=("Fixed-Full", "Fixed-Small"),
        default="Fixed-Full",
    )
    parser.add_argument("--snapshots-per-category", "--snapshots-per-phase", dest="snapshots_per_category", type=int, default=50)
    parser.add_argument("--minimum-per-category", type=int, default=40)
    parser.add_argument("--max-per-episode-per-phase", type=int, choices=(1, 2), default=2)
    parser.add_argument("--phase-sample-gap", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=400)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_steps <= 0 or (
        args.successful_episodes is not None and args.successful_episodes < 0
    ):
        raise ValueError("--max-steps must be positive and --successful-episodes non-negative")
    if args.watershed_pairs is not None:
        snapshots = collect_targeted_from_trajectories(
            args.episodes_csv.expanduser().resolve(),
            args.watershed_pairs.expanduser().resolve(),
            args.output_dir.expanduser().resolve(),
            trajectories_dir=(
                args.trajectories_dir.expanduser().resolve()
                if args.trajectories_dir is not None
                else None
            ),
            snapshot_limit=args.targeted_snapshot_limit,
            minimum_snapshots=args.minimum_targeted_snapshots,
            max_per_episode_per_phase=args.max_per_episode_per_phase,
            phase_sample_gap=args.phase_sample_gap,
        )
        print(f"Saved {len(snapshots)} snapshots under {args.output_dir}")
        return
    try:
        gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",")]
    except ValueError as exc:
        raise ValueError("--gpu-ids must be comma-separated CUDA indices") from exc
    devices = [f"cuda:{index}" for index in gpu_ids]
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="libero_snapshot_config_",
        dir=args.output_dir.parent,
    ) as runtime_root_text:
        runtime_config_dir, dataset_dir, trajectory_file_count, resolution = prepare_libero_config(
            PROJECT_ROOT / "configs" / "EXP-B000" / "libero",
            Path(runtime_root_text) / "libero_config",
            args.trajectories_dir,
        )
        if resolution in {"created-no-demonstrations-placeholder", "configured-empty-or-unindexed"}:
            dataset_dir.mkdir(parents=True, exist_ok=True)
        del dataset_dir, trajectory_file_count
        options = {
            "model_path": str(args.model_path.expanduser().resolve()),
            "task_config": args.task_config,
            "config_name": args.config_name,
            "cuda_devices": devices,
            "max_steps": args.max_steps,
            "libero_config_dir": str(runtime_config_dir),
        }
        snapshots = collect_from_csv(
            args.episodes_csv,
            args.output_dir,
            _load_factory(args.adapter),
            options=options,
            device=devices[0] if devices else "cpu",
            successful_episode_limit=args.successful_episodes,
            phase_success_method=args.phase_success_method,
            snapshots_per_category=args.snapshots_per_category,
            minimum_per_category=args.minimum_per_category,
            max_per_episode_per_phase=args.max_per_episode_per_phase,
            phase_sample_gap=args.phase_sample_gap,
            max_steps=args.max_steps,
        )
    print(f"Saved {len(snapshots)} snapshots under {args.output_dir}")


if __name__ == "__main__":
    main()
