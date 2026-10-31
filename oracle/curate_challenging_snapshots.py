"""Curate reaching controls and physically risky manipulation snapshots."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_fixed_baselines import load_trajectory
from oracle.collect_snapshots import _trajectory_snapshot_payload, _write_snapshot


def _metric(step: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    metrics = step.get("physical_metrics")
    value = metrics.get(name) if isinstance(metrics, Mapping) else None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _phase_segments(steps: Sequence[Mapping[str, Any]], phase: str) -> list[list[int]]:
    segments: list[list[int]] = []
    current: list[int] = []
    for index, step in enumerate(steps):
        if str(step.get("phase")) == phase:
            current.append(index)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def _reaching_indices(steps: Sequence[Mapping[str, Any]]) -> list[int]:
    segments = _phase_segments(steps, "reaching")
    if not segments:
        return []
    initial = segments[0]
    fractions = (0.35, 0.70)
    return sorted(
        {
            initial[min(len(initial) - 1, max(0, round((len(initial) - 1) * fraction)))]
            for fraction in fractions
        }
    )


def _manipulation_candidates(
    steps: Sequence[Mapping[str, Any]],
    *,
    source_method: str,
) -> list[tuple[float, int, str]]:
    candidates: list[tuple[float, int, str]] = []
    source_bonus = 5.0 if source_method == "No-Imagination" else 0.0
    for segment in _phase_segments(steps, "manipulation"):
        segment_length = len(segment)
        considered = segment[: min(3, segment_length)]
        for offset, index in enumerate(considered):
            step = steps[index]
            future = steps[index + 1 : min(len(steps), index + 11)]
            force = _metric(step, "contact_force_n")
            goal = _metric(step, "goal_distance_m", default=float("nan"))
            future_goals = [
                _metric(item, "goal_distance_m", default=float("nan")) for item in future
            ]
            future_goals = [value for value in future_goals if np.isfinite(value)]
            goal_regression = (
                max(0.0, max(future_goals) - goal)
                if np.isfinite(goal) and future_goals
                else 0.0
            )
            unstable_exit = any(
                str(item.get("phase")) in {"reaching", "precontact", "unknown", "None"}
                for item in future
            )
            short_contact = segment_length <= 5
            onset_bonus = 3.0 if offset == 0 else 1.0 / (offset + 1)
            light_touch_bonus = 1.0 if 0.0 < force <= 10.0 else 0.0
            score = (
                source_bonus
                + onset_bonus
                + 4.0 * float(unstable_exit)
                + 2.0 * float(short_contact)
                + light_touch_bonus
                + 30.0 * goal_regression
            )
            reasons = ["failed_policy_contact" if source_bonus else "successful_contact_boundary"]
            if offset == 0:
                reasons.append("contact_onset")
            if unstable_exit:
                reasons.append("contact_loss")
            if short_contact:
                reasons.append("short_contact_segment")
            if goal_regression > 0.005:
                reasons.append("goal_regression")
            candidates.append((score, index, "+".join(reasons)))
    return sorted(candidates, key=lambda item: (-item[0], item[1]))


def _round_robin(
    candidates: Mapping[str, Sequence[tuple[float, dict[str, Any], dict[str, Any]]]],
    limit: int,
    *,
    per_episode: int,
) -> list[tuple[float, dict[str, Any], dict[str, Any]]]:
    selected: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for rank in range(per_episode):
        ranked = sorted(
            (
                rows[rank]
                for rows in candidates.values()
                if rank < len(rows)
            ),
            key=lambda item: -item[0],
        )
        for item in ranked:
            if len(selected) >= limit:
                return selected
            selected.append(item)
    return selected


def curate(
    pairs_csv: Path,
    output_dir: Path,
    *,
    reaching_count: int = 12,
    manipulation_count: int = 16,
) -> list[Path]:
    if not (10 <= reaching_count <= 15):
        raise ValueError("reaching_count must be between 10 and 15")
    if manipulation_count <= 0:
        raise ValueError("manipulation_count must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory {output_dir}")
    with pairs_csv.open(newline="", encoding="utf-8") as source:
        pairs = list(csv.DictReader(source))
    if not pairs:
        raise ValueError(f"No watershed pairs in {pairs_csv}")

    reaching: dict[str, list[tuple[float, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    manipulation: dict[str, list[tuple[float, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        episode_key = f"{pair['task_suite']}:{pair['task_id']}:{pair['seed']}"
        low = load_trajectory(Path(pair["no_imagination_trajectory"]))
        full = load_trajectory(Path(pair["fixed_full_trajectory"]))
        full_steps = list(full["steps"])
        for rank, index in enumerate(_reaching_indices(full_steps)):
            step = full_steps[index]
            payload = _trajectory_snapshot_payload(full, step, phase="reaching")
            payload["physical_metadata"].update(
                {"selection_reason": "free_space_control", "risk_score": 0.0}
            )
            metadata = {
                "category": "reaching",
                "source_method": "Fixed-Full",
                "selection_reason": "free_space_control",
                "risk_score": 0.0,
                "rank_within_episode": rank,
            }
            reaching[episode_key].append((float(2 - rank), payload, metadata))

        method_candidates: list[tuple[float, int, str, Mapping[str, Any], str]] = []
        for trajectory, method in ((low, "No-Imagination"), (full, "Fixed-Full")):
            steps = list(trajectory["steps"])
            method_candidates.extend(
                (score, index, reason, trajectory, method)
                for score, index, reason in _manipulation_candidates(
                    steps, source_method=method
                )
            )
        method_candidates.sort(key=lambda item: (-item[0], item[1]))
        used_steps: set[tuple[str, int]] = set()
        for score, index, reason, trajectory, method in method_candidates:
            step = trajectory["steps"][index]
            identity = (method, int(step["step"]))
            if identity in used_steps:
                continue
            used_steps.add(identity)
            payload = _trajectory_snapshot_payload(trajectory, step, phase="manipulation")
            payload["physical_metadata"].update(
                {"selection_reason": reason, "risk_score": float(score)}
            )
            metadata = {
                "category": "manipulation",
                "source_method": method,
                "selection_reason": reason,
                "risk_score": float(score),
                "rank_within_episode": len(manipulation[episode_key]),
            }
            manipulation[episode_key].append((float(score), payload, metadata))
            if len(manipulation[episode_key]) >= 2:
                break

    selected = _round_robin(reaching, reaching_count, per_episode=2)
    selected.extend(_round_robin(manipulation, manipulation_count, per_episode=2))
    if sum(item[2]["category"] == "reaching" for item in selected) != reaching_count:
        raise RuntimeError("Insufficient independent reaching candidates")
    if sum(item[2]["category"] == "manipulation" for item in selected) != manipulation_count:
        raise RuntimeError("Insufficient independent manipulation candidates")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    manifest: list[dict[str, Any]] = []
    for _, payload, metadata in selected:
        path = _write_snapshot(output_dir, payload)
        paths.append(path)
        events = payload["physical_metadata"].get("events", {})
        manifest.append(
            {
                "snapshot_file": str(path.resolve()),
                "task_suite": payload["task_suite"],
                "task_id": payload["task_id"],
                "seed": payload["seed"],
                "step": payload["step"],
                "phase": payload["phase"],
                **metadata,
                "gripper_target_distance_m": events.get("gripper_target_distance_m"),
                "contact_force_n": events.get("contact_force_n"),
                "target_object_displacement_m": events.get("target_object_displacement_m"),
                "goal_distance_m": events.get("goal_distance_m"),
            }
        )
    manifest_path = output_dir / "snapshot_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watershed-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reaching-count", type=int, default=12)
    parser.add_argument("--manipulation-count", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = curate(
        args.watershed_pairs.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        reaching_count=args.reaching_count,
        manipulation_count=args.manipulation_count,
    )
    print(f"Saved {len(paths)} curated snapshots under {args.output_dir}")


if __name__ == "__main__":
    main()
