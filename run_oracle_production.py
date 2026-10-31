"""Run the gated production rollout, physical snapshot, and Oracle pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "checkpoints" / "libero_optional_idm_2cam224.pt"
DEFAULT_EXISTING_EPISODES = PROJECT_ROOT / "results" / "fixed_baselines_episodes.csv"


def read_episode_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Episode CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        rows = [dict(row) for row in csv.DictReader(source)]
    if not rows:
        raise ValueError(f"Episode CSV is empty: {path}")
    return rows


def baseline_capacity(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_suite: str,
    task_ids: Sequence[int],
    seed_start: int,
    seed_count: int,
    max_per_episode_per_phase: int = 2,
    watershed_offsets: int = 5,
) -> dict[str, int]:
    """Measure paired coverage and candidate capacity for the requested grid."""
    requested = {
        (task_suite, task_id, seed)
        for task_id in task_ids
        for seed in range(seed_start, seed_start + seed_count)
    }
    by_episode: dict[tuple[str, int, int], dict[str, int]] = defaultdict(dict)
    for row in rows:
        try:
            key = (str(row["task_suite"]), int(row["task_id"]), int(row["seed"]))
            method = str(row["method"])
            success = int(row["success"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed episode row: {row}") from exc
        if key in requested:
            if method in by_episode[key]:
                raise ValueError(f"Duplicate method {method!r} for episode {key}")
            by_episode[key][method] = success

    paired = {
        key: methods
        for key, methods in by_episode.items()
        if "No-Imagination" in methods and "Fixed-Full" in methods
    }
    full_successes = sum(methods["Fixed-Full"] == 1 for methods in paired.values())
    watersheds = sum(
        methods["No-Imagination"] == 0 and methods["Fixed-Full"] == 1
        for methods in paired.values()
    )
    return {
        "requested_episodes": len(requested),
        "paired_episodes": len(paired),
        "fixed_full_successes": full_successes,
        "watershed_episodes": watersheds,
        "phase_candidate_capacity": full_successes * max_per_episode_per_phase,
        "failure_recovery_candidate_capacity": watersheds * watershed_offsets,
    }


def snapshot_phase_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for snapshot_path in sorted(path.rglob("*.pkl")):
        with snapshot_path.open("rb") as source:
            payload = pickle.load(source)
        if payload.get("sampling_protocol") != "physical_events_v2":
            raise ValueError(f"Non-physical-v2 snapshot in production pool: {snapshot_path}")
        phase = str(payload.get("phase", ""))
        if phase.startswith("failure_recovery"):
            phase = "failure_recovery"
        elif phase.startswith("placement"):
            phase = "placement"
        counts[phase] += 1
    return counts


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run(command: Sequence[str], *, dry_run: bool, environment: Mapping[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, env=dict(environment), check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--existing-episodes-csv", type=Path, default=DEFAULT_EXISTING_EPISODES)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-ids", default="0-9")
    parser.add_argument("--initial-seeds", type=int, default=5)
    parser.add_argument("--maximum-seeds", type=int, default=10)
    parser.add_argument(
        "--snapshot-quota-per-category",
        type=int,
        default=60,
        help="Balanced target per physical category (60 gives about 300 total).",
    )
    parser.add_argument(
        "--minimum-per-category",
        type=int,
        default=50,
        help="Required floor per category before full Oracle evaluation.",
    )
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _parse_task_ids(value: str) -> list[int]:
    result: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if "-" in token:
            lower, upper = (int(item) for item in token.split("-", 1))
            result.update(range(lower, upper + 1))
        elif token:
            result.add(int(token))
    if not result:
        raise ValueError("--task-ids selects no tasks")
    return sorted(result)


def main() -> None:
    args = build_parser().parse_args()
    if not 0 < args.initial_seeds <= args.maximum_seeds:
        raise ValueError("Seed counts must satisfy 0 < initial <= maximum")
    if not 0 < args.minimum_per_category <= args.snapshot_quota_per_category:
        raise ValueError("Snapshot counts must satisfy 0 < minimum <= quota")
    if args.num_repeats <= 0:
        raise ValueError("--num-repeats must be positive")

    task_ids = _parse_task_ids(args.task_ids)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    episodes_csv = output_root / "fixed_baselines_episodes.csv"
    summary_csv = output_root / "fixed_baselines_summary.csv"
    snapshots_dir = output_root / "snapshots"
    oracle_dir = output_root / "oracle"
    oracle_csv = oracle_dir / "oracle_counterfactual_results.csv"
    state_path = output_root / "production_state.json"

    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    environment.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model_path = args.model_path.expanduser().resolve()
    existing_csv = args.existing_episodes_csv.expanduser().resolve()
    python = sys.executable

    def run_baselines(seed_start: int, seed_count: int) -> None:
        command = [
            python,
            str(PROJECT_ROOT / "eval_fixed_baselines.py"),
            "--model-path", str(model_path),
            "--task-suite", args.task_suite,
            "--task-ids", args.task_ids,
            "--seed-start", str(seed_start),
            "--num-seeds", str(seed_count),
            "--methods", "No-Imagination,Fixed-Full",
            "--gpu-ids", args.gpu_ids,
            "--episodes-csv", str(episodes_csv),
            "--summary-csv", str(summary_csv),
            "--resume",
        ]
        if existing_csv.is_file():
            command.extend(("--merge-episodes-csv", str(existing_csv)))
        _run(command, dry_run=args.dry_run, environment=environment)

    run_baselines(0, args.initial_seeds)
    if args.dry_run:
        run_baselines(args.initial_seeds, args.maximum_seeds - args.initial_seeds)
        return

    capacity = baseline_capacity(
        read_episode_rows(episodes_csv),
        task_suite=args.task_suite,
        task_ids=task_ids,
        seed_start=0,
        seed_count=args.initial_seeds,
    )
    needs_more = (
        capacity["paired_episodes"] < capacity["requested_episodes"]
        or capacity["phase_candidate_capacity"] < args.minimum_per_category
        or capacity["failure_recovery_candidate_capacity"] < args.minimum_per_category
    )
    if needs_more and args.maximum_seeds > args.initial_seeds:
        run_baselines(
            args.initial_seeds,
            args.maximum_seeds - args.initial_seeds,
        )
        capacity = baseline_capacity(
            read_episode_rows(episodes_csv),
            task_suite=args.task_suite,
            task_ids=task_ids,
            seed_start=0,
            seed_count=args.maximum_seeds,
        )

    _atomic_json(state_path, {"stage": "baselines_complete", "capacity": capacity})
    if (
        capacity["paired_episodes"] < capacity["requested_episodes"]
        or capacity["phase_candidate_capacity"] < args.minimum_per_category
        or capacity["failure_recovery_candidate_capacity"] < args.minimum_per_category
    ):
        required_watersheds = math.ceil(args.minimum_per_category / 5)
        raise RuntimeError(
            "Baseline pool cannot meet balanced snapshot minimum: "
            f"{capacity}; require at least {required_watersheds} watershed episodes"
        )

    existing_counts = snapshot_phase_counts(snapshots_dir) if snapshots_dir.exists() else Counter()
    required_phases = {"reaching", "precontact", "manipulation", "placement", "failure_recovery"}
    snapshots_complete = (
        required_phases.issubset(existing_counts)
        and all(existing_counts[phase] >= args.minimum_per_category for phase in required_phases)
        and 250 <= sum(existing_counts.values()) <= 350
    )
    if not snapshots_complete:
        if existing_counts:
            raise RuntimeError(
                f"Incomplete snapshot directory requires inspection before retry: {existing_counts}"
            )
        _run(
            [
                python,
                str(PROJECT_ROOT / "oracle" / "collect_snapshots.py"),
                "--episodes-csv", str(episodes_csv),
                "--output-dir", str(snapshots_dir),
                "--model-path", str(model_path),
                "--gpu-ids", args.gpu_ids,
                "--phase-success-method", "Fixed-Full",
                "--snapshots-per-category", str(args.snapshot_quota_per_category),
                "--minimum-per-category", str(args.minimum_per_category),
                "--max-per-episode-per-phase", "2",
                "--phase-sample-gap", "5",
            ],
            dry_run=False,
            environment=environment,
        )
        existing_counts = snapshot_phase_counts(snapshots_dir)
    if not (
        required_phases.issubset(existing_counts)
        and all(existing_counts[phase] >= args.minimum_per_category for phase in required_phases)
        and 250 <= sum(existing_counts.values()) <= 350
    ):
        raise RuntimeError(f"Collected snapshot pool failed production validation: {existing_counts}")
    _atomic_json(
        state_path,
        {"stage": "snapshots_complete", "capacity": capacity, "snapshot_counts": existing_counts},
    )

    _run(
        [
            python,
            str(PROJECT_ROOT / "oracle" / "counterfactual_oracle.py"),
            "--snapshots-dir", str(snapshots_dir),
            "--cost-table", str(PROJECT_ROOT / "configs" / "hardware_cost_table.json"),
            "--output-csv", str(oracle_csv),
            "--model-path", str(model_path),
            "--gpu-ids", args.gpu_ids,
            "--eval-mode", "sustained",
            "--num-repeats", str(args.num_repeats),
            "--resume",
        ],
        dry_run=False,
        environment=environment,
    )
    _run(
        [
            python,
            str(PROJECT_ROOT / "oracle" / "analyze_oracle.py"),
            "--results-csv", str(oracle_csv),
            "--cost-table", str(PROJECT_ROOT / "configs" / "hardware_cost_table.json"),
            "--output-dir", str(oracle_dir),
        ],
        dry_run=False,
        environment=environment,
    )
    _atomic_json(
        state_path,
        {
            "stage": "complete",
            "capacity": capacity,
            "snapshot_counts": existing_counts,
            "oracle_results": str(oracle_csv),
        },
    )


if __name__ == "__main__":
    main()
