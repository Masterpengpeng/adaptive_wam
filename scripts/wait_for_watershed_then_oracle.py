"""Wait for paired baseline trajectories, then run watershed smoke end to end."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty watershed CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_complete_pairs(
    path: Path,
    *,
    task_ids: Sequence[int],
    seeds: Sequence[int],
) -> tuple[list[dict[str, str]], bool]:
    if not path.is_file():
        return [], False
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    by_key: dict[tuple[str, int, int, str], dict[str, str]] = {}
    for row in rows:
        key = (
            str(row.get("task_suite", "")),
            int(row["task_id"]),
            int(row["seed"]),
            str(row["method"]),
        )
        if key in by_key:
            raise ValueError(f"Duplicate baseline result {key}")
        by_key[key] = row
    expected = {
        (task_id, seed, method)
        for task_id in task_ids
        for seed in seeds
        for method in ("No-Imagination", "Fixed-Full")
    }
    observed = {
        (task_id, seed, method)
        for _, task_id, seed, method in by_key
        if task_id in task_ids and seed in seeds
    }
    return rows, expected.issubset(observed)


def watershed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_episode: dict[tuple[str, int, int], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["task_suite"]), int(row["task_id"]), int(row["seed"]))
        by_episode.setdefault(key, {})[str(row["method"])] = row
    selected: list[dict[str, Any]] = []
    for (suite, task_id, seed), methods in sorted(by_episode.items()):
        low = methods.get("No-Imagination")
        full = methods.get("Fixed-Full")
        if low is None or full is None:
            continue
        if int(low["success"]) == 0 and int(full["success"]) == 1:
            selected.append(
                {
                    "task_suite": suite,
                    "task_id": task_id,
                    "seed": seed,
                    "no_imagination_trajectory": str(low.get("trajectory_path", "")),
                    "fixed_full_trajectory": str(full.get("trajectory_path", "")),
                }
            )
    return selected


def _run(command: Sequence[str], status_path: Path, stage: str) -> None:
    _atomic_json(status_path, {"stage": stage, "command": list(command)})
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-pid", type=int, required=True)
    parser.add_argument("--episodes-csv", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--trajectories-dir", type=Path, required=True)
    parser.add_argument("--snapshots-dir", type=Path, required=True)
    parser.add_argument("--oracle-csv", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--minimum-watersheds", type=int, default=8)
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=48.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.minimum_watersheds <= 0 or args.expected_episodes <= 0:
        raise ValueError("Watershed and episode counts must be positive")
    if args.poll_seconds <= 0 or args.timeout_hours <= 0:
        raise ValueError("Polling interval and timeout must be positive")
    task_ids = list(range(10))
    seeds = list(range(5))
    episodes_csv = args.episodes_csv.expanduser().resolve()
    pairs_csv = args.pairs_csv.expanduser().resolve()
    snapshots_dir = args.snapshots_dir.expanduser().resolve()
    oracle_csv = args.oracle_csv.expanduser().resolve()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    status_path = args.status_json.expanduser().resolve()
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    _atomic_json(
        status_path,
        {
            "stage": "waiting_for_paired_rollouts",
            "producer_pid": args.producer_pid,
            "expected_method_rows": args.expected_episodes * 2,
        },
    )

    while True:
        rows, complete = read_complete_pairs(
            episodes_csv,
            task_ids=task_ids,
            seeds=seeds,
        )
        if complete:
            break
        if not _process_alive(args.producer_pid):
            payload = {
                "stage": "baseline_failed",
                "observed_method_rows": len(rows),
                "reason": "baseline process exited before all paired episodes were recorded",
            }
            _atomic_json(status_path, payload)
            raise RuntimeError(payload["reason"])
        if time.monotonic() >= deadline:
            _atomic_json(status_path, {"stage": "baseline_timeout", "rows": len(rows)})
            raise TimeoutError("Paired baseline collection timed out")
        time.sleep(args.poll_seconds)

    watersheds = watershed_rows(rows)
    _atomic_csv(pairs_csv, watersheds) if watersheds else None
    if len(watersheds) < args.minimum_watersheds:
        payload = {
            "stage": "insufficient_watersheds",
            "watershed_count": len(watersheds),
            "minimum_watersheds": args.minimum_watersheds,
        }
        _atomic_json(status_path, payload)
        raise RuntimeError(str(payload))

    if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty snapshot directory {snapshots_dir}")
    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    if free_bytes < 3 * 1024**3:
        raise RuntimeError(f"Less than 3 GiB free before snapshot/Oracle stages: {free_bytes}")

    python = sys.executable
    _run(
        [
            python,
            str(PROJECT_ROOT / "oracle" / "collect_snapshots.py"),
            "--episodes-csv", str(episodes_csv),
            "--watershed-pairs", str(pairs_csv),
            "--trajectories-dir", str(args.trajectories_dir.expanduser().resolve()),
            "--output-dir", str(snapshots_dir),
            "--targeted-snapshot-limit", "60",
            "--minimum-targeted-snapshots", "40",
        ],
        status_path,
        "extracting_targeted_snapshots",
    )
    oracle_csv.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            python,
            str(PROJECT_ROOT / "oracle" / "counterfactual_oracle.py"),
            "--snapshots-dir", str(snapshots_dir),
            "--model-path", str(
                PROJECT_ROOT / "checkpoints" / "fastwam_release" / "libero_optional_idm_2cam224.pt"
            ),
            "--eval-mode", "sustained",
            "--output-csv", str(oracle_csv),
            "--num-repeats", "3",
            "--gpu-ids", args.gpu_ids,
            "--resume",
        ],
        status_path,
        "running_sustained_oracle",
    )
    _run(
        [
            python,
            str(PROJECT_ROOT / "oracle" / "analyze_oracle.py"),
            "--results-csv", str(oracle_csv),
            "--cost-table", str(PROJECT_ROOT / "configs" / "hardware_cost_table.json"),
            "--output-dir", str(analysis_dir),
        ],
        status_path,
        "analyzing_oracle",
    )
    report = json.loads((analysis_dir / "oracle_analysis.json").read_text(encoding="utf-8"))
    phases = {
        row["phase"]: row
        for row in report["rq2"]["phase_distribution"]
    }
    checks = {
        phase: {
            "mean_E": float(phases[phase]["mean_E"]),
            "mean_D": float(phases[phase]["mean_D"]),
            "passed": (
                float(phases[phase]["mean_E"]) < 10.0
                and float(phases[phase]["mean_D"]) >= 1.0
            ),
        }
        for phase in ("precontact", "manipulation")
    }
    _atomic_json(
        status_path,
        {
            "stage": "complete",
            "watershed_count": len(watersheds),
            "checks": checks,
            "passed": all(row["passed"] for row in checks.values()),
            "analysis": str(analysis_dir / "oracle_analysis.json"),
            "plot": str(analysis_dir / "rq2_phase_distribution.png"),
        },
    )


if __name__ == "__main__":
    main()
