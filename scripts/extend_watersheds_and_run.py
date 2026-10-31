"""Mine the minimum number of extra paired rollouts, then resume Oracle.

This helper is intentionally conservative: the required Task 0--9 / Seed 0--4
pool remains untouched, extra candidates are evaluated one at a time, and the
loop stops as soon as the requested watershed threshold is reached.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from wait_for_watershed_then_oracle import watershed_rows  # noqa: E402


def _episode_key(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row["task_suite"]),
        int(row["task_id"]),
        int(row["seed"]),
        str(row["method"]),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _merge_rows(*groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for rows in groups:
        for source in rows:
            row = dict(source)
            key = _episode_key(row)
            previous = merged.get(key)
            if previous is not None and previous != row:
                raise ValueError(f"Conflicting result rows for {key}")
            merged[key] = row
    return [merged[key] for key in sorted(merged, key=lambda x: (x[1], x[2], x[3]))]


def _parse_candidates(value: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for token in value.split(","):
        task_text, separator, seed_text = token.strip().partition(":")
        if not separator or not task_text.isdigit() or not seed_text.isdigit():
            raise ValueError(f"Invalid candidate {token!r}; expected TASK:SEED")
        candidate = (int(task_text), int(seed_text))
        if candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        raise ValueError("At least one extension candidate is required")
    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--extension-csv", type=Path, required=True)
    parser.add_argument("--combined-csv", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--snapshots-dir", type=Path, required=True)
    parser.add_argument("--oracle-csv", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="4,5,6,7")
    parser.add_argument("--minimum-watersheds", type=int, default=8)
    parser.add_argument(
        "--candidates",
        default=(
            "7:5,7:6,7:7,7:8,7:9,"
            "1:5,1:6,1:7,1:8,1:9,"
            "5:5,5:6,5:7,5:8,5:9,"
            "9:5,9:6,9:7,9:8,9:9"
        ),
    )
    parser.add_argument("--minimum-free-gib", type=float, default=2.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.minimum_watersheds <= 0 or args.minimum_free_gib <= 0:
        raise ValueError("Thresholds must be positive")

    base_csv = args.base_csv.expanduser().resolve()
    extension_csv = args.extension_csv.expanduser().resolve()
    combined_csv = args.combined_csv.expanduser().resolve()
    pairs_csv = args.pairs_csv.expanduser().resolve()
    status_json = args.status_json.expanduser().resolve()
    base_rows = _read_csv(base_csv)
    if len(base_rows) != 100:
        raise RuntimeError(f"Expected the required 100 base rows, found {len(base_rows)}")

    python = sys.executable
    candidates = _parse_candidates(args.candidates)
    for task_id, seed in candidates:
        extension_rows = _read_csv(extension_csv)
        combined_rows = _merge_rows(base_rows, extension_rows)
        watersheds = watershed_rows(combined_rows)
        _atomic_csv(combined_csv, combined_rows)
        if len(watersheds) >= args.minimum_watersheds:
            break

        pair_methods = {
            str(row["method"])
            for row in combined_rows
            if int(row["task_id"]) == task_id and int(row["seed"]) == seed
        }
        if {"No-Imagination", "Fixed-Full"} <= pair_methods:
            continue

        free_gib = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
        if free_gib < args.minimum_free_gib:
            raise RuntimeError(
                f"Only {free_gib:.2f} GiB free before candidate task={task_id} seed={seed}"
            )
        _atomic_json(
            status_json,
            {
                "stage": "extending_watershed_pool",
                "watershed_count": len(watersheds),
                "candidate": {"task_id": task_id, "seed": seed},
                "free_gib": free_gib,
            },
        )
        subprocess.run(
            [
                python,
                str(PROJECT_ROOT / "eval_fixed_baselines.py"),
                "--task-ids", str(task_id),
                "--seeds", str(seed),
                "--methods", "No-Imagination,Fixed-Full",
                "--save-trajectories",
                "--trajectory-dir", str(args.trajectory_dir.expanduser().resolve()),
                "--trajectory-stride", "1",
                "--episodes-csv", str(extension_csv),
                "--summary-csv", str(
                    extension_csv.with_name(extension_csv.stem + "_summary.csv")
                ),
                "--gpu-ids", args.gpu_ids,
                "--resume",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    else:
        combined_rows = _merge_rows(base_rows, _read_csv(extension_csv))
        watersheds = watershed_rows(combined_rows)

    combined_rows = _merge_rows(base_rows, _read_csv(extension_csv))
    watersheds = watershed_rows(combined_rows)
    _atomic_csv(combined_csv, combined_rows)
    if watersheds:
        _atomic_csv(pairs_csv, watersheds)
    if len(watersheds) < args.minimum_watersheds:
        _atomic_json(
            status_json,
            {
                "stage": "insufficient_watersheds_after_extension",
                "watershed_count": len(watersheds),
                "minimum_watersheds": args.minimum_watersheds,
            },
        )
        raise RuntimeError("Extension candidates exhausted before reaching threshold")

    _atomic_json(
        status_json,
        {
            "stage": "extension_complete",
            "watershed_count": len(watersheds),
            "method_rows": len(combined_rows),
        },
    )
    subprocess.run(
        [
            python,
            str(PROJECT_ROOT / "scripts" / "wait_for_watershed_then_oracle.py"),
            "--producer-pid", str(os.getpid()),
            "--episodes-csv", str(combined_csv),
            "--pairs-csv", str(pairs_csv),
            "--trajectories-dir", str(args.trajectory_dir.expanduser().resolve()),
            "--snapshots-dir", str(args.snapshots_dir.expanduser().resolve()),
            "--oracle-csv", str(args.oracle_csv.expanduser().resolve()),
            "--analysis-dir", str(args.analysis_dir.expanduser().resolve()),
            "--status-json", str(status_json),
            "--gpu-ids", args.gpu_ids,
            "--minimum-watersheds", str(args.minimum_watersheds),
            "--expected-episodes", "50",
            "--poll-seconds", "10",
            "--timeout-hours", "48",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
