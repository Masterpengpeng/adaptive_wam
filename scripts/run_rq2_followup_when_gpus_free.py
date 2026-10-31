"""Wait for a safe GPU window, then run RQ2 follow-up experiments."""

from __future__ import annotations

import argparse
import csv
import json
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
        raise ValueError(f"Cannot write empty CSV {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _gpu_stats() -> dict[int, tuple[int, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    stats: dict[int, tuple[int, int]] = {}
    for line in completed.stdout.splitlines():
        index, memory, utilization = (int(item.strip()) for item in line.split(","))
        stats[index] = (memory, utilization)
    return stats


def _wait_for_gpus(
    gpu_ids: Sequence[int] | None,
    status_path: Path,
    *,
    poll_seconds: float,
    timeout_hours: float,
) -> list[int]:
    deadline = time.monotonic() + timeout_hours * 3600
    free_checks: dict[int, int] = {}
    while time.monotonic() < deadline:
        stats = _gpu_stats()
        candidates = list(gpu_ids) if gpu_ids is not None else sorted(stats)
        missing = set(candidates) - set(stats)
        if missing:
            raise ValueError(f"Requested GPU IDs are not available: {sorted(missing)}")
        available = [
            index
            for index in candidates
            if stats[index][0] < 512 and stats[index][1] < 10
        ]
        free_checks = {
            index: free_checks.get(index, 0) + 1 if index in available else 0
            for index in candidates
        }
        ready = [index for index in candidates if free_checks[index] >= 3]
        selected = ready[:4]
        top_checks = sorted(free_checks.values(), reverse=True)
        consecutive = top_checks[3] if len(top_checks) >= 4 else 0
        _atomic_json(
            status_path,
            {
                "stage": "waiting_for_gpus",
                "gpu_selection": "auto" if gpu_ids is None else "fixed",
                "gpu_ids": selected if len(selected) == 4 else (list(gpu_ids) if gpu_ids is not None else []),
                "candidate_gpu_ids": candidates,
                "available_gpu_ids": available,
                "gpu_stats": {
                    str(index): {"memory_mib": value[0], "utilization_percent": value[1]}
                    for index, value in stats.items()
                    if index in candidates
                },
                "free_check_counts": free_checks,
                "consecutive_free_checks": consecutive,
                "required_free_checks": 3,
            },
        )
        if len(selected) == 4:
            return selected
        time.sleep(poll_seconds)
    raise TimeoutError("GPU wait timed out")


def _run(command: Sequence[str], status_path: Path, stage: str, **extra: Any) -> None:
    _atomic_json(status_path, {"stage": stage, "command": list(command), **extra})
    try:
        subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        _atomic_json(
            status_path,
            {
                "stage": "failed",
                "failed_stage": stage,
                "returncode": exc.returncode,
                "command": list(command),
                **extra,
            },
        )
        raise


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _merge_main_results(old_path: Path, expansion_path: Path, output_path: Path) -> int:
    rows = _read_csv(old_path) + _read_csv(expansion_path)
    keys: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row["snapshot_file"]),
            str(row["repeat"]),
            str(row["configuration"]),
            str(row["eval_mode"]),
        )
        if key in keys:
            raise ValueError(f"Duplicate merged result key {key}")
        keys.add(key)
    _atomic_csv(output_path, rows)
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-ids",
        default="auto",
        help="auto selects any four idle physical GPUs; alternatively supply four IDs",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=168.0)
    parser.add_argument(
        "--decoupled-only",
        action="store_true",
        help="Resume only the decoupled experiment without rerunning completed main stages",
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "rq2_followup_status.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    requested_gpu_ids = None
    if args.gpu_ids.strip().lower() != "auto":
        requested_gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
        if len(requested_gpu_ids) != 4:
            raise ValueError("FastWAM follow-up requires four physical GPUs")
        if len(set(requested_gpu_ids)) != len(requested_gpu_ids):
            raise ValueError("Physical GPU IDs must be unique")
    status = args.status_json.expanduser().resolve()
    physical_gpu_ids = _wait_for_gpus(
        requested_gpu_ids,
        status,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )
    logical_gpu_ids = physical_gpu_ids
    python = sys.executable
    gpu_text = ",".join(str(item) for item in logical_gpu_ids)
    model = PROJECT_ROOT / "checkpoints" / "fastwam_release" / "libero_optional_idm_2cam224.pt"
    if not args.decoupled_only:
        expansion_csv = PROJECT_ROOT / "results" / "oracle_watershed_expansion_v2.csv"
        _run(
            [
                python,
                str(PROJECT_ROOT / "oracle" / "counterfactual_oracle.py"),
                "--snapshots-dir", str(PROJECT_ROOT / "snapshots_watershed_expansion_v2"),
                "--model-path", str(model),
                "--eval-mode", "sustained",
                "--output-csv", str(expansion_csv),
                "--num-repeats", "3",
                "--gpu-ids", gpu_text,
                "--resume",
            ],
            status,
            "running_reaching_and_hard_manipulation",
            expected_result_rows=504,
            physical_gpu_ids=physical_gpu_ids,
            logical_cuda_placement=logical_gpu_ids,
        )

        combined_csv = PROJECT_ROOT / "results" / "oracle_watershed_v2_combined.csv"
        merged_count = _merge_main_results(
            PROJECT_ROOT / "results" / "oracle_watershed_smoke.csv",
            expansion_csv,
            combined_csv,
        )
        if merged_count != 1242:
            raise ValueError(
                f"Expected 1242 merged rows for 69 snapshots x 6 configs x 3 repeats, "
                f"found {merged_count}"
            )
        _run(
            [
                python,
                str(PROJECT_ROOT / "oracle" / "analyze_oracle.py"),
                "--results-csv", str(combined_csv),
                "--cost-table", str(PROJECT_ROOT / "configs" / "hardware_cost_table.json"),
                "--output-dir", str(PROJECT_ROOT / "results" / "oracle_watershed_v2"),
            ],
            status,
            "analyzing_rq2_followup",
            merged_result_rows=merged_count,
        )

    decoupled_csv = PROJECT_ROOT / "results" / "oracle_decoupled_core_v1.csv"
    _run(
        [
            python,
            str(PROJECT_ROOT / "oracle" / "counterfactual_oracle.py"),
            "--snapshots-dir", str(PROJECT_ROOT / "snapshots_decoupled_core_v1"),
            "--config-grid", str(PROJECT_ROOT / "configs" / "oracle_decoupled_grid.json"),
            "--model-path", str(model),
            "--eval-mode", "sustained",
            "--output-csv", str(decoupled_csv),
            "--num-repeats", "1",
            "--gpu-ids", gpu_text,
            "--resume",
        ],
        status,
        "running_decoupled_grid",
        expected_result_rows=200,
        physical_gpu_ids=physical_gpu_ids,
        logical_cuda_placement=logical_gpu_ids,
    )
    decoupled_output = PROJECT_ROOT / "results" / "oracle_decoupled_v1"
    _run(
        [
            python,
            str(PROJECT_ROOT / "oracle" / "analyze_decoupled.py"),
            "--results-csv", str(decoupled_csv),
            "--output-dir", str(decoupled_output),
        ],
        status,
        "analyzing_decoupled_grid",
    )
    _atomic_json(
        status,
        {
            "stage": "complete",
            "main_analysis": str(PROJECT_ROOT / "results" / "oracle_watershed_v2" / "oracle_analysis.json"),
            "decoupled_analysis": str(decoupled_output / "decoupled_axis_analysis.json"),
        },
    )


if __name__ == "__main__":
    main()
