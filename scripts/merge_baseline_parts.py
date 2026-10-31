"""Wait for disjoint baseline workers and atomically merge their episode CSVs."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _key(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row["task_suite"]),
        int(row["task_id"]),
        int(row["seed"]),
        str(row["method"]),
    )


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot merge an empty episode set")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError("Baseline part CSV schemas differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _part(value: str) -> tuple[int, Path, int]:
    fields = value.split(":", 2)
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("--part must be PID:CSV_PATH:EXPECTED_ROWS")
    try:
        pid = int(fields[0])
        expected = int(fields[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("part PID/EXPECTED_ROWS must be integers") from exc
    if pid <= 0 or expected <= 0:
        raise argparse.ArgumentTypeError("part PID/EXPECTED_ROWS must be positive")
    return pid, Path(fields[1]), expected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", action="append", type=_part, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=36.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.poll_seconds <= 0 or args.timeout_hours <= 0:
        raise ValueError("Polling interval and timeout must be positive")
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    completed: list[list[dict[str, str]] | None] = [None] * len(args.part)
    while any(rows is None for rows in completed):
        for index, (pid, source_path, expected_rows) in enumerate(args.part):
            if completed[index] is not None:
                continue
            rows = _read(source_path.expanduser().resolve())
            if len(rows) >= expected_rows:
                if len(rows) != expected_rows:
                    raise RuntimeError(
                        f"{source_path} has {len(rows)} rows, expected exactly {expected_rows}"
                    )
                completed[index] = rows
                continue
            if not _process_alive(pid):
                raise RuntimeError(
                    f"Baseline worker {pid} exited with {len(rows)}/{expected_rows} rows "
                    f"in {source_path}"
                )
        if all(rows is not None for rows in completed):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Baseline part merge timed out")
        time.sleep(args.poll_seconds)

    merged: dict[tuple[str, int, int, str], dict[str, str]] = {}
    for part_rows in completed:
        assert part_rows is not None
        for row in part_rows:
            key = _key(row)
            if key in merged:
                raise ValueError(f"Duplicate episode across baseline parts: {key}")
            merged[key] = row
    rows = [merged[key] for key in sorted(merged)]
    _write(args.output_csv.expanduser().resolve(), rows)
    print(f"Merged {len(rows)} episode rows into {args.output_csv.expanduser().resolve()}")


if __name__ == "__main__":
    main()
