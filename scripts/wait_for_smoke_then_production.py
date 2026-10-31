"""Wait for the smoke analysis gate, then launch the production Oracle pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_smoke_analysis(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report["rq2"]["phase_distribution"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot read smoke analysis {path}: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError("rq2.phase_distribution must be a list")
    by_phase = {
        str(row.get("phase")): row
        for row in rows
        if isinstance(row, Mapping)
    }
    checks: dict[str, Any] = {}
    for phase in ("precontact", "manipulation"):
        row = by_phase.get(phase)
        if row is None:
            raise ValueError(f"Smoke analysis is missing phase {phase!r}")
        try:
            state_count = int(row["state_count"])
            mean_e = float(row["mean_E"])
            mean_d = float(row["mean_D"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed {phase} phase summary: {row}") from exc
        checks[phase] = {
            "state_count": state_count,
            "mean_E": mean_e,
            "mean_D": mean_d,
            "passed": state_count > 0 and mean_e < 10.0 and mean_d > 0.0,
        }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "selected_lambda": report.get("rq2", {}).get("selected_lambda"),
        "best_success_rate": report.get("rq2", {}).get("best_success_rate"),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-analysis", type=Path, required=True)
    parser.add_argument("--smoke-process-pid", type=int, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--production-output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=72.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.poll_seconds <= 0 or args.timeout_hours <= 0:
        raise ValueError("Polling interval and timeout must be positive")
    smoke_analysis = args.smoke_analysis.expanduser().resolve()
    status_json = args.status_json.expanduser().resolve()
    production_output = args.production_output_root.expanduser().resolve()
    deadline = time.monotonic() + args.timeout_hours * 3600.0
    _atomic_json(
        status_json,
        {
            "stage": "waiting_for_smoke",
            "smoke_analysis": str(smoke_analysis),
            "smoke_process_pid": args.smoke_process_pid,
        },
    )

    while not smoke_analysis.is_file():
        if not _process_alive(args.smoke_process_pid):
            payload = {
                "stage": "smoke_failed",
                "reason": "smoke process exited without oracle_analysis.json",
            }
            _atomic_json(status_json, payload)
            raise RuntimeError(payload["reason"])
        if time.monotonic() >= deadline:
            payload = {"stage": "smoke_timeout", "timeout_hours": args.timeout_hours}
            _atomic_json(status_json, payload)
            raise TimeoutError(f"Smoke run exceeded {args.timeout_hours} hours")
        time.sleep(args.poll_seconds)

    gate = validate_smoke_analysis(smoke_analysis)
    plot_path = smoke_analysis.parent / "rq2_phase_distribution.png"
    gate["plot_exists"] = plot_path.is_file() and plot_path.stat().st_size > 0
    gate["stage"] = "smoke_passed" if gate["passed"] and gate["plot_exists"] else "smoke_rejected"
    _atomic_json(status_json, gate)
    if gate["stage"] != "smoke_passed":
        raise RuntimeError(f"Smoke trend gate rejected production launch: {gate}")

    command = [
        sys.executable,
        str(PROJECT_ROOT / "run_oracle_production.py"),
        "--output-root", str(production_output),
        "--model-path", str(PROJECT_ROOT / "checkpoints" / "libero_optional_idm_2cam224.pt"),
        "--gpu-ids", args.gpu_ids,
    ]
    _atomic_json(
        status_json,
        {**gate, "stage": "production_running", "command": command},
    )
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        _atomic_json(
            status_json,
            {**gate, "stage": "production_failed", "returncode": completed.returncode},
        )
        raise SystemExit(completed.returncode)
    _atomic_json(status_json, {**gate, "stage": "production_complete"})


if __name__ == "__main__":
    main()
