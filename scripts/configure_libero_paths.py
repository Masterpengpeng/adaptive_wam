#!/usr/bin/env python3
"""Generate a LIBERO config whose paths match this project checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_config(project_root: Path) -> dict[str, Path]:
    project_root = project_root.expanduser().resolve()
    benchmark_root = project_root / "repos/LIBERO/libero/libero"
    paths = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": project_root / "outputs/baseline/EXP-B000/no_demonstrations",
        "assets": benchmark_root / "assets",
    }
    for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
        if not paths[key].is_dir():
            raise FileNotFoundError(f"Missing LIBERO {key} directory: {paths[key]}")
    paths["datasets"].mkdir(parents=True, exist_ok=True)
    return paths


def write_config(project_root: Path, output: Path) -> Path:
    paths = build_config(project_root)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        f"{key}: {json.dumps(str(value))}\n" for key, value in paths.items()
    )
    output.write_text(content, encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / ".runtime/libero/config.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = write_config(args.project_root, args.output)
    print(f"LIBERO config written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
