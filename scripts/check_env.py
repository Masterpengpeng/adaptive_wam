#!/usr/bin/env python3
"""Smoke-test a LIBERO task and save its two EGL-rendered camera observations."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from real_fastwam_adapter import (
    FASTWAM_ROOT,
    LIBERO_ROOT,
    _add_project_import_paths,
)


CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


def _camera_image(observation: dict[str, Any], key: str, resolution: int) -> np.ndarray:
    if key not in observation:
        raise KeyError(f"LIBERO observation is missing camera key {key!r}")
    image = np.asarray(observation[key])
    expected_shape = (resolution, resolution, 3)
    if image.shape != expected_shape:
        raise ValueError(
            f"Camera {key!r} returned shape {image.shape}, expected {expected_shape}"
        )
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number):
            raise TypeError(f"Camera {key!r} must return numeric pixels, got {image.dtype}")
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
            raise ValueError(f"Camera {key!r} has invalid pixel values")
        image = image.astype(np.uint8)
    if not np.any(image):
        raise ValueError(f"Camera {key!r} returned an all-zero frame")
    return np.ascontiguousarray(image)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--fastwam-root", type=Path, default=FASTWAM_ROOT)
    parser.add_argument("--libero-root", type=Path, default=LIBERO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "phase1"
        / f"env_smoke_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.task_id < 0 or args.seed < 0 or args.resolution <= 0:
        raise ValueError("task-id and seed must be non-negative; resolution must be positive")

    paths = {
        "fastwam_root": str(args.fastwam_root.expanduser().resolve()),
        "libero_root": str(args.libero_root.expanduser().resolve()),
    }
    _add_project_import_paths(paths)
    libero = importlib.import_module("libero.libero")
    eval_helpers = importlib.import_module("eval_libero_single")

    benchmark = libero.benchmark.get_benchmark_dict()
    suite_factory = benchmark.get(args.task_suite)
    if suite_factory is None:
        raise ValueError(f"Unknown LIBERO task suite: {args.task_suite}")
    task = suite_factory().get_task(args.task_id)
    env, task_description = eval_helpers.get_libero_env(
        task, args.resolution, args.seed
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        env.close()
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        observation = env.reset()
        init_path = (
            Path(libero.get_libero_path("init_states"))
            / task.problem_folder
            / task.init_states_file
        )
        initial_states = torch.load(init_path, map_location="cpu", weights_only=False)
        if len(initial_states) == 0:
            raise RuntimeError(f"No LIBERO initial states found in {init_path}")
        observation = env.set_init_state(initial_states[args.seed % len(initial_states)])

        images = {
            key: _camera_image(observation, key, args.resolution)
            for key in CAMERA_KEYS
        }
        for key, image in images.items():
            Image.fromarray(image, mode="RGB").save(
                output_dir / f"{key}.png", format="PNG"
            )
        Image.fromarray(np.concatenate(tuple(images.values()), axis=1), mode="RGB").save(
            output_dir / "two_camera_preview.png", format="PNG"
        )
        report = {
            "status": "passed",
            "task_suite": args.task_suite,
            "task_id": args.task_id,
            "task_description": task_description,
            "seed": args.seed,
            "render_backend": os.environ["MUJOCO_GL"],
            "resolution": args.resolution,
            "camera_images": {
                key: {"shape": list(image.shape), "dtype": str(image.dtype)}
                for key, image in images.items()
            },
            "initial_state_count": len(initial_states),
            "initial_state_index": args.seed % len(initial_states),
            "initial_observation_success": bool(env.check_success()),
            "artifacts": {
                "agentview": "agentview_image.png",
                "wrist": "robot0_eye_in_hand_image.png",
                "side_by_side": "two_camera_preview.png",
            },
        }
        (output_dir / "env_check.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        env.close()

    print(json.dumps(report, indent=2))
    print(f"Saved EGL render checks to {output_dir}")


if __name__ == "__main__":
    main()
