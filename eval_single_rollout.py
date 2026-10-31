#!/usr/bin/env python3
"""Run and record one real FastWAM episode in a LIBERO task."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Protocol, cast

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from collect_counterfactual_data import InferenceConfig, WorkerContext
from real_fastwam_adapter import FASTWAM_ROOT, LIBERO_ROOT, create_adapter


PROJECT_ROOT = Path(__file__).resolve().parent


class _VideoWriter(Protocol):
    def append_data(self, image: np.ndarray) -> None: ...

    def close(self) -> None: ...


class _ImageIo(Protocol):
    def get_writer(
        self,
        uri: str,
        *,
        fps: int,
        codec: str,
        macro_block_size: int,
    ) -> _VideoWriter: ...


def _video_frame(observation: dict[str, Any], adapter: Any) -> np.ndarray:
    camera_images = adapter.eval_helpers.get_libero_image(observation)
    primary = np.asarray(camera_images["image"])
    wrist = np.asarray(camera_images["wrist_image"])
    if primary.ndim != 3 or wrist.ndim != 3:
        raise ValueError(
            f"Expected HWC RGB camera frames, got {primary.shape} and {wrist.shape}"
        )
    if primary.shape[2] != 3 or wrist.shape[2] != 3:
        raise ValueError(
            f"Expected three-channel RGB camera frames, got {primary.shape} and {wrist.shape}"
        )
    if primary.shape[:2] != wrist.shape[:2]:
        raise ValueError(
            f"Camera frame dimensions differ: {primary.shape} vs {wrist.shape}"
        )
    frame = np.concatenate((primary, wrist), axis=1)
    if frame.dtype != np.uint8:
        if not np.issubdtype(frame.dtype, np.number):
            raise TypeError(f"Video frame must have numeric pixels, got {frame.dtype}")
        if not np.isfinite(frame).all() or frame.min() < 0 or frame.max() > 255:
            raise ValueError("Video frame contains invalid pixel values")
        frame = frame.astype(np.uint8)
    return np.ascontiguousarray(frame)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        "--model-path",
        type=Path,
        default=PROJECT_ROOT
        / "checkpoints"
        / "fastwam_release"
        / "libero_optional_idm_2cam224.pt",
    )
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--H", type=int, default=0)
    parser.add_argument("--D", type=int, default=0)
    parser.add_argument("--W", type=int, default=0)
    parser.add_argument("--E", type=int, default=10)
    parser.add_argument("--direct-depth", type=int, default=10)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--fastwam-root", type=Path, default=FASTWAM_ROOT)
    parser.add_argument("--libero-root", type=Path, default=LIBERO_ROOT)
    parser.add_argument("--task-config", default="libero_optional_idm_2cam224_1e-4")
    parser.add_argument("--config-name", default="sim_libero")
    parser.add_argument("--dataset-stats-path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "phase1"
        / f"single_rollout_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.task_id < 0 or args.seed < 0 or args.max_steps <= 0 or args.fps <= 0:
        raise ValueError("task-id/seed must be non-negative; max-steps/fps must be positive")
    if args.direct_depth <= 0:
        raise ValueError("--direct-depth must be positive")
    config = InferenceConfig(H=args.H, D=args.D, W=args.W, E=args.E)
    config.validate()
    if config.E > args.max_steps:
        raise ValueError("--E cannot exceed --max-steps")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for real FastWAM inference")

    try:
        gpu_ids = [int(part.strip()) for part in args.gpu_ids.split(",")]
    except ValueError as exc:
        raise ValueError("--gpu-ids must be a comma-separated list of CUDA indices") from exc
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError("Real FastWAM inference requires exactly four distinct CUDA IDs")
    if any(gpu_id < 0 or gpu_id >= torch.cuda.device_count() for gpu_id in gpu_ids):
        raise ValueError(f"Invalid CUDA IDs {gpu_ids}; visible device count is {torch.cuda.device_count()}")

    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"FastWAM checkpoint does not exist: {model_path}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    devices = [f"cuda:{gpu_id}" for gpu_id in gpu_ids]
    imageio = cast(_ImageIo, importlib.import_module("imageio.v2"))
    options: dict[str, Any] = {
        "model_path": str(model_path),
        "fastwam_root": str(args.fastwam_root.expanduser().resolve()),
        "libero_root": str(args.libero_root.expanduser().resolve()),
        "task_config": args.task_config,
        "config_name": args.config_name,
        "task_suite": args.task_suite,
        "direct_depth": args.direct_depth,
        "max_steps": args.max_steps,
        "cuda_devices": devices,
    }
    if args.dataset_stats_path is not None:
        options["dataset_stats_path"] = str(
            args.dataset_stats_path.expanduser().resolve()
        )

    adapter = create_adapter(
        WorkerContext(device=devices[0], task_suite=args.task_suite, options=options)
    )
    video_path = output_dir / "episode.mp4"
    writer = None
    try:
        if adapter.model.torch_dtype != torch.float16:
            raise RuntimeError(
                f"Expected FP16 FastWAM model, got {adapter.model.torch_dtype}"
            )
        observation = adapter.start_task(args.task_id, args.seed)
        writer = imageio.get_writer(
            str(video_path),
            fps=args.fps,
            codec="libx264",
            macro_block_size=1,
        )
        writer.append_data(_video_frame(observation, adapter))

        inference_latencies_ms: list[float] = []
        episode_return = 0.0
        env_steps = 0
        success = bool(adapter.is_success(observation, {}))
        terminated = success
        while env_steps < args.max_steps and not success and not terminated:
            inference_started = time.perf_counter()
            actions = adapter.infer_actions(observation, config)
            for device in devices:
                torch.cuda.synchronize(torch.device(device))
            inference_latencies_ms.append(
                (time.perf_counter() - inference_started) * 1000.0
            )
            actions = np.asarray(actions)
            if actions.ndim != 2 or actions.shape[0] == 0:
                raise ValueError(f"Expected a non-empty [time, action] chunk, got {actions.shape}")
            if actions.shape[0] < config.E:
                raise ValueError(
                    f"Policy returned {actions.shape[0]} actions, fewer than E={config.E}"
                )

            commit = min(config.E, args.max_steps - env_steps)
            for action in actions[:commit]:
                step = adapter.step(action)
                observation = step.observation
                episode_return += float(step.reward)
                env_steps += 1
                success = bool(adapter.is_success(observation, step.info))
                terminated = bool(step.terminated or step.truncated)
                writer.append_data(_video_frame(observation, adapter))
                if success or terminated or env_steps >= args.max_steps:
                    break

        success = bool(success or adapter.is_success(observation, {}))
        if inference_latencies_ms:
            latency_summary = {
                "count": len(inference_latencies_ms),
                "mean_ms": float(np.mean(inference_latencies_ms)),
                "median_ms": float(np.median(inference_latencies_ms)),
                "min_ms": float(np.min(inference_latencies_ms)),
                "max_ms": float(np.max(inference_latencies_ms)),
            }
        else:
            latency_summary = {
                "count": 0,
                "mean_ms": None,
                "median_ms": None,
                "min_ms": None,
                "max_ms": None,
            }
        report = {
            "status": "completed",
            "task_suite": args.task_suite,
            "task_id": args.task_id,
            "task_description": adapter.task_description,
            "seed": args.seed,
            "checkpoint": str(model_path),
            "precision": "FP16",
            "gpus": gpu_ids,
            "config": config.as_dict(),
            "direct_path_denoising_steps": (
                args.direct_depth if config.H == 0 else None
            ),
            "max_steps": args.max_steps,
            "episode_length": env_steps,
            "episode_return": episode_return,
            "success": success,
            "terminated": terminated,
            "timed_out": bool(not success and not terminated and env_steps >= args.max_steps),
            "inference_calls": len(inference_latencies_ms),
            "inference_latency_ms": latency_summary,
            "video": video_path.name,
            "video_fps": args.fps,
            "video_frame_count": env_steps + 1,
            "video_frame_shape": list(_video_frame(observation, adapter).shape),
        }
    finally:
        if writer is not None:
            writer.close()
        adapter.close()
        gc.collect()
        for gpu_id in gpu_ids:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()

    (output_dir / "episode.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Saved episode report and video to {output_dir}")


if __name__ == "__main__":
    main()
