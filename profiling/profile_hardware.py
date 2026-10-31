"""Measure parameterized FastWAM latency and peak GPU memory for H/D/W configs.

Example:
    python profiling/profile_hardware.py \
        --model-path checkpoints/fastwam_release/libero_optional_idm_2cam224.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name == "torch" or (exc.name and exc.name.startswith("torch.")):
        raise SystemExit(
            "Unable to import a complete PyTorch installation. Use the project "
            "environment instead of (base):\n"
            "  ./.conda/fastwam/bin/python profiling/profile_hardware.py "
            "--model-path checkpoints/fastwam_release/"
            "libero_optional_idm_2cam224.pt --gpu-ids 0,1,2,3"
        ) from exc
    raise

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collect_counterfactual_data import WorkerContext


DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "hardware_cost_table.json"
HORIZONS = (2, 4, 8)
DEPTHS = (2, 4, 8, 16)
WIDTHS = (1, 2, 4)
FIXED_FULL_CONFIG = (8, 16, 4)
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


@dataclass(frozen=True)
class ProfileConfig:
    H: int
    D: int
    W: int
    E: int

    def validate(self) -> None:
        if self.H == 0:
            if self.D != 0 or self.W != 1:
                raise ValueError("Action-only config must be H=0, D=0, W=1")
        elif self.H not in HORIZONS or self.D not in DEPTHS or self.W not in WIDTHS:
            raise ValueError(f"Unsupported profile configuration: {self}")
        if self.E <= 0:
            raise ValueError("E must be positive")

    def as_dict(self) -> dict[str, int]:
        return {"H": self.H, "D": self.D, "W": self.W, "E": self.E}


def iter_configurations(execution_chunk: int = 10) -> list[ProfileConfig]:
    """Return the 37 requested configurations in deterministic order."""
    if execution_chunk <= 0:
        raise ValueError("execution_chunk must be positive")
    configurations = [ProfileConfig(0, 0, 1, execution_chunk)]
    configurations.extend(
        ProfileConfig(horizon, depth, width, execution_chunk)
        for horizon in HORIZONS
        for depth in DEPTHS
        for width in WIDTHS
    )
    return configurations


def make_dummy_observation(
    reference_observation: dict[str, Any],
    *,
    height: int = IMAGE_HEIGHT,
    width: int = IMAGE_WIDTH,
) -> dict[str, Any]:
    """Build a synthetic two-camera LIBERO observation in its raw input format."""
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        reference_image = reference_observation.get(key)
        if not isinstance(reference_image, np.ndarray) or reference_image.ndim != 3:
            raise ValueError(
                f"Reference observation must provide HWC NumPy image {key!r}"
            )
    result = dict(reference_observation)
    result["agentview_image"] = np.zeros((height, width, 3), dtype=np.uint8)
    result["robot0_eye_in_hand_image"] = np.zeros((height, width, 3), dtype=np.uint8)
    result["robot0_eef_pos"] = np.zeros((3,), dtype=np.float32)
    result["robot0_eef_quat"] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    gripper_shape = np.asarray(reference_observation["robot0_gripper_qpos"]).shape
    result["robot0_gripper_qpos"] = np.zeros(gripper_shape, dtype=np.float32)
    return result


def _synchronize_devices(devices: Sequence[torch.device]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def profile_configuration(
    config: ProfileConfig,
    forward: Callable[[], Any],
    cuda_devices: Sequence[torch.device],
    *,
    warmup: int = 10,
    iterations: int = 50,
) -> dict[str, int | float]:
    """Measure end-to-end inference with CUDA events on all participating GPUs.

    Per-iteration latency is the maximum elapsed CUDA-event time across the
    model's participating devices. Synchronization before and after each
    iteration ensures event results include completed GPU work, including the
    cross-device FastWAM stages.
    """
    config.validate()
    if not cuda_devices:
        raise ValueError("At least one CUDA device is required")
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if any(device.type != "cuda" for device in cuda_devices):
        raise ValueError("CUDA-event profiling requires CUDA devices")

    for device in cuda_devices:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for _ in range(warmup):
            result = forward()
            del result
        _synchronize_devices(cuda_devices)

        elapsed_ms: list[float] = []
        for _ in range(iterations):
            _synchronize_devices(cuda_devices)
            starts = [torch.cuda.Event(enable_timing=True) for _ in cuda_devices]
            ends = [torch.cuda.Event(enable_timing=True) for _ in cuda_devices]
            for event, device in zip(starts, cuda_devices):
                with torch.cuda.device(device):
                    event.record()

            result = forward()

            for event, device in zip(ends, cuda_devices):
                with torch.cuda.device(device):
                    event.record()
            del result
            _synchronize_devices(cuda_devices)
            elapsed_ms.append(
                max(start.elapsed_time(end) for start, end in zip(starts, ends))
            )

        peak_memory_bytes = max(
            torch.cuda.max_memory_allocated(device) for device in cuda_devices
        )
        return {
            **config.as_dict(),
            "latency_ms_mean": statistics.fmean(elapsed_ms),
            "latency_ms_std": statistics.stdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0,
            "peak_vram_mb": peak_memory_bytes / (1024.0 * 1024.0),
        }
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA out of memory while profiling "
            f"H={config.H}, D={config.D}, W={config.W}, E={config.E}"
        ) from exc
    finally:
        gc.collect()
        for device in cuda_devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


def add_relative_compute(
    results: list[dict[str, int | float]],
) -> None:
    """Set relative_compute to measured latency divided by Fixed-Full latency."""
    full = next(
        (
            row
            for row in results
            if (row["H"], row["D"], row["W"]) == FIXED_FULL_CONFIG
        ),
        None,
    )
    if full is None:
        raise ValueError(f"Missing Fixed-Full reference config {FIXED_FULL_CONFIG}")
    reference_latency = float(full["latency_ms_mean"])
    if not np.isfinite(reference_latency) or reference_latency <= 0:
        raise ValueError("Fixed-Full latency must be finite and positive")
    for row in results:
        row["relative_compute"] = float(row["latency_ms_mean"]) / reference_latency


def render_markdown_table(results: Sequence[dict[str, int | float]]) -> str:
    """Render measured configurations as a concise Markdown comparison table."""
    lines = [
        "| H | D | W | Latency (ms, mean ± std) | Peak VRAM (MB) | Relative compute |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['H']} | {row['D']} | {row['W']} | "
            f"{float(row['latency_ms_mean']):.3f} ± "
            f"{float(row['latency_ms_std']):.3f} | "
            f"{float(row['peak_vram_mb']):.1f} | "
            f"{float(row['relative_compute']):.4f} |"
        )
    return "\n".join(lines)


def save_cost_table(
    output_path: Path,
    *,
    device_names: Sequence[str],
    results: list[dict[str, int | float]],
    model_path: Path,
    warmup: int,
    iterations: int,
) -> None:
    """Write results atomically as a structured JSON table."""
    add_relative_compute(results)
    payload = {
        "schema_version": 1,
        "model_path": str(model_path),
        "device": list(device_names),
        "precision": "FP16",
        "input": {
            "camera_count": 2,
            "camera_shape": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
            "camera_dtype": "uint8",
            "execution_chunk": int(results[0]["E"]),
        },
        "protocol": {
            "warmup_iterations": warmup,
            "measurement_iterations": iterations,
            "timer": "torch.cuda.Event(enable_timing=True), synchronized per GPU",
            "latency_aggregation": "maximum CUDA event elapsed time across model GPUs",
            "peak_vram_definition": (
                "maximum per-GPU torch.cuda.max_memory_allocated across model GPUs"
            ),
            "relative_compute_reference": {
                "name": "Fixed-Full",
                "H": FIXED_FULL_CONFIG[0],
                "D": FIXED_FULL_CONFIG[1],
                "W": FIXED_FULL_CONFIG[2],
                "definition": "measured mean latency divided by Fixed-Full mean latency",
            },
        },
        "configurations": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def _parse_gpu_ids(value: str) -> list[int]:
    try:
        gpu_ids = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("--gpu-ids must be comma-separated CUDA indices") from exc
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError("--gpu-ids must contain exactly four distinct IDs")
    return gpu_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
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
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--execution-chunk", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--direct-depth", type=int, default=10)
    parser.add_argument("--fastwam-root", type=Path, default=PROJECT_ROOT / "repos" / "FastWAM")
    parser.add_argument("--libero-root", type=Path, default=PROJECT_ROOT / "repos" / "LIBERO")
    parser.add_argument("--task-config", default="libero_optional_idm_2cam224_1e-4")
    parser.add_argument("--config-name", default="sim_libero")
    parser.add_argument("--dataset-stats-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be non-negative and --iterations positive")
    if args.execution_chunk <= 0 or args.direct_depth <= 0:
        raise ValueError("--execution-chunk and --direct-depth must be positive")
    if args.task_id < 0 or args.seed < 0:
        raise ValueError("--task-id and --seed must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU is required for hardware profiling")

    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"FastWAM checkpoint does not exist: {model_path}")
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if any(gpu_id < 0 or gpu_id >= torch.cuda.device_count() for gpu_id in gpu_ids):
        raise ValueError(
            f"Invalid GPU IDs {gpu_ids}; visible device count is {torch.cuda.device_count()}"
        )
    devices = [torch.device(f"cuda:{gpu_id}") for gpu_id in gpu_ids]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from real_fastwam_adapter import RealFastWAMAdapter

    options: dict[str, Any] = {
        "model_path": str(model_path),
        "fastwam_root": str(args.fastwam_root.expanduser().resolve()),
        "libero_root": str(args.libero_root.expanduser().resolve()),
        "task_config": args.task_config,
        "config_name": args.config_name,
        "task_suite": args.task_suite,
        "direct_depth": args.direct_depth,
        "max_steps": 400,
        "cuda_devices": [str(device) for device in devices],
    }
    if args.dataset_stats_path is not None:
        options["dataset_stats_path"] = str(args.dataset_stats_path.expanduser().resolve())

    adapter = RealFastWAMAdapter(
        WorkerContext(
            device=str(devices[0]),
            task_suite=args.task_suite,
            options=options,
        )
    )
    results: list[dict[str, int | float]] = []
    try:
        reference_observation = adapter.start_task(args.task_id, args.seed)
        dummy_observation = make_dummy_observation(reference_observation)
        for config in iter_configurations(args.execution_chunk):
            if config.E > adapter.action_horizon:
                raise ValueError(
                    f"--execution-chunk={config.E} exceeds model action horizon "
                    f"{adapter.action_horizon}"
                )
            print(
                f"Profiling H={config.H}, D={config.D}, W={config.W}, E={config.E}",
                flush=True,
            )
            row = profile_configuration(
                config,
                lambda config=config: adapter.parameterized_wam(
                    dummy_observation, config.H, config.D, config.W, config.E
                ),
                devices,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            results.append(row)
    finally:
        adapter.close()

    device_names = [torch.cuda.get_device_name(device) for device in devices]
    save_cost_table(
        args.output.expanduser().resolve(),
        device_names=device_names,
        results=results,
        model_path=model_path,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print()
    print(render_markdown_table(results))
    print(f"\nSaved hardware cost table: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
