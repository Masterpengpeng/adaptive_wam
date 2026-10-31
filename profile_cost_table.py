"""Profile WAM inference configurations and write a JSON cost table."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from itertools import product
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch import nn

from collect_counterfactual_data import InferenceConfig, WorkerContext
from model_stub import WorldActionModel


HORIZONS = (0, 2, 4, 8)
DEPTHS = (2, 4, 8, 16)
WIDTHS = (1, 2, 4)
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("cost_table.json")


def iter_configurations() -> list[tuple[int, int, int]]:
    """Return the requested grid, normalizing the no-imagination fast path."""
    configs = [(0, 0, 0)]
    configs.extend(
        (horizon, depth, width)
        for horizon in HORIZONS
        if horizon > 0
        for depth, width in product(DEPTHS, WIDTHS)
    )
    return configs


def estimate_gflops(
    model: nn.Module,
    infer: Any,
) -> float:
    """Estimate multiply-add FLOPs from Linear and Conv2d module executions.

    This conventional MAC-based estimate counts one multiply and one add as
    two FLOPs; elementwise operations and random-number generation are omitted.
    """
    macs = 0
    handles = []

    def count_module(
        module: nn.Module,
        module_inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal macs
        if isinstance(module, nn.Linear):
            input_tensor = module_inputs[0]
            macs += input_tensor.numel() * module.out_features
        elif isinstance(module, nn.Conv2d):
            output_elements = output.numel()
            kernel_elements = module.kernel_size[0] * module.kernel_size[1]
            macs += (
                output_elements
                * (module.in_channels // module.groups)
                * kernel_elements
            )

    try:
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                handles.append(module.register_forward_hook(count_module))
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            infer()
    finally:
        for handle in handles:
            handle.remove()
    return (2 * macs) / 1_000_000_000


def profile_configuration(
    model: nn.Module,
    device: torch.device,
    config: tuple[int, int, int],
    warmup: int,
    iterations: int,
    image_height: int,
    image_width: int,
    infer: Any | None = None,
    sync_devices: tuple[torch.device, ...] = (),
) -> dict[str, int | float]:
    """Measure one configuration using CUDA events and bounded temporary memory."""
    horizon, depth, width = config
    torch.cuda.empty_cache()
    inputs = None
    try:
        if infer is None:
            inputs = {
                "image": torch.randn(
                    1, 3, image_height, image_width, device=device, dtype=torch.float32
                ),
                "state": torch.randn(
                    1, model.state_dim, device=device, dtype=torch.float32
                ),
            }

        def invoke() -> Any:
            if infer is not None:
                return infer(config)
            if inputs is None:
                raise RuntimeError("Stub profiling inputs were not initialized")
            return model.infer(inputs, horizon, depth, width)

        with torch.inference_mode():
            for _ in range(warmup):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    result = invoke()
                del result
            for sync_device in sync_devices or (device,):
                torch.cuda.synchronize(sync_device)

            gflops = estimate_gflops(model, invoke)
            for sync_device in sync_devices or (device,):
                torch.cuda.synchronize(sync_device)

            event_pairs = []
            elapsed_wall_ms = []
            for _ in range(iterations):
                if sync_devices:
                    for sync_device in sync_devices:
                        torch.cuda.synchronize(sync_device)
                    start_time = time.perf_counter()
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        result = invoke()
                    for sync_device in sync_devices:
                        torch.cuda.synchronize(sync_device)
                    elapsed_wall_ms.append((time.perf_counter() - start_time) * 1000)
                else:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        result = invoke()
                    end.record()
                    event_pairs.append((start, end))
                del result
            for sync_device in sync_devices or (device,):
                torch.cuda.synchronize(sync_device)

        latency_ms = (
            sum(elapsed_wall_ms) / iterations
            if sync_devices
            else sum(start.elapsed_time(end) for start, end in event_pairs) / iterations
        )
        return {
            "H": horizon,
            "D": depth,
            "W": width,
            "latency_ms": latency_ms,
            "gflops": gflops,
        }
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"CUDA out of memory while profiling H={horizon}, D={depth}, W={width}"
        ) from exc
    finally:
        if inputs is not None:
            del inputs
        gc.collect()
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_path", "--model-path", type=Path, default=None)
    parser.add_argument("--adapter", default="real_fastwam_adapter:create_adapter")
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--fastwam-root", type=Path, default=Path("repos/FastWAM"))
    parser.add_argument("--libero-root", type=Path, default=Path("repos/LIBERO"))
    parser.add_argument("--task-config", default="libero_optional_idm_2cam224_1e-4")
    parser.add_argument("--config-name", default="sim_libero")
    parser.add_argument("--dataset-stats-path", type=Path, default=None)
    parser.add_argument("--direct-depth", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be non-negative and --iterations must be positive")
    if args.image_height <= 0 or args.image_width <= 0:
        raise ValueError("image dimensions must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: CUDA-event latency profiling is GPU-only")

    if args.direct_depth <= 0:
        raise ValueError("--direct-depth must be positive")
    device = torch.device(args.device)
    gpu_ids: list[int] | None = None
    if args.model_path is not None:
        try:
            gpu_ids = [int(value.strip()) for value in args.gpu_ids.split(",")]
        except ValueError as exc:
            raise ValueError("--gpu-ids must be comma-separated CUDA device indices") from exc
        if len(gpu_ids) != 4 or len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError("--gpu-ids must list exactly four distinct IDs for real FastWAM")
        if any(gpu_id < 0 or gpu_id >= torch.cuda.device_count() for gpu_id in gpu_ids):
            raise ValueError(f"Invalid visible GPU IDs: {gpu_ids}")
        device = torch.device(f"cuda:{gpu_ids[0]}")
    if device.type != "cuda":
        raise ValueError("--device must select a CUDA device")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {device.index} is not available")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    adapter = None
    infer = None
    sync_devices: tuple[torch.device, ...] = ()
    if args.model_path is None:
        model: nn.Module = WorldActionModel().to(device).eval()
    else:
        if gpu_ids is None:
            raise RuntimeError("Real profiling GPU IDs were not initialized")
        devices = [f"cuda:{gpu_id}" for gpu_id in gpu_ids]
        options = {
            "model_path": str(args.model_path.expanduser().resolve()),
            "fastwam_root": str(args.fastwam_root.resolve()),
            "libero_root": str(args.libero_root.resolve()),
            "task_config": args.task_config,
            "config_name": args.config_name,
            "task_suite": args.task_suite,
            "direct_depth": args.direct_depth,
            "max_steps": 400,
            "cuda_devices": devices,
            "dataset_stats_path": (
                str(args.dataset_stats_path.expanduser().resolve())
                if args.dataset_stats_path is not None
                else None
            ),
        }
        factory_module, factory_name = args.adapter.split(":", maxsplit=1)
        factory = getattr(__import__(factory_module, fromlist=[factory_name]), factory_name)
        adapter = factory(
            WorkerContext(device=devices[0], task_suite=args.task_suite, options=options)
        )
        try:
            observation = adapter.start_task(args.task_id, args.seed)
            model = adapter.model
            infer = lambda config: adapter.infer_actions(
                observation,
                InferenceConfig(
                    H=config[0],
                    D=config[1],
                    W=config[2],
                    E=int(adapter.action_horizon),
                ),
            )
            sync_devices = tuple(torch.device(name) for name in devices)
        except Exception:
            adapter.close()
            raise

    configurations = []
    try:
        for config in iter_configurations():
            print(f"Profiling H={config[0]}, D={config[1]}, W={config[2]}", flush=True)
            configurations.append(
                profile_configuration(
                    model,
                    device,
                    config,
                    args.warmup,
                    args.iterations,
                    args.image_height,
                    args.image_width,
                    infer=infer,
                    sync_devices=sync_devices,
                )
            )
    finally:
        if adapter is not None:
            adapter.close()

    cost_table = {
        "device": torch.cuda.get_device_name(device),
        "precision": "FP16",
        "latency_method": (
            "synchronized_wall_clock"
            if sync_devices
            else "cuda_event"
        ),
        "gflops_method": "forward-hook MAC estimate for Linear and Conv2d modules",
        "model_path": (
            str(args.model_path.expanduser().resolve())
            if args.model_path is not None
            else None
        ),
        "direct_path_denoising_steps": (
            args.direct_depth if args.model_path is not None else None
        ),
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(cost_table, output_file, indent=2)
        output_file.write("\n")
    print(f"Wrote {len(configurations)} configurations to {args.output}")


if __name__ == "__main__":
    main()
