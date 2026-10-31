"""Run real FastWAM profiling, counterfactual collection, and oracle analysis."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _worker_gpu_count(num_gpus: int, model_gpus_per_worker: int) -> int:
    if num_gpus <= 0 or model_gpus_per_worker <= 0:
        raise ValueError("GPU counts must be positive")
    workers = num_gpus // model_gpus_per_worker
    if workers <= 0:
        raise ValueError(
            f"A FastWAM model replica requires {model_gpus_per_worker} GPUs"
        )
    return workers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", "--model-path", type=Path, required=True)
    parser.add_argument("--num_gpus", "--num-gpus", type=int, default=10)
    parser.add_argument("--model-gpus-per-worker", type=int, default=4)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-ids", default="all")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--execution-horizons", default="10")
    parser.add_argument("--scan-config", default="0,0,0,10")
    parser.add_argument("--scan-stride", type=int, default=10)
    parser.add_argument("--max-states-per-episode", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--direct-depth", type=int, default=10)
    parser.add_argument("--lambda-cost", type=float, default=0.0)
    parser.add_argument("--budget-ms", type=float, default=None)
    parser.add_argument("--profile-warmup", type=int, default=10)
    parser.add_argument("--profile-iterations", type=int, default=50)
    parser.add_argument("--task-config", default="libero_optional_idm_2cam224_1e-4")
    parser.add_argument("--config-name", default="sim_libero")
    parser.add_argument("--fastwam-root", type=Path, default=PROJECT_ROOT / "repos/FastWAM")
    parser.add_argument("--libero-root", type=Path, default=PROJECT_ROOT / "repos/LIBERO")
    parser.add_argument("--dataset-stats-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/phase2")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--adapter", default="real_fastwam_adapter:create_adapter")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"FastWAM checkpoint does not exist: {model_path}")
    workers = _worker_gpu_count(args.num_gpus, args.model_gpus_per_worker)
    if args.model_gpus_per_worker != 4:
        raise ValueError("The real FastWAM adapter currently requires four GPUs per worker")
    if args.profile_warmup < 0 or args.profile_iterations <= 0:
        raise ValueError("Profile warmup must be >= 0 and iterations must be > 0")

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_root / run_name).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Pipeline output directory is not empty: {output_dir}; choose --run-name"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cost_table = output_dir / "cost_table.json"
    collection_dir = output_dir / "counterfactual"
    analysis_dir = output_dir / "oracle"

    base_args = [
        "--model_path",
        str(model_path),
        "--adapter",
        args.adapter,
        "--task-suite",
        args.task_suite,
        "--fastwam-root",
        str(args.fastwam_root.expanduser().resolve()),
        "--libero-root",
        str(args.libero_root.expanduser().resolve()),
        "--task-config",
        args.task_config,
        "--config-name",
        args.config_name,
        "--direct-depth",
        str(args.direct_depth),
    ]
    if args.dataset_stats_path is not None:
        base_args.extend(
            ["--dataset-stats-path", str(args.dataset_stats_path.expanduser().resolve())]
        )

    profile_command = [
        sys.executable,
        str(PROJECT_ROOT / "profile_cost_table.py"),
        *base_args,
        "--gpu-ids",
        ",".join(str(gpu_id) for gpu_id in range(args.model_gpus_per_worker)),
        "--output",
        str(cost_table),
        "--warmup",
        str(args.profile_warmup),
        "--iterations",
        str(args.profile_iterations),
    ]
    collect_command = [
        sys.executable,
        str(PROJECT_ROOT / "collect_counterfactual_data.py"),
        *base_args,
        "--num_gpus",
        str(args.num_gpus),
        "--model-gpus-per-worker",
        str(args.model_gpus_per_worker),
        "--task-ids",
        args.task_ids,
        "--seeds",
        args.seeds,
        "--cost-table",
        str(cost_table),
        "--output-dir",
        str(collection_dir),
        "--execution-horizons",
        args.execution_horizons,
        "--scan-config",
        args.scan_config,
        "--scan-stride",
        str(args.scan_stride),
        "--max-states-per-episode",
        str(args.max_states_per_episode),
        "--max-steps",
        str(args.max_steps),
        "--gamma",
        str(args.gamma),
    ]
    analyze_command = [
        sys.executable,
        str(PROJECT_ROOT / "analyze_oracle.py"),
        "--results",
        str(collection_dir / "counterfactual_results.jsonl"),
        "--cost-table",
        str(cost_table),
        "--output-dir",
        str(analysis_dir),
        "--lambda-cost",
        str(args.lambda_cost),
    ]
    if args.budget_ms is not None:
        analyze_command.extend(["--budget-ms", str(args.budget_ms)])

    print(
        f"Run directory: {output_dir}\n"
        f"Physical GPUs requested: {args.num_gpus}; model replicas: {workers}; "
        f"GPUs per model: {args.model_gpus_per_worker}. "
        f"{args.num_gpus - workers * args.model_gpus_per_worker} GPU(s) remain unused.",
        flush=True,
    )
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    environment.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Keep all stages sequential so profiling finishes and releases its GPU memory
    # before scanning workers load their own model replicas.
    for command in (profile_command, collect_command, analyze_command):
        print("+ " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
