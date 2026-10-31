"""Collect state-level counterfactual WAM rollouts using simulator adapters.

An adapter factory passed as ``module:function`` must return an object that
implements ``start_task``, ``capture_checkpoint``, ``restore_checkpoint``,
``infer_actions``, ``step``, ``is_success`` and ``close``. See README.md for the
contract and the verified LIBERO checkpoint helpers below.
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import queue
import random
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parent
FASTWAM_ROOT = PROJECT_ROOT / "repos" / "FastWAM"
LIBERO_ROOT = PROJECT_ROOT / "repos" / "LIBERO"
_WORKER_ADAPTER: CounterfactualAdapter | None = None
_WORKER_DEVICE = "cpu"
_WORKER_GAMMA = 1.0
_WORKER_MAX_STEPS = 400


@dataclass(frozen=True, order=True)
class InferenceConfig:
    H: int
    D: int
    W: int
    E: int

    def validate(self) -> None:
        if self.H < 0 or self.E <= 0:
            raise ValueError("H must be non-negative and E must be positive")
        if self.H == 0:
            if self.D != 0 or self.W != 0:
                raise ValueError("H=0 configurations must use D=0 and W=0")
        elif self.D <= 0 or self.W <= 0:
            raise ValueError("D and W must be positive when H is positive")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerContext:
    device: str
    task_suite: str
    options: dict[str, Any]


@dataclass(frozen=True)
class StepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class CounterfactualAdapter(Protocol):
    def start_task(self, task_id: int, seed: int) -> Any: ...

    def capture_checkpoint(self) -> Any: ...

    def restore_checkpoint(self, checkpoint: Any) -> Any: ...

    def infer_actions(
        self, observation: Any, config: InferenceConfig
    ) -> np.ndarray | torch.Tensor: ...

    def step(self, action: Any) -> StepResult: ...

    def is_success(self, observation: Any, info: dict[str, Any]) -> bool: ...

    def close(self) -> None: ...


def capture_libero_checkpoint(env: Any) -> np.ndarray:
    """Copy the flattened MuJoCo state exposed by LIBERO's ControlEnv."""
    getter = getattr(env, "get_sim_state", None)
    if callable(getter):
        state = getter()
    else:
        sim = getattr(env, "sim", None)
        if sim is None:
            raise TypeError("LIBERO environment must expose get_sim_state() or sim")
        state = sim.get_state().flatten()
    return np.asarray(state).copy()


def restore_libero_checkpoint(env: Any, checkpoint: Any) -> Any:
    """Restore a LIBERO MuJoCo state and regenerate its observation."""
    state = np.asarray(checkpoint)
    restore = getattr(env, "set_init_state", None)
    if callable(restore):
        return restore(state)
    sim = getattr(env, "sim", None)
    if sim is None:
        raise TypeError("LIBERO environment must expose set_init_state() or sim")
    sim.set_state_from_flattened(state)
    sim.forward()
    post_process = getattr(env, "_post_process", None)
    if callable(post_process):
        post_process()
    update_observables = getattr(env, "_update_observables", None)
    if callable(update_observables):
        update_observables(force=True)
    get_observations = getattr(env, "_get_observations", None)
    if not callable(get_observations):
        raise TypeError("Cannot regenerate observation after restoring MuJoCo state")
    return get_observations()


def _capture_rng(device: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if device.startswith("cuda"):
        state["torch_cuda"] = torch.cuda.get_rng_state(torch.device(device)).cpu()
    return state


def _restore_rng(state: dict[str, Any], device: str) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.startswith("cuda"):
        cuda_state = state.get("torch_cuda")
        if cuda_state is None:
            raise ValueError(
                "Checkpoint has no CUDA RNG state; scan and rollout devices must match"
            )
        torch.cuda.set_rng_state(cuda_state.cpu(), torch.device(device))


def _load_factory(
    specification: str,
) -> Callable[[WorkerContext], CounterfactualAdapter]:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("--adapter must have the form python.module:factory")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError(f"Adapter factory {specification!r} is not callable")
    return factory


def _normalize_actions(actions: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()
    array = np.asarray(actions)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim < 2 or array.shape[0] == 0:
        raise ValueError("infer_actions must return a non-empty action chunk")
    if not np.isfinite(array).all():
        raise ValueError("infer_actions returned NaN or infinite actions")
    return array


def _advance_one(
    adapter: CounterfactualAdapter,
    observation: Any,
    config: InferenceConfig,
    action_chunk: np.ndarray | None,
    action_index: int,
) -> tuple[Any, np.ndarray | None, int, StepResult, int]:
    if action_chunk is None or action_index >= len(action_chunk):
        action_chunk = _normalize_actions(adapter.infer_actions(observation, config))
        action_chunk = action_chunk[: config.E]
        action_index = 0
        inference_calls = 1
    else:
        inference_calls = 0

    result = adapter.step(action_chunk[action_index])
    required_fields = ("observation", "reward", "terminated", "truncated", "info")
    if not all(hasattr(result, field) for field in required_fields):
        raise TypeError(
            "adapter.step() must return an object with observation, reward, "
            "terminated, truncated, and info fields"
        )
    result = StepResult(
        observation=result.observation,
        reward=float(result.reward),
        terminated=bool(result.terminated),
        truncated=bool(result.truncated),
        info=dict(result.info),
    )
    return (
        result.observation,
        action_chunk,
        action_index + 1,
        result,
        inference_calls,
    )


def run_rollout(
    adapter: CounterfactualAdapter,
    observation: Any,
    config: InferenceConfig,
    max_steps: int,
    gamma: float = 1.0,
    on_state: Callable[[int, Any], None] | None = None,
) -> dict[str, Any]:
    """Run from the current simulator state until termination or the step limit."""
    config.validate()
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between zero and one")

    episode_return = 0.0
    steps = 0
    inference_calls = 0
    action_chunk = None
    action_index = 0
    success = False
    while steps < max_steps:
        if on_state is not None:
            on_state(steps, observation)
        (
            observation,
            action_chunk,
            action_index,
            step_result,
            new_inference_calls,
        ) = _advance_one(
            adapter,
            observation,
            config,
            action_chunk,
            action_index,
        )
        inference_calls += new_inference_calls
        episode_return += (gamma**steps) * float(step_result.reward)
        steps += 1
        success = bool(adapter.is_success(observation, step_result.info))
        if success or step_result.terminated or step_result.truncated:
            break

    return {
        "episode_return": float(episode_return),
        "success": success,
        "steps": steps,
        "inference_calls": inference_calls,
    }


def scan_task(
    adapter: CounterfactualAdapter,
    *,
    task_suite: str,
    task_id: int,
    episode_seed: int,
    scan_config: InferenceConfig,
    output_dir: Path,
    scan_stride: int,
    max_steps: int,
    max_states: int | None,
    gamma: float,
    device: str,
) -> list[dict[str, Any]]:
    """Follow a scan policy and persist simulator/RNG snapshots along its path."""
    if scan_stride <= 0 or (max_states is not None and max_states <= 0):
        raise ValueError("scan_stride and max_states must be positive")
    scan_config.validate()
    nodes_dir = output_dir / "states"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    node_records: list[dict[str, Any]] = []

    def save_state(step_index: int, _: Any) -> None:
        node_id = f"{task_suite}_task{task_id}_seed{episode_seed}_step{step_index}"
        relative_path = Path("states") / f"{node_id}.pt"
        target = output_dir / relative_path
        temporary = target.with_suffix(".pt.tmp")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "task_suite": task_suite,
            "task_id": task_id,
            "episode_seed": episode_seed,
            "scan_step": step_index,
            "scan_config": scan_config.as_dict(),
            "checkpoint": adapter.capture_checkpoint(),
            "rng": _capture_rng(device),
        }
        torch.save(payload, temporary)
        os.replace(temporary, target)
        node_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "state_id": node_id,
                "task_suite": task_suite,
                "task_id": task_id,
                "episode_seed": episode_seed,
                "scan_step": step_index,
                "scan_config": scan_config.as_dict(),
                "checkpoint_file": relative_path.as_posix(),
            }
        )

    observation = adapter.start_task(task_id, episode_seed)
    scan_stats = run_rollout(
        adapter,
        observation,
        scan_config,
        max_steps,
        gamma=gamma,
        on_state=lambda step, obs: (
            save_state(step, obs)
            if step % scan_stride == 0
            and (max_states is None or len(node_records) < max_states)
            else None
        ),
    )
    if not node_records:
        raise RuntimeError(
            f"No states scanned for task={task_id}, seed={episode_seed}; "
            f"scan ended after {scan_stats['steps']} steps"
        )
    return node_records


def _worker_initialize(
    adapter_specification: str,
    device: str,
    task_suite: str,
    options: dict[str, Any],
    gamma: float,
    max_steps: int,
) -> None:
    global _WORKER_ADAPTER, _WORKER_DEVICE, _WORKER_GAMMA, _WORKER_MAX_STEPS
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    _WORKER_DEVICE = device
    _WORKER_GAMMA = gamma
    _WORKER_MAX_STEPS = max_steps
    context = WorkerContext(device=device, task_suite=task_suite, options=options)
    _WORKER_ADAPTER = _load_factory(adapter_specification)(context)


def _worker_entry(
    device: str,
    physical_gpu_ids: list[int] | None,
    adapter_specification: str,
    task_suite: str,
    options: dict[str, Any],
    gamma: float,
    max_steps: int,
    work_queue: Any,
    result_queue: Any,
) -> None:
    try:
        if physical_gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(gpu_id) for gpu_id in physical_gpu_ids
            )
            options["cuda_devices"] = [f"cuda:{index}" for index in range(len(physical_gpu_ids))]
        _worker_initialize(
            adapter_specification,
            device,
            task_suite,
            options,
            gamma,
            max_steps,
        )
        while True:
            item = work_queue.get()
            if item is None:
                break
            try:
                result_queue.put(("result", _evaluate_work_item(item)))
            except queue.Empty:
                result_queue.put(("error", traceback.format_exc()))
                break
    except Exception:
        result_queue.put(("error", traceback.format_exc()))
    finally:
        if _WORKER_ADAPTER is not None:
            _WORKER_ADAPTER.close()


def _evaluate_work_item(item: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_ADAPTER is None:
        raise RuntimeError("Counterfactual worker was not initialized")
    node_path = Path(item["output_dir"]) / item["checkpoint_file"]
    node = torch.load(node_path, map_location="cpu", weights_only=False)
    if node.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported state schema in {node_path}")

    adapter = _WORKER_ADAPTER
    adapter.start_task(int(node["task_id"]), int(node["episode_seed"]))
    observation = adapter.restore_checkpoint(node["checkpoint"])
    _restore_rng(node["rng"], _WORKER_DEVICE)
    config = InferenceConfig(**item["config"])
    start_time = time.perf_counter()
    metrics = run_rollout(
        adapter,
        observation,
        config,
        max_steps=_WORKER_MAX_STEPS,
        gamma=_WORKER_GAMMA,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "state_id": item["state_id"],
        "task_suite": node["task_suite"],
        "task_id": int(node["task_id"]),
        "episode_seed": int(node["episode_seed"]),
        "scan_step": int(node["scan_step"]),
        "config": config.as_dict(),
        **metrics,
        "Q": metrics["episode_return"],
        "success_rate": float(metrics["success"]),
        "wall_time_s": time.perf_counter() - start_time,
    }


def _parse_int_list(value: str, name: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not parsed or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError(f"{name} must contain unique integer values")
    return parsed


def _parse_config(value: str) -> InferenceConfig:
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--scan-config must contain integers") from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("--scan-config must contain exactly H,D,W,E")
    config = InferenceConfig(*values)
    config.validate()
    return config


def _resolve_task_ids(value: str, suite_name: str, options: dict[str, Any]) -> list[int]:
    if value.strip().lower() != "all":
        return _parse_int_list(value, "--task-ids")
    module_name, separator, _ = options["adapter"].partition(":")
    if not separator:
        raise ValueError("--adapter must have the form python.module:factory")
    module = importlib.import_module(module_name)
    list_task_ids = getattr(module, "list_task_ids", None)
    if not callable(list_task_ids):
        raise ValueError(
            "--task-ids all requires the adapter module to provide list_task_ids()"
        )
    task_ids = list_task_ids(
        suite_name,
        fastwam_root=options["fastwam_root"],
        libero_root=options["libero_root"],
    )
    if not task_ids:
        raise ValueError(f"No tasks found in LIBERO suite {suite_name}")
    return [int(task_id) for task_id in task_ids]


def _parse_device_groups(
    num_gpus: int | None,
    gpu_ids: str | None,
    model_gpus_per_worker: int,
) -> list[list[int] | None]:
    if model_gpus_per_worker <= 0:
        raise ValueError("--model-gpus-per-worker must be positive")
    if gpu_ids is not None and gpu_ids.strip().lower() == "cpu":
        if num_gpus is not None:
            raise ValueError("--num_gpus cannot be combined with --gpu-ids cpu")
        return [None]
    ids = (
        _parse_int_list(gpu_ids, "--gpu-ids")
        if gpu_ids is not None
        else list(range(num_gpus if num_gpus is not None else model_gpus_per_worker))
    )
    if num_gpus is not None and gpu_ids is not None:
        raise ValueError("Use either --num_gpus or --gpu-ids, not both")
    if num_gpus is not None and num_gpus <= 0:
        raise ValueError("--num_gpus must be positive")
    if any(gpu_id < 0 for gpu_id in ids):
        raise ValueError("GPU IDs must be non-negative")
    if len(ids) < model_gpus_per_worker:
        raise ValueError(
            f"Need at least {model_gpus_per_worker} GPUs per model worker, got {len(ids)}"
        )
    if len(ids) % model_gpus_per_worker:
        unused = ids[len(ids) // model_gpus_per_worker * model_gpus_per_worker :]
        print(
            f"Warning: {len(unused)} GPU(s) will be unused; model replicas require "
            f"{model_gpus_per_worker} GPUs each: {unused}",
            flush=True,
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --gpu-ids cpu or fix the CUDA runtime")
    count = torch.cuda.device_count()
    if any(gpu_id >= count for gpu_id in ids):
        raise ValueError(f"Requested GPU IDs {ids}, but only {count} CUDA devices are visible")
    if len(ids) != len(set(ids)):
        raise ValueError("GPU IDs must be unique")
    workers = len(ids) // model_gpus_per_worker
    return [
        ids[index * model_gpus_per_worker : (index + 1) * model_gpus_per_worker]
        for index in range(workers)
    ]


def _load_cost_grid(path: Path, execution_horizons: Sequence[int]) -> list[InferenceConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("configurations"), list):
        raise ValueError(f"{path} must contain a configurations array")
    configs: set[InferenceConfig] = set()
    for row in payload["configurations"]:
        horizon, depth, width = int(row["H"]), int(row["D"]), int(row["W"])
        for execution_horizon in execution_horizons:
            config = InferenceConfig(horizon, depth, width, execution_horizon)
            config.validate()
            configs.add(config)
    if not configs:
        raise ValueError(f"No configurations found in {path}")
    return sorted(configs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default="real_fastwam_adapter:create_adapter",
        help="Adapter factory as module:function",
    )
    parser.add_argument("--task-suite", required=True)
    parser.add_argument("--task-ids", default="0", help="Comma-separated task IDs")
    parser.add_argument("--seeds", default="0", help="Comma-separated episode seeds")
    parser.add_argument("--cost-table", type=Path, default=Path("cost_table.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num_gpus", "--num-gpus", type=int, default=None)
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated CUDA IDs, or cpu")
    parser.add_argument("--model-gpus-per-worker", type=int, default=4)
    parser.add_argument("--execution-horizons", default="10")
    parser.add_argument("--scan-config", default="0,0,0,10", help="H,D,W,E")
    parser.add_argument("--scan-stride", type=int, default=10)
    parser.add_argument("--max-states-per-episode", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_path", "--model-path", type=Path, default=None)
    parser.add_argument("--fastwam-root", type=Path, default=FASTWAM_ROOT)
    parser.add_argument("--libero-root", type=Path, default=LIBERO_ROOT)
    parser.add_argument("--task-config", default="libero_optional_idm_2cam224_1e-4")
    parser.add_argument("--config-name", default="sim_libero")
    parser.add_argument("--dataset-stats-path", type=Path, default=None)
    parser.add_argument("--direct-depth", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = _parse_int_list(args.seeds, "--seeds")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.task_suite):
        raise ValueError("--task-suite may contain only letters, digits, '_' and '-'")
    execution_horizons = _parse_int_list(
        args.execution_horizons, "--execution-horizons"
    )
    device_groups = _parse_device_groups(
        args.num_gpus,
        args.gpu_ids,
        args.model_gpus_per_worker,
    )
    scan_config = _parse_config(args.scan_config)
    if args.scan_stride <= 0 or args.max_steps <= 0:
        raise ValueError("--scan-stride and --max-steps must be positive")
    if args.max_states_per_episode <= 0:
        raise ValueError("--max-states-per-episode must be positive")
    if not 0 <= args.gamma <= 1:
        raise ValueError("--gamma must be between zero and one")
    if args.direct_depth <= 0:
        raise ValueError("--direct-depth must be positive")
    if args.adapter.startswith("real_fastwam_adapter:"):
        if args.model_path is None:
            raise ValueError("--model_path is required for the real FastWAM adapter")
        if args.model_gpus_per_worker != 4:
            raise ValueError("Real FastWAM adapter currently requires four GPUs per worker")
    _load_factory(args.adapter)
    collection_options = {
        "adapter": args.adapter,
        "fastwam_root": str(args.fastwam_root.resolve()),
        "libero_root": str(args.libero_root.resolve()),
    }
    task_ids = _resolve_task_ids(args.task_ids, args.task_suite, collection_options)
    configs = _load_cost_grid(args.cost_table, execution_horizons)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty collection directory {output_dir}; "
            "choose a new --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    states_manifest = output_dir / "states.jsonl"
    results_path = output_dir / "counterfactual_results.jsonl"
    if states_manifest.exists() or results_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing collection under {output_dir}; "
            "choose a new --output-dir"
        )

    scanner_gpu_ids = device_groups[0]
    scan_device = (
        "cpu" if scanner_gpu_ids is None else f"cuda:{scanner_gpu_ids[0]}"
    )
    if scan_device.startswith("cuda"):
        torch.cuda.set_device(torch.device(f"cuda:{scanner_gpu_ids[0]}"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if scan_device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    context = WorkerContext(
        device=scan_device,
        task_suite=args.task_suite,
        options={
            "task_suite": args.task_suite,
            "cost_table": str(args.cost_table.resolve()),
            "seed": args.seed,
            "max_steps": args.max_steps,
            "direct_depth": args.direct_depth,
            "task_config": args.task_config,
            "config_name": args.config_name,
            "fastwam_root": str(args.fastwam_root.resolve()),
            "libero_root": str(args.libero_root.resolve()),
            "cuda_devices": (
                [f"cuda:{gpu_id}" for gpu_id in scanner_gpu_ids]
                if scanner_gpu_ids is not None
                else ["cpu"]
            ),
            "model_path": (
                str(args.model_path.expanduser().resolve())
                if args.model_path is not None
                else None
            ),
            "dataset_stats_path": (
                str(args.dataset_stats_path.expanduser().resolve())
                if args.dataset_stats_path is not None
                else None
            ),
        },
    )
    scanner: CounterfactualAdapter = _load_factory(args.adapter)(context)
    try:
        with states_manifest.open("w", encoding="utf-8") as manifest:
            all_nodes = []
            for task_id in task_ids:
                for episode_seed in seeds:
                    nodes = scan_task(
                        scanner,
                        task_suite=args.task_suite,
                        task_id=task_id,
                        episode_seed=episode_seed,
                        scan_config=scan_config,
                        output_dir=output_dir,
                        scan_stride=args.scan_stride,
                        max_steps=args.max_steps,
                        max_states=args.max_states_per_episode,
                        gamma=args.gamma,
                        device=scan_device,
                    )
                    for node in nodes:
                        manifest.write(json.dumps(node) + "\n")
                        all_nodes.append(node)
                    manifest.flush()
    finally:
        scanner.close()

    work_items = [
        {
            **node,
            "output_dir": str(output_dir),
            "config": config.as_dict(),
        }
        for node in all_nodes
        for config in configs
    ]
    options = {
        "task_suite": args.task_suite,
        "cost_table": str(args.cost_table.resolve()),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "direct_depth": args.direct_depth,
        "task_config": args.task_config,
        "config_name": args.config_name,
        "fastwam_root": str(args.fastwam_root.resolve()),
        "libero_root": str(args.libero_root.resolve()),
        "model_path": (
            str(args.model_path.expanduser().resolve())
            if args.model_path is not None
            else None
        ),
        "dataset_stats_path": (
            str(args.dataset_stats_path.expanduser().resolve())
            if args.dataset_stats_path is not None
            else None
        ),
    }
    process_context = mp.get_context("spawn")
    work_queue = process_context.Queue()
    result_queue = process_context.Queue()
    for work_item in work_items:
        work_queue.put(work_item)
    for _ in device_groups:
        work_queue.put(None)

    workers = [
        process_context.Process(
            target=_worker_entry,
            args=(
                "cpu" if physical_gpu_ids is None else "cuda:0",
                physical_gpu_ids,
                args.adapter,
                args.task_suite,
                options,
                args.gamma,
                args.max_steps,
                work_queue,
                result_queue,
            ),
        )
        for physical_gpu_ids in device_groups
    ]
    with results_path.open("w", encoding="utf-8") as results_file:
        for worker in workers:
            worker.start()
        completed = 0
        try:
            while completed < len(work_items):
                try:
                    result_type, payload = result_queue.get(timeout=5)
                except Exception:
                    failed = [
                        (worker.pid, worker.exitcode)
                        for worker in workers
                        if worker.exitcode not in (None, 0)
                    ]
                    if failed:
                        raise RuntimeError(f"Counterfactual workers exited: {failed}")
                    continue
                if result_type == "error":
                    raise RuntimeError(f"Counterfactual worker failed:\n{payload}")
                results_file.write(json.dumps(payload) + "\n")
                results_file.flush()
                completed += 1
        finally:
            for worker in workers:
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            work_queue.close()
            result_queue.close()

    print(
        f"Collected {len(all_nodes)} states × {len(configs)} configurations using "
        f"{len(device_groups)} worker(s). "
        f"States: {states_manifest}; results: {results_path}"
    )


if __name__ == "__main__":
    main()
