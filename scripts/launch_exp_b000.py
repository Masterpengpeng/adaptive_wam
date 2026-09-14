#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/EXP-B000/demo.yaml"


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Demo config must contain a mapping.")
    return data


def resolve_project_path(value: str) -> Path:
    return (PROJECT_ROOT / value).resolve()


def validate_gpu_groups(groups: Any, num_gpus: int) -> list[list[int]]:
    if not isinstance(groups, list) or not groups:
        raise ValueError("physical_gpu_groups must be a non-empty list.")
    normalized: list[list[int]] = []
    used: set[int] = set()
    for index, group in enumerate(groups):
        if not isinstance(group, list) or len(group) != 4:
            raise ValueError(f"GPU group {index} must contain exactly four IDs.")
        normalized_group: list[int] = []
        for gpu in group:
            if isinstance(gpu, bool) or not isinstance(gpu, int):
                raise TypeError("GPU IDs must be integers, not booleans or strings.")
            if gpu < 0 or gpu >= num_gpus:
                raise ValueError(f"GPU ID {gpu} is outside [0, {num_gpus}).")
            if gpu in used:
                raise ValueError(f"GPU ID {gpu} is repeated across groups.")
            used.add(gpu)
            normalized_group.append(gpu)
        normalized.append(normalized_group)
    return normalized


def validate_demo_config(config: dict[str, Any]) -> None:
    if config.get("experiment_id") != "EXP-B000":
        raise ValueError("This launcher accepts only EXP-B000.")
    if config.get("mode") != "fastwam_base_first_frame":
        raise ValueError("Only the FastWAM base first-frame path is permitted.")

    evaluation = config["evaluation"]
    required_values = {
        "task_suite": "libero_spatial",
        "task_id": 0,
        "seed": 42,
        "num_trials": 1,
        "sigma_shift": 5.0,
        "num_inference_steps": 10,
        "replan_steps": 10,
        "action_horizon": 32,
        "visualize_future_video": False,
        "compile_action_infer": False,
    }
    for key, expected in required_values.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"evaluation.{key} must be {expected!r}.")

    placement = config["placement"]
    logical = placement["logical_devices"]
    expected_logical = {
        "video": "cuda:0",
        "action": "cuda:1",
        "vae": "cuda:1",
        "text_encoder": ["cuda:2", "cuda:3"],
    }
    if logical != expected_logical:
        raise ValueError(f"Logical placement must be {expected_logical!r}.")
    if config.get("model", {}).get("redirect_common_files") is not False:
        raise ValueError("model.redirect_common_files must be false for official HF assets.")
    validate_gpu_groups(
        placement["physical_gpu_groups"], int(config["manager"]["num_gpus"])
    )

    task_file = resolve_project_path(config["paths"]["task_file"])
    if task_file.read_text(encoding="utf-8").splitlines() != ["libero_spatial,0"]:
        raise ValueError("EXP-B000 task file must contain only `libero_spatial,0`.")


def git_head(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def validate_repositories(config: dict[str, Any]) -> None:
    expected = {
        PROJECT_ROOT / "repos/FastWAM": config["pins"]["fastwam_commit"],
        PROJECT_ROOT / "repos/LIBERO": config["pins"]["libero_commit"],
    }
    for repository, commit in expected.items():
        actual = git_head(repository)
        if actual != commit:
            raise RuntimeError(f"{repository} HEAD is {actual}, expected {commit}.")


def hash_file(path: Path, hash_type: str) -> str:
    if hash_type == "sha256":
        digest = hashlib.sha256()
        prefix = b""
    elif hash_type == "git_blob_sha1":
        digest = hashlib.sha1()
        prefix = f"blob {path.stat().st_size}\0".encode()
    else:
        raise ValueError(f"Unsupported hash_type: {hash_type}")
    digest.update(prefix)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = resolve_project_path(config["paths"]["asset_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "EXP-B000":
        raise ValueError("Asset manifest experiment_id mismatch.")
    if len(manifest.get("asset_classes", [])) != 4:
        raise ValueError("EXP-B000 requires exactly four asset classes.")
    return manifest


def iter_manifest_files(manifest: dict[str, Any]):
    for asset_class in manifest["asset_classes"]:
        identifier = str(asset_class["id"]).lower()
        if "optional" in identifier or "idm" in identifier or "demonstration" in identifier:
            raise ValueError(f"Prohibited asset class: {asset_class['id']}")
        for entry in asset_class["files"]:
            yield asset_class, entry


def validate_assets(manifest: dict[str, Any]) -> None:
    expected_total = 0
    for asset_class, entry in iter_manifest_files(manifest):
        path = resolve_project_path(entry["local_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing {asset_class['id']} asset: {path}")
        actual_size = path.stat().st_size
        expected_size = int(entry["size"])
        if actual_size != expected_size:
            raise RuntimeError(
                f"Size mismatch for {path}: {actual_size}, expected {expected_size}."
            )
        actual_hash = hash_file(path, entry["hash_type"])
        if actual_hash != entry["hash"]:
            raise RuntimeError(
                f"Hash mismatch for {path}: {actual_hash}, expected {entry['hash']}."
            )
        expected_total += expected_size
    if expected_total != int(manifest["total_bytes"]):
        raise ValueError(
            f"Manifest total is {manifest['total_bytes']}, file sum is {expected_total}."
        )


def build_environment(config: dict[str, Any]) -> dict[str, str]:
    groups = config["placement"]["physical_gpu_groups"]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join(
                [
                    str(PROJECT_ROOT / "repos/LIBERO"),
                    str(PROJECT_ROOT / "repos/FastWAM/src"),
                ]
            ),
            "PIP_CACHE_DIR": str(PROJECT_ROOT / ".conda/pip-cache"),
            "XDG_CACHE_HOME": str(PROJECT_ROOT / ".conda/xdg-cache"),
            "MPLCONFIGDIR": str(PROJECT_ROOT / ".conda/matplotlib"),
            "HF_HOME": str(PROJECT_ROOT / ".conda/huggingface"),
            "MODELSCOPE_CACHE": str(PROJECT_ROOT / ".conda/modelscope"),
            "LIBERO_CONFIG_PATH": str(
                resolve_project_path(config["paths"]["libero_config_dir"])
            ),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in groups[0]),
            "PYTHONHASHSEED": str(config["evaluation"]["seed"]),
            "DIFFSYNTH_DOWNLOAD_SOURCE": "modelscope",
            "DIFFSYNTH_SKIP_DOWNLOAD": "true",
            "DIFFSYNTH_MODEL_BASE_PATH": str(
                resolve_project_path(config["paths"]["diffsynth_model_base"])
            ),
        }
    )
    return environment


def build_manager_command(
    config: dict[str, Any], *, compose_only: bool = False
) -> list[str]:
    evaluation = config["evaluation"]
    placement = config["placement"]
    logical = placement["logical_devices"]
    output_dir = (
        PROJECT_ROOT / "outputs/baseline/EXP-B000/compose_check"
        if compose_only
        else resolve_project_path(config["paths"]["output_dir"])
    )
    checkpoint = PROJECT_ROOT / "checkpoints/fastwam_release/libero_uncond_2cam224.pt"
    statistics = (
        PROJECT_ROOT
        / "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
    )
    groups = json.dumps(placement["physical_gpu_groups"], separators=(",", ":"))
    command = [
        str(PROJECT_ROOT / ".conda/fastwam/bin/python"),
        str(PROJECT_ROOT / "repos/FastWAM/experiments/libero/run_libero_manager.py"),
        f"task={config['manager']['task_config']}",
        f"ckpt={checkpoint}",
        f"seed={evaluation['seed']}",
        f"mixed_precision={evaluation['mixed_precision']}",
        "model.load_text_encoder=true",
        "model.skip_dit_load_from_pretrain=true",
        "model.redirect_common_files=false",
        f"EVALUATION.task_suite_name={evaluation['task_suite']}",
        f"EVALUATION.task_id={evaluation['task_id']}",
        f"EVALUATION.dataset_stats_path={statistics}",
        f"EVALUATION.num_trials={evaluation['num_trials']}",
        f"EVALUATION.output_dir={output_dir}",
        f"EVALUATION.sigma_shift={evaluation['sigma_shift']}",
        f"EVALUATION.num_inference_steps={evaluation['num_inference_steps']}",
        f"EVALUATION.replan_steps={evaluation['replan_steps']}",
        f"EVALUATION.action_horizon={evaluation['action_horizon']}",
        f"EVALUATION.visualize_future_video={str(evaluation['visualize_future_video']).lower()}",
        f"EVALUATION.compile_action_infer={str(evaluation['compile_action_infer']).lower()}",
        f"EVALUATION.device={logical['video']}",
        f"EVALUATION.video_device={logical['video']}",
        f"EVALUATION.action_device={logical['action']}",
        f"EVALUATION.vae_device={logical['vae']}",
        "EVALUATION.text_encoder_device=null",
        "EVALUATION.text_encoder_devices=[cuda:2,cuda:3]",
        f"MULTIRUN.num_gpus={config['manager']['num_gpus']}",
        f"MULTIRUN.gpu_groups={groups}",
        f"MULTIRUN.task_file={resolve_project_path(config['paths']['task_file'])}",
    ]
    if compose_only:
        command.append("MULTIRUN.create_only=true")
    return command


def check_gpu_count(environment: dict[str, str], required: int) -> None:
    command = [
        str(PROJECT_ROOT / ".conda/fastwam/bin/python"),
        "-c",
        "import torch; print(torch.cuda.device_count())",
    ]
    count = int(subprocess.check_output(command, env=environment, text=True).strip())
    if count < required:
        raise RuntimeError(f"Only {count} visible GPUs; EXP-B000 requires {required}.")


def run_and_tee(command: list[str], environment: dict[str, str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT / "repos/FastWAM",
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--validate-only", action="store_true")
    actions.add_argument("--compose-only", action="store_true")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    validate_demo_config(config)
    validate_repositories(config)
    manifest = load_manifest(config)
    list(iter_manifest_files(manifest))
    environment = build_environment(config)
    command = build_manager_command(config, compose_only=args.compose_only)

    if args.compose_only:
        print("Validated pinned repositories, task, placement, and manifest schema.")
        print("Compose command:", subprocess.list2cmdline(command))
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT / "repos/FastWAM",
            env=environment,
            check=False,
        ).returncode

    validate_assets(manifest)
    print("All required base-only asset sizes and hashes match the manifest.")
    if args.validate_only:
        return 0

    check_gpu_count(environment, int(config["manager"]["num_gpus"]))
    output_dir = resolve_project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(PROJECT_ROOT).free
    print(f"Free disk before launch: {free} bytes")
    print("Launch command:", subprocess.list2cmdline(command))
    return run_and_tee(
        command, environment, resolve_project_path(config["paths"]["log_file"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
