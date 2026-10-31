"""Real FastWAM/LIBERO adapter for counterfactual collection and profiling."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from collect_counterfactual_data import (
    CounterfactualAdapter,
    InferenceConfig,
    StepResult,
    WorkerContext,
)
from models.parameterized_wam import ParameterizedWAM


PROJECT_ROOT = Path(__file__).resolve().parent
FASTWAM_ROOT = PROJECT_ROOT / "repos" / "FastWAM"
LIBERO_ROOT = PROJECT_ROOT / "repos" / "LIBERO"
LIBERO_RESOLUTION = 256


def _add_project_import_paths(options: dict[str, Any]) -> tuple[Path, Path]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    diffsynth_model_base = Path(
        options.get(
            "diffsynth_model_base_path",
            PROJECT_ROOT / "checkpoints" / "diffsynth",
        )
    ).expanduser().resolve()
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(diffsynth_model_base))
    fastwam_root = Path(options.get("fastwam_root", FASTWAM_ROOT)).resolve()
    libero_root = Path(options.get("libero_root", LIBERO_ROOT)).resolve()
    config_dir = Path(
        options.get("libero_config_dir", PROJECT_ROOT / "configs" / "EXP-B000" / "libero")
    ).resolve()
    if "libero_config_dir" in options:
        os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    else:
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_dir))
    import_paths = (
        fastwam_root / "src",
        fastwam_root / "experiments" / "libero",
        libero_root,
    )
    for import_path in import_paths:
        if not import_path.is_dir():
            raise FileNotFoundError(f"Required source directory does not exist: {import_path}")
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))
    return fastwam_root, libero_root


def _mujoco_state_spec(mujoco: Any) -> Any:
    state_enum = mujoco.mjtState
    state_spec = getattr(state_enum, "mjSTATE_INTEGRATION", None)
    if state_spec is None:
        raise RuntimeError("Installed MuJoCo does not expose mjSTATE_INTEGRATION")
    return state_spec


def capture_full_libero_state(env: Any) -> dict[str, Any]:
    """Capture the complete MuJoCo integration state and key mjData fields."""
    mujoco = importlib.import_module("mujoco")
    sim = env.sim
    model = sim.model
    data = sim.data
    state_spec = _mujoco_state_spec(mujoco)
    full_state = np.empty(mujoco.mj_stateSize(model, state_spec), dtype=np.float64)
    mujoco.mj_getState(model, data, full_state, state_spec)
    return {
        "state_spec": int(state_spec),
        "full_state": full_state.copy(),
        "qpos": np.asarray(data.qpos).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "ctrl": np.asarray(data.ctrl).copy(),
        "qacc_warmstart": np.asarray(data.qacc_warmstart).copy(),
        "time": float(data.time),
        "mocap_pos": np.asarray(data.mocap_pos).copy(),
        "mocap_quat": np.asarray(data.mocap_quat).copy(),
        "userdata": np.asarray(data.userdata).copy(),
    }


def restore_full_libero_state(env: Any, checkpoint: dict[str, Any]) -> Any:
    """Restore a saved MuJoCo state and regenerate an observation from mjData."""
    mujoco = importlib.import_module("mujoco")
    sim = env.sim
    model = sim.model
    data = sim.data
    state_spec = _mujoco_state_spec(mujoco)
    if int(checkpoint["state_spec"]) != int(state_spec):
        raise ValueError(
            "Saved MuJoCo state specification differs from the current runtime"
        )
    full_state = np.asarray(checkpoint["full_state"], dtype=np.float64)
    expected_size = mujoco.mj_stateSize(model, state_spec)
    if full_state.size != expected_size:
        raise ValueError(
            f"MuJoCo state size mismatch: got {full_state.size}, expected {expected_size}"
        )
    mujoco.mj_setState(model, data, full_state, state_spec)
    for name in (
        "qpos",
        "qvel",
        "ctrl",
        "qacc_warmstart",
        "mocap_pos",
        "mocap_quat",
        "userdata",
    ):
        target = getattr(data, name)
        source = np.asarray(checkpoint[name])
        if target.shape != source.shape:
            raise ValueError(
                f"Saved MuJoCo field {name} shape {source.shape} "
                f"does not match current shape {target.shape}"
            )
        target[...] = source
    data.time = float(checkpoint["time"])
    mujoco.mj_forward(model, data)
    # Forward restores derived data; reset the integrator inputs it may modify.
    mujoco.mj_setState(model, data, full_state, state_spec)
    for name in ("ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat", "userdata"):
        getattr(data, name)[...] = np.asarray(checkpoint[name])
    data.time = float(checkpoint["time"])
    post_process = getattr(env, "_post_process", None)
    if callable(post_process):
        post_process()
    update_observables = getattr(env, "_update_observables", None)
    if callable(update_observables):
        update_observables(force=True)
    observations = getattr(env, "_get_observations", None)
    if not callable(observations):
        raise TypeError("LIBERO environment cannot regenerate observations")
    return observations()


class RealFastWAMAdapter(CounterfactualAdapter):
    """Loads a real FastWAM checkpoint and evaluates it in a LIBERO task.

    A model replica occupies four GPUs: video on device 0, action/VAE on device
    1, and the text encoder split across devices 2 and 3. W samples are
    independently seeded FastWAM action inferences whose normalized action
    chunks are averaged before denormalization. H maps to IDM video frames using
    the dataset's action/video temporal ratio; it does not alter action horizon.
    """

    def __init__(self, context: WorkerContext) -> None:
        self.context = context
        self.options = context.options
        self.fastwam_root, self.libero_root = _add_project_import_paths(self.options)
        self.cuda_devices = self._resolve_cuda_devices()
        self.device = str(self.cuda_devices[0])
        self.model_path = Path(self.options["model_path"]).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"FastWAM checkpoint does not exist: {self.model_path}")

        self._import_baseline_api()
        self.cfg, self.processor, self.model = self._load_policy()
        self.env: Any = None
        self.task_id: int | None = None
        self.task: Any = None
        self.task_description = ""
        self.initial_states: list[Any] = []
        self.observation: Any = None
        self._infer_call_index = 0
        self.action_horizon = int(self.cfg.EVALUATION.get("action_horizon") or 32)
        self.action_video_freq_ratio = int(self.cfg.data.train.action_video_freq_ratio)
        video_size = self.cfg.data.train.video_size
        self.input_height, self.input_width = int(video_size[0]), int(video_size[1])
        self.max_steps = int(self.options.get("max_steps", 400))
        self.parameterized_wam = ParameterizedWAM(
            self.model,
            self._prepare_policy_observation,
            self._postprocess_policy_actions,
            action_horizon=self.action_horizon,
            action_video_freq_ratio=self.action_video_freq_ratio,
            direct_depth=int(self.options.get("direct_depth", 10)),
            negative_prompt=str(self.cfg.EVALUATION.get("negative_prompt", "")),
            text_cfg_scale=float(self.cfg.EVALUATION.get("text_cfg_scale", 1.0)),
            sigma_shift=self.cfg.EVALUATION.get("sigma_shift"),
            tiled=bool(self.cfg.EVALUATION.get("tiled", False)),
            compile_action_infer=False,
        )

    def _resolve_cuda_devices(self) -> list[str]:
        devices = [str(device) for device in self.options.get("cuda_devices", [self.context.device])]
        if len(devices) != 4 or any(not device.startswith("cuda:") for device in devices):
            raise ValueError(
                "Real FastWAM placement requires exactly four CUDA devices per model replica"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to load a real FastWAM model")
        visible_count = torch.cuda.device_count()
        indices = [int(device.partition(":")[2]) for device in devices]
        if any(index < 0 or index >= visible_count for index in indices):
            raise ValueError(
                f"Requested logical devices {devices}, but only {visible_count} CUDA devices are visible"
            )
        return devices

    def _import_baseline_api(self) -> None:
        self.hydra = importlib.import_module("hydra")
        self.hydra_utils = importlib.import_module("hydra.utils")
        self.omegaconf = importlib.import_module("omegaconf")
        self.eval_helpers = importlib.import_module("eval_libero_single")
        self.libero = importlib.import_module("libero.libero")

    def _compose_config(self) -> Any:
        config_dir = Path(self.options.get("config_dir", self.fastwam_root / "configs"))
        config_name = str(self.options.get("config_name", "sim_libero"))
        task_config = str(
            self.options.get("task_config", "libero_optional_idm_2cam224_1e-4")
        )
        overrides = [
            f"task={task_config}",
            f"ckpt={self.model_path}",
            "mixed_precision=fp16",
            f"EVALUATION.device={self.cuda_devices[0]}",
            "EVALUATION.compile_action_infer=false",
            "EVALUATION.visualize_future_video=false",
            "EVALUATION.action_horizon=32",
            "model.load_text_encoder=true",
            "model.redirect_common_files=false",
            "model.skip_dit_load_from_pretrain=true",
            f"EVALUATION.device={self.cuda_devices[0]}",
            f"EVALUATION.video_device={self.cuda_devices[0]}",
            f"EVALUATION.action_device={self.cuda_devices[1]}",
            f"EVALUATION.vae_device={self.cuda_devices[1]}",
            "EVALUATION.text_encoder_devices=["
            + ",".join(self.cuda_devices[2:])
            + "]",
        ]
        dataset_stats_path = self.options.get("dataset_stats_path")
        if dataset_stats_path is None:
            matching_stats = self.model_path.with_name(
                f"{self.model_path.stem}_dataset_stats.json"
            )
            if matching_stats.is_file():
                dataset_stats_path = matching_stats
        if dataset_stats_path:
            overrides.append(
                f"EVALUATION.dataset_stats_path={Path(dataset_stats_path).expanduser().resolve()}"
            )
        with self.hydra.initialize_config_dir(
            version_base="1.3", config_dir=str(config_dir.resolve())
        ):
            cfg = self.hydra.compose(config_name=config_name, overrides=overrides)
        cfg.EVALUATION.task_suite_name = self.context.task_suite
        return cfg

    def _load_policy(self) -> tuple[Any, Any, torch.nn.Module]:
        self.cfg = self._compose_config()
        instantiate = self.hydra_utils.instantiate
        model = instantiate(
            self.cfg.model,
            model_dtype=torch.float16,
            device=self.cuda_devices[0],
            video_device=self.cuda_devices[0],
            action_device=self.cuda_devices[1],
            vae_device=self.cuda_devices[1],
            text_encoder_devices=self.cuda_devices[2:],
        )
        model.load_checkpoint(str(self.model_path))
        model.eval()

        stats_path = self.eval_helpers._resolve_dataset_stats_path(self.cfg)
        stats_loader = importlib.import_module(
            "fastwam.datasets.lerobot.utils.normalizer"
        ).load_dataset_stats_from_json
        dataset_stats = stats_loader(str(stats_path))
        self.processor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(dataset_stats)

        infer_action = model.infer_action
        if not callable(infer_action):
            raise TypeError("Loaded FastWAM does not implement infer_action")
        return self.cfg, self.processor, model

    def start_task(self, task_id: int, seed: int) -> Any:
        benchmark = self.libero.benchmark.get_benchmark_dict()
        suite_factory = benchmark.get(self.context.task_suite)
        if suite_factory is None:
            raise ValueError(f"Unknown LIBERO suite: {self.context.task_suite}")
        suite = suite_factory()
        task = suite.get_task(int(task_id))
        if self.env is None or self.task_id != task_id:
            self.close_environment()
            self.env, self.task_description = self.eval_helpers.get_libero_env(
                task, LIBERO_RESOLUTION, int(seed)
            )
            self.task_id = int(task_id)
            init_path = (
                Path(self.libero.get_libero_path("init_states"))
                / task.problem_folder
                / task.init_states_file
            )
            init_states = torch.load(init_path, map_location="cpu", weights_only=False)
            if len(init_states) == 0:
                raise RuntimeError(f"No LIBERO initial states in {init_path}")
            self.initial_states = list(init_states)
        else:
            self.env.seed(int(seed))
        self.task = task
        self.env.reset()
        init_state = self.initial_states[int(seed) % len(self.initial_states)]
        self.observation = self.env.set_init_state(init_state)
        self._infer_call_index = 0
        return self.observation

    def capture_checkpoint(self) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("Call start_task() before capturing a LIBERO state")
        if self.observation is None:
            raise RuntimeError("No current observation is available")
        return {
            "mujoco": capture_full_libero_state(self.env),
            "observation": self._copy_observation(self.observation),
        }

    def restore_checkpoint(self, checkpoint: dict[str, Any]) -> Any:
        if self.env is None:
            raise RuntimeError("Call start_task() before restoring a LIBERO state")
        observation = restore_full_libero_state(self.env, checkpoint["mujoco"])
        self.observation = observation
        return observation

    def infer_actions(
        self, observation: dict[str, Any], config: InferenceConfig
    ) -> np.ndarray:
        config.validate()
        if self.env is None:
            raise RuntimeError("Call start_task() before running inference")
        actions = self.parameterized_wam(
            observation,
            config.H,
            config.D,
            max(config.W, 1),
            config.E,
        )
        self._infer_call_index += 1
        return actions.numpy()

    def _prepare_policy_observation(
        self, observation: dict[str, Any]
    ) -> dict[str, Any]:
        image, proprio, _ = self.eval_helpers._obs_to_model_input(
            observation,
            cfg=self.cfg,
            processor=self.processor,
            width=self.input_width,
            height=self.input_height,
            device=self.cuda_devices[0],
            dtype=self.model.torch_dtype,
        )
        prompt = self.eval_helpers.DEFAULT_PROMPT.format(task=self.task_description)
        return {
            "prompt": prompt,
            "input_image": image,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
        }

    def _postprocess_policy_actions(
        self, normalized_actions: torch.Tensor, execution_chunk: int
    ) -> np.ndarray:
        denormalized = self.eval_helpers._denormalize_action(
            normalized_actions[:execution_chunk], self.processor
        )[0]
        denormalized[..., -1] = denormalized[..., -1] * 2.0 - 1.0
        denormalized = self.eval_helpers.invert_gripper_action(denormalized)
        if bool(self.cfg.EVALUATION.get("binarize_gripper", True)):
            denormalized[..., -1] = np.sign(denormalized[..., -1])
        return np.asarray(denormalized, dtype=np.float32)

    def step(self, action: np.ndarray) -> StepResult:
        if self.env is None:
            raise RuntimeError("Call start_task() before stepping LIBERO")
        observation, reward, done, info = self.env.step(action.tolist())
        self.observation = observation
        return StepResult(
            observation=observation,
            reward=float(reward),
            terminated=bool(done),
            truncated=False,
            info=dict(info),
        )

    def is_success(self, observation: Any, info: dict[str, Any]) -> bool:
        del observation, info
        return bool(self.env.check_success())

    def close_environment(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
            self.task_id = None

    def close(self) -> None:
        self.close_environment()
        if hasattr(self, "parameterized_wam"):
            del self.parameterized_wam
        if hasattr(self, "model"):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _copy_observation(observation: Any) -> Any:
        if isinstance(observation, dict):
            return {
                key: RealFastWAMAdapter._copy_observation(value)
                for key, value in observation.items()
            }
        if isinstance(observation, np.ndarray):
            return observation.copy()
        if isinstance(observation, torch.Tensor):
            return observation.detach().cpu().clone()
        return observation


def create_adapter(context: WorkerContext) -> RealFastWAMAdapter:
    return RealFastWAMAdapter(context)


def list_task_ids(
    task_suite: str,
    *,
    fastwam_root: str | Path = FASTWAM_ROOT,
    libero_root: str | Path = LIBERO_ROOT,
) -> list[int]:
    _add_project_import_paths(
        {"fastwam_root": fastwam_root, "libero_root": libero_root}
    )
    libero = importlib.import_module("libero.libero")
    suite_factory = libero.benchmark.get_benchmark_dict().get(task_suite)
    if suite_factory is None:
        raise ValueError(f"Unknown LIBERO suite: {task_suite}")
    suite = suite_factory()
    return list(range(int(suite.n_tasks)))
