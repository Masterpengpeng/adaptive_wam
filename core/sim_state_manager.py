"""Full-state snapshot and restoration for LIBERO / robosuite MuJoCo envs."""

from __future__ import annotations

import copy
import importlib
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


_MUJOCO_DATA_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "userdata",
    "eq_active",
)
_MODEL_SCENE_FIELDS = (
    "body_pos",
    "body_quat",
    "geom_pos",
    "geom_quat",
    "geom_rgba",
    "site_pos",
    "site_quat",
    "cam_pos",
    "cam_quat",
    "light_pos",
    "light_dir",
    "light_diffuse",
    "light_specular",
    "light_ambient",
    "mat_rgba",
)
_COUNTER_TOKENS = ("step", "time", "episode", "done")
_RNG_ATTRIBUTE_NAMES = {
    "np_random",
    "rng",
    "random_generator",
    "_rng",
    "generator",
}
_COMPONENT_CHILD_NAMES = (
    "controller",
    "interpolator",
    "_interpolator",
    "gripper",
    "gripper_controller",
    "corrupter",
    "delayer",
)


def _copy_supported(value: Any) -> Any:
    """Copy values commonly used for simulation state without copying handles."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {
            key: _copy_supported(item)
            for key, item in value.items()
            if isinstance(key, (str, int, float, bool, tuple))
            and _is_supported(item)
        }
    if isinstance(value, list):
        return [_copy_supported(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_supported(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None), np.generic)):
        return copy.deepcopy(value)
    return copy.deepcopy(value)


def _is_supported(value: Any) -> bool:
    if isinstance(value, (np.ndarray, torch.Tensor, str, int, float, bool, type(None), np.generic)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_supported(item) for item in value)
    if isinstance(value, dict):
        return all(_is_supported(item) for item in value.values())
    return False


def _environment_chain(env: Any) -> list[Any]:
    chain: list[Any] = []
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = getattr(current, "env", None)
    return chain


def _capture_rng_object(rng: Any) -> dict[str, Any] | None:
    if isinstance(rng, np.random.Generator):
        return {
            "kind": "numpy_generator",
            "state": copy.deepcopy(rng.bit_generator.state),
        }
    if isinstance(rng, np.random.RandomState):
        return {"kind": "numpy_random_state", "state": rng.get_state()}
    if isinstance(rng, random.Random):
        return {"kind": "python_random", "state": rng.getstate()}
    if isinstance(rng, torch.Generator):
        return {"kind": "torch_generator", "state": rng.get_state().clone()}
    return None


def _restore_rng_object(rng: Any, snapshot: Mapping[str, Any]) -> None:
    kind = snapshot.get("kind")
    if kind == "numpy_generator" and isinstance(rng, np.random.Generator):
        rng.bit_generator.state = copy.deepcopy(snapshot["state"])
    elif kind == "numpy_random_state" and isinstance(rng, np.random.RandomState):
        rng.set_state(snapshot["state"])
    elif kind == "python_random" and isinstance(rng, random.Random):
        rng.setstate(snapshot["state"])
    elif kind == "torch_generator" and isinstance(rng, torch.Generator):
        rng.set_state(snapshot["state"])
    else:
        raise TypeError(f"RNG type changed since snapshot: {kind}")


def _capture_global_rng() -> dict[str, Any]:
    cuda_states = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": cuda_states,
    }


def _restore_global_rng(snapshot: Mapping[str, Any]) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"].cpu())
    cuda_states = snapshot.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("CUDA device topology differs from the saved RNG snapshot")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def _capture_mujoco_state(sim: Any) -> dict[str, Any]:
    model = sim.model
    data = sim.data
    fields = {
        name: np.asarray(getattr(data, name)).copy()
        for name in _MUJOCO_DATA_FIELDS
        if hasattr(data, name)
    }
    scene_fields = {
        name: np.asarray(getattr(model, name)).copy()
        for name in _MODEL_SCENE_FIELDS
        if hasattr(model, name)
    }
    try:
        mujoco = importlib.import_module("mujoco")
        state_spec = getattr(mujoco.mjtState, "mjSTATE_INTEGRATION", None)
        if state_spec is not None:
            size = int(mujoco.mj_stateSize(model, state_spec))
            integration_state = np.empty(size, dtype=np.float64)
            mujoco.mj_getState(model, data, integration_state, state_spec)
            return {
                "backend": "mujoco",
                "state_spec": int(state_spec),
                "integration_state": integration_state.copy(),
                "time": float(data.time),
                "data_fields": fields,
                "model_scene_fields": scene_fields,
            }
    except (ImportError, AttributeError, TypeError):
        pass

    getter = getattr(sim, "get_state", None)
    if not callable(getter):
        raise TypeError("Simulation must support MuJoCo integration state or get_state()")
    return {
        "backend": "robosuite",
        "flattened_state": np.asarray(getter().flatten()).copy(),
        "time": float(getattr(data, "time", 0.0)),
        "data_fields": fields,
        "model_scene_fields": scene_fields,
    }


def _capture_counters(chain: list[Any]) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    for index, item in enumerate(chain):
        values: dict[str, Any] = {}
        try:
            attributes = vars(item)
        except TypeError:
            continue
        for name, value in attributes.items():
            normalized = name.lower().lstrip("_")
            if any(token in normalized for token in _COUNTER_TOKENS) and isinstance(
                value, (int, float, bool, np.integer, np.floating)
            ):
                values[name] = value.item() if isinstance(value, np.generic) else value
        if values:
            captured.append({"chain_index": index, "values": values})
    return captured


def _capture_environment_rng(chain: list[Any]) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    for index, item in enumerate(chain):
        for name in _RNG_ATTRIBUTE_NAMES:
            if not hasattr(item, name):
                continue
            rng_snapshot = _capture_rng_object(getattr(item, name))
            if rng_snapshot is not None:
                captured.append(
                    {"chain_index": index, "attribute": name, "snapshot": rng_snapshot}
                )
    return captured


def _capture_component(component: Any, depth: int = 0) -> dict[str, Any] | None:
    if depth > 3 or not hasattr(component, "__dict__"):
        return None
    values = {
        name: _copy_supported(value)
        for name, value in vars(component).items()
        if _is_supported(value)
    }
    children = {}
    for name in _COMPONENT_CHILD_NAMES:
        child = getattr(component, name, None)
        if child is not None and hasattr(child, "__dict__"):
            state = _capture_component(child, depth + 1)
            if state:
                children[name] = state
    return {"values": values, "children": children} if values or children else None


def _restore_component(component: Any, snapshot: Mapping[str, Any]) -> None:
    for name, value in snapshot.get("values", {}).items():
        setattr(component, name, _copy_supported(value))
    for name, child_snapshot in snapshot.get("children", {}).items():
        child = getattr(component, name, None)
        if child is None:
            raise RuntimeError(f"Simulation component {name!r} no longer exists")
        _restore_component(child, child_snapshot)


def _capture_robots(chain: list[Any]) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    for chain_index, item in enumerate(chain):
        robots = getattr(item, "robots", None)
        if not isinstance(robots, (list, tuple)):
            continue
        for robot_index, robot in enumerate(robots):
            state = _capture_component(robot)
            if state is not None:
                captured.append(
                    {
                        "chain_index": chain_index,
                        "robot_index": robot_index,
                        "state": state,
                    }
                )
    return captured


def _capture_tracking_states(chain: list[Any]) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    for chain_index, item in enumerate(chain):
        trackers = getattr(item, "tracking_object_states_change", None)
        if not isinstance(trackers, (list, tuple)):
            continue
        for tracker_index, tracker in enumerate(trackers):
            state = _capture_component(tracker)
            if state is not None:
                captured.append(
                    {
                        "chain_index": chain_index,
                        "tracker_index": tracker_index,
                        "state": state,
                    }
                )
    return captured


def _capture_observables(chain: list[Any]) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    for chain_index, item in enumerate(chain):
        observables = getattr(item, "_observables", None)
        if not isinstance(observables, Mapping):
            continue
        for name, observable in observables.items():
            state = _capture_component(observable)
            if state is not None:
                captured.append(
                    {"chain_index": chain_index, "name": name, "state": state}
                )
        obs_cache = getattr(item, "_obs_cache", None)
        if isinstance(obs_cache, dict) and _is_supported(obs_cache):
            captured.append(
                {
                    "chain_index": chain_index,
                    "obs_cache": _copy_supported(obs_cache),
                }
            )
    return captured


def _capture_object_poses(chain: list[Any], sim: Any) -> dict[str, dict[str, np.ndarray]]:
    for item in chain:
        objects = getattr(item, "objects_dict", None)
        if not isinstance(objects, dict):
            continue
        poses: dict[str, dict[str, np.ndarray]] = {}
        for name, obj in objects.items():
            body_name = getattr(obj, "root_body", None)
            if not isinstance(body_name, str):
                continue
            model = sim.model
            if hasattr(model, "body_name2id"):
                body_id = int(model.body_name2id(body_name))
            else:
                mujoco = importlib.import_module("mujoco")
                body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
            if body_id < 0:
                continue
            poses[name] = {
                "position": np.asarray(sim.data.body_xpos[body_id]).copy(),
                "quaternion": np.asarray(sim.data.body_xquat[body_id]).copy(),
            }
        return poses
    return {}


class SimStateManager:
    """Capture and restore deterministic LIBERO / robosuite simulation state.

    MuJoCo joint state includes movable-object poses and robot/gripper joints.
    Mutable model body/geom transforms are also saved for LIBERO scene
    randomization. Observations are regenerated after ``mj_forward`` rather
    than reusing stale camera or observable caches.
    """

    @staticmethod
    def save_state(env: Any) -> dict[str, Any]:
        """Capture engine state, counters, environment/global RNGs, and controls."""
        chain = _environment_chain(env)
        sim = getattr(env, "sim", None)
        if sim is None:
            for item in chain:
                sim = getattr(item, "sim", None)
                if sim is not None:
                    break
        if sim is None or not hasattr(sim, "data") or not hasattr(sim, "model"):
            raise TypeError("Environment must expose a MuJoCo/robosuite sim with model and data")

        return {
            "schema_version": 1,
            "simulation": _capture_mujoco_state(sim),
            "counters": _capture_counters(chain),
            "environment_rng": _capture_environment_rng(chain),
            "global_rng": _capture_global_rng(),
            "robots": _capture_robots(chain),
            "tracking_object_states": _capture_tracking_states(chain),
            "observables": _capture_observables(chain),
            "object_poses": _capture_object_poses(chain, sim),
        }

    @staticmethod
    def restore_state(env: Any, state_snapshot: Mapping[str, Any]) -> Any:
        """Restore a snapshot, refresh MuJoCo derived data, and regenerate obs."""
        if state_snapshot.get("schema_version") != 1:
            raise ValueError("Unsupported or missing simulation snapshot schema_version")
        chain = _environment_chain(env)
        sim = getattr(env, "sim", None)
        if sim is None:
            for item in chain:
                sim = getattr(item, "sim", None)
                if sim is not None:
                    break
        if sim is None or not hasattr(sim, "data") or not hasattr(sim, "model"):
            raise TypeError("Environment must expose a MuJoCo/robosuite sim with model and data")
        simulation = state_snapshot["simulation"]
        model = sim.model
        data = sim.data

        for name, source in simulation["model_scene_fields"].items():
            target = getattr(model, name, None)
            if target is None or target.shape != source.shape:
                raise ValueError(f"MuJoCo model field {name!r} is incompatible with snapshot")
            target[...] = source

        if simulation["backend"] == "mujoco":
            mujoco = importlib.import_module("mujoco")
            state_spec = _mujoco_state_spec(mujoco)
            if int(state_spec) != int(simulation["state_spec"]):
                raise ValueError("MuJoCo state specification differs from snapshot")
            expected_size = int(mujoco.mj_stateSize(model, state_spec))
            integration_state = np.asarray(
                simulation["integration_state"], dtype=np.float64
            )
            if integration_state.size != expected_size:
                raise ValueError(
                    f"MuJoCo state size mismatch: got {integration_state.size}, "
                    f"expected {expected_size}"
                )
            mujoco.mj_setState(model, data, integration_state, state_spec)
        elif simulation["backend"] == "robosuite":
            restore = getattr(sim, "set_state_from_flattened", None)
            if not callable(restore):
                raise TypeError("robosuite sim does not support flattened state restore")
            restore(np.asarray(simulation["flattened_state"]).copy())
        else:
            raise ValueError(f"Unknown simulation backend {simulation['backend']!r}")

        for name, source in simulation["data_fields"].items():
            target = getattr(data, name, None)
            if target is None or target.shape != source.shape:
                raise ValueError(f"MuJoCo data field {name!r} is incompatible with snapshot")
            target[...] = source
        data.time = float(simulation["time"])

        forward = getattr(sim, "forward", None)
        if callable(forward):
            forward()
        else:
            mujoco = importlib.import_module("mujoco")
            mujoco.mj_forward(model, data)
        if "qacc_warmstart" in simulation["data_fields"]:
            data.qacc_warmstart[...] = simulation["data_fields"]["qacc_warmstart"]

        post_process = next(
            (
                method
                for item in chain
                if callable(method := getattr(item, "_post_process", None))
            ),
            None,
        )
        if post_process is not None:
            post_process()

        for name, source in simulation["model_scene_fields"].items():
            getattr(model, name)[...] = source
        if callable(forward):
            forward()
        else:
            mujoco.mj_forward(model, data)

        update_observables = next(
            (
                method
                for item in chain
                if callable(method := getattr(item, "_update_observables", None))
            ),
            None,
        )
        if update_observables is not None:
            update_observables(force=True)
        if "qacc_warmstart" in simulation["data_fields"]:
            data.qacc_warmstart[...] = simulation["data_fields"]["qacc_warmstart"]

        for entry in state_snapshot["observables"]:
            item = chain[int(entry["chain_index"])]
            if "obs_cache" in entry:
                cache = getattr(item, "_obs_cache", None)
                if not isinstance(cache, dict):
                    raise RuntimeError("Environment observation cache no longer exists")
                cache.clear()
                cache.update(_copy_supported(entry["obs_cache"]))
            else:
                observables = getattr(item, "_observables", None)
                name = str(entry["name"])
                if not isinstance(observables, Mapping) or name not in observables:
                    raise RuntimeError(f"Environment observable {name!r} no longer exists")
                _restore_component(observables[name], entry["state"])

        observation_getter = None
        for item in chain:
            candidate = getattr(item, "_get_observations", None)
            if callable(candidate):
                observation_getter = candidate
                break
        observation = observation_getter() if observation_getter is not None else None

        for entry in state_snapshot["counters"]:
            item = chain[int(entry["chain_index"])]
            for name, value in entry["values"].items():
                setattr(item, name, value)

        for entry in state_snapshot["robots"]:
            item = chain[int(entry["chain_index"])]
            robots = getattr(item, "robots", None)
            robot_index = int(entry["robot_index"])
            if not isinstance(robots, (list, tuple)) or robot_index >= len(robots):
                raise RuntimeError("Robot topology changed since snapshot")
            _restore_component(robots[robot_index], entry["state"])

        for entry in state_snapshot["tracking_object_states"]:
            item = chain[int(entry["chain_index"])]
            trackers = getattr(item, "tracking_object_states_change", None)
            tracker_index = int(entry["tracker_index"])
            if not isinstance(trackers, (list, tuple)) or tracker_index >= len(trackers):
                raise RuntimeError("Tracked object topology changed since snapshot")
            _restore_component(trackers[tracker_index], entry["state"])

        for entry in state_snapshot["environment_rng"]:
            item = chain[int(entry["chain_index"])]
            rng = getattr(item, entry["attribute"], None)
            if rng is None:
                raise RuntimeError(
                    f"Environment RNG {entry['attribute']!r} no longer exists"
                )
            _restore_rng_object(rng, entry["snapshot"])

        _restore_global_rng(state_snapshot["global_rng"])
        return observation


def _mujoco_state_spec(mujoco: Any) -> Any:
    state_spec = getattr(mujoco.mjtState, "mjSTATE_INTEGRATION", None)
    if state_spec is None:
        raise RuntimeError("Installed MuJoCo does not expose mjSTATE_INTEGRATION")
    return state_spec
