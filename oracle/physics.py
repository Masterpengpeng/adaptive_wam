"""Physical event measurements shared by snapshot collection and rollouts."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PhysicalMetrics:
    target_object: str | None
    goal_object: str | None
    gripper_target_distance_m: float | None
    contact_force_n: float | None
    target_object_displacement_m: float | None
    goal_distance_m: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_object": self.target_object,
            "goal_object": self.goal_object,
            "gripper_target_distance_m": self.gripper_target_distance_m,
            "contact_force_n": self.contact_force_n,
            "target_object_displacement_m": self.target_object_displacement_m,
            "goal_distance_m": self.goal_distance_m,
        }


def _environment_chain(env: Any) -> list[Any]:
    chain: list[Any] = []
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = getattr(current, "env", None)
    return chain


def _find_task_env(chain: list[Any]) -> Any | None:
    return next(
        (item for item in chain if isinstance(getattr(item, "objects_dict", None), Mapping)),
        None,
    )


def _goal_pair(task_env: Any) -> tuple[str | None, str | None]:
    problem = getattr(task_env, "parsed_problem", {})
    goals = problem.get("goal_state", []) if isinstance(problem, Mapping) else []
    for goal in goals:
        if isinstance(goal, (list, tuple)) and len(goal) >= 3:
            return str(goal[1]), str(goal[2])
    return None, None


def _object_position(task_env: Any, name: str | None, sim: Any) -> np.ndarray | None:
    if not name:
        return None
    sites = getattr(task_env, "object_sites_dict", {})
    if name in sites:
        try:
            site_id = int(sim.model.site_name2id(name))
            return np.asarray(sim.data.site_xpos[site_id], dtype=np.float64).copy()
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
    body_ids = getattr(task_env, "obj_body_id", {})
    if name in body_ids:
        try:
            return np.asarray(sim.data.body_xpos[int(body_ids[name])], dtype=np.float64).copy()
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
    objects = getattr(task_env, "objects_dict", {})
    fixtures = getattr(task_env, "fixtures_dict", {})
    obj = objects.get(name, fixtures.get(name)) if isinstance(objects, Mapping) else None
    body_name = getattr(obj, "root_body", None)
    if isinstance(body_name, str):
        try:
            body_id = int(sim.model.body_name2id(body_name))
            return np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).copy()
        except (AttributeError, TypeError, ValueError):
            return None
    return None


def _gripper_position(chain: list[Any], sim: Any, observation: Any = None) -> np.ndarray | None:
    if isinstance(observation, Mapping):
        for key in ("robot0_eef_pos", "eef_pos"):
            value = observation.get(key)
            if value is not None:
                position = np.asarray(value, dtype=np.float64).reshape(-1)
                if position.size >= 3:
                    return position[:3].copy()
    for item in chain:
        robots = getattr(item, "robots", None)
        if not isinstance(robots, (list, tuple)) or not robots:
            continue
        site_id = getattr(robots[0], "eef_site_id", None)
        if site_id is not None:
            try:
                return np.asarray(sim.data.site_xpos[int(site_id)], dtype=np.float64).copy()
            except (AttributeError, IndexError, TypeError, ValueError):
                pass
    return None


def _body_is_descendant(model: Any, body_id: int, root_id: int) -> bool:
    parent_ids = getattr(model, "body_parentid", None)
    if parent_ids is None:
        return body_id == root_id
    current = int(body_id)
    while current > 0:
        if current == root_id:
            return True
        current = int(parent_ids[current])
    return current == root_id


def _target_geom_ids(task_env: Any, target_name: str, sim: Any) -> set[int]:
    body_ids = getattr(task_env, "obj_body_id", {})
    if target_name not in body_ids:
        return set()
    root_id = int(body_ids[target_name])
    geom_body_ids = getattr(sim.model, "geom_bodyid", ())
    return {
        geom_id
        for geom_id, body_id in enumerate(geom_body_ids)
        if _body_is_descendant(sim.model, int(body_id), root_id)
    }


def _gripper_geom_ids(chain: list[Any], sim: Any) -> set[int]:
    model = sim.model
    result: set[int] = set()
    for item in chain:
        robots = getattr(item, "robots", None)
        if not isinstance(robots, (list, tuple)) or not robots:
            continue
        gripper = getattr(robots[0], "gripper", None)
        important_geoms = getattr(gripper, "_important_geoms", {})
        naming_prefix = str(getattr(gripper, "naming_prefix", ""))
        if not isinstance(important_geoms, Mapping):
            continue
        for names in important_geoms.values():
            if isinstance(names, str):
                names = [names]
            for name in names:
                try:
                    geom_id = int(model.geom_name2id(f"{naming_prefix}{name}"))
                except (AttributeError, TypeError, ValueError):
                    continue
                if geom_id >= 0:
                    result.add(geom_id)
        if result:
            break
    return result


def _contact_force_n(
    chain: list[Any],
    task_env: Any,
    target_name: str | None,
    sim: Any,
) -> float | None:
    if not target_name:
        return None
    object_geom_ids = _target_geom_ids(task_env, target_name, sim)
    gripper_geom_ids = _gripper_geom_ids(chain, sim)
    if not object_geom_ids or not gripper_geom_ids:
        return None
    mujoco = importlib.import_module("mujoco")
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    force = np.zeros(6, dtype=np.float64)
    total_force = 0.0
    for contact_index in range(int(sim.data.ncon)):
        contact = sim.data.contact[contact_index]
        geom_pair = {int(contact.geom1), int(contact.geom2)}
        if not (geom_pair & object_geom_ids and geom_pair & gripper_geom_ids):
            continue
        mujoco.mj_contactForce(model, data, contact_index, force)
        total_force += float(np.linalg.norm(force[:3]))
    return total_force


def collect_object_positions(env: Any) -> dict[str, np.ndarray]:
    chain = _environment_chain(env)
    task_env = _find_task_env(chain)
    if task_env is None:
        return {}
    sim = next((getattr(item, "sim", None) for item in chain if getattr(item, "sim", None) is not None), None)
    if sim is None:
        return {}
    names = set(getattr(task_env, "obj_of_interest", ()))
    names.update(getattr(task_env, "objects_dict", {}).keys())
    positions: dict[str, np.ndarray] = {}
    for name in names:
        position = _object_position(task_env, str(name), sim)
        if position is not None:
            positions[str(name)] = position
    return positions


def measure_physics(
    env: Any,
    *,
    initial_object_positions: Mapping[str, np.ndarray] | None = None,
    observation: Any = None,
) -> PhysicalMetrics:
    chain = _environment_chain(env)
    task_env = _find_task_env(chain)
    sim = next((getattr(item, "sim", None) for item in chain if getattr(item, "sim", None) is not None), None)
    if task_env is None or sim is None:
        return PhysicalMetrics(None, None, None, None, None, None)

    target_name, goal_name = _goal_pair(task_env)
    available_objects = getattr(task_env, "objects_dict", {})
    if target_name not in available_objects:
        interests = [str(name) for name in getattr(task_env, "obj_of_interest", ())]
        target_name = next((name for name in interests if name in available_objects), None)

    target_position = _object_position(task_env, target_name, sim)
    goal_position = _object_position(task_env, goal_name, sim)
    gripper_position = _gripper_position(chain, sim, observation)
    gripper_distance = (
        float(np.linalg.norm(gripper_position - target_position))
        if gripper_position is not None and target_position is not None
        else None
    )
    goal_distance = (
        float(np.linalg.norm(goal_position - target_position))
        if goal_position is not None and target_position is not None
        else None
    )
    displacement = None
    if target_name and initial_object_positions and target_name in initial_object_positions and target_position is not None:
        displacement = float(
            np.linalg.norm(target_position - np.asarray(initial_object_positions[target_name]))
        )
    contact_force = _contact_force_n(chain, task_env, target_name, sim)
    return PhysicalMetrics(
        target_object=target_name,
        goal_object=goal_name,
        gripper_target_distance_m=gripper_distance,
        contact_force_n=contact_force,
        target_object_displacement_m=displacement,
        goal_distance_m=goal_distance,
    )


def classify_physical_phase(metrics: PhysicalMetrics) -> str | None:
    goal_distance = metrics.goal_distance_m
    contact_force = metrics.contact_force_n
    target_distance = metrics.gripper_target_distance_m
    displacement = metrics.target_object_displacement_m
    if goal_distance is not None and goal_distance <= 0.05:
        return "placement"
    if displacement is not None and displacement > 0.01 and contact_force is not None and contact_force > 0.0:
        return "manipulation"
    if target_distance is not None and target_distance < 0.08 and contact_force == 0.0:
        return "precontact"
    if target_distance is not None and target_distance > 0.05:
        return "reaching"
    return None