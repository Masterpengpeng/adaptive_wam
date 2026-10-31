from __future__ import annotations

import csv
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import yaml

from collect_counterfactual_data import InferenceConfig, StepResult, WorkerContext
from real_fastwam_adapter import FASTWAM_ROOT, LIBERO_ROOT, _add_project_import_paths
from oracle.collect_snapshots import collect_from_csv, collect_targeted_from_trajectories
from oracle.collect_snapshots import prepare_libero_config, resolve_trajectories_dir
from oracle.counterfactual_oracle import (
    CONFIG_GRID,
    OracleConfig,
    _run_branch,
    evaluate_snapshot,
    load_config_grid,
    load_relative_costs,
    load_snapshots,
    read_results,
    write_results,
)
from oracle.physics import PhysicalMetrics, classify_physical_phase


class FakeStateManager:
    @staticmethod
    def save_state(env: Any) -> dict[str, int]:
        return {"position": env.position}

    @staticmethod
    def restore_state(env: Any, state: dict[str, int]) -> dict[str, np.ndarray]:
        env.position = int(state["position"])
        return {
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
        }


class FakeOracleAdapter:
    def __init__(self, context: WorkerContext) -> None:
        self.context = context
        self.env = SimpleNamespace(
            position=0,
            sim=SimpleNamespace(
                data=SimpleNamespace(
                    time=0.0,
                    qpos=np.zeros(1, dtype=np.float64),
                    qvel=np.zeros(1, dtype=np.float64),
                    ctrl=np.zeros(1, dtype=np.float64),
                )
            ),
        )
        self.observation: dict[str, np.ndarray] | None = None
        self.inference_configs: list[InferenceConfig] = []

    def start_task(self, task_id: int, seed: int) -> dict[str, np.ndarray]:
        del task_id
        self.succeeds = seed == 1
        self.env.position = 0
        self.env.sim.data.qpos[0] = 0
        self.env.sim.data.time = 0.0
        self.observation = {
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        return self.observation

    def infer_actions(self, observation: Any, config: InferenceConfig) -> np.ndarray:
        del observation
        self.inference_configs.append(config)
        return np.ones((config.E, 1), dtype=np.float32)

    def step(self, action: np.ndarray) -> StepResult:
        self.env.position += int(np.asarray(action).reshape(-1)[0])
        self.env.sim.data.qpos[0] = self.env.position
        self.env.sim.data.time += 0.05
        observation = {
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        self.observation = observation
        succeeded = self.succeeds and self.env.position >= 6
        return StepResult(observation, 1.0, succeeded, False, {"success": succeeded})

    def is_success(self, observation: Any, info: dict[str, Any]) -> bool:
        del observation
        return bool(info.get("success", False) or (self.succeeds and self.env.position >= 6))

    def close(self) -> None:
        return None


class OraclePipelineSmokeTest(unittest.TestCase):
    def test_loads_custom_decoupled_configuration_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.json"
            path.write_text(
                '{"configurations": ['
                '{"name":"base","H":2,"D":2,"W":1,"E":4},'
                '{"name":"width","H":2,"D":2,"W":4,"E":4}'
                ']}',
                encoding="utf-8",
            )
            configs = load_config_grid(path)
            self.assertEqual([row.name for row in configs], ["base", "width"])
            self.assertEqual(configs[1].as_dict()["W"], 4)

    def test_collects_targeted_snapshots_directly_from_saved_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectories = root / "trajectories"
            trajectories.mkdir()

            def step(index: int, phase: str, goal: float, displacement: float) -> dict[str, Any]:
                return {
                    "step": index + 1,
                    "action": np.zeros(2, dtype=np.float32),
                    "reward": 0.0,
                    "done": False,
                    "success": False,
                    "sim_state": {"schema_version": 1, "position": index},
                    "obs": {
                        "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
                        "robot0_eye_in_hand_image": np.zeros((16, 16, 3), dtype=np.uint8),
                    },
                    "contact": None,
                    "phase": phase,
                    "physical_metrics": {
                        "goal_distance_m": goal,
                        "target_object_displacement_m": displacement,
                    },
                }

            low_steps = [step(i, "reaching", 0.5, 0.0) for i in range(10)]
            full_steps = [
                step(i, "precontact" if i < 2 else "manipulation", 0.5 - i * 0.06, i * 0.05)
                for i in range(8)
            ]
            low_path = trajectories / "low.pt"
            full_path = trajectories / "full.pt"
            for path, method, success, steps in (
                (low_path, "No-Imagination", False, low_steps),
                (full_path, "Fixed-Full", True, full_steps),
            ):
                torch.save(
                    {
                        "schema_version": 1,
                        "storage": "content_addressed_arrays_v1",
                        "task_suite": "fake",
                        "task_id": 0,
                        "seed": 3,
                        "method": method,
                        "config": {},
                        "success": success,
                        "episode_steps": len(steps),
                        "episode_return": float(success),
                        "trajectory_stride": 1,
                        "steps": steps,
                        "array_blobs": {},
                    },
                    path,
                )

            episodes = root / "episodes.csv"
            with episodes.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(
                    destination,
                    fieldnames=(
                        "task_suite", "task_id", "seed", "method", "success", "steps",
                        "trajectory_path",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {
                            "task_suite": "fake", "task_id": 0, "seed": 3,
                            "method": "No-Imagination", "success": 0, "steps": 10,
                            "trajectory_path": str(low_path),
                        },
                        {
                            "task_suite": "fake", "task_id": 0, "seed": 3,
                            "method": "Fixed-Full", "success": 1, "steps": 8,
                            "trajectory_path": str(full_path),
                        },
                    )
                )
            pairs = root / "pairs.csv"
            with pairs.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(
                    destination,
                    fieldnames=(
                        "task_suite", "task_id", "seed",
                        "no_imagination_trajectory", "fixed_full_trajectory",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_suite": "fake", "task_id": 0, "seed": 3,
                        "no_imagination_trajectory": str(low_path),
                        "fixed_full_trajectory": str(full_path),
                    }
                )

            saved = collect_targeted_from_trajectories(
                episodes,
                pairs,
                root / "snapshots",
                trajectories_dir=trajectories,
                snapshot_limit=8,
                minimum_snapshots=8,
            )

            self.assertEqual(len(saved), 8)
            with saved[0].open("rb") as source:
                payload = pickle.load(source)
            self.assertEqual(payload["sampling_protocol"], "physical_events_v2")
            phases = [path.stem.rsplit("_", 1)[-1] for path in saved]
            self.assertTrue(any("precontact" in path.name for path in saved))
            self.assertTrue(any("manipulation" in path.name for path in saved))
            self.assertTrue(any("failure_recovery" in path.name for path in saved))

    def test_collects_paired_watershed_and_physical_full_phases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = root / "episodes.csv"
            with episodes.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(
                    destination,
                    fieldnames=("task_suite", "task_id", "seed", "method", "success", "steps"),
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {"task_suite": "fake", "task_id": 0, "seed": 1, "method": "No-Imagination", "success": 1, "steps": 6},
                        {"task_suite": "fake", "task_id": 0, "seed": 1, "method": "Fixed-Full", "success": 1, "steps": 6},
                        {"task_suite": "fake", "task_id": 0, "seed": 2, "method": "No-Imagination", "success": 0, "steps": 9},
                        {"task_suite": "fake", "task_id": 0, "seed": 2, "method": "Fixed-Full", "success": 1, "steps": 6},
                    )
                )
            snapshots_dir = root / "snapshots"
            adapter = FakeOracleAdapter(WorkerContext("cpu", "fake", {}))
            def fake_measure(env: Any, **_: Any) -> PhysicalMetrics:
                position = env.position
                if position < 2:
                    return PhysicalMetrics("obj", "goal", 0.10 - position * 0.005, 0.0, 0.0, 0.5)
                if position < 4:
                    return PhysicalMetrics("obj", "goal", 0.07, 0.0, 0.0, 0.3)
                if position < 6:
                    return PhysicalMetrics("obj", "goal", 0.03, 1.0, 0.02, 0.2)
                return PhysicalMetrics("obj", "goal", 0.01, 1.0, 0.04, 0.01)

            with patch("oracle.collect_snapshots.measure_physics", side_effect=fake_measure):
                saved = collect_from_csv(
                    episodes,
                    snapshots_dir,
                    lambda _: adapter,
                    options={},
                    device="cpu",
                    successful_episode_limit=1,
                    snapshots_per_category=6,
                    max_per_episode_per_phase=2,
                    phase_sample_gap=1,
                    state_manager=FakeStateManager,
                    max_steps=10,
                )

            self.assertEqual(len(saved), 12)
            payloads = []
            for path in saved:
                with path.open("rb") as source:
                    payloads.append(pickle.load(source))
            failure_steps = sorted(item["step"] for item in payloads if item["phase"].startswith("failure_recovery"))
            self.assertEqual(failure_steps, [4, 5, 6, 7, 8])
            success_phases = {item["phase"] for item in payloads if not item["phase"].startswith("failure_recovery")}
            self.assertEqual(success_phases, {"reaching", "precontact", "manipulation", "placement"})
            phase_counts = {phase: sum(item["phase"] == phase for item in payloads) for phase in success_phases}
            self.assertEqual(phase_counts, {"reaching": 2, "precontact": 2, "manipulation": 2, "placement": 1})
            self.assertEqual(set(payloads[0]["observation_images"]), {"agentview", "wrist"})
            self.assertTrue(all(item["sampling_protocol"] == "physical_events_v2" for item in payloads))

    def test_evaluates_all_configs_and_writes_costed_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_file = root / "snapshot.pkl"
            snapshot = {
                "task_suite": "fake",
                "task_id": 0,
                "seed": 1,
                "step": 0,
                "phase": "approach",
                "state": {"position": 0},
            }
            with snapshot_file.open("wb") as destination:
                pickle.dump(snapshot, destination)
            loaded = load_snapshots(snapshot_file)
            table = root / "costs.json"
            table.write_text(
                '{"configurations": ['
                '{"H":0,"D":0,"W":1,"E":10,"relative_compute":0.1},'
                '{"H":2,"D":2,"W":1,"E":10,"relative_compute":0.2},'
                '{"H":2,"D":16,"W":1,"E":10,"relative_compute":0.3},'
                '{"H":4,"D":4,"W":4,"E":10,"relative_compute":0.4},'
                '{"H":8,"D":4,"W":1,"E":10,"relative_compute":0.5},'
                '{"H":8,"D":16,"W":4,"E":10,"relative_compute":1.0}'
                ']}'
            )
            costs = load_relative_costs(table)
            adapter = FakeOracleAdapter(WorkerContext("cpu", "fake", {}))
            records = evaluate_snapshot(
                loaded[0], adapter, costs, state_manager=FakeStateManager, max_episode_steps=12
            )
            output = root / "result.csv"
            write_results(output, records)

            self.assertEqual(len(records), len(CONFIG_GRID))
            self.assertTrue(all(row["success"] == 1 for row in records))
            self.assertTrue(all(row["remaining_steps"] == 6 for row in records))
            self.assertTrue(all(row["eval_mode"] == "sustained" for row in records))
            self.assertTrue(all("actual_elapsed_seconds" in row for row in records))
            self.assertTrue(all("dense_reward" in row for row in records))
            self.assertAlmostEqual(records[1]["estimated_total_relative_compute"], 0.3)
            with output.open(newline="", encoding="utf-8") as source:
                self.assertEqual(len(list(csv.DictReader(source))), 6)
            self.assertEqual(len(read_results(output)), 6)
            self.assertEqual(adapter.inference_configs[0].W, 0)

            duplicate = root / "duplicate.csv"
            with duplicate.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows((records[0], records[0]))
            with self.assertRaises(ValueError):
                read_results(duplicate)

    def test_sustained_mode_keeps_candidate_config_for_every_chunk(self) -> None:
        context = WorkerContext("cpu", "fake", {})
        adapter = FakeOracleAdapter(context)
        observation = adapter.start_task(0, 1)
        config = OracleConfig("B_deep", 2, 16, 1, 2)

        sustained = _run_branch(adapter, observation, config, max_steps=8, eval_mode="sustained")

        self.assertEqual(sustained["success"], 1)
        self.assertEqual(len(adapter.inference_configs), 3)
        self.assertTrue(all(item == InferenceConfig(2, 16, 1, 2) for item in adapter.inference_configs))

    def test_dense_reward_includes_goal_distance_progress(self) -> None:
        adapter = FakeOracleAdapter(WorkerContext("cpu", "fake", {}))
        observation = adapter.start_task(0, 1)
        distances = iter((0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0))

        def next_physical_metrics(*args: Any, **kwargs: Any) -> PhysicalMetrics:
            if not args or "observation" not in kwargs:
                raise AssertionError("Expected environment and observation measurements")
            return PhysicalMetrics("obj", "goal", 0.1, 0.0, 0.0, next(distances))

        with patch(
            "oracle.counterfactual_oracle.measure_physics",
            side_effect=next_physical_metrics,
        ):
            result = _run_branch(
                adapter,
                observation,
                OracleConfig("B_small", 2, 2, 1, 2),
                max_steps=8,
                eval_mode="sustained",
            )

        self.assertAlmostEqual(result["distance_reward"], 0.6)
        self.assertAlmostEqual(result["dense_reward"], result["reward"] + 0.6)
        self.assertGreaterEqual(result["actual_elapsed_seconds"], 0.0)

    def test_physical_phase_thresholds(self) -> None:
        self.assertEqual(
            classify_physical_phase(PhysicalMetrics("obj", "goal", 0.081, 0.0, 0.0, 0.5)),
            "reaching",
        )
        self.assertEqual(
            classify_physical_phase(PhysicalMetrics("obj", "goal", 0.07, 0.0, 0.0, 0.2)),
            "precontact",
        )
        self.assertEqual(
            classify_physical_phase(PhysicalMetrics("obj", "goal", 0.03, 0.2, 0.02, 0.2)),
            "manipulation",
        )

    def test_sustained_cli_loader_excludes_legacy_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy.pkl"
            physical = root / "physical.pkl"
            for path, protocol in ((legacy, None), (physical, "physical_events_v2")):
                with path.open("wb") as destination:
                    payload = {"state": {}, "sampling_protocol": protocol}
                    pickle.dump(payload, destination)

            loaded = load_snapshots(root, required_sampling_protocol="physical_events_v2")

            self.assertEqual([Path(item["snapshot_file"]).name for item in loaded], ["physical.pkl"])

    def test_trajectory_directory_autodiscovery_and_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectories = root / "recorded_episodes"
            trajectories.mkdir()
            (trajectories / "task0_seed3.hdf5").touch()
            configured = root / "missing_no_demonstrations"
            resolved, count, mode = resolve_trajectories_dir(
                root / "missing_requested",
                configured,
                search_root=root,
            )
            self.assertEqual(resolved, trajectories)
            self.assertEqual(count, 1)
            self.assertEqual(mode, "auto-discovered")

            source_config = root / "source_config"
            source_config.mkdir()
            (source_config / "config.yaml").write_text(
                yaml.safe_dump({"benchmark_root": str(root), "datasets": str(configured)}),
                encoding="utf-8",
            )
            runtime_config, dataset_path, file_count, resolution = prepare_libero_config(
                source_config,
                root / "runtime_config",
                root / "missing_requested",
                search_root=root,
            )
            saved_config = yaml.safe_load((runtime_config / "config.yaml").read_text())
            self.assertEqual(Path(saved_config["datasets"]), trajectories)
            self.assertEqual(dataset_path, trajectories)
            self.assertEqual(file_count, 1)
            self.assertEqual(resolution, "auto-discovered")

    def test_no_demonstrations_path_is_created_as_empty_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configured = Path(directory) / "no_demonstrations"
            resolved, count, mode = resolve_trajectories_dir(
                None,
                configured,
                search_root=Path(directory) / "no_outputs",
            )
            self.assertEqual(resolved, configured)
            self.assertEqual(count, 0)
            self.assertEqual(mode, "created-no-demonstrations-placeholder")
            self.assertTrue(configured.is_dir())

    def test_explicit_libero_config_overrides_stale_environment_variable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory) / "runtime_config"
            config_dir.mkdir()
            with patch.dict(os.environ, {"LIBERO_CONFIG_PATH": "/stale/libero/config"}):
                _add_project_import_paths(
                    {
                        "fastwam_root": FASTWAM_ROOT,
                        "libero_root": LIBERO_ROOT,
                        "libero_config_dir": str(config_dir),
                    }
                )
                self.assertEqual(os.environ["LIBERO_CONFIG_PATH"], str(config_dir))


if __name__ == "__main__":
    unittest.main()
