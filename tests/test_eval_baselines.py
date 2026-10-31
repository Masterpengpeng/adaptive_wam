from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from collect_counterfactual_data import StepResult
from eval_fixed_baselines import (
    FIXED_CONFIGS,
    aggregate_results,
    build_parser,
    evaluate_episode,
    load_trajectory,
    load_hardware_costs,
    merge_episode_records,
    parse_methods,
    parse_seeds,
    parse_task_ids,
    read_episode_csv,
    render_markdown_table,
    write_csv,
)


class MockSim:
    def __init__(self) -> None:
        self.model = SimpleNamespace()
        self.data = SimpleNamespace(
            qpos=np.zeros(2, dtype=np.float64),
            qvel=np.zeros(2, dtype=np.float64),
            act=np.zeros(0, dtype=np.float64),
            ctrl=np.zeros(1, dtype=np.float64),
            qacc_warmstart=np.zeros(2, dtype=np.float64),
            qfrc_applied=np.zeros(2, dtype=np.float64),
            xfrc_applied=np.zeros((0, 6), dtype=np.float64),
            mocap_pos=np.zeros((0, 3), dtype=np.float64),
            mocap_quat=np.zeros((0, 4), dtype=np.float64),
            userdata=np.zeros(0, dtype=np.float64),
            eq_active=np.zeros(0, dtype=bool),
            time=0.0,
        )

    def get_state(self) -> Any:
        return SimpleNamespace(
            flatten=lambda: np.concatenate(
                (self.data.qpos, self.data.qvel, np.asarray([self.data.time]))
            )
        )

    def set_state_from_flattened(self, state: np.ndarray) -> None:
        self.data.qpos[...] = state[:2]
        self.data.qvel[...] = state[2:4]
        self.data.time = float(state[4])

    def forward(self) -> None:
        return None


class MockEnv:
    def __init__(self) -> None:
        self.sim = MockSim()
        self.timestep = 0


class MockAdapter:
    def __init__(self) -> None:
        self.env = MockEnv()
        self.observation: dict[str, int] = {"step": 0}
        self.cuda_devices: tuple[str, ...] = ()
        self.calls: list[tuple[int, int, int, int]] = []

    def start_task(self, task_id: int, seed: int) -> dict[str, int]:
        self.env.sim.data.qpos[:] = (float(task_id), float(seed))
        self.observation = {"step": 0}
        return self.observation

    def parameterized_wam(
        self,
        observation: Any,
        H: int,
        D: int,
        W: int,
        E: int,
    ) -> Tensor:
        del observation
        self.calls.append((H, D, W, E))
        return torch.zeros((E, 2), dtype=torch.float32)

    def step(self, action: np.ndarray) -> StepResult:
        del action
        self.observation = {"step": self.observation["step"] + 1}
        self.env.timestep += 1
        self.env.sim.data.time += 0.05
        self.env.sim.data.qpos[0] += 0.01
        return StepResult(
            observation=self.observation,
            reward=1.0,
            terminated=False,
            truncated=False,
            info={},
        )

    def is_success(self, observation: Any, info: dict[str, Any]) -> bool:
        del info
        return int(observation["step"]) >= 3

    def close(self) -> None:
        return None


class FixedBaselineEvaluationTest(unittest.TestCase):
    def test_parser_and_task_id_ranges(self) -> None:
        args = build_parser().parse_args(
            [
                "--model-path",
                "fake.pt",
                "--task-ids",
                "0-2,4",
                "--num-seeds",
                "2",
                "--merge-episodes-csv",
                "old.csv",
            ]
        )

        self.assertEqual(parse_task_ids(args.task_ids), [0, 1, 2, 4])
        self.assertEqual(args.num_seeds, 2)
        self.assertEqual(args.task_suite, "libero_spatial")
        self.assertEqual(args.merge_episodes_csv, [Path("old.csv")])
        task_id_args = build_parser().parse_args(["--task-id", "3"])
        self.assertEqual(parse_task_ids(task_id_args.task_ids), [3])
        with self.assertRaises(ValueError):
            parse_task_ids("2-1")
        self.assertEqual(
            [config.name for config in parse_methods("No-Imagination,Fixed-Full")],
            ["No-Imagination", "Fixed-Full"],
        )
        with self.assertRaises(ValueError):
            parse_methods("unknown")
        self.assertEqual(parse_seeds("0-2,4"), [0, 1, 2, 4])
        trajectory_args = build_parser().parse_args(
            ["--seeds", "0,2", "--save-trajectories", "--trajectory-stride", "2"]
        )
        self.assertEqual(trajectory_args.seeds, "0,2")
        self.assertTrue(trajectory_args.save_trajectories)
        self.assertEqual(trajectory_args.trajectory_stride, 2)

    def test_trajectory_capture_is_replayable_and_linked_from_csv_row(self) -> None:
        adapter = MockAdapter()
        with tempfile.TemporaryDirectory() as directory:
            trajectory_root = Path(directory) / "trajectories"
            records = evaluate_episode(
                adapter,
                task_suite="mock",
                task_id=1,
                seed=2,
                max_steps=8,
                configs=(FIXED_CONFIGS[0],),
                warmup_iterations=0,
                save_trajectories=True,
                trajectory_dir=trajectory_root,
                trajectory_stride=1,
            )

            self.assertEqual(len(records), 1)
            path = Path(records[0]["trajectory_path"])
            self.assertTrue(path.is_file())
            self.assertEqual(
                path.relative_to(trajectory_root).as_posix(),
                "mock/task_1/seed_2/No-Imagination_ep0.pt",
            )
            compact = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(len(compact["steps"]), 3)
            self.assertEqual(
                set(compact["steps"][0]),
                {
                    "step", "action", "reward", "done", "success",
                    "sim_state", "obs", "contact", "phase", "physical_metrics",
                },
            )
            restored = load_trajectory(path)
            self.assertEqual(restored["steps"][-1]["step"], 3)
            self.assertTrue(restored["steps"][-1]["success"])
            self.assertIn("simulation", restored["steps"][0]["sim_state"])

    def test_episode_methods_restore_same_initial_state_and_record_metrics(self) -> None:
        adapter = MockAdapter()
        warmed_configs: set[tuple[int, int, int, int]] = set()

        records = evaluate_episode(
            adapter,
            task_suite="mock",
            task_id=0,
            seed=7,
            max_steps=8,
            warmed_configs=warmed_configs,
        )

        self.assertEqual([row["method"] for row in records], [
            "No-Imagination",
            "Fixed-Small",
            "Fixed-Full",
        ])
        self.assertTrue(all(row["success"] == 1 for row in records))
        self.assertTrue(all(row["steps"] == 3 for row in records))
        self.assertTrue(all(row["episode_return"] == 3.0 for row in records))
        self.assertEqual([row["wam_calls"] for row in records], [1, 1, 2])
        self.assertTrue(
            all("local_latency_ms_total" in row for row in records)
        )
        self.assertEqual(adapter.env.sim.data.qpos[1], 7.0)
        self.assertEqual(len(adapter.calls), 19)
        self.assertEqual(
            adapter.calls.count((0, 0, 1, 10)),
            6,
        )
        self.assertEqual(
            adapter.calls.count((2, 2, 1, 4)),
            6,
        )
        self.assertEqual(
            adapter.calls.count((8, 16, 4, 2)),
            7,
        )
        evaluate_episode(
            adapter,
            task_suite="mock",
            task_id=0,
            seed=8,
            max_steps=8,
            warmed_configs=warmed_configs,
        )
        self.assertEqual(len(adapter.calls), 23)

    def test_hardware_cost_aggregation_and_csv_export(self) -> None:
        call_counts = (1, 3, 2)
        records = [
            {
                "method": config.name,
                "success": int(config.name != "Fixed-Full"),
                "steps": 5,
                "episode_return": 2.0,
                "wam_calls": calls,
                "local_latency_ms_mean": 10.0,
                "local_latency_ms_total": 10.0 * calls,
            }
            for config, calls in zip(FIXED_CONFIGS, call_counts)
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            cost_path = directory_path / "hardware.json"
            cost_path.write_text(
                """
                {"configurations": [
                  {"H": 0, "D": 0, "W": 1, "E": 10,
                   "latency_ms_mean": 2.0, "relative_compute": 0.1},
                  {"H": 2, "D": 2, "W": 1, "E": 4,
                   "latency_ms_mean": 4.0, "relative_compute": 0.2},
                  {"H": 8, "D": 16, "W": 4, "E": 2,
                   "latency_ms_mean": 20.0, "relative_compute": 1.0}
                ]}
                """,
                encoding="utf-8",
            )
            costs = load_hardware_costs(cost_path)
            summary = aggregate_results(records, costs)
            self.assertEqual(len(summary), 3)
            self.assertEqual(
                [row["avg_episode_compute_ms"] for row in summary],
                [2.0, 12.0, 40.0],
            )
            self.assertEqual(
                [row["relative_compute_percent"] for row in summary],
                [5.0, 30.0, 100.0],
            )
            self.assertEqual(
                [row["episode_latency_s"] for row in summary],
                [0.002, 0.012, 0.04],
            )
            markdown = render_markdown_table(summary)
            self.assertIn(
                "| Method | Success Rate | Episode Latency (s) | Relative Compute (%) | Avg WAM Calls |",
                markdown,
            )

            csv_path = directory_path / "summary.csv"
            write_csv(csv_path, summary)
            with csv_path.open(newline="", encoding="utf-8") as source:
                saved_rows = list(csv.DictReader(source))
            self.assertEqual(len(saved_rows), 3)
            self.assertEqual(saved_rows[2]["method"], "Fixed-Full")

    def test_without_hardware_table_uses_local_timing(self) -> None:
        calls_per_episode = (1, 2, 4)
        records = [
            {
                "method": config.name,
                "success": 1,
                "steps": 4,
                "episode_return": 1.0,
                "wam_calls": calls,
                "local_latency_ms_mean": latency,
                "local_latency_ms_total": latency * calls,
            }
            for config, calls, latency in zip(
                FIXED_CONFIGS, calls_per_episode, (2.0, 4.0, 8.0)
            )
        ]
        summary = aggregate_results(records, {})

        self.assertEqual(
            [row["avg_episode_compute_ms"] for row in summary],
            [2.0, 8.0, 32.0],
        )
        self.assertEqual(
            [row["relative_compute_percent"] for row in summary],
            [6.25, 25.0, 100.0],
        )
        self.assertEqual(
            [row["episode_latency_s"] for row in summary],
            [0.002, 0.008, 0.032],
        )
        self.assertTrue(
            all(row["cost_source"] == "measured_local_wall_time" for row in summary)
        )

    def test_resume_csv_merge_deduplicates_and_rejects_conflicts(self) -> None:
        first = {
            "task_suite": "mock",
            "task_id": 0,
            "seed": 1,
            "method": "No-Imagination",
            "success": 1,
        }
        second = {
            "task_suite": "mock",
            "task_id": 0,
            "seed": 1,
            "method": "Fixed-Full",
            "success": 1,
        }
        self.assertEqual(len(merge_episode_records([first], [first, second])), 2)
        with self.assertRaises(ValueError):
            merge_episode_records([first], [{**first, "success": 0}])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.csv"
            write_csv(path, [first, second])
            self.assertEqual(len(read_episode_csv(path)), 2)


if __name__ == "__main__":
    unittest.main()
