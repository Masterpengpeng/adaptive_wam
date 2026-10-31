from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from analyze_oracle import analyze
from collect_counterfactual_data import (
    InferenceConfig,
    WorkerContext,
    scan_task,
)
from tests.fake_counterfactual_adapter import FakeCounterfactualAdapter


class CounterfactualOracleTest(unittest.TestCase):
    def test_scan_saves_recoverable_state_checkpoints(self):
        adapter = FakeCounterfactualAdapter(
            WorkerContext(device="cpu", task_suite="fake", options={})
        )
        with tempfile.TemporaryDirectory() as directory:
            nodes = scan_task(
                adapter,
                task_suite="fake",
                task_id=3,
                episode_seed=4,
                scan_config=InferenceConfig(0, 0, 0, 1),
                output_dir=Path(directory),
                scan_stride=2,
                max_steps=5,
                max_states=3,
                gamma=1.0,
                device="cpu",
            )
            self.assertEqual([node["scan_step"] for node in nodes], [0, 2, 4])
            payload = torch.load(
                Path(directory) / nodes[1]["checkpoint_file"],
                map_location="cpu",
                weights_only=False,
            )
            observation = adapter.restore_checkpoint(payload["checkpoint"])
            self.assertEqual(observation, 2.0)

    def test_statewise_oracle_beats_fixed_config_under_same_budget(self):
        slow_config = {"H": 0, "D": 0, "W": 0, "E": 1}
        planned_config = {"H": 2, "D": 2, "W": 1, "E": 1}
        records = [
            {
                "task_suite": "fake",
                "task_id": 0,
                "state_id": "state-a",
                "config": slow_config,
                "episode_return": 1.0,
                "success": False,
                "inference_calls": 1,
            },
            {
                "task_suite": "fake",
                "task_id": 0,
                "state_id": "state-a",
                "config": planned_config,
                "episode_return": 2.0,
                "success": True,
                "inference_calls": 1,
            },
            {
                "task_suite": "fake",
                "task_id": 0,
                "state_id": "state-b",
                "config": slow_config,
                "episode_return": 2.0,
                "success": True,
                "inference_calls": 1,
            },
            {
                "task_suite": "fake",
                "task_id": 0,
                "state_id": "state-b",
                "config": planned_config,
                "episode_return": 1.0,
                "success": False,
                "inference_calls": 1,
            },
        ]
        summary = analyze(
            records,
            {(0, 0, 0): 1.0, (2, 2, 1): 1.0},
            lambda_cost=0.0,
            budget_ms=1.0,
        )
        strategies = summary["strategies"]
        self.assertEqual(strategies["global_fixed_oracle"]["success_rate"], 0.5)
        self.assertEqual(strategies["state_wise_oracle"]["success_rate_at_budget"], 1.0)
        self.assertEqual(strategies["task_wise"]["success_rate"], 0.5)

    def test_taskwise_selection_can_improve_over_one_global_config(self):
        config_a = {"H": 0, "D": 0, "W": 0, "E": 1}
        config_b = {"H": 2, "D": 2, "W": 1, "E": 1}
        records = []
        for task_id, better_config in ((0, config_a), (1, config_b)):
            other_config = config_b if better_config == config_a else config_a
            for state_index in range(2):
                records.extend(
                    (
                        {
                            "task_suite": "fake",
                            "task_id": task_id,
                            "state_id": f"{task_id}-{state_index}",
                            "config": better_config,
                            "episode_return": 2.0,
                            "success": True,
                            "inference_calls": 1,
                        },
                        {
                            "task_suite": "fake",
                            "task_id": task_id,
                            "state_id": f"{task_id}-{state_index}",
                            "config": other_config,
                            "episode_return": 1.0,
                            "success": False,
                            "inference_calls": 1,
                        },
                    )
                )
        summary = analyze(
            records,
            {(0, 0, 0): 1.0, (2, 2, 1): 1.0},
            budget_ms=1.0,
        )
        strategies = summary["strategies"]
        self.assertEqual(strategies["global_fixed_oracle"]["success_rate"], 0.5)
        self.assertEqual(strategies["task_wise"]["success_rate_at_budget"], 1.0)
        self.assertEqual(strategies["state_wise_oracle"]["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
