from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_rq2_followup_when_gpus_free import _wait_for_gpus, main


class FollowupSchedulerTest(unittest.TestCase):
    def wait(self, samples, gpu_ids=None):
        with tempfile.TemporaryDirectory() as directory:
            status = Path(directory) / "status.json"
            with patch(
                "scripts.run_rq2_followup_when_gpus_free._gpu_stats",
                side_effect=samples,
            ), patch("scripts.run_rq2_followup_when_gpus_free.time.sleep"):
                selected = _wait_for_gpus(
                    gpu_ids, status, poll_seconds=1, timeout_hours=1
                )
            return selected, json.loads(status.read_text())

    def test_auto_uses_idle_cards_outside_original_group(self):
        stats = {i: (9, 0) if i in (1, 2, 8, 9) else (900, 45) for i in range(10)}
        selected, status = self.wait([stats] * 3)
        self.assertEqual(selected, [1, 2, 8, 9])
        self.assertEqual(status["gpu_selection"], "auto")
        self.assertEqual(status["candidate_gpu_ids"], list(range(10)))

    def test_each_selected_card_must_have_three_consecutive_free_checks(self):
        samples = [
            {i: (9, 0) if i in free else (900, 40) for i in range(5)}
            for free in (
                {0, 1, 2, 3},
                {0, 1, 2, 4},
                {0, 1, 2, 3},
                {0, 1, 2, 3},
                {0, 1, 2, 3},
            )
        ]
        selected, status = self.wait(samples)
        self.assertEqual(selected, [0, 1, 2, 3])
        self.assertEqual(status["free_check_counts"]["3"], 3)
        self.assertEqual(status["free_check_counts"]["0"], 5)

    def test_fixed_mode_preserves_requested_placement_order(self):
        stats = {i: (9, 0) for i in range(10)}
        selected, status = self.wait([stats] * 3, [6, 0, 5, 4])
        self.assertEqual(selected, [6, 0, 5, 4])
        self.assertEqual(status["gpu_selection"], "fixed")

    def test_low_memory_active_cards_are_not_selected(self):
        stats = {i: (9, 0) for i in range(6)}
        stats[0] = (450, 50)
        stats[1] = (1000, 0)
        selected, _ = self.wait([stats] * 3)
        self.assertEqual(selected, [2, 3, 4, 5])

    def test_decoupled_only_starts_directly_on_selected_gpus(self):
        module = "scripts.run_rq2_followup_when_gpus_free"
        with patch("sys.argv", ["followup", "--gpu-ids", "auto", "--decoupled-only"]), patch(
            f"{module}._wait_for_gpus", return_value=[1, 3, 7, 9]
        ), patch(f"{module}._run") as run, patch(f"{module}._atomic_json"):
            main()
        self.assertEqual(run.call_count, 2)
        command = run.call_args_list[0].args[0]
        self.assertIn("1,3,7,9", command)
        self.assertTrue(any("snapshots_decoupled_core_v1" in item for item in command))
        self.assertEqual(run.call_args_list[0].args[2], "running_decoupled_grid")
        self.assertEqual(run.call_args_list[1].args[2], "analyzing_decoupled_grid")


if __name__ == "__main__":
    unittest.main()
