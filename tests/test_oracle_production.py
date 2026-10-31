from __future__ import annotations

import unittest

from run_oracle_production import baseline_capacity, build_parser, _parse_task_ids


class OracleProductionTest(unittest.TestCase):
    def test_capacity_counts_only_requested_paired_episodes(self) -> None:
        rows = []
        for task_id, seed, no_imagination, fixed_full in (
            (0, 0, 0, 1),
            (0, 1, 1, 1),
            (1, 0, 0, 0),
        ):
            rows.extend(
                (
                    {
                        "task_suite": "suite",
                        "task_id": task_id,
                        "seed": seed,
                        "method": "No-Imagination",
                        "success": no_imagination,
                    },
                    {
                        "task_suite": "suite",
                        "task_id": task_id,
                        "seed": seed,
                        "method": "Fixed-Full",
                        "success": fixed_full,
                    },
                )
            )
        rows.append(
            {
                "task_suite": "other",
                "task_id": 0,
                "seed": 0,
                "method": "Fixed-Full",
                "success": 1,
            }
        )

        capacity = baseline_capacity(
            rows,
            task_suite="suite",
            task_ids=(0, 1),
            seed_start=0,
            seed_count=2,
        )

        self.assertEqual(capacity["requested_episodes"], 4)
        self.assertEqual(capacity["paired_episodes"], 3)
        self.assertEqual(capacity["fixed_full_successes"], 2)
        self.assertEqual(capacity["watershed_episodes"], 1)
        self.assertEqual(capacity["phase_candidate_capacity"], 4)
        self.assertEqual(capacity["failure_recovery_candidate_capacity"], 5)

    def test_task_id_parser(self) -> None:
        self.assertEqual(_parse_task_ids("0-2,4"), [0, 1, 2, 4])

    def test_production_defaults_target_about_three_hundred_snapshots(self) -> None:
        args = build_parser().parse_args(["--output-root", "results/test"])
        self.assertEqual(args.snapshot_quota_per_category, 60)
        self.assertEqual(args.minimum_per_category, 50)


if __name__ == "__main__":
    unittest.main()
