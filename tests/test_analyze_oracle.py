from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from oracle.analyze_oracle import analyze_oracle, read_cost_table, read_counterfactual_results


MOCK_CONFIGS = (
    ("B_action", 0, 0, 1, 10, 0.04),
    ("B_small", 2, 2, 1, 4, 0.08),
    ("B_deep", 2, 16, 1, 2, 0.12),
    ("B_wide", 4, 4, 4, 4, 0.20),
    ("B_far", 8, 4, 1, 4, 0.30),
    ("B_full", 8, 16, 4, 2, 1.00),
)


def _write_mock_inputs(root: Path) -> tuple[Path, Path]:
    cost_path = root / "hardware.json"
    cost_rows = []
    for _, horizon, depth, width, _, relative in MOCK_CONFIGS:
        key = (horizon, depth, width)
        if not any((row["H"], row["D"], row["W"]) == key for row in cost_rows):
            cost_rows.append(
                {
                    "H": horizon,
                    "D": depth,
                    "W": width,
                    "E": 10,
                    "relative_compute": relative,
                }
            )
    cost_path.write_text(
        json.dumps({"device": ["Mock GPU"], "precision": "FP32", "configurations": cost_rows}),
        encoding="utf-8",
    )

    results_path = root / "counterfactual.csv"
    fields = (
        "snapshot_file",
        "task_suite",
        "task_id",
        "seed",
        "repeat",
        "step",
        "phase",
        "configuration",
        "H",
        "D",
        "W",
        "E",
        "success",
        "remaining_steps",
        "reward",
        "estimated_total_relative_compute",
    )
    phase_success_config = {
        "approach": "B_action",
        "pre_contact": "B_deep",
        "grasp_interaction": "B_deep",
        "placement_goal": "B_action",
        "failure_pre_5": "B_deep",
    }
    with results_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for phase_index, (phase, winning_config) in enumerate(phase_success_config.items()):
            for state_index in range(3):
                seed = phase_index * 3 + state_index
                snapshot = f"mock_state_{seed}.pkl"
                for name, horizon, depth, width, execution, relative in MOCK_CONFIGS:
                    for repetition_index in range(2):
                        row: dict[str, Any] = {
                            "snapshot_file": snapshot,
                            "task_suite": "mock_suite",
                            "task_id": 0,
                            "seed": seed,
                            "repeat": repetition_index,
                            "step": 10 + state_index,
                            "phase": phase,
                            "configuration": name,
                            "H": horizon,
                            "D": depth,
                            "W": width,
                            "E": execution,
                            "success": int(name == winning_config),
                            "remaining_steps": 10,
                            "reward": (
                                1.0
                                if name == winning_config
                                else 0.5
                                if snapshot == "mock_state_0.pkl" and name == "B_small"
                                else 0.0
                            ),
                            "estimated_total_relative_compute": relative * 10,
                        }
                        if snapshot == "mock_state_0.pkl" and name == "B_action":
                            row["estimated_total_relative_compute"] = ""
                        writer.writerow(row)
    return results_path, cost_path


class AnalyzeOracleTest(unittest.TestCase):
    def test_full_analysis_aggregates_selects_and_writes_plots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path, cost_path = _write_mock_inputs(root)
            output_dir = root / "results"

            report = analyze_oracle(results_path, cost_path, output_dir)

            self.assertEqual(report["state_count"], 15)
            self.assertEqual(report["config_count"], 6)
            self.assertEqual(report["value_metric"], "hybrid")
            self.assertEqual(
                report["hybrid_value_definition"],
                "0.8 * success_rate + 0.2 * global_minmax_normalized_dense_reward",
            )
            self.assertEqual(len(report["lambda_grid"]), 85)
            self.assertIn(report["rq2"]["selected_lambda"], report["lambda_grid"])
            self.assertIn("knee", report["rq2"]["lambda_selection_method"])
            self.assertEqual(report["rq2"]["best_success_rate"], 1.0)
            self.assertEqual(report["rq1"]["comparison_target_success_rate"], 0.6)
            self.assertIsNotNone(
                report["rq1"]["state_wise_compute_savings_percent_vs_global"]
            )
            self.assertEqual(
                [row["budget_fraction_of_b_full"] for row in report["rq1"]["budget_comparisons"]],
                [0.008, 0.009, 0.010, 0.012, 0.015],
            )
            self.assertTrue(
                all("q_value_lead" in row for row in report["rq1"]["budget_comparisons"])
            )
            phase_means = {
                row["phase"]: row for row in report["rq2"]["phase_distribution"]
            }
            self.assertEqual(phase_means["reaching"]["mean_E"], 10.0)
            self.assertEqual(phase_means["precontact"]["mean_E"], 2.0)
            self.assertEqual(phase_means["manipulation"]["mean_D"], 16.0)
            self.assertEqual(phase_means["placement"]["state_count"], 3)
            self.assertIn("E", report["rq2"]["directional_mann_whitney_tests"])
            self.assertEqual(
                report["rq2"]["directional_mann_whitney_tests"]["E"]["alternative"],
                "less",
            )
            for filename in (
                "oracle_pareto_curve.png",
                "rq2_phase_distribution.png",
                "rq2_config_selection_stacked.png",
                "oracle_lambda_scan.csv",
                "rq2_phase_summary.csv",
                "rq2_statewise_choices.csv",
                "oracle_analysis.json",
            ):
                artifact = output_dir / filename
                self.assertTrue(artifact.is_file(), filename)
                self.assertGreater(artifact.stat().st_size, 0, filename)

    def test_cost_fallback_uses_hardware_cost_per_configured_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path, cost_path = _write_mock_inputs(root)
            costs, _ = read_cost_table(cost_path)
            estimates = read_counterfactual_results(results_path, costs)
            action_estimate = next(
                estimate
                for estimate in estimates
                if estimate.state_id == "mock_state_0.pkl"
                and estimate.config.H == 0
            )
            self.assertAlmostEqual(action_estimate.relative_compute, 0.04)
            self.assertEqual(action_estimate.repeat_count, 2)
            self.assertEqual(action_estimate.value, 1.0)
            unsuccessful_estimate = next(
                estimate
                for estimate in estimates
                if estimate.state_id == "mock_state_0.pkl"
                and estimate.config.H == 2
                and estimate.config.D == 2
            )
            self.assertAlmostEqual(unsuccessful_estimate.value, 0.1)


if __name__ == "__main__":
    unittest.main()
