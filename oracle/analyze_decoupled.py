"""Analyze single-axis H/D/W counterfactual responses on a compact hard-state pool."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AXES = {
    "H": (("Base_H2_D2_W1", 2), ("H4", 4), ("H8", 8)),
    "D": (("Base_H2_D2_W1", 2), ("D4", 4), ("D8", 8), ("D16", 16)),
    "W": (("Base_H2_D2_W1", 1), ("W2", 2), ("W4", 4)),
}
ANCHORS = ("ActionOnly", "Base_H2_D2_W1", "JointFull")


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0,
    )


def analyze(results_csv: Path, output_dir: Path) -> dict[str, Any]:
    with results_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No decoupled results in {results_csv}")
    by_state_config: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_state_config[(row["snapshot_file"], row["configuration"])].append(row)
    state_rows: list[dict[str, Any]] = []
    for (snapshot, configuration), repeats in by_state_config.items():
        first = repeats[0]
        state_rows.append(
            {
                "snapshot_file": snapshot,
                "phase": first["phase"],
                "configuration": configuration,
                "H": int(first["H"]),
                "D": int(first["D"]),
                "W": int(first["W"]),
                "E": int(first["E"]),
                "success_rate": float(np.mean([float(row["success"]) for row in repeats])),
                "dense_reward": float(np.mean([float(row["dense_reward"]) for row in repeats])),
                "relative_compute": float(
                    np.mean([float(row["estimated_total_relative_compute"]) for row in repeats])
                ),
                "repeat_count": len(repeats),
            }
        )

    phases = sorted({str(row["phase"]) for row in state_rows})
    summary: list[dict[str, Any]] = []
    for configuration in sorted({str(row["configuration"]) for row in state_rows}):
        for phase in (*phases, "all"):
            selected = [
                row
                for row in state_rows
                if row["configuration"] == configuration
                and (phase == "all" or row["phase"] == phase)
            ]
            if not selected:
                continue
            success_mean, success_se = _mean_se([row["success_rate"] for row in selected])
            reward_mean, reward_se = _mean_se([row["dense_reward"] for row in selected])
            summary.append(
                {
                    "configuration": configuration,
                    "phase": phase,
                    "state_count": len(selected),
                    "H": selected[0]["H"],
                    "D": selected[0]["D"],
                    "W": selected[0]["W"],
                    "E": selected[0]["E"],
                    "success_rate": success_mean,
                    "success_se": success_se,
                    "dense_reward": reward_mean,
                    "dense_reward_se": reward_se,
                    "relative_compute": float(np.mean([row["relative_compute"] for row in selected])),
                }
            )

    by_config_phase = {
        (row["configuration"], row["phase"]): row for row in summary
    }
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = {"all": "#222222", "manipulation": "#e45756", "failure_recovery": "#4c78a8"}
    for axis, (parameter, settings) in zip(axes.flat[:3], AXES.items()):
        for phase in ("all", *phases):
            available = [
                (value, by_config_phase[(name, phase)])
                for name, value in settings
                if (name, phase) in by_config_phase
            ]
            if not available:
                continue
            axis.errorbar(
                [item[0] for item in available],
                [float(item[1]["success_rate"]) for item in available],
                yerr=[float(item[1]["success_se"]) for item in available],
                marker="o",
                capsize=4,
                linewidth=2,
                color=colors.get(phase),
                label=phase,
            )
        axis.set_title(f"Single-axis {parameter} response")
        axis.set_xlabel(parameter)
        axis.set_ylabel("Success rate (mean ± SE)")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)

    anchor_axis = axes.flat[3]
    anchor_rows = [by_config_phase[(name, "all")] for name in ANCHORS]
    anchor_axis.bar(
        range(len(anchor_rows)),
        [float(row["success_rate"]) for row in anchor_rows],
        yerr=[float(row["success_se"]) for row in anchor_rows],
        capsize=4,
        color=("#4c78a8", "#f58518", "#b279a2"),
    )
    anchor_axis.set_xticks(range(len(anchor_rows)), ("Action", "Single-axis base", "Joint full"))
    anchor_axis.set_ylim(-0.05, 1.05)
    anchor_axis.set_ylabel("Success rate (mean ± SE)")
    anchor_axis.set_title("Action-only vs single-axis base vs joint")
    anchor_axis.grid(True, axis="y", alpha=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "decoupled_axis_response.png", dpi=180)
    plt.close(figure)
    with (output_dir / "decoupled_axis_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report = {
        "state_count": len({row["snapshot_file"] for row in state_rows}),
        "configuration_count": len({row["configuration"] for row in state_rows}),
        "result_rows": len(rows),
        "phases": phases,
        "repeat_counts": sorted({row["repeat_count"] for row in state_rows}),
        "summary": summary,
        "interpretation": (
            "Exploratory single-axis response analysis; one-repeat results are descriptive "
            "and require replication before inferential claims."
        ),
    }
    (output_dir / "decoupled_axis_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = analyze(args.results_csv, args.output_dir)
    print(
        f"Analyzed {report['state_count']} states and "
        f"{report['configuration_count']} decoupled configurations"
    )


if __name__ == "__main__":
    main()
