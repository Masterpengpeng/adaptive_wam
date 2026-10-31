"""Analyze Phase 2 counterfactual results and test RQ1/RQ2 hypotheses."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results" / "oracle_counterfactual_results.csv"
DEFAULT_COST_TABLE = PROJECT_ROOT / "configs" / "hardware_cost_table.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
LAMBDA_GRID = np.unique(
    np.concatenate(
        (
            np.logspace(-4, -2, 41),
            np.linspace(0.01, 0.10, 19),
            np.linspace(0.10, 0.50, 17),
            np.linspace(0.50, 1.00, 11),
        )
    )
)
RQ1_BUDGETS = (0.008, 0.009, 0.010, 0.012, 0.015)
PHASE_ORDER = ("reaching", "precontact", "manipulation", "placement", "failure_recovery")
PARAMETERS = ("H", "D", "W", "E")


@dataclass(frozen=True, order=True)
class Config:
    H: int
    D: int
    W: int
    E: int

    def key(self) -> tuple[int, int, int, int]:
        return self.H, self.D, self.W, self.E


@dataclass(frozen=True)
class StateConfigEstimate:
    state_id: str
    task_suite: str
    task_id: int
    phase: str
    config: Config
    success_rate: float
    mean_return: float
    normalized_reward: float
    value: float
    relative_compute: float
    repeat_count: int


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def read_cost_table(path: Path) -> tuple[dict[tuple[int, int, int], dict[str, float]], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read hardware cost table {path}: {exc}") from exc
    rows = payload.get("configurations")
    if not isinstance(rows, list):
        raise ValueError(f"Hardware cost table {path} has no configurations list")
    costs: dict[tuple[int, int, int], dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            key = (int(row["H"]), int(row["D"]), int(row["W"]))
            relative = float(row["relative_compute"])
            table_e = float(row.get("E", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed hardware cost row: {row}") from exc
        if not np.isfinite(relative) or relative < 0 or table_e < 0:
            raise ValueError(f"Invalid hardware cost for H/D/W={key}")
        if key in costs:
            raise ValueError(f"Duplicate hardware cost for H/D/W={key}")
        costs[key] = {"relative_compute": relative, "table_E": table_e}
    if not costs:
        raise ValueError(f"No usable hardware costs in {path}")
    return costs, payload


def _state_id(row: Mapping[str, Any]) -> str:
    if row.get("snapshot_file"):
        return str(row["snapshot_file"])
    required = ("task_suite", "task_id", "seed", "step", "phase")
    if not all(row.get(key) not in (None, "") for key in required):
        raise ValueError(f"Result row has neither snapshot_file nor complete state identity: {row}")
    return "|".join(str(row[key]) for key in required)


def _row_cost(
    row: Mapping[str, Any],
    config: Config,
    hardware_costs: Mapping[tuple[int, int, int], Mapping[str, float]],
) -> float:
    cumulative = row.get("estimated_total_relative_compute")
    if cumulative not in (None, ""):
        try:
            value = float(cumulative)
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value) and value >= 0:
            return value

    key = (config.H, config.D, config.W)
    if key not in hardware_costs:
        raise ValueError(f"No hardware relative_compute for H/D/W={key}")
    if config.E <= 0:
        raise ValueError(f"E must be positive for config {config.key()}")
    steps_value = row.get("remaining_steps", row.get("steps"))
    try:
        steps = float(steps_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot determine branch step count for state {_state_id(row)}") from exc
    if not np.isfinite(steps) or steps < 0:
        raise ValueError(f"Invalid branch step count {steps_value!r}")
    step_cost = float(hardware_costs[key]["relative_compute"]) / config.E
    return step_cost * steps


def read_counterfactual_results(
    results_path: Path,
    hardware_costs: Mapping[tuple[int, int, int], Mapping[str, float]],
    *,
    value_metric: str = "hybrid",
) -> list[StateConfigEstimate]:
    if value_metric not in {"hybrid", "success", "return"}:
        raise ValueError("value_metric must be 'hybrid', 'success', or 'return'")
    with results_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No counterfactual results in {results_path}")

    grouped: dict[tuple[str, Config], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            config = Config(*(int(row[key]) for key in PARAMETERS))
            success = float(_as_bool(row["success"]))
            reward_value_raw = next(
                (
                    row[key]
                    for key in ("dense_reward", "reward", "episode_return")
                    if row.get(key) not in (None, "")
                ),
                None,
            )
            if reward_value_raw is None:
                raise ValueError("missing dense_reward/reward/episode_return")
            reward_value = float(reward_value_raw)
            task_id = int(row["task_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed counterfactual result row: {row}") from exc
        if config.E <= 0 or min(config.H, config.D, config.W) < 0:
            raise ValueError(f"Invalid inference configuration {config.key()}")
        if not np.isfinite(reward_value):
            raise ValueError(f"Non-finite reward in state {_state_id(row)}")
        row["_success_value"] = success
        row["_return_value"] = reward_value
        row["_cost_value"] = _row_cost(row, config, hardware_costs)
        row["_task_id_value"] = task_id
        grouped[(_state_id(row), config)].append(row)

    branch_reward_means = {
        key: float(np.mean([row["_return_value"] for row in repeats]))
        for key, repeats in grouped.items()
    }
    reward_min = min(branch_reward_means.values())
    reward_max = max(branch_reward_means.values())
    reward_range = reward_max - reward_min

    estimates: list[StateConfigEstimate] = []
    for (state_id, config), repeats in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        first = repeats[0]
        task_suite = str(first.get("task_suite", ""))
        phase = str(first.get("phase", "unknown"))
        if any(str(row.get("task_suite", "")) != task_suite for row in repeats):
            raise ValueError(f"Inconsistent task suite for state {state_id}")
        success_rate = float(np.mean([row["_success_value"] for row in repeats]))
        mean_return = float(np.mean([row["_return_value"] for row in repeats]))
        normalized_reward = (
            (mean_return - reward_min) / reward_range if reward_range > 0 else 0.5
        )
        if value_metric == "hybrid":
            value = 0.8 * success_rate + 0.2 * normalized_reward
        elif value_metric == "success":
            value = success_rate
        else:
            value = mean_return
        estimates.append(
            StateConfigEstimate(
                state_id=state_id,
                task_suite=task_suite,
                task_id=int(first["_task_id_value"]),
                phase=phase,
                config=config,
                success_rate=success_rate,
                mean_return=mean_return,
                normalized_reward=normalized_reward,
                value=value,
                relative_compute=float(np.mean([row["_cost_value"] for row in repeats])),
                repeat_count=len(repeats),
            )
        )
    return estimates


def _choose(estimates: Sequence[StateConfigEstimate], lambda_value: float) -> StateConfigEstimate:
    return max(
        estimates,
        key=lambda item: (
            item.value - lambda_value * item.relative_compute,
            item.value,
            -item.relative_compute,
            tuple(-value for value in item.config.key()),
        ),
    )


def _strategy_selection(
    estimates: Sequence[StateConfigEstimate], lambda_value: float, strategy: str
) -> dict[str, StateConfigEstimate]:
    by_state: dict[str, list[StateConfigEstimate]] = defaultdict(list)
    for estimate in estimates:
        by_state[estimate.state_id].append(estimate)
    state_metadata = {state_id: rows[0] for state_id, rows in by_state.items()}

    if strategy == "state_wise":
        return {state_id: _choose(rows, lambda_value) for state_id, rows in by_state.items()}

    task_states: dict[tuple[str, int], list[str]] = defaultdict(list)
    for state_id, metadata in state_metadata.items():
        task_states[(metadata.task_suite, metadata.task_id)].append(state_id)

    def select_shared(state_ids: Sequence[str]) -> Config:
        option_maps = {
            state_id: {item.config: item for item in by_state[state_id]}
            for state_id in state_ids
        }
        configs = set.intersection(*(set(options) for options in option_maps.values()))
        if not configs:
            raise ValueError(f"No common configuration across states {state_ids}")
        return max(
            configs,
            key=lambda config: (
                float(np.mean([
                    option_maps[state_id][config].value
                    - lambda_value * option_maps[state_id][config].relative_compute
                    for state_id in state_ids
                ])),
                tuple(-value for value in config.key()),
            ),
        )

    selections: dict[str, StateConfigEstimate] = {}
    if strategy == "task_wise":
        for state_ids in task_states.values():
            selected_config = select_shared(state_ids)
            for state_id in state_ids:
                selections[state_id] = next(
                    item for item in by_state[state_id] if item.config == selected_config
                )
        return selections
    if strategy == "global_fixed":
        selected_config = select_shared(list(by_state))
        for state_id, options in by_state.items():
            selections[state_id] = next(item for item in options if item.config == selected_config)
        return selections
    raise ValueError(f"Unknown oracle strategy {strategy!r}")


def _metrics(selections: Mapping[str, StateConfigEstimate]) -> dict[str, float | int]:
    selected = list(selections.values())
    return {
        "state_count": len(selected),
        "success_rate": float(np.mean([item.success_rate for item in selected])),
        "mean_return": float(np.mean([item.mean_return for item in selected])),
        "q_value": float(np.mean([item.value for item in selected])),
        "relative_compute": float(np.mean([item.relative_compute for item in selected])),
    }


def scan_lambdas(
    estimates: Sequence[StateConfigEstimate],
    lambdas: Sequence[float] = LAMBDA_GRID,
) -> tuple[list[dict[str, Any]], dict[float, dict[str, dict[str, StateConfigEstimate]]]]:
    scan_rows: list[dict[str, Any]] = []
    selections_by_lambda: dict[float, dict[str, dict[str, StateConfigEstimate]]] = {}
    for lambda_value in lambdas:
        selections_by_strategy = {
            strategy: _strategy_selection(estimates, float(lambda_value), strategy)
            for strategy in ("state_wise", "task_wise", "global_fixed")
        }
        selections_by_lambda[float(lambda_value)] = selections_by_strategy
        for strategy, selections in selections_by_strategy.items():
            scan_rows.append(
                {"lambda": float(lambda_value), "strategy": strategy, **_metrics(selections)}
            )
    return scan_rows, selections_by_lambda


def _pareto_points(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    ordered = sorted(
        ({"relative_compute": float(row["relative_compute"]), "q_value": float(row["q_value"])} for row in rows),
        key=lambda point: (point["relative_compute"], -point["q_value"]),
    )
    frontier: list[dict[str, float]] = []
    best_success = -float("inf")
    for point in ordered:
        if point["q_value"] > best_success + 1e-12:
            frontier.append(point)
            best_success = point["q_value"]
    return frontier


def _fixed_config_metrics(
    estimates: Sequence[StateConfigEstimate],
) -> list[dict[str, Any]]:
    by_state: dict[str, dict[Config, StateConfigEstimate]] = defaultdict(dict)
    for estimate in estimates:
        by_state[estimate.state_id][estimate.config] = estimate
    configs = sorted(set.intersection(*(set(rows) for rows in by_state.values())))
    labels = {
        Config(0, 0, 1, 10): "B_action",
        Config(2, 2, 1, 4): "B_small",
        Config(2, 16, 1, 2): "B_deep",
        Config(4, 4, 4, 4): "B_wide",
        Config(8, 4, 1, 4): "B_far",
        Config(8, 16, 4, 2): "B_full",
    }
    rows: list[dict[str, Any]] = []
    for config in configs:
        metrics = _metrics(
            {state_id: options[config] for state_id, options in by_state.items()}
        )
        rows.append({"configuration": labels.get(config, str(config.key())), **metrics})
    return sorted(rows, key=lambda row: float(row["relative_compute"]))


def plot_pareto(
    scan_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    estimates: Sequence[StateConfigEstimate] = (),
    selected_lambda: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = {
        "state_wise": "State-wise Oracle",
        "task_wise": "Task-wise Oracle",
        "global_fixed": "Global Fixed Oracle",
    }
    colors = {"state_wise": "#007f73", "task_wise": "#e28e2c", "global_fixed": "#b23a48"}
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for strategy, label in labels.items():
        rows = [row for row in scan_rows if row["strategy"] == strategy]
        axis.scatter(
            [float(row["relative_compute"]) for row in rows],
            [float(row["q_value"]) for row in rows],
            s=12,
            alpha=0.25,
            color=colors[strategy],
        )
        frontier = _pareto_points(rows)
        axis.plot(
            [point["relative_compute"] for point in frontier],
            [point["q_value"] for point in frontier],
            marker="o",
            linewidth=2,
            markersize=4,
            color=colors[strategy],
            label=label,
        )
    if estimates:
        fixed_rows = _fixed_config_metrics(estimates)
        axis.plot(
            [float(row["relative_compute"]) for row in fixed_rows],
            [float(row["q_value"]) for row in fixed_rows],
            marker="D",
            linestyle="--",
            linewidth=1.2,
            markersize=4,
            color="#555555",
            alpha=0.85,
            label="Fixed configurations",
        )
        for row in fixed_rows:
            axis.annotate(
                str(row["configuration"]),
                (float(row["relative_compute"]), float(row["q_value"])),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7,
                color="#444444",
            )
    if selected_lambda is not None:
        knee = next(
            row
            for row in scan_rows
            if row["strategy"] == "state_wise"
            and np.isclose(float(row["lambda"]), selected_lambda)
        )
        axis.scatter(
            [float(knee["relative_compute"])],
            [float(knee["q_value"])],
            marker="*",
            s=130,
            color="#f2c14e",
            edgecolor="black",
            linewidth=0.6,
            zorder=10,
            label=f"Knee λ={selected_lambda:.3g}",
        )
    axis.set_xlabel("Relative Compute (mean cumulative branch cost)")
    axis.set_ylabel("Value Q (0.8 success + 0.2 normalized reward)")
    axis.set_xscale("log")
    axis.set_ylim(0.75, 0.82)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _normalize_phase(phase: str) -> str:
    value = phase.strip().lower().replace("-", "_").replace(" ", "_")
    if "fail" in value or "recover" in value:
        return "failure_recovery"
    if value in {"approach", "reaching", "reach"}:
        return "reaching"
    if value in {"pre_contact", "precontact", "contact_approach"}:
        return "precontact"
    if value in {"grasp_interaction", "manipulation", "grasping", "interaction"}:
        return "manipulation"
    if value in {"placement", "placement_goal"}:
        return "placement"
    return "unknown"


def _choose_rq2_lambda(
    scan_rows: Sequence[Mapping[str, Any]],
    *,
    success_tolerance: float,
) -> tuple[float, float]:
    state_rows = [row for row in scan_rows if row["strategy"] == "state_wise"]
    best_success = max(float(row["success_rate"]) for row in state_rows)
    eligible = [
        row for row in state_rows
        if float(row["success_rate"]) >= best_success - success_tolerance
    ]
    ordered = sorted(
        eligible,
        key=lambda row: (float(row["relative_compute"]), float(row["q_value"])),
    )
    unique: list[Mapping[str, Any]] = []
    for row in ordered:
        if not unique or not (
            np.isclose(
                float(row["relative_compute"]),
                float(unique[-1]["relative_compute"]),
                rtol=1e-12,
                atol=1e-14,
            )
            and np.isclose(
                float(row["q_value"]),
                float(unique[-1]["q_value"]),
                rtol=1e-12,
                atol=1e-14,
            )
        ):
            unique.append(row)
    costs = np.asarray([float(row["relative_compute"]) for row in unique])
    values = np.asarray([float(row["q_value"]) for row in unique])
    if len(unique) < 3 or np.ptp(costs) <= 0 or np.ptp(values) <= 0:
        selected = max(unique, key=lambda row: float(row["q_value"]))
    else:
        normalized_cost = (costs - costs.min()) / np.ptp(costs)
        normalized_value = (values - values.min()) / np.ptp(values)
        knee_scores = normalized_value - normalized_cost
        selected = unique[int(np.argmax(knee_scores))]
    return float(selected["lambda"]), best_success


def _phase_statistics(
    selections: Mapping[str, StateConfigEstimate],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[float]]]]:
    grouped: dict[str, list[StateConfigEstimate]] = defaultdict(list)
    for estimate in selections.values():
        grouped[_normalize_phase(estimate.phase)].append(estimate)
    summary: list[dict[str, Any]] = []
    distributions: dict[str, dict[str, list[float]]] = {}
    for phase in (*PHASE_ORDER, "unknown"):
        rows = grouped.get(phase, [])
        if not rows:
            continue
        distributions[phase] = {
            parameter: [float(getattr(item.config, parameter)) for item in rows]
            for parameter in PARAMETERS
        }
        summary.append(
            {
                "phase": phase,
                "state_count": len(rows),
                **{
                    f"mean_{parameter}": float(np.mean(distributions[phase][parameter]))
                    for parameter in PARAMETERS
                },
                **{
                    f"se_{parameter}": (
                        float(np.std(distributions[phase][parameter], ddof=1))
                        / np.sqrt(len(distributions[phase][parameter]))
                        if len(distributions[phase][parameter]) > 1
                        else 0.0
                    )
                    for parameter in PARAMETERS
                },
            }
        )
    return summary, distributions


def _directional_tests(
    distributions: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, Any]:
    reaching = distributions.get("reaching", {})
    deliberate = [
        values
        for phase in ("precontact", "manipulation")
        if (values := distributions.get(phase, {}))
    ]
    combined = {
        parameter: [float(value) for phase in deliberate for value in phase.get(parameter, [])]
        for parameter in PARAMETERS
    }
    hypotheses = {
        "E": ("less", "precontact/manipulation execution horizon is lower than reaching"),
        "D": ("greater", "precontact/manipulation denoising depth is higher than reaching"),
        "W": ("greater", "precontact/manipulation rollout count is higher than reaching"),
    }
    results: dict[str, Any] = {}
    for parameter, (alternative, hypothesis) in hypotheses.items():
        first = combined[parameter]
        second = [float(value) for value in reaching.get(parameter, [])]
        if not first or not second:
            results[parameter] = {
                "hypothesis": hypothesis,
                "alternative": alternative,
                "n_deliberate": len(first),
                "n_reaching": len(second),
                "u_statistic": None,
                "p_value": None,
            }
            continue
        test = mannwhitneyu(first, second, alternative=alternative, method="auto")
        results[parameter] = {
            "hypothesis": hypothesis,
            "alternative": alternative,
            "n_deliberate": len(first),
            "n_reaching": len(second),
            "mean_deliberate": float(np.mean(first)),
            "mean_reaching": float(np.mean(second)),
            "u_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
        }
    return results


def plot_phase_distributions(
    distributions: Mapping[str, Mapping[str, Sequence[float]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    phases = [phase for phase in PHASE_ORDER if phase in distributions]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    positions = np.arange(len(phases), dtype=float)
    width = 0.34
    for offset, parameter, color in (
        (-width / 2, "H", "#4c78a8"),
        (width / 2, "D", "#f58518"),
    ):
        values = [np.asarray(distributions[phase][parameter], dtype=float) for phase in phases]
        means = [float(np.mean(value)) for value in values]
        errors = [
            float(np.std(value, ddof=1) / np.sqrt(len(value))) if len(value) > 1 else 0.0
            for value in values
        ]
        axes[0].bar(
            positions + offset,
            means,
            width,
            yerr=errors,
            capsize=4,
            color=color,
            label=parameter,
        )
    execution = [np.asarray(distributions[phase]["E"], dtype=float) for phase in phases]
    axes[1].bar(
        positions,
        [float(np.mean(value)) for value in execution],
        0.55,
        yerr=[
            float(np.std(value, ddof=1) / np.sqrt(len(value))) if len(value) > 1 else 0.0
            for value in execution
        ],
        capsize=4,
        color="#54a24b",
    )
    axes[0].set_title("Deliberate more: imagination level")
    axes[0].set_ylabel("Selected H / D (mean ± SE)")
    axes[0].legend(frameon=False)
    axes[1].set_title("Commit less: execution horizon")
    axes[1].set_ylabel("Selected E (mean ± SE)")
    for axis in axes:
        axis.set_xticks(positions, phases, rotation=22, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_config_selection_proportions(
    selections: Mapping[str, StateConfigEstimate],
    output_path: Path,
) -> None:
    labels = {
        Config(0, 0, 1, 10): "B_action",
        Config(2, 2, 1, 4): "B_small",
        Config(2, 16, 1, 2): "B_deep",
        Config(4, 4, 4, 4): "B_wide",
        Config(8, 4, 1, 4): "B_far",
        Config(8, 16, 4, 2): "B_full",
    }
    grouped: dict[str, list[StateConfigEstimate]] = defaultdict(list)
    for estimate in selections.values():
        grouped[_normalize_phase(estimate.phase)].append(estimate)
    phases = [phase for phase in PHASE_ORDER if grouped.get(phase)]
    configs = [config for config in labels if any(config == row.config for rows in grouped.values() for row in rows)]
    colors = ("#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#b279a2")
    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    bottoms = np.zeros(len(phases), dtype=float)
    for config, color in zip(configs, colors):
        proportions = np.asarray(
            [
                sum(row.config == config for row in grouped[phase]) / len(grouped[phase])
                for phase in phases
            ],
            dtype=float,
        )
        axis.bar(phases, proportions, bottom=bottoms, color=color, label=labels[config])
        bottoms += proportions
    axis.set_ylim(0, 1)
    axis.set_ylabel("Selection proportion")
    axis.set_title("State-wise Oracle configuration choices at Pareto knee")
    axis.tick_params(axis="x", labelrotation=22)
    axis.legend(frameon=False, ncol=3)
    axis.grid(True, axis="y", alpha=0.2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _reference_compute(estimates: Sequence[StateConfigEstimate]) -> float:
    full_config = Config(8, 16, 4, 2)
    by_state: dict[str, list[float]] = defaultdict(list)
    for estimate in estimates:
        if estimate.config == full_config:
            by_state[estimate.state_id].append(estimate.relative_compute)
    if not by_state:
        raise ValueError("Cannot normalize RQ1 budgets: B_full results are missing")
    state_costs = [float(np.mean(costs)) for costs in by_state.values()]
    return float(np.mean(state_costs))


def _budget_best(
    rows: Sequence[Mapping[str, Any]], strategy: str, budget: float
) -> Mapping[str, Any] | None:
    feasible = [
        row
        for row in rows
        if row["strategy"] == strategy and float(row["relative_compute"]) <= budget
    ]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda row: (
            float(row["q_value"]),
            float(row["success_rate"]),
            -float(row["relative_compute"]),
        ),
    )


def _rq1_summary(scan_rows: Sequence[Mapping[str, Any]], reference_compute: float) -> dict[str, Any]:
    best_success_by_strategy = {
        strategy: max(
            float(row["success_rate"])
            for row in scan_rows
            if row["strategy"] == strategy
        )
        for strategy in ("state_wise", "task_wise", "global_fixed")
    }
    target_success = min(
        best_success_by_strategy["state_wise"],
        best_success_by_strategy["global_fixed"],
    )
    success_costs: dict[str, float | None] = {}
    for strategy in ("state_wise", "task_wise", "global_fixed"):
        feasible = [
            float(row["relative_compute"])
            for row in scan_rows
            if row["strategy"] == strategy
            and float(row["success_rate"]) >= target_success - 1e-12
        ]
        success_costs[strategy] = min(feasible) if feasible else None
    state_cost = success_costs["state_wise"]
    global_cost = success_costs["global_fixed"]
    savings = (
        100.0 * (global_cost - state_cost) / global_cost
        if state_cost is not None and global_cost is not None and global_cost > 0
        else None
    )
    budgets: list[dict[str, Any]] = []
    for fraction in RQ1_BUDGETS:
        budget = reference_compute * fraction
        state_best = _budget_best(scan_rows, "state_wise", budget)
        global_best = _budget_best(scan_rows, "global_fixed", budget)
        state_success = float(state_best["success_rate"]) if state_best is not None else None
        global_success = float(global_best["success_rate"]) if global_best is not None else None
        state_q = float(state_best["q_value"]) if state_best is not None else None
        global_q = float(global_best["q_value"]) if global_best is not None else None
        budgets.append(
            {
                "budget_fraction_of_b_full": fraction,
                "budget_relative_compute": budget,
                "state_wise_success_rate": state_success,
                "global_fixed_success_rate": global_success,
                "success_lead": (
                    state_success - global_success
                    if state_success is not None and global_success is not None
                    else None
                ),
                "state_wise_q_value": state_q,
                "global_fixed_q_value": global_q,
                "q_value_lead": (
                    state_q - global_q
                    if state_q is not None and global_q is not None
                    else None
                ),
            }
        )
    return {
        "b_full_reference_relative_compute": reference_compute,
        "best_success_rate_by_strategy": best_success_by_strategy,
        "comparison_target_success_rate": target_success,
        "comparison_target_definition": (
            "highest success rate jointly attainable by state-wise and global-fixed strategies"
        ),
        "minimum_compute_at_target_success": success_costs,
        "state_wise_compute_savings_percent_vs_global": savings,
        "budget_comparisons": budgets,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_rq1_table(summary: Mapping[str, Any]) -> None:
    savings = summary["state_wise_compute_savings_percent_vs_global"]
    savings_text = "n/a" if savings is None else f"{savings:.1f}%"
    target = float(summary["comparison_target_success_rate"])
    print(
        f"RQ1: State-wise savings at jointly attainable success {target:.3f} "
        f"vs Global Fixed: {savings_text}"
    )
    print("budget  state_success  global_success  state_lead  q_lead")
    for row in summary["budget_comparisons"]:
        values = [
            row[key]
            for key in (
                "state_wise_success_rate",
                "global_fixed_success_rate",
                "success_lead",
                "q_value_lead",
            )
        ]
        formatted = ["n/a" if value is None else f"{value:.3f}" for value in values]
        print(
            f"{row['budget_fraction_of_b_full']:>5.1%}  {formatted[0]:>13}  "
            f"{formatted[1]:>14}  {formatted[2]:>10}  {formatted[3]:>7}"
        )


def analyze_oracle(
    results_csv: Path,
    cost_table_path: Path,
    output_dir: Path,
    *,
    value_metric: str = "hybrid",
    success_tolerance: float = 0.01,
) -> dict[str, Any]:
    if not 0 <= success_tolerance < 1:
        raise ValueError("success_tolerance must be in [0, 1)")
    hardware_costs, cost_metadata = read_cost_table(cost_table_path)
    estimates = read_counterfactual_results(
        results_csv,
        hardware_costs,
        value_metric=value_metric,
    )
    scan_rows, selections = scan_lambdas(estimates)
    reward_min = min(item.mean_return for item in estimates)
    reward_max = max(item.mean_return for item in estimates)
    reference_compute = _reference_compute(estimates)
    rq1 = _rq1_summary(scan_rows, reference_compute)
    rq2_lambda, best_success = _choose_rq2_lambda(
        scan_rows,
        success_tolerance=success_tolerance,
    )
    rq2_selection = selections[rq2_lambda]["state_wise"]
    phase_summary, distributions = _phase_statistics(rq2_selection)
    tests = _directional_tests(distributions)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_pareto(
        scan_rows,
        output_dir / "oracle_pareto_curve.png",
        estimates=estimates,
        selected_lambda=rq2_lambda,
    )
    plot_phase_distributions(distributions, output_dir / "rq2_phase_distribution.png")
    plot_config_selection_proportions(
        rq2_selection,
        output_dir / "rq2_config_selection_stacked.png",
    )
    _write_csv(output_dir / "oracle_lambda_scan.csv", scan_rows)
    _write_csv(output_dir / "rq2_phase_summary.csv", phase_summary)
    choice_rows = [
        {
            "state_id": state_id,
            "task_suite": estimate.task_suite,
            "task_id": estimate.task_id,
            "phase": estimate.phase,
            "normalized_phase": _normalize_phase(estimate.phase),
            "H": estimate.config.H,
            "D": estimate.config.D,
            "W": estimate.config.W,
            "E": estimate.config.E,
            "success_rate": estimate.success_rate,
            "mean_return": estimate.mean_return,
            "normalized_reward": estimate.normalized_reward,
            "q_value": estimate.value,
            "relative_compute": estimate.relative_compute,
        }
        for state_id, estimate in rq2_selection.items()
    ]
    _write_csv(output_dir / "rq2_statewise_choices.csv", choice_rows)
    report = {
        "objective": "Q(s,B) - lambda * C(s,B)",
        "value_metric": value_metric,
        "hybrid_value_definition": "0.8 * success_rate + 0.2 * global_minmax_normalized_dense_reward",
        "reward_normalization_range": [reward_min, reward_max],
        "cost_metric": "mean estimated_total_relative_compute when available; otherwise relative_compute(H,D,W) / E * remaining_steps",
        "lambda_grid": [float(value) for value in LAMBDA_GRID],
        "state_count": len({estimate.state_id for estimate in estimates}),
        "config_count": len({estimate.config for estimate in estimates}),
        "cost_table_metadata": {
            "device": cost_metadata.get("device"),
            "precision": cost_metadata.get("precision"),
        },
        "rq1": rq1,
        "rq2": {
            "selected_lambda": rq2_lambda,
            "lambda_selection_method": (
                "normalized maximum-chord-distance knee on the state-wise Pareto "
                "frontier, restricted to success within tolerance of the best"
            ),
            "best_success_rate": best_success,
            "success_tolerance": success_tolerance,
            "phase_distribution": phase_summary,
            "directional_mann_whitney_tests": tests,
            "interpretation": "Exploratory in-sample tests over state-level selected configurations; independent-state assumption and small sample sizes limit inferential claims.",
        },
    }
    (output_dir / "oracle_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_rq1_table(rq1)
    print(
        f"RQ2 lambda={rq2_lambda:.5g}, state-wise success={best_success:.3f}; "
        f"wrote plots and reports to {output_dir}"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--cost-table", type=Path, default=DEFAULT_COST_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--value-metric", choices=("hybrid", "success", "return"), default="hybrid")
    parser.add_argument("--success-tolerance", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analyze_oracle(
        args.results_csv,
        args.cost_table,
        args.output_dir,
        value_metric=args.value_metric,
        success_tolerance=args.success_tolerance,
    )


if __name__ == "__main__":
    main()
