"""Analyze state-wise, global-fixed, and task-wise counterfactual policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, order=True)
class InferenceConfig:
    H: int
    D: int
    W: int
    E: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    def label(self) -> str:
        return f"({self.H},{self.D},{self.W},{self.E})"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(record)
    return records


def load_cost_table(
    path: Path,
) -> tuple[dict[tuple[int, int, int], float], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("configurations"), list):
        raise ValueError(f"{path} must contain a configurations array")
    latency_by_hdw = {}
    for row in payload["configurations"]:
        try:
            key = (int(row["H"]), int(row["D"]), int(row["W"]))
            latency = float(row["latency_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed cost-table row in {path}: {row!r}") from exc
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"Non-positive inference latency for {key} in {path}")
        if key in latency_by_hdw:
            raise ValueError(f"Duplicate cost-table configuration {key}")
        latency_by_hdw[key] = latency
    if not latency_by_hdw:
        raise ValueError(f"No configurations found in {path}")
    return latency_by_hdw, payload


def _config_from_record(record: dict[str, Any]) -> InferenceConfig:
    config = record.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Counterfactual record has no config object: {record}")
    try:
        return InferenceConfig(
            H=int(config["H"]),
            D=int(config["D"]),
            W=int(config["W"]),
            E=int(config["E"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed config in counterfactual record: {config!r}") from exc


def add_costs(
    records: Iterable[dict[str, Any]],
    latency_by_hdw: dict[tuple[int, int, int], float],
) -> list[dict[str, Any]]:
    scored = []
    configs_by_state: dict[tuple[str, int, str], set[InferenceConfig]] = defaultdict(set)
    for record in records:
        if not {"state_id", "task_id", "episode_return", "success", "inference_calls"} <= (
            record.keys()
        ):
            raise ValueError(f"Counterfactual record is missing required fields: {record}")
        config = _config_from_record(record)
        config_values = (config.H, config.D, config.W, config.E)
        if config.H < 0 or config.E <= 0:
            raise ValueError(f"Invalid inference config {config_values}")
        if (config.H == 0 and (config.D != 0 or config.W != 0)) or (
            config.H > 0 and (config.D <= 0 or config.W <= 0)
        ):
            raise ValueError(f"Invalid inference config {config_values}")
        latency_key = (config.H, config.D, config.W)
        if latency_key not in latency_by_hdw:
            raise ValueError(f"No Phase 1 cost-table latency for H/D/W={latency_key}")
        calls = int(record["inference_calls"])
        if calls < 0:
            raise ValueError("inference_calls must not be negative")
        episode_return = float(record["episode_return"])
        if not math.isfinite(episode_return):
            raise ValueError(f"Non-finite Q return for state {record['state_id']}")
        if not isinstance(record["success"], bool):
            raise ValueError(f"success must be a boolean for state {record['state_id']}")
        configs_by_state[_state_key(record)].add(config)
        cost_ms = calls * latency_by_hdw[latency_key]
        scored.append(
            {
                **record,
                "config": config.as_dict(),
                "config_key": config,
                "cost_ms": cost_ms,
                "utility": episode_return,
            }
        )
    if not scored:
        raise ValueError("No counterfactual result records were provided")
    all_configs = set().union(*configs_by_state.values())
    incomplete = {
        state: sorted(all_configs - state_configs)
        for state, state_configs in configs_by_state.items()
        if state_configs != all_configs
    }
    if incomplete:
        state, missing = next(iter(incomplete.items()))
        raise ValueError(
            f"Incomplete counterfactual grid for state {state}; "
            f"missing {len(missing)} configuration(s), first missing: {missing[0]}"
        )
    return scored


def _candidate_summary(
    rows: list[dict[str, Any]],
    config: InferenceConfig,
    lambda_cost: float,
) -> dict[str, Any]:
    selected = [row for row in rows if row["config_key"] == config]
    if not selected:
        raise ValueError(f"No observations for configuration {config}")
    mean_return = sum(float(row["episode_return"]) for row in selected) / len(selected)
    mean_cost = sum(float(row["cost_ms"]) for row in selected) / len(selected)
    mean_utility = sum(
        float(row["episode_return"]) - lambda_cost * float(row["cost_ms"])
        for row in selected
    ) / len(selected)
    return {
        "config": config,
        "count": len(selected),
        "success_rate": sum(bool(row["success"]) for row in selected) / len(selected),
        "mean_return": mean_return,
        "mean_cost_ms": mean_cost,
        "mean_utility": mean_utility,
    }


def _choose_config(
    rows: list[dict[str, Any]],
    configs: list[InferenceConfig],
    lambda_cost: float,
    budget_ms: float | None,
) -> dict[str, Any]:
    summaries = []
    for config in configs:
        try:
            summary = _candidate_summary(rows, config, lambda_cost)
        except ValueError:
            continue
        if budget_ms is None or summary["mean_cost_ms"] <= budget_ms:
            summaries.append(summary)
    if not summaries:
        raise ValueError(
            "No configuration satisfies the requested cost budget for an evaluation group"
        )
    return max(
        summaries,
        key=lambda summary: (
            summary["mean_utility"],
            summary["success_rate"],
            -summary["mean_cost_ms"],
            tuple(-value for value in asdict(summary["config"]).values()),
        ),
    )


def _task_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("task_suite", "")), int(row["task_id"])


def _state_key(row: dict[str, Any]) -> tuple[str, int, str]:
    suite, task_id = _task_key(row)
    return suite, task_id, str(row["state_id"])


def _strategy_metrics(
    rows: list[dict[str, Any]],
    lambda_cost: float,
    budget_ms: float | None,
) -> dict[str, Any]:
    count = len(rows)
    config_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        config_counts[row["config_key"].label()] += 1
    return {
        "sample_count": count,
        "success_rate": sum(bool(row["success"]) for row in rows) / count,
        "success_rate_at_budget": (
            sum(bool(row["success"]) for row in rows) / count
            if budget_ms is not None
            else None
        ),
        "mean_return": sum(float(row["episode_return"]) for row in rows) / count,
        "mean_cost_ms": sum(float(row["cost_ms"]) for row in rows) / count,
        "mean_utility": sum(
            float(row["episode_return"]) - lambda_cost * float(row["cost_ms"])
            for row in rows
        )
        / count,
        "config_counts": dict(sorted(config_counts.items())),
    }


def analyze(
    records: list[dict[str, Any]],
    latency_by_hdw: dict[tuple[int, int, int], float],
    lambda_cost: float = 0.0,
    budget_ms: float | None = None,
) -> dict[str, Any]:
    """Select empirical configurations and compare episode success/return/cost."""
    if not math.isfinite(lambda_cost) or lambda_cost < 0:
        raise ValueError("lambda_cost must be finite and non-negative")
    if budget_ms is not None and (not math.isfinite(budget_ms) or budget_ms <= 0):
        raise ValueError("budget_ms must be positive")
    scored = add_costs(records, latency_by_hdw)
    configs = sorted({row["config_key"] for row in scored})
    global_config = _choose_config(scored, configs, lambda_cost, budget_ms)["config"]
    global_rows = [
        row for row in scored if row["config_key"] == global_config
    ]

    by_task: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_state: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_task[_task_key(row)].append(row)
        by_state[_state_key(row)].append(row)

    task_rows = []
    task_configs = {}
    for task, task_records in sorted(by_task.items()):
        chosen = _choose_config(task_records, configs, lambda_cost, budget_ms)
        task_configs[f"{task[0]}:{task[1]}"] = chosen["config"].as_dict()
        task_rows.extend(
            row for row in task_records if row["config_key"] == chosen["config"]
        )

    state_rows = []
    statewise_choices = []
    for state, state_records in sorted(by_state.items()):
        chosen = _choose_config(state_records, configs, lambda_cost, budget_ms)
        selected = [
            row for row in state_records if row["config_key"] == chosen["config"]
        ]
        state_rows.extend(selected)
        statewise_choices.append(
            {
                "task_suite": state[0],
                "task_id": state[1],
                "state_id": state[2],
                "config": chosen["config"].as_dict(),
                "mean_return": chosen["mean_return"],
                "success_rate": chosen["success_rate"],
                "mean_cost_ms": chosen["mean_cost_ms"],
                "mean_utility": chosen["mean_utility"],
            }
        )

    strategies = {
        "state_wise_oracle": _strategy_metrics(state_rows, lambda_cost, budget_ms),
        "global_fixed_oracle": _strategy_metrics(global_rows, lambda_cost, budget_ms),
        "task_wise": _strategy_metrics(task_rows, lambda_cost, budget_ms),
    }
    for metrics in strategies.values():
        metrics["gain_vs_global_success_rate"] = (
            metrics["success_rate"]
            - strategies["global_fixed_oracle"]["success_rate"]
        )
        metrics["gain_vs_global_return"] = (
            metrics["mean_return"] - strategies["global_fixed_oracle"]["mean_return"]
        )
    statewise_gain = (
        strategies["state_wise_oracle"]["success_rate"]
        - strategies["global_fixed_oracle"]["success_rate"]
    )
    taskwise_gain = (
        strategies["task_wise"]["success_rate"]
        - strategies["global_fixed_oracle"]["success_rate"]
    )
    go_threshold = 0.05
    return {
        "selection": {
            "objective": "mean(episode_return - lambda_cost * inference_cost_ms)",
            "lambda_cost_per_ms": lambda_cost,
            "budget_ms_per_continuation": budget_ms,
            "global_fixed_config": global_config.as_dict(),
            "task_wise_configs": task_configs,
            "note": (
                "Empirical in-sample counterfactual oracle; use held-out states for "
                "generalization claims. Cost uses Phase 1 H/D/W latency multiplied "
                "by observed policy inference calls; E affects calls through rollout."
            ),
        },
        "strategies": strategies,
        "state_wise_choices": statewise_choices,
        "go_no_go": {
            "decision": "GO" if statewise_gain >= go_threshold else "NO-GO",
            "state_wise_success_gain": statewise_gain,
            "task_wise_success_gain": taskwise_gain,
            "minimum_success_gain_threshold": go_threshold,
            "recommendation": (
                "Proceed to a held-out state-dependent controller evaluation."
                if statewise_gain >= go_threshold
                else "Do not claim adaptive-inference benefit yet; collect more "
                "states/seeds or test task-level adaptation."
            ),
            "caveat": (
                "This is an in-sample heuristic recommendation. Oracle gains are "
                "optimistic and are not a statistical significance test."
            ),
        },
    }


def _write_outputs(
    summary: dict[str, Any], summary_path: Path, comparison_path: Path, statewise_path: Path
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    statewise_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    statewise_path.write_text(
        json.dumps(summary["state_wise_choices"], indent=2) + "\n",
        encoding="utf-8",
    )

    columns = (
        "strategy",
        "sample_count",
        "success_rate",
        "success_rate_at_budget",
        "mean_return",
        "mean_cost_ms",
        "mean_utility",
        "gain_vs_global_success_rate",
        "gain_vs_global_return",
    )
    with comparison_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for strategy, metrics in summary["strategies"].items():
            writer.writerow({"strategy": strategy, **metrics})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cost-table", type=Path, default=Path("cost_table.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lambda-cost",
        type=float,
        default=0.0,
        help="Reward penalty per predicted inference-cost millisecond",
    )
    parser.add_argument(
        "--budget-ms",
        type=float,
        default=None,
        help="Maximum mean inference cost in ms per continuation rollout",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    latency_by_hdw, cost_table_metadata = load_cost_table(args.cost_table)
    summary = analyze(
        read_jsonl(args.results),
        latency_by_hdw,
        lambda_cost=args.lambda_cost,
        budget_ms=args.budget_ms,
    )
    output_dir = args.output_dir
    _write_outputs(
        summary,
        output_dir / "oracle_summary.json",
        output_dir / "oracle_comparison.csv",
        output_dir / "statewise_oracle.json",
    )
    print(
        f"Cost table device: {cost_table_metadata.get('device', 'unknown')} "
        f"({cost_table_metadata.get('precision', 'unknown')})"
    )
    print(
        "strategy              success  success@budget  mean return  mean cost(ms)  "
        "success gain"
    )
    for strategy, metrics in summary["strategies"].items():
        budget_rate = metrics["success_rate_at_budget"]
        print(
            f"{strategy:21} {metrics['success_rate']:7.3f}  "
            f"{budget_rate if budget_rate is not None else float('nan'):14.3f}  "
            f"{metrics['mean_return']:11.3f}  {metrics['mean_cost_ms']:13.3f}  "
            f"{metrics['gain_vs_global_success_rate']:+12.3f}"
        )
    recommendation = summary["go_no_go"]
    print(
        f"Go/No-Go: {recommendation['decision']} "
        f"(state-wise success gain "
        f"{recommendation['state_wise_success_gain']:+.3f}; "
        f"threshold {recommendation['minimum_success_gain_threshold']:.3f})"
    )
    print(f"Wrote oracle comparison to {output_dir}")


if __name__ == "__main__":
    main()
