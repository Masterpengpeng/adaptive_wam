"""Tiny deterministic adapter for exercising the collection CLI without a simulator."""

from __future__ import annotations

from typing import Any

import numpy as np

from collect_counterfactual_data import InferenceConfig, StepResult, WorkerContext


class FakeCounterfactualAdapter:
    def __init__(self, context: WorkerContext) -> None:
        self.context = context
        self.position = 0.0

    def start_task(self, task_id: int, seed: int) -> float:
        self.position = float(seed % 2)
        return self.position

    def capture_checkpoint(self) -> dict[str, float]:
        return {"position": self.position}

    def restore_checkpoint(self, checkpoint: dict[str, float]) -> float:
        self.position = float(checkpoint["position"])
        return self.position

    def infer_actions(
        self, observation: Any, config: InferenceConfig
    ) -> np.ndarray:
        increment = 2.0 if config.H > 0 else 1.0
        return np.full((config.E, 1), increment, dtype=np.float32)

    def step(self, action: np.ndarray) -> StepResult:
        self.position += float(np.asarray(action).reshape(-1)[0])
        succeeded = self.position >= 6
        return StepResult(
            observation=self.position,
            reward=1.0 if succeeded else 0.1,
            terminated=succeeded,
            truncated=False,
            info={"success": succeeded},
        )

    def is_success(self, observation: Any, info: dict[str, Any]) -> bool:
        return bool(info.get("success", False))

    def close(self) -> None:
        pass


def create_adapter(context: WorkerContext) -> FakeCounterfactualAdapter:
    return FakeCounterfactualAdapter(context)
