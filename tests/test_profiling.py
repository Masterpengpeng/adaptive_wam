from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from models.parameterized_wam import ParameterizedWAM
from profiling.profile_hardware import (
    FIXED_FULL_CONFIG,
    ProfileConfig,
    add_relative_compute,
    iter_configurations,
    make_dummy_observation,
    profile_configuration,
    render_markdown_table,
    save_cost_table,
)


class TinyFastWAM(nn.Module):
    def infer_action(
        self,
        *,
        input_image: Tensor,
        action_horizon: int,
        num_inference_steps: int,
        seed: int,
        action_infer_mode: str = "first_frame",
        num_video_frames: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        del num_inference_steps, seed, action_infer_mode, num_video_frames, kwargs
        return {
            "action": torch.ones(
                (action_horizon, 7),
                device=input_image.device,
                dtype=input_image.dtype,
            )
        }


class HardwareProfilingSmokeTest(unittest.TestCase):
    def test_grid_contains_action_only_and_all_36_world_configs(self) -> None:
        configurations = iter_configurations(execution_chunk=6)

        self.assertEqual(len(configurations), 37)
        self.assertEqual(
            (configurations[0].H, configurations[0].D, configurations[0].W),
            (0, 0, 1),
        )
        self.assertIn(
            ProfileConfig(*FIXED_FULL_CONFIG, E=6),
            configurations,
        )

    def test_dummy_observation_uses_two_uint8_cameras_and_valid_state(self) -> None:
        observation = make_dummy_observation(
            {
                "agentview_image": torch.zeros((224, 224, 3)).numpy(),
                "robot0_eye_in_hand_image": torch.zeros((224, 224, 3)).numpy(),
                "robot0_gripper_qpos": torch.zeros((2,)).numpy(),
            }
        )

        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            self.assertEqual(observation[key].shape, (224, 224, 3))
            self.assertEqual(observation[key].dtype.name, "uint8")
        self.assertEqual(tuple(observation["robot0_eef_pos"].shape), (3,))

    def test_cost_table_serialization_and_relative_compute(self) -> None:
        rows: list[dict[str, int | float]] = [
            {
                "H": 0,
                "D": 0,
                "W": 1,
                "E": 10,
                "latency_ms_mean": 5.0,
                "latency_ms_std": 0.2,
                "peak_vram_mb": 100.0,
            },
            {
                "H": 8,
                "D": 16,
                "W": 4,
                "E": 10,
                "latency_ms_mean": 20.0,
                "latency_ms_std": 1.0,
                "peak_vram_mb": 500.0,
            },
        ]
        add_relative_compute(rows)
        self.assertEqual(rows[0]["relative_compute"], 0.25)
        self.assertEqual(rows[1]["relative_compute"], 1.0)
        self.assertIn("| H | D | W |", render_markdown_table(rows))

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "hardware_cost_table.json"
            save_cost_table(
                destination,
                device_names=["NVIDIA Test GPU"],
                results=rows,
                model_path=Path("/tmp/test-checkpoint.pt"),
                warmup=10,
                iterations=50,
            )
            self.assertTrue(destination.is_file())
            import json

            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(payload["precision"], "FP16")
            self.assertEqual(len(payload["configurations"]), 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA event smoke test requires CUDA")
    def test_cuda_event_profile_runs_parameterized_wrapper(self) -> None:
        device = torch.device("cuda:0")
        model = TinyFastWAM().to(device).eval()

        def prepare(observation: dict[str, Tensor]) -> dict[str, Tensor]:
            return {
                "input_image": observation["image"],
                "proprio": observation["state"],
            }

        def postprocess(actions: Tensor, execution_chunk: int) -> Tensor:
            return actions[:execution_chunk]

        policy = ParameterizedWAM(
            model,
            prepare,
            postprocess,
            action_horizon=10,
            action_video_freq_ratio=4,
        )
        observation = {
            "image": torch.zeros((1, 3, 224, 448), device=device),
            "state": torch.zeros((1, 14), device=device),
        }
        config = ProfileConfig(0, 0, 1, 4)
        result = profile_configuration(
            config,
            lambda: policy(observation, config.H, config.D, config.W, config.E),
            [device],
            warmup=2,
            iterations=3,
        )

        self.assertGreater(float(result["latency_ms_mean"]), 0.0)
        self.assertGreaterEqual(float(result["latency_ms_std"]), 0.0)
        self.assertGreater(float(result["peak_vram_mb"]), 0.0)
        self.assertEqual(result["E"], 4)


if __name__ == "__main__":
    unittest.main()
