from __future__ import annotations

import gc
import unittest
from typing import Any

import torch
from torch import Tensor, nn

from models.parameterized_wam import ParameterizedWAM


class FakeFastWAM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls: list[dict[str, Any]] = []
        self.non_finite = False

    def infer_action(
        self,
        *,
        prompt: str,
        input_image: Tensor,
        proprio: Tensor,
        action_horizon: int,
        num_inference_steps: int,
        seed: int,
        action_infer_mode: str = "first_frame",
        num_video_frames: int | None = None,
        negative_prompt: str = "",
        text_cfg_scale: float = 1.0,
        sigma_shift: float | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
    ) -> dict[str, Tensor]:
        self.calls.append(
            {
                "prompt": prompt,
                "proprio_shape": tuple(proprio.shape),
                "action_horizon": action_horizon,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
                "action_infer_mode": action_infer_mode,
                "num_video_frames": num_video_frames,
                "negative_prompt": negative_prompt,
                "text_cfg_scale": text_cfg_scale,
                "sigma_shift": sigma_shift,
                "rand_device": rand_device,
                "tiled": tiled,
                "compile_action_infer": compile_action_infer,
            }
        )
        sample_value = float(len(self.calls))
        action = torch.full(
            (action_horizon, 7),
            sample_value,
            dtype=torch.float32,
            device=input_image.device,
        )
        if self.non_finite:
            action[0, 0] = torch.nan
        return {"action": action + self.anchor}


class ParameterizedWAMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeFastWAM()
        self.wam = ParameterizedWAM(
            self.model,
            self._prepare,
            self._postprocess,
            action_horizon=10,
            action_video_freq_ratio=4,
            direct_depth=10,
        )

    @staticmethod
    def _prepare(obs: Any) -> dict[str, Any]:
        return {
            "prompt": "move the object",
            "input_image": obs["image"],
            "proprio": obs["state"],
        }

    @staticmethod
    def _postprocess(actions: Tensor, execution_chunk: int) -> Tensor:
        return actions[:execution_chunk] * 2.0 + 0.5

    def _observation(self, device: torch.device | str = "cpu") -> dict[str, Tensor]:
        return {
            "image": torch.zeros((1, 3, 224, 224), device=device),
            "state": torch.zeros((1, 14), device=device),
        }

    def test_direct_path_returns_requested_chunk_and_ignores_w(self) -> None:
        actions = self.wam(self._observation(), 0, 0, 1, 10)

        self.assertEqual(tuple(actions.shape), (10, 7))
        self.assertEqual(actions.dtype, torch.float32)
        self.assertEqual(actions.device.type, "cpu")
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(self.model.calls[0]["action_infer_mode"], "first_frame")
        self.assertEqual(self.model.calls[0]["num_inference_steps"], 10)
        self.assertIsNone(self.model.calls[0]["num_video_frames"])

    def test_horizon_two_uses_idm_and_truncates_execution_chunk(self) -> None:
        actions = self.wam(self._observation(), 2, 2, 1, 4)

        self.assertEqual(tuple(actions.shape), (4, 7))
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(self.model.calls[0]["action_infer_mode"], "idm")
        self.assertEqual(self.model.calls[0]["num_inference_steps"], 2)
        self.assertEqual(self.model.calls[0]["num_video_frames"], 9)
        torch.testing.assert_close(actions, torch.full((4, 7), 2.5))

    def test_parallel_futures_are_averaged_before_postprocessing(self) -> None:
        actions = self.wam(self._observation(), 4, 4, 2, 2)

        self.assertEqual(tuple(actions.shape), (2, 7))
        self.assertEqual(len(self.model.calls), 2)
        self.assertTrue(
            all(call["action_infer_mode"] == "idm" for call in self.model.calls)
        )
        self.assertTrue(
            all(call["num_inference_steps"] == 4 for call in self.model.calls)
        )
        self.assertTrue(
            all(call["num_video_frames"] == 17 for call in self.model.calls)
        )
        torch.testing.assert_close(actions, torch.full((2, 7), 3.5))

    def test_non_finite_action_predictions_raise(self) -> None:
        self.model.non_finite = True
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.wam(self._observation(), 0, 0, 1, 10)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA memory check requires CUDA")
    def test_repeated_cuda_inference_releases_temporary_allocations(self) -> None:
        self.wam.to("cuda")
        warmup = self.wam(self._observation("cuda"), 4, 4, 2, 2)
        del warmup
        torch.cuda.synchronize()
        gc.collect()
        baseline = torch.cuda.memory_allocated()

        for _ in range(3):
            output = self.wam(self._observation("cuda"), 4, 4, 2, 2)
            self.assertEqual(tuple(output.shape), (2, 7))
            del output
        gc.collect()
        torch.cuda.synchronize()

        self.assertLessEqual(torch.cuda.memory_allocated(), baseline + 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
