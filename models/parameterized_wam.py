"""Parameterized FastWAM inference with adaptive H/D/W/E controls."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


ObservationPreprocessor = Callable[[Any], Mapping[str, Any]]
ActionPostprocessor = Callable[[Tensor, int], Tensor | np.ndarray]


class ParameterizedWAM(nn.Module):
    """Wrap a FastWAM policy with a single parameterized inference interface.

    ``prepare_observation`` returns the fixed FastWAM ``infer_action`` inputs
    (for example ``prompt``, ``input_image``, ``proprio`` and guidance options).
    ``postprocess_actions`` denormalizes the W-averaged action prediction and
    applies the policy's action conventions. The returned action chunk is a
    CPU float32 tensor with shape ``[E, action_dim]``.

    H=0 selects FastWAM's first-frame/direct-action branch, uses
    ``direct_depth`` denoising steps, and performs exactly one action sample.
    H>0 selects the IDM branch, maps H to ``1 + H * action_video_freq_ratio``
    video frames, and averages W independently seeded normalized action samples.
    """

    def __init__(
        self,
        model: nn.Module,
        prepare_observation: ObservationPreprocessor,
        postprocess_actions: ActionPostprocessor,
        *,
        action_horizon: int,
        action_video_freq_ratio: int,
        direct_depth: int = 10,
        negative_prompt: str = "",
        text_cfg_scale: float = 1.0,
        sigma_shift: float | None = None,
        tiled: bool = False,
        compile_action_infer: bool = False,
    ) -> None:
        super().__init__()
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if action_video_freq_ratio <= 0:
            raise ValueError("action_video_freq_ratio must be positive")
        if direct_depth <= 0:
            raise ValueError("direct_depth must be positive")
        if not callable(getattr(model, "infer_action", None)):
            raise TypeError("model must implement infer_action(**kwargs)")

        self.model = model
        self.prepare_observation = prepare_observation
        self.postprocess_actions = postprocess_actions
        self.action_horizon = action_horizon
        self.action_video_freq_ratio = action_video_freq_ratio
        self.direct_depth = direct_depth
        self.negative_prompt = negative_prompt
        self.text_cfg_scale = text_cfg_scale
        self.sigma_shift = sigma_shift
        self.tiled = tiled
        self.compile_action_infer = compile_action_infer

    def forward(
        self,
        obs: Any,
        H: int,
        D: int,
        W: int,
        E: int,
    ) -> Tensor:
        """Predict and postprocess one action chunk under the supplied config."""
        self._validate_config(H, D, W, E)
        if E > self.action_horizon:
            raise ValueError(
                f"E={E} exceeds the model action horizon {self.action_horizon}"
            )

        infer_kwargs = dict(self.prepare_observation(obs))
        input_image = infer_kwargs.get("input_image")
        if not isinstance(input_image, Tensor):
            raise TypeError("prepare_observation must provide a Tensor input_image")

        infer_action = self.model.infer_action
        parameters = inspect.signature(infer_action).parameters
        if H > 0:
            for required_parameter in ("action_infer_mode", "num_video_frames"):
                if required_parameter not in parameters:
                    raise TypeError(
                        f"{type(self.model).__name__}.infer_action does not support "
                        f"{required_parameter}; H>0 requires an IDM-capable model"
                    )
            action_mode = "idm"
        else:
            action_mode = "first_frame"
            if "action_infer_mode" in parameters:
                infer_kwargs["action_infer_mode"] = action_mode

        if "action_horizon" in parameters:
            infer_kwargs.setdefault("action_horizon", self.action_horizon)
        if "negative_prompt" in parameters:
            infer_kwargs.setdefault("negative_prompt", self.negative_prompt)
        if "text_cfg_scale" in parameters:
            infer_kwargs.setdefault("text_cfg_scale", self.text_cfg_scale)
        if "sigma_shift" in parameters:
            infer_kwargs.setdefault("sigma_shift", self.sigma_shift)
        if "rand_device" in parameters:
            infer_kwargs.setdefault("rand_device", "cpu")
        if "tiled" in parameters:
            infer_kwargs.setdefault("tiled", self.tiled)
        if "compile_action_infer" in parameters:
            infer_kwargs.setdefault(
                "compile_action_infer", self.compile_action_infer
            )
        if H > 0:
            infer_kwargs["action_infer_mode"] = action_mode
            infer_kwargs["num_video_frames"] = (
                1 + H * self.action_video_freq_ratio
            )

        sampling_depth = D if H > 0 else self.direct_depth
        samples: list[Tensor] = []
        with torch.inference_mode():
            for _ in range(W if H > 0 else 1):
                kwargs = dict(infer_kwargs)
                kwargs["num_inference_steps"] = sampling_depth
                kwargs["seed"] = int(np.random.randint(0, 2**31 - 1))
                with torch.autocast(
                    device_type=input_image.device.type,
                    dtype=torch.float16,
                    enabled=input_image.device.type == "cuda",
                ):
                    prediction = infer_action(**kwargs)
                if not isinstance(prediction, Mapping):
                    raise TypeError("FastWAM infer_action must return a mapping")
                action = prediction.get("action")
                if not isinstance(action, Tensor):
                    raise TypeError(
                        "FastWAM infer_action must return a Tensor in ['action']"
                    )
                if action.ndim != 2 or action.shape[0] < E:
                    raise ValueError(
                        f"FastWAM returned action shape {tuple(action.shape)} for E={E}"
                    )
                if not torch.isfinite(action[:E]).all().item():
                    raise ValueError("FastWAM returned non-finite action values")
                samples.append(
                    action[:E].detach().to(device="cpu", dtype=torch.float32)
                )
                del prediction, action

        normalized_actions = torch.stack(samples, dim=0).mean(dim=0)
        actions = self.postprocess_actions(normalized_actions, E)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(np.asarray(actions))
        if not isinstance(actions, Tensor):
            raise TypeError("postprocess_actions must return a Tensor or numpy array")
        if actions.ndim != 2 or actions.shape[0] != E:
            raise ValueError(
                f"postprocess_actions must return shape [E, action_dim], got "
                f"{tuple(actions.shape)} for E={E}"
            )
        if not torch.isfinite(actions).all().item():
            raise ValueError("postprocess_actions returned non-finite values")
        return actions.detach().to(device="cpu", dtype=torch.float32)

    @staticmethod
    def _validate_config(H: int, D: int, W: int, E: int) -> None:
        for name, value in (("H", H), ("D", D), ("W", W), ("E", E)):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
        if H < 0:
            raise ValueError("H must be non-negative")
        if D < 0:
            raise ValueError("D must be non-negative")
        if W <= 0:
            raise ValueError("W must be positive")
        if E <= 0:
            raise ValueError("E must be positive")
        if H == 0 and D != 0:
            raise ValueError("H=0 direct-path configurations must use D=0")
        if H > 0 and D <= 0:
            raise ValueError("D must be positive when H is positive")
