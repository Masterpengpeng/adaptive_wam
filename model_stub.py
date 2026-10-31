"""Composable WAM inference interface with lightweight profiling components."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


class VisionEncoder(nn.Module):
    """Small image encoder used by the default profiling stub."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.encoder(image)


class StateEncoder(nn.Module):
    """Encodes a proprioceptive state vector."""

    def __init__(self, state_dim: int = 14, feature_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.SiLU(),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.encoder(state)


class FutureGenerator(nn.Module):
    """Refines independent latent rollouts and decodes state/video features."""

    def __init__(
        self,
        feature_dim: int = 128,
        latent_dim: int = 128,
        video_feature_dim: int = 64,
        state_dim: int = 14,
    ) -> None:
        super().__init__()
        self.condition = nn.Linear(feature_dim * 2, latent_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )
        self.state_decoder = nn.Linear(latent_dim, state_dim)
        self.video_decoder = nn.Linear(latent_dim, video_feature_dim)

    def forward(
        self,
        vision_features: Tensor,
        state_features: Tensor,
        H: int,
        D: int,
        W: int,
    ) -> dict[str, Tensor]:
        batch_size = vision_features.shape[0]
        condition = self.condition(torch.cat((vision_features, state_features), dim=-1))
        latents = torch.randn(
            batch_size,
            W,
            H,
            condition.shape[-1],
            device=condition.device,
            dtype=condition.dtype,
        )
        expanded_condition = condition[:, None, None, :].expand_as(latents)
        for _ in range(D):
            denoiser_input = torch.cat((latents, expanded_condition), dim=-1)
            latents = latents - self.denoiser(denoiser_input) / D

        return {
            "states": self.state_decoder(latents),
            "video_latents": self.video_decoder(latents),
            "features": latents.mean(dim=(1, 2)),
        }


class ActionHead(nn.Module):
    """Action head with separate fast and imagination-conditioned paths."""

    def __init__(
        self,
        feature_dim: int = 128,
        action_dim: int = 7,
    ) -> None:
        super().__init__()
        self.fast_path = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.SiLU(),
            nn.Linear(128, action_dim),
        )
        self.planned_path = nn.Sequential(
            nn.Linear(feature_dim * 3, 128),
            nn.SiLU(),
            nn.Linear(128, action_dim),
        )

    def forward(
        self,
        vision_features: Tensor,
        state_features: Tensor,
        future_features: Tensor | None = None,
    ) -> Tensor:
        observed_features = torch.cat((vision_features, state_features), dim=-1)
        if future_features is None:
            return self.fast_path(observed_features)
        return self.planned_path(torch.cat((observed_features, future_features), dim=-1))


class WorldActionModel(nn.Module):
    """Modular world-action policy with an observation-only H=0 fast path.

    The default modules are deliberately small synthetic components for profiling.
    A real model can replace any component with a module implementing the same
    input/output contract.

    ``x_t`` must contain ``image`` with shape ``[B, 3, height, width]`` and
    ``state`` with shape ``[B, state_dim]``. The result contains an ``action``
    tensor and either ``None`` or generated ``trajectories``.
    """

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 7,
        feature_dim: int = 128,
        vision_encoder: nn.Module | None = None,
        state_encoder: nn.Module | None = None,
        future_generator: nn.Module | None = None,
        action_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.vision_encoder = (
            vision_encoder if vision_encoder is not None else VisionEncoder(feature_dim)
        )
        self.state_encoder = (
            state_encoder
            if state_encoder is not None
            else StateEncoder(state_dim, feature_dim)
        )
        self.future_generator = (
            future_generator
            if future_generator is not None
            else FutureGenerator(
                feature_dim=feature_dim,
                latent_dim=feature_dim,
                state_dim=state_dim,
            )
        )
        self.action_head = (
            action_head if action_head is not None else ActionHead(feature_dim, action_dim)
        )

    def infer(
        self,
        x_t: Mapping[str, Tensor],
        H: int,
        D: int,
        W: int,
    ) -> dict[str, Tensor | dict[str, Tensor] | None]:
        """Infer an action and optionally generate W future trajectories.

        For ``H == 0``, D and W are normalized to zero and the future generator
        is never called. For positive H, D and W must both be positive.
        """
        H, D, W = self._normalize_config(H, D, W)
        image, state = self._validate_inputs(x_t)
        vision_features = self.vision_encoder(image)
        state_features = self.state_encoder(state)

        trajectories = None
        future_features = None
        if H > 0:
            trajectories = self.future_generator(
                vision_features,
                state_features,
                H,
                D,
                W,
            )
            future_features = trajectories["features"]

        action = self.action_head(vision_features, state_features, future_features)
        return {"action": action, "trajectories": trajectories}

    def forward(
        self,
        x_t: Mapping[str, Tensor],
        H: int,
        D: int,
        W: int,
    ) -> dict[str, Tensor | dict[str, Tensor] | None]:
        return self.infer(x_t, H, D, W)

    def _validate_inputs(self, x_t: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        try:
            image = x_t["image"]
            state = x_t["state"]
        except KeyError as exc:
            raise ValueError("x_t must contain 'image' and 'state' tensors") from exc

        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[2:]) <= 0:
            raise ValueError("image must have shape [B, 3, height, width]")
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must have shape [B, {self.state_dim}]")
        if image.shape[0] != state.shape[0]:
            raise ValueError("image and state must have the same batch size")
        if image.device != state.device:
            raise ValueError("image and state must be on the same device")
        return image, state

    @staticmethod
    def _normalize_config(H: int, D: int, W: int) -> tuple[int, int, int]:
        for name, value in (("H", H), ("D", D), ("W", W)):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
        if H < 0:
            raise ValueError("H must be non-negative")
        if H == 0:
            return 0, 0, 0
        if D <= 0 or W <= 0:
            raise ValueError("D and W must be positive when H is positive")
        return H, D, W
