from __future__ import annotations

import torch
import torch.nn as nn

from .common import LayerNorm2d


class BlurAwareSkipFusion(nn.Module):
    """
    Blur-aware skip fusion for encoder-decoder connections.

    The module keeps the original skip addition at initialization:
        out = x_dec + x_skip

    During training it learns a bounded residual modulation on skip features,
    allowing reliable skip information to be enhanced and blur-corrupted skip
    information to be suppressed.
    """

    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 1.0,
        scale_limit: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_ratio <= 0:
            raise ValueError("hidden_ratio must be positive.")
        if scale_limit <= 0:
            raise ValueError("scale_limit must be positive.")

        hidden_channels = max(1, int(channels * hidden_ratio))
        self.scale_limit = scale_limit
        self.norm_dec = LayerNorm2d(channels)
        self.norm_skip = LayerNorm2d(channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._last_aux: dict[str, torch.Tensor] = {}

    def forward(
        self,
        x_dec: torch.Tensor,
        x_skip: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        gate_input = torch.cat([self.norm_dec(x_dec), self.norm_skip(x_skip)], dim=1)
        gate = self.gate(gate_input)
        alpha = self.scale_limit * torch.tanh(self.scale)
        out = x_dec + x_skip + alpha * gate * x_skip

        self._last_aux = {
            "gate_mean": gate.mean().detach(),
            "gate_std": gate.std().detach(),
            "alpha_abs_mean": alpha.abs().mean().detach(),
        }

        if return_gate:
            return out, gate
        return out

    def get_last_aux(self) -> dict[str, torch.Tensor]:
        return self._last_aux


__all__ = ["BlurAwareSkipFusion"]
