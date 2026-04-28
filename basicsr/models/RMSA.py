from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm2d


class RotaryMotionAwareSkipAlignment(nn.Module):
    """
    Rotary motion-aware skip alignment for encoder-decoder fusion.

    The module projects decoder and skip features into a rotary phase space,
    uses their phase discrepancy to predict a local offset field, and aligns
    skip features before fusion. It keeps the original skip addition at
    initialization:
        out = x_dec + x_skip
    """

    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 1.0,
        pos_bands: int = 4,
        offset_limit: float = 2.0,
        scale_limit: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_ratio <= 0:
            raise ValueError("hidden_ratio must be positive.")
        if pos_bands < 1:
            raise ValueError("pos_bands must be at least 1.")
        if offset_limit <= 0:
            raise ValueError("offset_limit must be positive.")
        if scale_limit <= 0:
            raise ValueError("scale_limit must be positive.")

        hidden_channels = max(4, int(channels * hidden_ratio))
        hidden_channels = ((hidden_channels + 3) // 4) * 4
        self.pos_bands = pos_bands
        self.offset_limit = offset_limit
        self.scale_limit = scale_limit

        self.norm_dec = LayerNorm2d(channels)
        self.norm_skip = LayerNorm2d(channels)
        self.dec_proj = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        self.skip_proj = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        self.context = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=True),
            nn.GELU(),
        )
        self.offset_head = nn.Conv2d(hidden_channels, 2, kernel_size=1, bias=True)
        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._last_aux: dict[str, torch.Tensor] = {}

        nn.init.constant_(self.offset_head.weight, 0.0)
        nn.init.constant_(self.offset_head.bias, 0.0)

    def _build_base_grid(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1).unsqueeze(0)

    def _build_rotary_phase(
        self,
        height: int,
        width: int,
        pair_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        band_ids = torch.arange(pair_count, device=device, dtype=dtype)
        freqs = math.pi * torch.pow(2.0, torch.remainder(band_ids, self.pos_bands))

        phase_x = freqs.view(1, pair_count, 1, 1) * x.view(1, 1, 1, width)
        phase_y = freqs.view(1, pair_count, 1, 1) * y.view(1, 1, height, 1)
        return torch.sin(phase_x), torch.cos(phase_x), torch.sin(phase_y), torch.cos(phase_y)

    def _rotate_pairs(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        even = x[:, 0::2]
        odd = x[:, 1::2]
        rot_even = even * cos - odd * sin
        rot_odd = even * sin + odd * cos
        return torch.stack([rot_even, rot_odd], dim=2).flatten(1, 2)

    def _apply_2d_rope(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h, w = x.shape
        x_part, y_part = x.chunk(2, dim=1)
        pair_count = x_part.shape[1] // 2
        sin_x, cos_x, sin_y, cos_y = self._build_rotary_phase(
            h,
            w,
            pair_count,
            x.device,
            x.dtype,
        )
        x_rot = self._rotate_pairs(x_part, sin_x, cos_x)
        y_rot = self._rotate_pairs(y_part, sin_y, cos_y)
        return torch.cat([x_rot, y_rot], dim=1)

    def forward(
        self,
        x_dec: torch.Tensor,
        x_skip: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, _, h, w = x_dec.shape
        dec_phase = self._apply_2d_rope(self.dec_proj(self.norm_dec(x_dec)))
        skip_phase = self._apply_2d_rope(self.skip_proj(self.norm_skip(x_skip)))
        phase_delta = dec_phase - skip_phase
        phase_product = dec_phase * skip_phase
        context = self.context(torch.cat([dec_phase, skip_phase, phase_delta, phase_product], dim=1))

        offset_px = self.offset_limit * torch.tanh(self.offset_head(context))
        norm_x = 0.0 if w <= 1 else 2.0 / (w - 1)
        norm_y = 0.0 if h <= 1 else 2.0 / (h - 1)
        offset_norm = torch.stack(
            [offset_px[:, 0] * norm_x, offset_px[:, 1] * norm_y],
            dim=-1,
        )

        base_grid = self._build_base_grid(h, w, x_dec.device, x_dec.dtype)
        sample_grid = base_grid + offset_norm
        skip_aligned = F.grid_sample(
            x_skip,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        gate = self.gate_head(context)
        alpha = self.scale_limit * torch.tanh(self.scale)
        correction = gate * skip_aligned
        out = x_dec + x_skip + alpha * correction

        self._last_aux = {
            "gate_mean": gate.mean().detach(),
            "gate_std": gate.std().detach(),
            "offset_abs_mean": offset_px.abs().mean().detach(),
            "alpha_abs_mean": alpha.abs().mean().detach(),
        }

        if return_aux:
            aux = {
                "gate": gate,
                "offset_px": offset_px,
                "skip_aligned": skip_aligned,
                "stats": self._last_aux,
            }
            return out, aux
        return out

    def get_last_aux(self) -> dict[str, torch.Tensor]:
        return self._last_aux


__all__ = ["RotaryMotionAwareSkipAlignment"]
