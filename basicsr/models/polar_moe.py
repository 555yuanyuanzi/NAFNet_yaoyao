from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.arch_util import LayerNorm2d


class SpatialMoE(nn.Module):
    def __init__(self, channels: int, k: int = 4) -> None:
        super().__init__()
        self.k = k
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True),
                    nn.Conv2d(channels, channels, kernel_size=1, bias=True),
                )
                for _ in range(k)
            ]
        )

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        weights = routing_weights[:, :, None, None, None]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.sum(weights * expert_outputs, dim=1)


class PolarRouter(nn.Module):
    def __init__(
        self,
        channels: int,
        k: int = 4,
        n_theta: int = 32,
        n_r: int = 16,
        gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.k = k
        self.n_theta = n_theta
        self.n_r = n_r
        self._cache_key = None
        self._cached_grid = None

        self.router_head = nn.Sequential(
            nn.Linear(n_theta, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, k),
        )

    def _build_cache(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta = torch.linspace(0, 2 * math.pi, steps=self.n_theta, device=device, dtype=dtype).view(
            1, 1, self.n_theta
        )
        radius_max = 0.5 * math.sqrt(height**2 + width**2)
        radius = torch.linspace(0.0, radius_max, steps=self.n_r, device=device, dtype=dtype).view(
            1, self.n_r, 1
        )

        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        x_norm = x / (width * 0.5)
        y_norm = y / (height * 0.5)
        grid = torch.stack([x_norm, y_norm], dim=-1)

        yy, xx = torch.meshgrid(
            torch.linspace(-(height - 1) / 2, (height - 1) / 2, height, device=device, dtype=dtype),
            torch.linspace(-(width - 1) / 2, (width - 1) / 2, width, device=device, dtype=dtype),
            indexing="ij",
        )
        theta_map = (torch.atan2(yy, xx) + 2 * math.pi) % (2 * math.pi)
        theta_idx = torch.clamp((theta_map / (2 * math.pi) * self.n_theta).long(), 0, self.n_theta - 1)
        return grid, theta_idx

    def _get_cache(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (height, width, device, dtype)
        if self._cache_key != key:
            self._cached_grid, self._cached_theta_idx = self._build_cache(height, width, device, dtype)
            self._cache_key = key
        return self._cached_grid, self._cached_theta_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        fft_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        spectrum = torch.fft.fftshift(torch.fft.fft2(fft_input, dim=(-2, -1)), dim=(-2, -1))
        magnitude = torch.log1p(torch.abs(spectrum)).to(x.dtype)

        grid, _ = self._get_cache(height, width, x.device, x.dtype)
        polar = F.grid_sample(
            magnitude,
            grid.repeat(batch, 1, 1, 1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        theta_energy = polar.mean(dim=(1, 2))
        theta_energy = theta_energy / theta_energy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return F.softmax(self.router_head(theta_energy), dim=-1)


class PolarMoEDecoderBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        k: int = 4,
        n_theta: int = 32,
        n_r: int = 16,
        gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.router = PolarRouter(
            channels=channels,
            k=k,
            n_theta=n_theta,
            n_r=n_r,
            gate_hidden=gate_hidden,
        )
        self.moe = SpatialMoE(channels=channels, k=k)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._last_aux: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x_norm = self.norm(x)
        routing_weights = self.router(x_norm)
        residual = self.moe(x_norm, routing_weights)
        out = identity + self.scale * residual

        self._last_aux = {
            "routing_weight_mean": routing_weights.mean(dim=0).detach(),
            "routing_weight_max": routing_weights.max(dim=-1).values.mean().detach(),
            "scale_abs_mean": self.scale.abs().mean().detach(),
        }
        return out

    def get_last_aux(self) -> dict[str, torch.Tensor]:
        return self._last_aux


__all__ = [
    "SpatialMoE",
    "PolarRouter",
    "PolarMoEDecoderBlock",
]
