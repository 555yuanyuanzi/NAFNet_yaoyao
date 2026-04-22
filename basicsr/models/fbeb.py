from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm2d


def _inverse_sigmoid(value: float) -> float:
    value = min(max(value, 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class SEBranch(nn.Module):
    """
    Lightweight per-band channel recalibration.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.gelu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class BandCompensationBlock(nn.Module):
    """
    Band-specific compensation block used after inverse FFT.

    Each band keeps a lightweight local operator so low / mid / high frequency
    responses are not treated identically after decomposition.
    """

    def __init__(
        self,
        channels: int,
        *,
        first_kernel_size: int,
        extra_depthwise: bool,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        padding = first_kernel_size // 2
        self.se = SEBranch(channels, reduction=reduction)
        self.dwconv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=first_kernel_size,
            padding=padding,
            groups=channels,
            bias=True,
        )
        self.pwconv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.extra_depthwise = extra_depthwise
        if extra_depthwise:
            self.dwconv2 = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=True,
            )
            self.pwconv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        else:
            self.dwconv2 = None
            self.pwconv2 = None
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.se(x)
        y = self.dwconv1(residual)
        y = self.pwconv1(y)
        y = self.act(y)
        if self.extra_depthwise:
            assert self.dwconv2 is not None and self.pwconv2 is not None
            y = self.dwconv2(y)
            y = self.pwconv2(y)
        return residual + self.scale * y


class BandRedistributor(nn.Module):
    """
    Predict sample-level low / mid / high redistribution weights.

    The output is a per-sample softmax over the three bands. This keeps the
    module lightweight while making band usage input-adaptive.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, 3, kernel_size=1, bias=True)

    def forward(
        self,
        low: torch.Tensor,
        mid: torch.Tensor,
        high: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        descriptor = torch.cat(
            [
                self.pool(low),
                self.pool(mid),
                self.pool(high),
                self.pool(context),
            ],
            dim=1,
        )
        logits = self.fc2(F.gelu(self.fc1(descriptor)))
        return torch.softmax(logits, dim=1)


class FrequencyBandEnhancementBlock(nn.Module):
    """
    FBEB-v2: learnable radial decomposition with band-specific compensation and
    sample-adaptive band redistribution.

    Flow:
        LayerNorm2d
        -> FFT2 in float32
        -> learnable low / mid / high soft masks
        -> inverse FFT to spatial bands
        -> per-band compensation blocks
        -> sample-level band redistribution
        -> concat + 1x1 conv + 3x3 conv
        -> residual add with learnable scaling
    """

    def __init__(
        self,
        channels: int,
        init_r1: float = 0.22,
        init_r2: float = 0.58,
        init_tau: float = 0.05,
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)

        self.p1 = nn.Parameter(torch.tensor(_inverse_sigmoid((init_r1 - 0.10) / 0.25), dtype=torch.float32))
        self.p2 = nn.Parameter(torch.tensor(_inverse_sigmoid((init_r2 - 0.50) / 0.25), dtype=torch.float32))
        self.p3 = nn.Parameter(torch.tensor(_inverse_sigmoid((init_tau - 0.03) / 0.07), dtype=torch.float32))

        self.comp_low = BandCompensationBlock(
            channels,
            first_kernel_size=5,
            extra_depthwise=False,
            reduction=se_reduction,
        )
        self.comp_mid = BandCompensationBlock(
            channels,
            first_kernel_size=3,
            extra_depthwise=False,
            reduction=se_reduction,
        )
        self.comp_high = BandCompensationBlock(
            channels,
            first_kernel_size=3,
            extra_depthwise=True,
            reduction=se_reduction,
        )
        self.redistributor = BandRedistributor(channels, reduction=se_reduction)

        # Keep a lightweight spatial path so the block still carries local cues
        # alongside the decomposed frequency bands.
        self.spa_conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )
        self.spa_conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )

        self.fuse_in = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True)
        self.fuse_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._last_band_stats: dict[str, torch.Tensor] = {}
        self._last_band_visuals: dict[str, torch.Tensor] = {}

    def _build_radius_map(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        Build a normalized radius map on the shifted FFT plane.
        """

        y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        radius = radius / radius.max().clamp_min(1e-6)
        return radius.view(1, 1, height, width)

    def get_band_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Map unconstrained parameters to bounded, interpretable values.

        r1: low / mid cutoff in [0.10, 0.35]
        r2: mid / high cutoff in [0.50, 0.75]
        tau: transition softness in [0.03, 0.10]
        """

        r1 = 0.10 + 0.25 * torch.sigmoid(self.p1)
        r2 = 0.50 + 0.25 * torch.sigmoid(self.p2)
        tau = 0.03 + 0.07 * torch.sigmoid(self.p3)
        return r1, r2, tau

    def _build_soft_masks(
        self,
        radius: torch.Tensor,
        r1: torch.Tensor,
        r2: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build normalized low / mid / high soft radial masks.
        """

        low = torch.sigmoid((r1 - radius) / tau)
        high = torch.sigmoid((radius - r2) / tau)
        mid = torch.sigmoid((radius - r1) / tau) * torch.sigmoid((r2 - radius) / tau)
        denom = (low + mid + high).clamp_min(1e-6)
        return low / denom, mid / denom, high / denom

    def _to_visual_map(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Reduce a multi-channel feature map to a single-channel visualization map.
        """

        return feature.detach().abs().mean(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x_norm = self.norm(x)

        fft_x = torch.fft.fftshift(torch.fft.fft2(x_norm.float(), dim=(-2, -1)), dim=(-2, -1))

        radius = self._build_radius_map(x.shape[-2], x.shape[-1], x.device)
        r1, r2, tau = self.get_band_params()
        mask_low, mask_mid, mask_high = self._build_soft_masks(radius, r1, r2, tau)

        low_fft = fft_x * mask_low
        mid_fft = fft_x * mask_mid
        high_fft = fft_x * mask_high

        low = torch.fft.ifft2(torch.fft.ifftshift(low_fft, dim=(-2, -1)), dim=(-2, -1)).real.to(dtype=x.dtype)
        mid = torch.fft.ifft2(torch.fft.ifftshift(mid_fft, dim=(-2, -1)), dim=(-2, -1)).real.to(dtype=x.dtype)
        high = torch.fft.ifft2(torch.fft.ifftshift(high_fft, dim=(-2, -1)), dim=(-2, -1)).real.to(dtype=x.dtype)

        low = self.comp_low(low)
        mid = self.comp_mid(mid)
        high = self.comp_high(high)

        alphas = self.redistributor(low, mid, high, x_norm).to(dtype=x.dtype)
        low = low * alphas[:, 0:1]
        mid = mid * alphas[:, 1:2]
        high = high * alphas[:, 2:3]

        fused_freq = torch.cat([low, mid, high], dim=1)
        fused_freq = self.fuse_out(self.fuse_in(fused_freq))

        spatial = self.spa_conv1(x_norm)
        spatial = F.gelu(spatial)
        spatial = self.spa_conv2(spatial)
        fused = fused_freq + spatial

        self._last_band_stats = {
            "r1": r1.detach(),
            "r2": r2.detach(),
            "tau": tau.detach(),
            "low_energy": low.abs().mean().detach(),
            "mid_energy": mid.abs().mean().detach(),
            "high_energy": high.abs().mean().detach(),
            "alpha_low": alphas[:, 0:1].mean().detach(),
            "alpha_mid": alphas[:, 1:2].mean().detach(),
            "alpha_high": alphas[:, 2:3].mean().detach(),
        }
        self._last_band_visuals = {
            "low": self._to_visual_map(low),
            "mid": self._to_visual_map(mid),
            "high": self._to_visual_map(high),
        }
        return identity + self.gamma * fused

    def get_last_band_stats(self) -> dict[str, torch.Tensor]:
        return self._last_band_stats

    def get_last_band_visuals(self) -> dict[str, torch.Tensor]:
        return self._last_band_visuals