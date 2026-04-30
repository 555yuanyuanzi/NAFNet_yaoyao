from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm2d

try:
    from torchvision.ops import deform_conv2d as tv_deform_conv2d
except Exception:  # pragma: no cover - fallback depends on runtime environment
    tv_deform_conv2d = None


class FFTFrequencyExtractor(nn.Module):
    """
    Split features into low/high frequency parts with a centered FFT mask.

    The fixed radius in fft_dfpb.py is converted to a ratio by default, because
    DFPB operates on feature maps whose spatial sizes vary by stage.
    """

    def __init__(
        self,
        radius_ratio: float = 0.3,
        radius: int | float | None = None,
    ) -> None:
        super().__init__()
        if radius is None and radius_ratio <= 0:
            raise ValueError("radius_ratio must be positive when radius is not set.")
        if radius is None and radius_ratio > 0.5:
            raise ValueError("radius_ratio should be <= 0.5 for a meaningful low/high split.")
        if radius is not None and radius < 0:
            raise ValueError("radius must be non-negative.")
        self.radius_ratio = radius_ratio
        self.radius = radius

    def _build_low_mask(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        center_y, center_x = height // 2, width // 2
        radius = self.radius
        if radius is None:
            radius = min(height, width) * self.radius_ratio

        y = torch.arange(height, device=device, dtype=dtype)
        x = torch.arange(width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        distance = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        return (distance <= radius).to(dtype=dtype).view(1, 1, height, width)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        fft_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        _, _, height, width = fft_input.shape

        f_transform = torch.fft.fft2(fft_input, dim=(-2, -1))
        f_shift = torch.fft.fftshift(f_transform, dim=(-2, -1))

        mask_low = self._build_low_mask(
            height,
            width,
            device=fft_input.device,
            dtype=fft_input.real.dtype,
        )
        mask_high = 1.0 - mask_low

        f_shift_low = f_shift * mask_low
        f_shift_high = f_shift * mask_high

        x_low = torch.fft.ifft2(
            torch.fft.ifftshift(f_shift_low, dim=(-2, -1)),
            dim=(-2, -1),
        ).real
        x_high = torch.fft.ifft2(
            torch.fft.ifftshift(f_shift_high, dim=(-2, -1)),
            dim=(-2, -1),
        ).real

        return x_low.to(orig_dtype), x_high.to(orig_dtype)


class ResidualConvBlock(nn.Module):
    """Lightweight residual conv block used inside low/high branches."""

    def __init__(self, channels: int, expand_ratio: int = 2) -> None:
        super().__init__()
        hidden = channels * expand_ratio
        self.norm = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=True)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.act(y)
        y = self.pw2(y)
        return residual + self.scale * y


class LowFrequencyRestorer(nn.Module):
    """Structure-oriented low-frequency restoration branch."""

    def __init__(self, channels: int, num_blocks: int = 1, expand_ratio: int = 2) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResidualConvBlock(channels, expand_ratio=expand_ratio) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class HighFrequencyRectifier(nn.Module):
    """
    Low-guided high-frequency rectification branch.

    The branch predicts a spatial gate from both high-frequency residual and
    restored low-frequency structure, then uses the gate to control how much
    detail correction should be injected.
    """

    def __init__(self, channels: int, expand_ratio: int = 2) -> None:
        super().__init__()
        hidden = channels * expand_ratio
        self.norm = LayerNorm2d(channels)
        self.high_proj = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.high_dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=True)
        self.high_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(
        self,
        x_high: torch.Tensor,
        x_low: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate(torch.cat([x_high, x_low], dim=1))

        y = self.norm(x_high)
        y = self.high_proj(y)
        y = self.high_dw(y)
        y = self.act(y)
        y = self.high_out(y)
        out = x_high + self.scale * gate * y

        if return_gate:
            return out, gate
        return out


class ConditionedDeformableConv2d(nn.Module):
    """
    Deformable conv whose offsets and masks are predicted from a condition map.

    If torchvision deformable convolution is unavailable at runtime, it falls
    back to a standard convolution so the surrounding module still remains
    usable for ablation and integration.
    """

    def __init__(
        self,
        channels: int,
        condition_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_deformable: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_deformable = use_deformable and tv_deform_conv2d is not None
        self.weight = nn.Parameter(torch.empty(channels, channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        self.offset_mask = nn.Conv2d(
            condition_channels,
            3 * kernel_size * kernel_size,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        nn.init.constant_(self.offset_mask.weight, 0.0)
        nn.init.constant_(self.offset_mask.bias, 0.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if not self.use_deformable:
            return F.conv2d(x, self.weight, self.bias, padding=self.padding)

        offset_mask = self.offset_mask(cond)
        k2 = self.kernel_size * self.kernel_size
        offset = offset_mask[:, : 2 * k2]
        mask = torch.sigmoid(offset_mask[:, 2 * k2 :])
        return tv_deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            padding=(self.padding, self.padding),
            mask=mask,
        )


class LowGuidedDeformableFusion(nn.Module):
    """
    Fuse base, low-frequency, and high-frequency features with low-guided
    deformable alignment on the high-frequency branch.
    """

    def __init__(self, channels: int, use_deformable: bool = True) -> None:
        super().__init__()
        self.base_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.low_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.high_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.align_high = ConditionedDeformableConv2d(
            channels=channels,
            condition_channels=channels * 2,
            kernel_size=3,
            padding=1,
            use_deformable=use_deformable,
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(
        self,
        x_base: torch.Tensor,
        x_low: torch.Tensor,
        x_high: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base = self.base_proj(x_base)
        low = self.low_proj(x_low)
        high = self.high_proj(x_high)

        align_cond = torch.cat([base, low], dim=1)
        high_aligned = self.align_high(high, align_cond)
        gate = self.fusion_gate(torch.cat([low, high_aligned], dim=1))
        fused = self.fuse(torch.cat([base, low, gate * high_aligned], dim=1))

        if return_aux:
            aux = {
                "fusion_gate": gate,
                "high_aligned": high_aligned,
            }
            return fused, aux
        return fused


class FFTDualFrequencyProgressiveBlock(nn.Module):
    """
    Dual-frequency progressive block for motion deblurring.

    Flow:
        LayerNorm2d
        -> FFT low/high-frequency decomposition
        -> low-frequency structure restoration
        -> low-guided high-frequency rectification
        -> low-guided deformable fusion
        -> residual add with learnable scaling
    """

    def __init__(
        self,
        channels: int,
        low_blocks: int = 1,
        branch_expand_ratio: int = 2,
        use_deformable_fusion: bool = True,
        fft_radius_ratio: float = 0.3,
        fft_radius: int | float | None = None,
    ) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.frequency_extractor = FFTFrequencyExtractor(
            radius_ratio=fft_radius_ratio,
            radius=fft_radius,
        )
        self.low_restorer = LowFrequencyRestorer(
            channels,
            num_blocks=low_blocks,
            expand_ratio=branch_expand_ratio,
        )
        self.high_rectifier = HighFrequencyRectifier(
            channels,
            expand_ratio=branch_expand_ratio,
        )
        self.fusion = LowGuidedDeformableFusion(
            channels,
            use_deformable=use_deformable_fusion,
        )
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._last_aux: dict[str, torch.Tensor] = {}

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        identity = x
        x_norm = self.norm(x)

        x_low, x_high = self.frequency_extractor(x_norm)

        x_low_hat = self.low_restorer(x_low)
        x_high_hat, rect_gate = self.high_rectifier(x_high, x_low_hat, return_gate=True)
        fused, fusion_aux = self.fusion(x_norm, x_low_hat, x_high_hat, return_aux=True)

        out = identity + self.gamma * fused

        self._last_aux = {
            "low_energy": x_low.abs().mean().detach(),
            "high_energy": x_high.abs().mean().detach(),
            "rect_gate_mean": rect_gate.mean().detach(),
            "fusion_gate_mean": fusion_aux["fusion_gate"].mean().detach(),
            "deformable_enabled": torch.tensor(
                float(self.fusion.align_high.use_deformable),
                device=x.device,
                dtype=x.dtype,
            ),
        }

        if return_aux:
            aux = {
                "x_low": x_low,
                "x_high": x_high,
                "x_low_hat": x_low_hat,
                "x_high_hat": x_high_hat,
                "rect_gate": rect_gate,
                **fusion_aux,
                "stats": self._last_aux,
            }
            return out, aux
        return out

    def get_last_aux(self) -> dict[str, torch.Tensor]:
        return self._last_aux


__all__ = [
    "FFTFrequencyExtractor",
    "LowFrequencyRestorer",
    "HighFrequencyRectifier",
    "LowGuidedDeformableFusion",
    "FFTDualFrequencyProgressiveBlock",
]
