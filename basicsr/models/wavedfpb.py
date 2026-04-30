from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm2d
from .dfpb import HighFrequencyRectifier, LowFrequencyRestorer, LowGuidedDeformableFusion


class HaarWaveletFrequencyExtractor(nn.Module):
    """
    Split features with one-level Haar wavelet decomposition.

    The low component is reconstructed from LL only, and the high component is
    reconstructed from LH/HL/HH. Both outputs keep the input spatial size.
    """

    def __init__(self) -> None:
        super().__init__()
        filters = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[-0.5, -0.5], [0.5, 0.5]],
                [[-0.5, 0.5], [-0.5, 0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        ).view(4, 1, 2, 2)
        self.register_buffer("filters", filters)

    def _filters(self, channels: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.filters.to(device=device, dtype=dtype).repeat(channels, 1, 1, 1)

    def _dwt(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, channels, height, width = x.shape
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        filters = self._filters(channels, x.dtype, x.device)
        coeffs = F.conv2d(x, filters, stride=2, groups=channels)
        coeffs = coeffs.view(x.shape[0], channels, 4, x.shape[2] // 2, x.shape[3] // 2)
        return coeffs, (height, width)

    def _idwt(self, coeffs: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        batch, channels, _, _, _ = coeffs.shape
        filters = self._filters(channels, coeffs.dtype, coeffs.device)
        x = F.conv_transpose2d(
            coeffs.view(batch, channels * 4, coeffs.shape[-2], coeffs.shape[-1]),
            filters,
            stride=2,
            groups=channels,
        )
        height, width = output_size
        return x[:, :, :height, :width]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        wave_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

        coeffs, output_size = self._dwt(wave_input)
        low_coeffs = torch.zeros_like(coeffs)
        high_coeffs = torch.zeros_like(coeffs)
        low_coeffs[:, :, 0] = coeffs[:, :, 0]
        high_coeffs[:, :, 1:] = coeffs[:, :, 1:]

        x_low = self._idwt(low_coeffs, output_size)
        x_high = self._idwt(high_coeffs, output_size)
        return x_low.to(orig_dtype), x_high.to(orig_dtype)


class WaveletDualFrequencyProgressiveBlock(nn.Module):
    """
    Dual-frequency progressive block with one-level Haar wavelet split.
    """

    def __init__(
        self,
        channels: int,
        low_blocks: int = 1,
        branch_expand_ratio: int = 2,
        use_deformable_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.frequency_extractor = HaarWaveletFrequencyExtractor()
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
    "HaarWaveletFrequencyExtractor",
    "WaveletDualFrequencyProgressiveBlock",
]
