from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class BasicConv(nn.Module):
    """Minimal Conv + optional ReLU block used by DGFM variants."""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int = 1,
        *,
        bias: bool = False,
        relu: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=groups,
            )
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class DeformableConv2d(nn.Module):
    """
    Deformable conv wrapper (copied from DeRestormer with small cleanup).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        kernel_h = kernel_size
        kernel_w = kernel_size
        self.stride = (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_h * kernel_w,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        self.modulator_conv = nn.Conv2d(
            in_channels,
            kernel_h * kernel_w,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        nn.init.constant_(self.modulator_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.bias, 0.0)

        self.regular_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))
        return deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
        )


class DGFM(nn.Module):
    """
    Deformable Gated Fusion Module from DeRestormer.
    Input tensors are three aligned feature maps at the same spatial size.
    """

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.conv_max = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            DeformableConv2d(out_channel, out_channel, kernel_size=7, padding=3, stride=1),
        )
        self.conv_mid = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            DeformableConv2d(out_channel, out_channel, kernel_size=5, padding=2, stride=1),
        )
        self.conv_small = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            DeformableConv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1),
        )

        # Keep the original DeRestormer channel convention for compatibility.
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True)
        self.conv2 = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True)

    def forward(
        self,
        x_max: torch.Tensor,
        x_mid: torch.Tensor,
        x_small: torch.Tensor,
    ) -> torch.Tensor:
        y_max = x_max + x_mid + x_small

        x_max = self.conv_max(x_max)
        x_mid = self.conv_mid(x_mid)
        x_small = self.conv_small(x_small)

        x = torch.tanh(x_mid) * x_max
        x = self.conv1(x)

        x = torch.tanh(x_small) * x
        x = self.conv2(x)
        return x + y_max


__all__ = ["DGFM"]
