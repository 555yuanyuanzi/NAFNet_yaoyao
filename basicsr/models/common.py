from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    """自定义 2D LayerNorm 的底层实现。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        ctx.eps = eps
        n, c, h, w = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, c, 1, 1) * y + bias.view(1, c, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        eps = ctx.eps
        n, c, h, w = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, c, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        d_weight = (grad_output * y).sum(dim=(0, 2, 3))
        d_bias = grad_output.sum(dim=(0, 2, 3))
        return gx, d_weight, d_bias, None


class LayerNorm2d(nn.Module):
    """对每个空间位置在通道维上做 LayerNorm。"""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """将通道一分为二后做逐元素相乘。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class MaskedDWConv2d(nn.Module):
    """带固定二值掩码的深度可分离卷积。"""

    def __init__(self, channels: int, kernel_size: int, mask: torch.Tensor, bias: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None
        self.register_buffer("mask", mask.view(1, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight * self.mask
        return F.conv2d(x, masked_weight, self.bias, stride=1, padding=self.padding, groups=self.channels)


def make_direction_masks(kernel_size: int) -> dict[str, torch.Tensor]:
    """生成水平、垂直、主对角线、副对角线 4 个方向掩码。"""

    if kernel_size % 2 == 0:
        raise ValueError("Directional masks require an odd kernel size.")
    center = kernel_size // 2
    horizontal = torch.zeros(kernel_size, kernel_size)
    horizontal[center, :] = 1.0
    vertical = torch.zeros(kernel_size, kernel_size)
    vertical[:, center] = 1.0
    main_diagonal = torch.eye(kernel_size)
    anti_diagonal = torch.flip(main_diagonal, dims=[1])
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "main_diagonal": main_diagonal,
        "anti_diagonal": anti_diagonal,
    }


def window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
    """按固定窗口大小把特征图切成不重叠窗口。"""

    b, c, h, w = x.shape
    if h % window_size != 0 or w % window_size != 0:
        raise ValueError(f"Input spatial size {(h, w)} must be divisible by window_size={window_size}.")
    hg = h // window_size
    wg = w // window_size
    windows = (
        x.view(b, c, hg, window_size, wg, window_size)
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .view(b * hg * wg, c, window_size, window_size)
    )
    return windows, hg, wg


def pad_to_multiple(x: torch.Tensor, multiple: int, mode: str = "replicate") -> tuple[torch.Tensor, int, int]:
    """将空间尺寸补到指定倍数，返回补齐后的张量和 padding 大小。"""

    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, pad_h, pad_w
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return padded, pad_h, pad_w
