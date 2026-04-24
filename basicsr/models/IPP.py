import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpolatedPatchPrior(nn.Module):
    def __init__(self, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size

    def _pool_with_padding(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        k = self.patch_size
        _, _, h, w = x.shape
        pad_h = (k - h % k) % k
        pad_w = (k - w % k) % k
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        pooled = F.avg_pool2d(x, kernel_size=k, stride=k)
        return pooled, h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, h, w = self._pool_with_padding(x)
        return F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
