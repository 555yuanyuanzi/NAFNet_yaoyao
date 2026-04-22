import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchAveraging(nn.Module):
    def __init__(self, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.patch_size
        b, c, h, w = x.shape

        assert h % k == 0 and w % k == 0, "H and W must be divisible by patch_size"

        pooled = F.avg_pool2d(x, kernel_size=k, stride=k)
        out = pooled.repeat_interleave(k, dim=2).repeat_interleave(k, dim=3)
        return out