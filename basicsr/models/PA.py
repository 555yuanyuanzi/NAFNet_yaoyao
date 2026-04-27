import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchAveraging(nn.Module):
    def __init__(self, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.patch_size
        _, _, h, w = x.shape

        pad_h = (k - h % k) % k
        pad_w = (k - w % k) % k

        if pad_h > 0 or pad_w > 0:
            pad_mode = 'reflect'
            # F.pad(..., mode='reflect') requires the padded amount to stay
            # smaller than the source size. Fall back only for degenerate maps.
            if (pad_h > 0 and h <= pad_h) or (pad_w > 0 and w <= pad_w):
                pad_mode = 'replicate'
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)

        pooled = F.avg_pool2d(x, kernel_size=k, stride=k)
        out = pooled.repeat_interleave(k, dim=2).repeat_interleave(k, dim=3)
        return out[:, :, :h, :w]
