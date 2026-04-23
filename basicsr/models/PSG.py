import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchStatisticsExtractor(nn.Module):
    def __init__(self, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size

    def _broadcast(self, x: torch.Tensor) -> torch.Tensor:
        k = self.patch_size
        return x.repeat_interleave(k, dim=2).repeat_interleave(k, dim=3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.patch_size
        _, _, h, w = x.shape

        assert h % k == 0 and w % k == 0, "H and W must be divisible by patch_size"

        patch_mean = F.avg_pool2d(x, kernel_size=k, stride=k)
        patch_mean_up = self._broadcast(patch_mean)

        # Mean absolute deviation inside each patch acts as a light structure-strength cue.
        patch_contrast = F.avg_pool2d((x - patch_mean_up).abs(), kernel_size=k, stride=k)
        patch_contrast_up = self._broadcast(patch_contrast)

        return patch_mean_up, patch_contrast_up


class PatchStatisticsGate(nn.Module):
    def __init__(self, channels: int, patch_size: int = 8):
        super().__init__()
        self.patch_stats = PatchStatisticsExtractor(patch_size=patch_size)
        self.patch_size = patch_size
        self.structure_fuse = nn.Conv2d(
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=channels,
            bias=True,
        )
        nn.init.zeros_(self.structure_fuse.weight)
        nn.init.zeros_(self.structure_fuse.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        if x1.shape[-2] % self.patch_size == 0 and x1.shape[-1] % self.patch_size == 0:
            patch_mean, patch_contrast = self.patch_stats(x1)
            patch_structure = self.structure_fuse(torch.cat([patch_mean, patch_contrast], dim=1))
            x1 = x1 + patch_structure
        return x1 * x2
