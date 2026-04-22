from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalDirectionalPriorModulation(nn.Module):
    """
    GDPM: Global Directional Prior Modulation

    A lightweight spectral prompt module for motion deblurring.

    Pipeline:
        input blur image
        -> downsample
        -> FFT + fftshift
        -> log magnitude spectrum
        -> radial mean normalization for direction contrast
        -> directional pooling on a mid-high frequency ring
        -> low/high band statistics
        -> compact spectral prompt vector
        -> MLP -> bounded gamma
        -> residual feature scaling

    Args:
        feat_channels: feature channels to modulate
        in_channels: input image channels
        prior_size: low-resolution size for FFT prior extraction
        num_dirs: number of direction bins, default 4
        dir_band_start: start radius of the directional pooling ring
        dir_band_end: end radius of the directional pooling ring
        low_cutoff: cutoff for the low-frequency soft mask
        high_band_start: start radius of the high-frequency ring
        high_band_end: end radius of the high-frequency ring
        band_tau: softness shared by the radial masks
        dir_sigma: softness of directional masks; if None, auto-set
        mlp_hidden: hidden dim of the small MLP
        gamma_limit: tanh-bounded amplitude for gamma modulation
        num_radial_bins: number of bins used for radial mean normalization
        use_grayscale: convert input image to grayscale before FFT
    """

    def __init__(
        self,
        feat_channels: int,
        in_channels: int = 3,
        prior_size: int = 64,
        num_dirs: int = 4,
        dir_band_start: float = 0.35,
        dir_band_end: float = 0.80,
        low_cutoff: float = 0.22,
        high_band_start: float = 0.45,
        high_band_end: float = 0.85,
        band_tau: float = 0.04,
        dir_sigma: float | None = None,
        mlp_hidden: int = 32,
        gamma_limit: float = 0.25,
        num_radial_bins: int = 32,
        use_grayscale: bool = True,
    ) -> None:
        super().__init__()
        if not (0.0 < low_cutoff < dir_band_start < dir_band_end < high_band_end <= 1.0):
            raise ValueError("Expected 0 < low_cutoff < dir_band_start < dir_band_end < high_band_end <= 1.")
        if not (0.0 < high_band_start < high_band_end <= 1.0):
            raise ValueError("Expected 0 < high_band_start < high_band_end <= 1.")
        if band_tau <= 0.0:
            raise ValueError("band_tau must be positive.")
        if gamma_limit <= 0.0:
            raise ValueError("gamma_limit must be positive.")
        if num_radial_bins < 4:
            raise ValueError("num_radial_bins must be at least 4.")

        self.feat_channels = feat_channels
        self.in_channels = in_channels
        self.prior_size = prior_size
        self.num_dirs = num_dirs
        self.dir_band_start = dir_band_start
        self.dir_band_end = dir_band_end
        self.low_cutoff = low_cutoff
        self.high_band_start = high_band_start
        self.high_band_end = high_band_end
        self.band_tau = band_tau
        self.gamma_limit = gamma_limit
        self.num_radial_bins = num_radial_bins
        self.use_grayscale = use_grayscale

        if dir_sigma is None:
            dir_sigma = math.pi / max(2 * num_dirs, 4)
        self.dir_sigma = dir_sigma

        # The prompt uses K directional contrast values plus two band statistics:
        #   [direction_prior, high_energy, high_minus_low]
        self.prior_norm = nn.LayerNorm(num_dirs + 2, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(num_dirs + 2, mlp_hidden, bias=True),
            nn.GELU(),
            nn.Linear(mlp_hidden, feat_channels, bias=True),
        )

        # Learnable residual scale for stable prompt injection.
        self.scale = nn.Parameter(torch.zeros(1))

        # Optional cache for geometry tensors.
        self._cache_key: tuple[object, ...] | None = None
        self._cached_radius: torch.Tensor | None = None
        self._cached_dir_masks: torch.Tensor | None = None
        self._cached_radial_bins: torch.Tensor | None = None

    def _build_radius_map(self, size: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        radius = radius / radius.max().clamp_min(1e-6)
        return radius.view(1, 1, size, size)

    def _build_angle_map(self, size: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        theta = torch.atan2(yy, xx)              # [-pi, pi]
        theta = torch.remainder(theta, math.pi) # orientation in [0, pi)
        return theta.view(1, 1, size, size)

    def _build_direction_masks(self, angle_map: torch.Tensor) -> torch.Tensor:
        """
        Soft directional masks with K orientation centers.
        Output: [1, K, H, W]
        """
        centers = torch.linspace(
            0.0,
            math.pi * (self.num_dirs - 1) / self.num_dirs,
            steps=self.num_dirs,
            device=angle_map.device,
            dtype=angle_map.dtype,
        ).view(1, self.num_dirs, 1, 1)

        diff = torch.abs(angle_map - centers)
        diff = torch.minimum(diff, math.pi - diff)
        masks = torch.exp(-(diff.square()) / (2.0 * (self.dir_sigma ** 2)))
        masks = masks / masks.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return masks

    def _build_ring_mask(
        self,
        radius: torch.Tensor,
        inner: float,
        outer: float,
    ) -> torch.Tensor:
        return torch.sigmoid((radius - inner) / self.band_tau) * torch.sigmoid((outer - radius) / self.band_tau)

    def _build_low_mask(self, radius: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((self.low_cutoff - radius) / self.band_tau)

    def _get_cached_geometry(
        self,
        size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (
            size,
            str(device),
            self.num_dirs,
            self.num_radial_bins,
            float(self.dir_sigma),
        )
        if self._cache_key != key:
            with torch.no_grad():
                radius = self._build_radius_map(size, device)
                angle_map = self._build_angle_map(size, device)
                dir_masks = self._build_direction_masks(angle_map)
                radial_bins = torch.clamp(
                    torch.round(radius[0, 0] * (self.num_radial_bins - 1)).long(),
                    min=0,
                    max=self.num_radial_bins - 1,
                ).view(1, 1, size, size)
            self._cache_key = key
            self._cached_radius = radius
            self._cached_dir_masks = dir_masks
            self._cached_radial_bins = radial_bins

        assert self._cached_radius is not None
        assert self._cached_dir_masks is not None
        assert self._cached_radial_bins is not None
        return self._cached_radius, self._cached_dir_masks, self._cached_radial_bins

    def clear_cache(self) -> None:
        self._cache_key = None
        self._cached_radius = None
        self._cached_dir_masks = None
        self._cached_radial_bins = None

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=(-2, -1)).clamp_min(1e-6)
        value = (x * mask).sum(dim=(-2, -1)) / denom
        return value.squeeze(1)

    def _compute_radial_mean(self, x: torch.Tensor, radial_bins: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        flat_x = x.view(b, -1)
        flat_bins = radial_bins.view(1, -1).expand(b, -1)

        radial_sum = torch.zeros(
            b,
            self.num_radial_bins,
            device=x.device,
            dtype=x.dtype,
        )
        radial_count = torch.zeros_like(radial_sum)
        radial_sum.scatter_add_(1, flat_bins, flat_x)
        radial_count.scatter_add_(1, flat_bins, torch.ones_like(flat_x))
        radial_mean = radial_sum / radial_count.clamp_min(1.0)
        gathered = radial_mean.gather(1, flat_bins).view(b, 1, h, w)
        return gathered

    def _extract_direction_prior(
        self,
        x_img: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Extract a compact spectral prompt from the input blur image.

        Args:
            x_img: [B, C, H, W]

        Returns:
            prior: [B, K + 2]
            aux: cached statistics for inspection
        """
        if self.use_grayscale and x_img.shape[1] > 1:
            x_gray = x_img.mean(dim=1, keepdim=True)
        else:
            x_gray = x_img[:, :1]

        x_small = F.interpolate(
            x_gray,
            size=(self.prior_size, self.prior_size),
            mode="bilinear",
            align_corners=False,
        )

        # zero-mean before FFT
        x_small = x_small - x_small.mean(dim=(-2, -1), keepdim=True)

        fft = torch.fft.fft2(x_small.float(), dim=(-2, -1))
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        log_mag = torch.log1p(torch.abs(fft))  # [B,1,S,S]

        radius, dir_masks, radial_bins = self._get_cached_geometry(self.prior_size, x_img.device)
        radial_mean = self._compute_radial_mean(log_mag, radial_bins)
        log_mag_norm = log_mag - radial_mean

        dir_band_mask = self._build_ring_mask(radius, self.dir_band_start, self.dir_band_end).to(dtype=log_mag.dtype)

        p_list = []
        for i in range(self.num_dirs):
            dmask = dir_masks[:, i : i + 1] * dir_band_mask
            p_list.append(self._masked_mean(log_mag_norm, dmask))

        direction_prior = torch.stack(p_list, dim=1)
        direction_prior = direction_prior - direction_prior.mean(dim=1, keepdim=True)

        low_mask = self._build_low_mask(radius).to(dtype=log_mag.dtype)
        high_mask = self._build_ring_mask(radius, self.high_band_start, self.high_band_end).to(dtype=log_mag.dtype)
        low_energy = self._masked_mean(log_mag, low_mask)
        high_energy = self._masked_mean(log_mag, high_mask)
        high_minus_low = high_energy - low_energy

        band_prior = torch.stack([high_energy, high_minus_low], dim=1)
        prior = torch.cat([direction_prior, band_prior], dim=1)

        aux = {
            "direction_prior": direction_prior.detach(),
            "high_energy": high_energy.detach(),
            "low_energy": low_energy.detach(),
            "high_minus_low": high_minus_low.detach(),
            "log_mag": log_mag.detach(),
            "log_mag_norm": log_mag_norm.detach(),
        }
        return prior, aux

    def forward(
        self,
        x_img: torch.Tensor,
        feat: torch.Tensor,
        return_prior: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x_img: input blur image, [B, Cin, H, W]
            feat: feature to modulate, [B, C, Hf, Wf]

        Returns:
            modulated feature, or (feature, aux_stats)
        """
        assert feat.shape[1] == self.feat_channels, (
            f"Expected feat channels {self.feat_channels}, got {feat.shape[1]}"
        )

        prior, aux = self._extract_direction_prior(x_img)
        prior_norm = self.prior_norm(prior)
        gamma = self.mlp(prior_norm).view(feat.shape[0], self.feat_channels, 1, 1)
        gamma = self.gamma_limit * torch.tanh(gamma)
        out = feat * (1.0 + self.scale * gamma)

        if return_prior:
            aux["spectral_prior_raw"] = prior.detach()
            aux["spectral_prior_norm"] = prior_norm.detach()
            aux["gamma_mean"] = gamma.mean().detach()
            aux["scale"] = self.scale.detach()
            return out, aux

        return out
