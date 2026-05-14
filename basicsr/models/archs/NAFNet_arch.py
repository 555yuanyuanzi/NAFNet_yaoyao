# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.GDPM import GlobalDirectionalPriorModulation
from basicsr.models.IPP import InterpolatedPatchPrior
from basicsr.models.PA import PatchAveraging
from basicsr.models.dfpb import AdaptiveLowPassExtractor, DualFrequencyProgressiveBlock, FrequencyAwareBlock
from basicsr.models.fftdfpb import FFTDualFrequencyProgressiveBlock
from basicsr.models.wavedfpb import WaveletDualFrequencyProgressiveBlock, WaveletFrequencyAwareBlock
from basicsr.models.polar_moe import PolarMoEDecoderBlock
from basicsr.models.RMSA import RotaryMotionAwareSkipAlignment

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class PatchAwareGate(nn.Module):
    def __init__(self, channels, patch_size=8):
        super().__init__()
        self.pa = PatchAveraging(patch_size=patch_size)
        self.pa_scale = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = x1 + self.pa_scale * self.pa(x1)
        return x1 * x2


class IPPGate(nn.Module):
    def __init__(self, channels, patch_size=8):
        super().__init__()
        self.ipp = InterpolatedPatchPrior(patch_size=patch_size)
        self.patch_size = patch_size
        self.ipp_scale = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = x1 + self.ipp_scale * self.ipp(x1)
        return x1 * x2


class NAFBlockFrequencyGate(nn.Module):
    def __init__(self, channels, low_kernel_size=5, hidden_channels=16):
        super().__init__()
        self.low_pass = AdaptiveLowPassExtractor(channels, kernel_size=low_kernel_size)
        self.gate = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        x_low = self.low_pass(x)
        x_high = x - x_low
        energy = x_high.square().mean(dim=1, keepdim=True)
        gate = self.gate(energy)
        return x + self.scale * gate * x


class ManifoldCurvatureEnhance(nn.Module):
    """
    Curvature-guided decoder enhancement.

    The module estimates local curvature responses with finite differences and
    uses them to mix a smooth branch and a detail branch. A zero-initialized
    residual scale keeps the decoder unchanged at initialization.
    """

    def __init__(
        self,
        channels,
        kernel_size=5,
        scale_limit=0.5,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd.')
        if scale_limit <= 0:
            raise ValueError('scale_limit must be positive.')

        self.scale_limit = scale_limit
        self.smooth_branch = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )
        self.enhance_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True),
        )
        self.curvature_map = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=True)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._last_aux = {}

    def _calc_curvature_weight(self, x):
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')

        grad_xx = grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1]
        grad_yy = grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :]
        grad_xx = F.pad(grad_xx, (0, 1, 0, 0), mode='replicate')
        grad_yy = F.pad(grad_yy, (0, 0, 0, 1), mode='replicate')

        curvature_feat = torch.cat(
            [
                grad_xx.abs().mean(dim=1, keepdim=True),
                grad_yy.abs().mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        return torch.sigmoid(self.curvature_map(curvature_feat))

    def forward(self, x, return_aux=False):
        weight = self._calc_curvature_weight(x)
        smooth = self.smooth_branch(x)
        enhance = self.enhance_branch(x)
        residual = weight * enhance + (1.0 - weight) * smooth
        alpha = self.scale_limit * torch.tanh(self.scale)
        out = x + alpha * residual

        self._last_aux = {
            'weight_mean': weight.mean().detach(),
            'weight_std': weight.std().detach(),
            'alpha_abs_mean': alpha.abs().mean().detach(),
        }
        if return_aux:
            return out, {'weight': weight, 'stats': self._last_aux}
        return out

    def get_last_aux(self):
        return self._last_aux


class NAFBlock(nn.Module):
    def __init__(
        self,
        c,
        DW_Expand=2,
        FFN_Expand=2,
        drop_out_rate=0.,
        use_pa=False,
        pa_patch_size=8,
        use_ipp=False,
        ipp_patch_size=8,
        use_ipp_in_ffn=False,
        use_freq_gate=False,
        freq_gate_low_kernel_size=5,
        freq_gate_hidden_channels=16,
    ):
        super().__init__()

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        def build_gate(channels, enable_ipp):
            if enable_ipp:
                return IPPGate(channels, patch_size=ipp_patch_size)
            if use_pa:
                return PatchAwareGate(channels, patch_size=pa_patch_size)
            return SimpleGate()

        ffn_channel = FFN_Expand * c
        self.sg = build_gate(dw_channel // 2, use_ipp)
        self.ffn_sg = build_gate(ffn_channel // 2, use_ipp and use_ipp_in_ffn)
        self.freq_gate = (
            NAFBlockFrequencyGate(
                dw_channel // 2,
                low_kernel_size=freq_gate_low_kernel_size,
                hidden_channels=freq_gate_hidden_channels,
            )
            if use_freq_gate
            else nn.Identity()
        )
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.freq_gate(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.ffn_sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        use_gdpm=False,
        gdpm_kwargs=None,
        use_pa=False,
        pa_patch_size=8,
        pa_stages=None,
        use_nafblock_freq_gate=False,
        nafblock_freq_gate_stages=None,
        nafblock_freq_gate_kwargs=None,
        use_ipp=False,
        ipp_patch_size=8,
        ipp_stages=None,
        ipp_in_ffn=False,
        use_dfpb=False,
        dfpb_kwargs=None,
        dfpb_stages=None,
        use_fftdfpb=False,
        fftdfpb_kwargs=None,
        fftdfpb_stages=None,
        use_wavedfpb=False,
        wavedfpb_kwargs=None,
        wavedfpb_stages=None,
        use_layerwise_dfpb=False,
        layerwise_dfpb_kwargs=None,
        layerwise_dfpb_stages=None,
        use_layerwise_wavedfpb=False,
        layerwise_wavedfpb_kwargs=None,
        layerwise_wavedfpb_stages=None,
        use_polar_moe=False,
        polar_moe_kwargs=None,
        polar_moe_stages=None,
        use_decoder_mce=False,
        decoder_mce_kwargs=None,
        use_rmsa=False,
        rmsa_kwargs=None,
        rmsa_stages=None,
        rmsa_use_raw_skip_ref=False,
    ):
        super().__init__()
        if use_pa and use_ipp:
            raise ValueError('use_pa and use_ipp cannot be enabled at the same time.')

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.gdpm = None
        if use_gdpm:
            gdpm_kwargs = {} if gdpm_kwargs is None else gdpm_kwargs
            self.gdpm = GlobalDirectionalPriorModulation(
                feat_channels=width,
                in_channels=img_channel,
                **gdpm_kwargs,
            )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoder_mces = nn.ModuleList()
        pa_stages = set(pa_stages or [])
        nafblock_freq_gate_stages = set(nafblock_freq_gate_stages or [])
        ipp_stages = set(ipp_stages or [])
        dfpb_stages = set(dfpb_stages or [])
        fftdfpb_stages = set(fftdfpb_stages or [])
        wavedfpb_stages = set(wavedfpb_stages or [])
        layerwise_dfpb_stages = set(layerwise_dfpb_stages or [])
        layerwise_wavedfpb_stages = set(layerwise_wavedfpb_stages or [])
        polar_moe_stages = set(polar_moe_stages or [])
        rmsa_stages = set(rmsa_stages or [])
        dfpb_kwargs = {} if dfpb_kwargs is None else dict(dfpb_kwargs)
        fftdfpb_kwargs = {} if fftdfpb_kwargs is None else dict(fftdfpb_kwargs)
        wavedfpb_kwargs = {} if wavedfpb_kwargs is None else dict(wavedfpb_kwargs)
        layerwise_dfpb_kwargs = {} if layerwise_dfpb_kwargs is None else dict(layerwise_dfpb_kwargs)
        layerwise_wavedfpb_kwargs = {} if layerwise_wavedfpb_kwargs is None else dict(layerwise_wavedfpb_kwargs)
        polar_moe_kwargs = {} if polar_moe_kwargs is None else dict(polar_moe_kwargs)
        decoder_mce_kwargs = {} if decoder_mce_kwargs is None else dict(decoder_mce_kwargs)
        dfpb_stage_kwargs = dfpb_kwargs.pop('stage_kwargs', {})
        fftdfpb_stage_kwargs = fftdfpb_kwargs.pop('stage_kwargs', {})
        wavedfpb_stage_kwargs = wavedfpb_kwargs.pop('stage_kwargs', {})
        layerwise_dfpb_stage_kwargs = layerwise_dfpb_kwargs.pop('stage_kwargs', {})
        layerwise_wavedfpb_stage_kwargs = layerwise_wavedfpb_kwargs.pop('stage_kwargs', {})
        polar_moe_stage_kwargs = polar_moe_kwargs.pop('stage_kwargs', {})
        layerwise_dfpb_tier_map = layerwise_dfpb_kwargs.pop('tier_map', {})
        layerwise_wavedfpb_tier_map = layerwise_wavedfpb_kwargs.pop('tier_map', {})
        nafblock_freq_gate_kwargs = {} if nafblock_freq_gate_kwargs is None else dict(nafblock_freq_gate_kwargs)
        nafblock_freq_gate_stage_kwargs = nafblock_freq_gate_kwargs.pop('stage_kwargs', {})
        rmsa_kwargs = {} if rmsa_kwargs is None else dict(rmsa_kwargs)
        if use_dfpb and use_fftdfpb and (dfpb_stages & fftdfpb_stages):
            raise ValueError('dfpb_stages and fftdfpb_stages cannot overlap when both blocks are enabled.')
        if use_dfpb and use_wavedfpb and (dfpb_stages & wavedfpb_stages):
            raise ValueError('dfpb_stages and wavedfpb_stages cannot overlap when both blocks are enabled.')
        if use_fftdfpb and use_wavedfpb and (fftdfpb_stages & wavedfpb_stages):
            raise ValueError('fftdfpb_stages and wavedfpb_stages cannot overlap when both blocks are enabled.')
        if use_layerwise_dfpb and use_dfpb and (layerwise_dfpb_stages & dfpb_stages):
            raise ValueError('layerwise_dfpb_stages and dfpb_stages cannot overlap when both blocks are enabled.')
        if use_layerwise_dfpb and use_fftdfpb and (layerwise_dfpb_stages & fftdfpb_stages):
            raise ValueError('layerwise_dfpb_stages and fftdfpb_stages cannot overlap when both blocks are enabled.')
        if use_layerwise_dfpb and use_wavedfpb and (layerwise_dfpb_stages & wavedfpb_stages):
            raise ValueError('layerwise_dfpb_stages and wavedfpb_stages cannot overlap when both blocks are enabled.')
        if use_layerwise_wavedfpb and use_dfpb and (layerwise_wavedfpb_stages & dfpb_stages):
            raise ValueError('layerwise_wavedfpb_stages and dfpb_stages cannot overlap when both blocks are enabled.')
        if use_layerwise_wavedfpb and use_fftdfpb and (layerwise_wavedfpb_stages & fftdfpb_stages):
            raise ValueError('layerwise_wavedfpb_stages and fftdfpb_stages cannot overlap when both blocks are enabled.')
        if use_layerwise_wavedfpb and use_wavedfpb and (layerwise_wavedfpb_stages & wavedfpb_stages):
            raise ValueError('layerwise_wavedfpb_stages and wavedfpb_stages cannot overlap when both blocks are enabled.')
        if use_layerwise_wavedfpb and use_layerwise_dfpb and (layerwise_wavedfpb_stages & layerwise_dfpb_stages):
            raise ValueError('layerwise_wavedfpb_stages and layerwise_dfpb_stages cannot overlap when both blocks are enabled.')
        self.dfpb_modules = nn.ModuleDict()
        self.fftdfpb_modules = nn.ModuleDict()
        self.wavedfpb_modules = nn.ModuleDict()
        self.layerwise_dfpb_modules = nn.ModuleDict()
        self.layerwise_wavedfpb_modules = nn.ModuleDict()
        self.polar_moe_modules = nn.ModuleDict()
        self.rmsa_modules = nn.ModuleDict()
        self.rmsa_use_raw_skip_ref = rmsa_use_raw_skip_ref

        def register_dfpb(channels, stage_name):
            if use_dfpb and stage_name in dfpb_stages:
                kwargs = dict(dfpb_kwargs)
                kwargs.update(dfpb_stage_kwargs.get(stage_name, {}))
                self.dfpb_modules[stage_name] = DualFrequencyProgressiveBlock(
                    channels=channels,
                    **kwargs,
                )

        def register_fftdfpb(channels, stage_name):
            if use_fftdfpb and stage_name in fftdfpb_stages:
                kwargs = dict(fftdfpb_kwargs)
                kwargs.update(fftdfpb_stage_kwargs.get(stage_name, {}))
                self.fftdfpb_modules[stage_name] = FFTDualFrequencyProgressiveBlock(
                    channels=channels,
                    **kwargs,
                )

        def register_wavedfpb(channels, stage_name):
            if use_wavedfpb and stage_name in wavedfpb_stages:
                kwargs = dict(wavedfpb_kwargs)
                kwargs.update(wavedfpb_stage_kwargs.get(stage_name, {}))
                self.wavedfpb_modules[stage_name] = WaveletDualFrequencyProgressiveBlock(
                    channels=channels,
                    **kwargs,
                )

        def register_layerwise_dfpb(channels, stage_name):
            if use_layerwise_dfpb and stage_name in layerwise_dfpb_stages:
                kwargs = dict(layerwise_dfpb_kwargs)
                kwargs.update(layerwise_dfpb_stage_kwargs.get(stage_name, {}))
                tier = kwargs.pop('tier', layerwise_dfpb_tier_map.get(stage_name, None))
                if tier is None:
                    raise ValueError(f'layerwise_dfpb tier is required for stage {stage_name}.')
                self.layerwise_dfpb_modules[stage_name] = FrequencyAwareBlock(
                    channels=channels,
                    tier=int(tier),
                    **kwargs,
                )

        def register_layerwise_wavedfpb(channels, stage_name):
            if use_layerwise_wavedfpb and stage_name in layerwise_wavedfpb_stages:
                kwargs = dict(layerwise_wavedfpb_kwargs)
                kwargs.update(layerwise_wavedfpb_stage_kwargs.get(stage_name, {}))
                tier = kwargs.pop('tier', layerwise_wavedfpb_tier_map.get(stage_name, None))
                if tier is None:
                    raise ValueError(f'layerwise_wavedfpb tier is required for stage {stage_name}.')
                self.layerwise_wavedfpb_modules[stage_name] = WaveletFrequencyAwareBlock(
                    channels=channels,
                    tier=int(tier),
                    **kwargs,
                )

        def register_polar_moe(channels, stage_name):
            if use_polar_moe and stage_name in polar_moe_stages:
                kwargs = dict(polar_moe_kwargs)
                kwargs.update(polar_moe_stage_kwargs.get(stage_name, {}))
                self.polar_moe_modules[stage_name] = PolarMoEDecoderBlock(
                    channels=channels,
                    **kwargs,
                )

        def register_rmsa(channels, stage_name):
            if use_rmsa and stage_name in rmsa_stages:
                self.rmsa_modules[stage_name] = RotaryMotionAwareSkipAlignment(
                    channels=channels,
                    **rmsa_kwargs,
                )

        def build_block(channels, stage_name):
            pa_active = use_pa and stage_name in pa_stages
            ipp_active = use_ipp and stage_name in ipp_stages
            freq_gate_active = use_nafblock_freq_gate and stage_name in nafblock_freq_gate_stages
            freq_gate_kwargs = dict(nafblock_freq_gate_kwargs)
            freq_gate_kwargs.update(nafblock_freq_gate_stage_kwargs.get(stage_name, {}))
            return NAFBlock(
                channels,
                use_pa=pa_active,
                pa_patch_size=pa_patch_size,
                use_ipp=ipp_active,
                ipp_patch_size=ipp_patch_size,
                use_ipp_in_ffn=ipp_in_ffn and ipp_active,
                use_freq_gate=freq_gate_active,
                **freq_gate_kwargs,
            )

        chan = width
        for stage_idx, num in enumerate(enc_blk_nums, start=1):
            stage_name = f'enc{stage_idx}'
            self.encoders.append(
                nn.Sequential(
                    *[build_block(chan, stage_name) for _ in range(num)]
                )
            )
            register_dfpb(chan, stage_name)
            register_fftdfpb(chan, stage_name)
            register_wavedfpb(chan, stage_name)
            register_layerwise_dfpb(chan, stage_name)
            register_layerwise_wavedfpb(chan, stage_name)
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        register_dfpb(chan, 'middle')
        register_fftdfpb(chan, 'middle')
        register_wavedfpb(chan, 'middle')
        register_layerwise_dfpb(chan, 'middle')
        register_layerwise_wavedfpb(chan, 'middle')
        self.middle_blks = \
            nn.Sequential(
                *[build_block(chan, 'middle') for _ in range(middle_blk_num)]
            )

        for stage_idx, num in enumerate(dec_blk_nums, start=1):
            stage_name = f'dec{stage_idx}'
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[build_block(chan, stage_name) for _ in range(num)]
                )
            )
            self.decoder_mces.append(
                ManifoldCurvatureEnhance(chan, **decoder_mce_kwargs)
                if use_decoder_mce
                else nn.Identity()
            )
            register_rmsa(chan, stage_name)
            register_dfpb(chan, stage_name)
            register_fftdfpb(chan, stage_name)
            register_wavedfpb(chan, stage_name)
            register_layerwise_dfpb(chan, stage_name)
            register_layerwise_wavedfpb(chan, stage_name)
            register_polar_moe(chan, stage_name)

        self.padder_size = 2 ** len(self.encoders)

    def _apply_dfpb(self, stage_name, x):
        if stage_name not in self.dfpb_modules:
            return x
        return self.dfpb_modules[stage_name](x)

    def _apply_fftdfpb(self, stage_name, x):
        if stage_name not in self.fftdfpb_modules:
            return x
        return self.fftdfpb_modules[stage_name](x)

    def _apply_wavedfpb(self, stage_name, x):
        if stage_name not in self.wavedfpb_modules:
            return x
        return self.wavedfpb_modules[stage_name](x)

    def _apply_layerwise_dfpb(self, stage_name, x):
        if stage_name not in self.layerwise_dfpb_modules:
            return x
        return self.layerwise_dfpb_modules[stage_name](x)

    def _apply_layerwise_wavedfpb(self, stage_name, x):
        if stage_name not in self.layerwise_wavedfpb_modules:
            return x
        return self.layerwise_wavedfpb_modules[stage_name](x)

    def _apply_polar_moe(self, stage_name, x):
        if stage_name not in self.polar_moe_modules:
            return x
        return self.polar_moe_modules[stage_name](x)

    def _fuse_skip(self, stage_name, x_dec, x_skip, x_skip_ref=None):
        if stage_name in self.rmsa_modules:
            return self.rmsa_modules[stage_name](x_dec, x_skip, x_skip_ref=x_skip_ref)
        return x_dec + x_skip

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        if self.gdpm is not None:
            x = self.gdpm(inp, x)

        encs = []

        for stage_idx, (encoder, down) in enumerate(zip(self.encoders, self.downs), start=1):
            x = encoder(x)
            x_raw_skip = x
            x = self._apply_dfpb(f'enc{stage_idx}', x)
            x = self._apply_fftdfpb(f'enc{stage_idx}', x)
            x = self._apply_wavedfpb(f'enc{stage_idx}', x)
            x = self._apply_layerwise_dfpb(f'enc{stage_idx}', x)
            x = self._apply_layerwise_wavedfpb(f'enc{stage_idx}', x)
            if self.rmsa_use_raw_skip_ref:
                encs.append((x, x_raw_skip))
            else:
                encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        x = self._apply_dfpb('middle', x)
        x = self._apply_fftdfpb('middle', x)
        x = self._apply_wavedfpb('middle', x)
        x = self._apply_layerwise_dfpb('middle', x)
        x = self._apply_layerwise_wavedfpb('middle', x)

        for stage_idx, (decoder, up, enc_skip, decoder_mce) in enumerate(
            zip(self.decoders, self.ups, encs[::-1], self.decoder_mces),
            start=1,
        ):
            stage_name = f'dec{stage_idx}'
            x = up(x)
            if isinstance(enc_skip, tuple):
                enc_skip, enc_skip_ref = enc_skip
            else:
                enc_skip_ref = None
            x = self._fuse_skip(stage_name, x, enc_skip, x_skip_ref=enc_skip_ref)
            x = decoder(x)
            x = self._apply_polar_moe(stage_name, x)
            x = decoder_mce(x)
            x = self._apply_dfpb(stage_name, x)
            x = self._apply_fftdfpb(stage_name, x)
            x = self._apply_wavedfpb(stage_name, x)
            x = self._apply_layerwise_dfpb(stage_name, x)
            x = self._apply_layerwise_wavedfpb(stage_name, x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
