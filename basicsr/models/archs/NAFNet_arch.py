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
from basicsr.models.dfpb import DualFrequencyProgressiveBlock

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class PatchAwareGate(nn.Module):
    def __init__(self, channels, patch_size=8):
        super().__init__()
        self.pa = PatchAveraging(patch_size=patch_size)
        self.patch_size = patch_size
        self.pa_scale = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        if x1.shape[-2] % self.patch_size == 0 and x1.shape[-1] % self.patch_size == 0:
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
        use_ipp=False,
        ipp_patch_size=8,
        ipp_stages=None,
        ipp_in_ffn=False,
        use_dfpb=False,
        dfpb_kwargs=None,
        dfpb_stages=None,
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
        pa_stages = set(pa_stages or [])
        ipp_stages = set(ipp_stages or [])
        dfpb_stages = set(dfpb_stages or [])
        dfpb_kwargs = {} if dfpb_kwargs is None else dict(dfpb_kwargs)
        self.dfpb_modules = nn.ModuleDict()

        def register_dfpb(channels, stage_name):
            if use_dfpb and stage_name in dfpb_stages:
                self.dfpb_modules[stage_name] = DualFrequencyProgressiveBlock(
                    channels=channels,
                    **dfpb_kwargs,
                )

        def build_block(channels, stage_name):
            pa_active = use_pa and stage_name in pa_stages
            ipp_active = use_ipp and stage_name in ipp_stages
            return NAFBlock(
                channels,
                use_pa=pa_active,
                pa_patch_size=pa_patch_size,
                use_ipp=ipp_active,
                ipp_patch_size=ipp_patch_size,
                use_ipp_in_ffn=ipp_in_ffn and ipp_active,
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
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        register_dfpb(chan, 'middle')
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
            register_dfpb(chan, stage_name)

        self.padder_size = 2 ** len(self.encoders)

    def _apply_dfpb(self, stage_name, x):
        if stage_name not in self.dfpb_modules:
            return x
        return self.dfpb_modules[stage_name](x)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        if self.gdpm is not None:
            x = self.gdpm(inp, x)

        encs = []

        for stage_idx, (encoder, down) in enumerate(zip(self.encoders, self.downs), start=1):
            x = encoder(x)
            x = self._apply_dfpb(f'enc{stage_idx}', x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        x = self._apply_dfpb('middle', x)

        for stage_idx, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1]), start=1):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            x = self._apply_dfpb(f'dec{stage_idx}', x)

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
