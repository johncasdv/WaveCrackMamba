import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class WCMDecoderHead(BaseDecodeHead):
    """Simplified U-shaped decoder with semantic wavelet gates (Spec §8.7/§8.6).

    Levels (M0-6 full form):

    - D3: ``1/16 -> 1/8`` + trunk ``s2`` lateral
    - D2: ``-> 1/4``  + trunk ``s1`` lateral + HF2 semantic gate
    - D1: ``-> 1/2``  + detail ``skip2``    + HF1 semantic gate
    - D0: ``-> 1/1``  + detail ``skip1`` (full-res skip)
    - mask head: ``out_channels=1`` logits (no sigmoid inside)

    Semantic gate (Spec §8.6): ``gate = σ(Conv(cat(ctx, proj)))``,
    ``out = ctx + gate ⊙ Val(proj)``. ``Val`` is a bias-free 1x1 mapping the
    wavelet projection to ``channels``; the wavelet projections are zero at init
    (near-zero bypass), and bias-free ``Val`` keeps them exactly zero, so the
    decoder forward is unchanged at init regardless of gate values.

    Switches are EXPLICIT config, not silent adaptation (M0-3 decision): arm
    configs must set ``detail_skips=False`` for E0, ``full_res_skip=False`` for
    E1/E2, ``wavelet_gate=False`` for C0/C1. A mismatch between switches and
    the features actually present raises instead of degrading.
    """

    def __init__(self, trunk_channels=(96, 192, 384), detail_channels=(24, 48),
                 wavelet_proj_channels=(48, 96), channels=128,
                 detail_skips=True, full_res_skip=True, wavelet_gate=True,
                 norm_cfg=dict(type='GN', num_groups=16),
                 act_cfg=dict(type='SiLU'), **kwargs):
        super().__init__(
            in_channels=trunk_channels[-1], channels=channels,
            norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)
        self.detail_skips = detail_skips
        self.full_res_skip = full_res_skip
        self.wavelet_gate = wavelet_gate
        c1, c2, c3 = trunk_channels
        cm = dict(norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.lateral1 = ConvModule(c1, channels, 1, **cm)
        self.lateral2 = ConvModule(c2, channels, 1, **cm)
        self.lateral3 = ConvModule(c3, channels, 1, **cm)
        self.fuse3 = ConvModule(channels, channels, 3, padding=1, **cm)  # 1/8
        self.fuse2 = ConvModule(channels, channels, 3, padding=1, **cm)  # 1/4
        self.fuse1 = ConvModule(channels, channels, 3, padding=1, **cm)  # 1/2
        self.fuse0 = ConvModule(channels, channels, 3, padding=1, **cm)  # 1/1
        if detail_skips:
            self.lat_d2 = ConvModule(detail_channels[1], channels, 1, **cm)
            if full_res_skip:
                self.lat_d1 = ConvModule(detail_channels[0], channels, 1, **cm)
        if wavelet_gate:
            w1, w2 = wavelet_proj_channels
            self.gate2 = nn.Conv2d(channels + w2, channels, 3, padding=1)
            self.val2 = nn.Conv2d(w2, channels, 1, bias=False)
            self.gate1 = nn.Conv2d(channels + w1, channels, 3, padding=1)
            self.val1 = nn.Conv2d(w1, channels, 1, bias=False)

    def _up_to(self, x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode='bilinear',
                             align_corners=self.align_corners)

    def _up2(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear',
                             align_corners=self.align_corners)

    def _gated(self, ctx, proj, gate_conv, val_conv):
        g = torch.sigmoid(gate_conv(torch.cat([ctx, proj], dim=1)))
        return ctx + g * val_conv(proj)

    def forward(self, feats):
        s1, s2, s3 = feats['s1'], feats['s2'], feats['s3']
        detail = feats.get('detail')
        wavelet = feats.get('wavelet')
        if self.detail_skips:
            assert detail is not None, (
                'detail_skips=True but no detail features -- the E0 arm must '
                'set decode_head.detail_skips=False')
        if self.wavelet_gate:
            assert wavelet is not None, (
                'wavelet_gate=True but no wavelet features -- C0/C1 arms must '
                'set decode_head.wavelet_gate=False')

        x = self.lateral3(s3)                                        # 1/16
        x = self.fuse3(self._up_to(x, s2) + self.lateral2(s2))       # 1/8
        x = self._up_to(x, s1) + self.lateral1(s1)
        if self.wavelet_gate:
            x = self._gated(x, wavelet['proj2'], self.gate2, self.val2)
        x = self.fuse2(x)                                            # 1/4
        x = self._up2(x)
        if self.detail_skips:
            x = x + self.lat_d2(detail['skip2'])
        if self.wavelet_gate:
            x = self._gated(x, wavelet['proj1'], self.gate1, self.val1)
        x = self.fuse1(x)                                            # 1/2
        x = self._up2(x)
        if self.detail_skips and self.full_res_skip:
            assert detail['skip1'] is not None, (
                'full_res_skip=True but the detail form has no 1/1 skip -- '
                'E1/E2 arms must set decode_head.full_res_skip=False')
            x = x + self.lat_d1(detail['skip1'])
        x = self.fuse0(x)                                            # 1/1
        return self.cls_seg(x)
