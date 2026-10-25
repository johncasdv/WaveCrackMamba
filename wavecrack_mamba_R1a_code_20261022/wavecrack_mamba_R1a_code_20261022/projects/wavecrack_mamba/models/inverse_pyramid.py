"""G1 R-arms: inverse-pyramid fusion head (R1a) and its DySample variant (R1b).

Spec grounding:

- §982/§983: R1a = "low-resolution entry + inverse-pyramid fusion head with
  FIXED upsampling", the architecture-level low-cost high-resolution recovery
  reference; R1b = "R1a + DySample", isolating the dynamic-upsampling increment.
- §447: the inverse-pyramid arm MUST split "multi-scale projection / fusion
  topology" and "DySample dynamic upsampling" into two variants -- hence one
  head class with an ``upsample=`` switch instead of two divergent classes.
- §207/§110: the head is the architecture reference closest to SCSegamba's
  form. Its MFS head does: per-level projection -> upsample every level to the
  finest grid (DySample, single step per level) -> concat -> fuse -> predict
  (projects/SCSegamba/models/MFS.py). We mirror that topology on the WCM trunk
  pyramid (s1@1/4, s2@1/8, s3@1/16) with WCM's GN/SiLU conventions, then one
  more x4 recovery to 1/1 so the mask head stays protocol-identical to the
  E-arms (full-res logits into the same WCMRegionLoss).

"Low-resolution entry" means the R arms are trunk-only: resolution is recovered
at the DECODE end from the pyramid alone -- the architectural alternative to
E1-E3's high-resolution entry branches. detail/wavelet features must therefore
be absent; per the M0-3 explicit-switch rule a mismatch raises instead of
silently ignoring the extra features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


def _normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    """Dynamic upsampling by content-aware point resampling.

    Copyright (c) 2023 Wenze Liu. Licensed under the MIT License; see the
    repository's THIRD_PARTY_NOTICES.md for the full license text.

    Vendored from https://github.com/tiny-smart/dysample (ICCV'23), the same
    copy SCSegamba ships (projects/SCSegamba/models/DySample.py) -- imported
    here rather than from that tree because its files use ``from models.X``
    absolute imports that only resolve from SCSegamba's own root. 'lp' style,
    ``groups=4``, no dyscope: SCSegamba's defaults, kept verbatim so R1b uses
    the operator exactly as the reference architecture does. Only change:
    ``indexing='ij'`` made explicit on meshgrid (today's default; silences the
    future-default deprecation warning without touching semantics).

    Init property the R1b tests rely on: offsets start at ``init_pos`` plus
    N(0, 0.001^2) noise, i.e. the exact bilinear sub-pixel grid -- so at init
    DySample ~= fixed bilinear and R1b starts from R1a behaviour, keeping the
    arm pair a clean single-variable contrast.
    """

    def __init__(self, in_channels, scale=2, style='lp', groups=4,
                 dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        _normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            nn.init.constant_(self.scope.weight, 0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2,
                         (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing='ij')
                           ).transpose(1, 2).repeat(1, self.groups, 1).reshape(
                               1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.reshape(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(
            torch.meshgrid([coords_w, coords_h], indexing='ij')
        ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype,
                                  device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W),
                                 self.scale).reshape(
            B, 2, -1, self.scale * H, self.scale * W).permute(
            0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
            align_corners=False, padding_mode='border').reshape(
                B, -1, self.scale * H, self.scale * W)

    def forward(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 \
                + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)


@MODELS.register_module()
class WCMInversePyramidHead(BaseDecodeHead):
    """Inverse-pyramid fusion head for the G1 R arms (module docstring above).

    Topology (identical for both variants): 1x1 project s1/s2/s3 to a common
    width -> lift s2 (x2) and s3 (x4) to the 1/4 grid -> concat -> 1x1 fuse +
    3x3 refine at 1/4 -> one x4 lift to 1/1 -> ``cls_seg``. Cheaper than
    WCMDecoderHead by construction (no 3x3 convs at 1/2 or 1/1), matching the
    "low-cost recovery" role.

    Args:
        upsample: ``'fixed'`` (R1a) -- all three lifts are bilinear;
            ``'dysample'`` (R1b) -- the same three lifts become DySample.
            The lift SITES are the only thing the switch changes.
    """

    def __init__(self, trunk_channels=(96, 192, 384), channels=128,
                 upsample='fixed', refiner=None,
                 norm_cfg=dict(type='GN', num_groups=16),
                 act_cfg=dict(type='SiLU'), **kwargs):
        super().__init__(
            in_channels=trunk_channels[-1], channels=channels,
            norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)
        assert upsample in ('fixed', 'dysample'), upsample
        self.upsample = upsample
        c1, c2, c3 = trunk_channels
        cm = dict(norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.proj1 = ConvModule(c1, channels, 1, **cm)
        self.proj2 = ConvModule(c2, channels, 1, **cm)
        self.proj3 = ConvModule(c3, channels, 1, **cm)
        self.fuse = ConvModule(3 * channels, channels, 1, **cm)
        self.refine = ConvModule(channels, channels, 3, padding=1, **cm)
        self.refiner = None
        if refiner is not None:
            cfg = dict(refiner)
            cfg.setdefault('channels', channels)
            self.refiner = MODELS.build(cfg)
            if self.refiner.channels != channels:
                raise ValueError('refiner channels must match the fused head')
        if upsample == 'dysample':
            self.up2 = DySample(channels, scale=2)
            self.up3 = DySample(channels, scale=4)
            self.up0 = DySample(channels, scale=4)

    def _up(self, x, scale, dysample=None):
        # Inputs are padded to a multiple of 16 in the segmentor, so the s3/s2/
        # s1 grids are exact x4/x2/x1 of each other and integer-scale lifts are
        # always size-exact -- DySample only supports integer scales.
        if self.upsample == 'dysample':
            return dysample(x)
        return F.interpolate(x, scale_factor=scale, mode='bilinear',
                             align_corners=self.align_corners)

    def fuse_features(self, feats):
        """Return the reusable quarter-resolution feature before refinement."""
        assert feats.get('detail') is None and feats.get('wavelet') is None, (
            'R arms are trunk-only (low-resolution entry): detail/wavelet '
            'features must not be produced -- set detail_branch=None and '
            'wavelet_bypass=None in the arm config')
        s1, s2, s3 = feats['s1'], feats['s2'], feats['s3']
        p1 = self.proj1(s1)
        p2 = self._up(self.proj2(s2), 2, getattr(self, 'up2', None))
        p3 = self._up(self.proj3(s3), 4, getattr(self, 'up3', None))
        x = self.refine(self.fuse(torch.cat([p1, p2, p3], dim=1)))   # 1/4
        return x

    def predict_from_fused(self, features, steps=None, return_intermediates=False):
        """Decode online or cached features without duplicating old weights."""
        coarse = ()
        if self.refiner is not None:
            result = self.refiner(features, self.conv_seg, steps=steps,
                                  return_intermediates=return_intermediates)
            if return_intermediates:
                features, coarse = result
            else:
                features = result
        elif steps not in (None, 0):
            raise ValueError('steps > 0 requires an enabled refiner')
        elif return_intermediates:
            coarse = (self.conv_seg(features),)
        logits = self.cls_seg(self._up(features, 4, getattr(self, 'up0', None)))
        return (logits, coarse) if return_intermediates else logits

    def forward(self, feats):
        return self.predict_from_fused(self.fuse_features(feats))
