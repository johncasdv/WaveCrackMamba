import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS

from .wavelet import _HAAR


def _cbr(i, o, k, s, norm_cfg, act_cfg):
    return ConvModule(i, o, k, stride=s, padding=k // 2,
                      norm_cfg=norm_cfg, act_cfg=act_cfg)


class _HWD(BaseModule):
    """Haar Wavelet Downsampling -- a drop-in for a stride-2 conv (survey A).

    HWD (Pattern Recognition 2023, ~400 cites) replaces a stride-2 operator with
    ``Haar DWT -> concat(LL,LH,HL,HH) -> 1x1 conv``. The point is WHERE the
    discarding happens, not extra capacity: a 3x3 stride-2 conv drops three of
    every four samples before anything has looked at them, whereas the DWT is a
    lossless orthonormal rearrangement of H*W*C into (H/2)*(W/2)*4C and the 1x1
    then chooses what to keep AFTER the subbands have been separated.

    This is exactly the distinction that the "a 4x4 stem can already express a
    2x2 Haar" objection does NOT cover. That objection is about expressiveness
    and correctly sinks the input-side bypass (survey class E); information
    preservation is a different property, and it is not in a stride-2 conv's
    hypothesis space at all.

    It is placed ONLY on the detail branch's own downsamplers, which are
    randomly initialised -- so unlike HWD's original use (swapping a backbone's
    stride-2 stages) it costs nothing in pretrained-weight alignment, and
    pretraining is the largest measured main effect on this task.

    Note it is also SMALLER than what it replaces: 4*Ci*Co parameters against
    9*Ci*Co for the 3x3 stride-2 conv, so a win here cannot be read as capacity.
    """

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        # Orthonormal (Parseval) -- reuse the bypass's kernels so there is one
        # Haar convention in the codebase, pinned by its synthetic tests.
        self.register_buffer('haar', _HAAR.clone())        # [4, 1, 2, 2]
        self.proj = ConvModule(4 * in_channels, out_channels, 1,
                               norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        c = x.shape[1]
        # Match the output size of the 3x3/stride-2/pad-1 conv this replaces:
        # that conv gives ceil(H/2), a bare 2x2/stride-2 gives floor(H/2). A
        # no-op at the 512x512 training size, but sliding-window inference does
        # hand in odd crops, and a silent one-pixel drift would misalign the
        # decoder skips rather than raise.
        ph, pw = x.shape[-2] % 2, x.shape[-1] % 2
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        # groups=c => channel c_i produces [LL, LH, HL, HH] contiguously; the
        # 1x1 is a full map over all 4C, so the interleaving does not matter.
        k = self.haar.to(x.dtype).repeat(c, 1, 1, 1)       # [4C, 1, 2, 2]
        return self.proj(F.conv2d(x, k, stride=2, groups=c))


@MODELS.register_module()
class WCMDetailBranch(BaseModule):
    """High-resolution detail branch (Spec §8.3).

    Preserves the ~2 px geometry that the stride-4 trunk stem discards and
    injects it, near-zero, into the trunk's 1/4 features. Returns a dict:

    - ``skip1``  ``[B, base, H, W]``       (1/1 decoder skip; ``None`` for the
      downsampling forms)
    - ``skip2``  ``[B, 2*base, H/2, W/2]`` (1/2 decoder skip)
    - ``inject`` ``[B, inject_channels, H/4, W/4]`` (added to the trunk stage-0
      output; ``0`` at init so the trunk forward is unchanged -- 硬约束 5)

    Forms (the H1/G1 detail-entry comparison, exec-plan §8.3):

    - ``'stride1'`` (E3): stride-1 stem keeps full resolution -> ``skip1 @ 1/1``.
    - ``'stride2_aa'`` (E1): anti-aliased (blur + stride-2) stem, no full-res skip.
    - ``'pixel_unshuffle'`` (E2): space-to-depth stem, no full-res skip.

    All layers are GroupNorm/SiLU -- from-scratch new params, so the task-side
    norm/act applies (Spec §8.3), unlike the pretrained trunk. E1/E2 are seed
    variants for G1; their exact anti-aliasing/space-to-depth may be refined when
    that arm runs. E3 is the faithful core candidate.

    ``downsample`` selects the branch's own stride-2 operator:

    - ``'conv'`` (default, unchanged): 3x3 stride-2 conv.
    - ``'hwd'``: Haar Wavelet Downsampling (survey class A). See :class:`_HWD`.

    That the branch's downsamplers are randomly initialised is precisely why
    class A can be tested here at all: the literature's own form of it swaps a
    pretrained backbone's stride-2 stages, which would misalign the pretrained
    weights this task depends on most. Nothing else about the arm changes, so
    the contrast against the same base isolates the sampling operator.
    """

    def __init__(self, form='stride1', in_channels=3, base_channels=24,
                 inject_channels=96, norm_cfg=dict(type='GN', num_groups=4),
                 act_cfg=dict(type='SiLU'), downsample='conv', init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert form in ('stride1', 'stride2_aa', 'pixel_unshuffle'), form
        assert downsample in ('conv', 'hwd'), downsample
        self.form = form
        self.downsample = downsample
        c1, c2 = base_channels, base_channels * 2

        def _down(i, o):
            """The branch's own stride-2 operator -- the A-class swap point."""
            if downsample == 'hwd':
                return _HWD(i, o, norm_cfg, act_cfg)
            return _cbr(i, o, 3, 2, norm_cfg, act_cfg)

        if form == 'stride1':
            self.stem = nn.Sequential(
                _cbr(in_channels, c1, 3, 1, norm_cfg, act_cfg),
                _cbr(c1, c1, 3, 1, norm_cfg, act_cfg))       # c1 @ 1/1 -> skip1
            self.to_half = _down(c1, c2)                     # c2 @ 1/2
        elif form == 'stride2_aa':
            # An anti-alias pre-filter in front of HWD would confound the
            # contrast: the DWT already separates the aliasing band, so the two
            # changes could not be attributed. Refused rather than silently
            # stacked -- and unnecessary, since the C2 base uses 'stride1'.
            assert downsample == 'conv', \
                "downsample='hwd' with form='stride2_aa' confounds the " \
                'anti-alias pre-filter with the sampling operator'
            self.stem = _cbr(in_channels, c1, 3, 1, norm_cfg, act_cfg)  # c1 @ 1/1
            self.to_half = nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1),         # cheap anti-alias
                _cbr(c1, c2, 3, 2, norm_cfg, act_cfg))        # c2 @ 1/2
        else:  # pixel_unshuffle
            self.unshuffle = nn.PixelUnshuffle(2)             # in*4 @ 1/2
            self.stem = _cbr(in_channels * 4, c2, 3, 1, norm_cfg, act_cfg)  # c2 @ 1/2

        self.to_quarter = _down(c2, inject_channels)          # 1/4
        # Near-zero injection: last conv zero-initialised so inject == 0 at init.
        self.inject_proj = nn.Conv2d(inject_channels, inject_channels, 1)
        nn.init.zeros_(self.inject_proj.weight)
        nn.init.zeros_(self.inject_proj.bias)

    def forward(self, img):
        skip1 = None
        if self.form == 'stride1':
            skip1 = self.stem(img)             # c1 @ 1/1
            skip2 = self.to_half(skip1)        # c2 @ 1/2
        elif self.form == 'stride2_aa':
            skip2 = self.to_half(self.stem(img))            # c2 @ 1/2
        else:  # pixel_unshuffle
            skip2 = self.stem(self.unshuffle(img))          # c2 @ 1/2
        inject = self.inject_proj(self.to_quarter(skip2))   # ~0 at init
        return dict(skip1=skip1, skip2=skip2, inject=inject)
