import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS

# 2x2 stride-2 Haar filters, orthonormal (factor 0.5). Rows of a 2x2 block are
# (a=TL, b=TR, c=BL, d=BR). The subband CONVENTION below is pinned by the
# synthetic-pattern tests in tests/test_models/test_wcm_wavelet.py (Spec §8.2:
# must not be inferred from names). Downstream consumers (direction router,
# decoder gates) must index descriptors via SUBBAND_ORDER, not magic numbers.
#
#   LL = a+b+c+d           low-low
#   LH = (a+b) - (c+d)     vertical high-pass   -> excited by HORIZONTAL structure
#   HL = (a+c) - (b+d)     horizontal high-pass -> excited by VERTICAL structure
#   HH = a - b - c + d     diagonal high-pass   -> excited by DIAGONAL structure
#
# Signed-channel semantics (all verified by tests):
# - Polarity: inverting the image negates all signed coefficients; dark cracks
#   on bright pavement therefore mirror the sign pattern of bright lines.
# - HH sign vs diagonal direction: ONLY under phase-aligned sampling a 45°
#   (TL-BR) line gives HH>0 and a 135° (TR-BL) line gives HH<0; off-phase
#   diagonals flip sign and spread energy into LH/HL. Routers must therefore
#   use magnitudes/statistics for direction cues, never a single HH sign.
SUBBAND_ORDER = ('LH', 'HL', 'HH')  # signed channels 0..2; magnitudes 3..5

_HAAR = torch.tensor([
    [[1., 1.], [1., 1.]],
    [[1., 1.], [-1., -1.]],
    [[1., -1.], [1., -1.]],
    [[1., -1.], [-1., 1.]],
], dtype=torch.float32).unsqueeze(1) * 0.5  # [4, 1, 2, 2]


class _LevelProj(BaseModule):
    """Project a 6-ch (signed 3 + magnitude 3) descriptor to ``out_ch``.

    Signed and magnitude are projected *separately* (Spec §8.2: must not lose the
    sign before projection) then fused; the fuse conv is zero-initialised so the
    injected feature is 0 at init (near-zero injection).

    ``zero_init_fuse=False`` (R1 rewire arm, 小波设计审查 v0.1 F1): the audit
    showed the zero fuse, in series with the router's zero output layer, makes
    the wavelet->router gradient exactly zero at init -- the routing pathway
    never bootstraps. A live fuse does NOT break the identity-at-init
    discipline: that discipline (Spec §7.2) covers branches merged into the
    PRETRAINED trunk, and the projections feed only the router and the decoder
    gates; the mixer's residual projection stays zero-initialised regardless.
    """

    def __init__(self, out_ch, norm_cfg, act_cfg, zero_init_fuse=True):
        super().__init__()
        self.signed = ConvModule(3, out_ch, 3, padding=1,
                                 norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.mag = ConvModule(3, out_ch, 3, padding=1,
                              norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.fuse = nn.Conv2d(out_ch, out_ch, 1)
        if zero_init_fuse:
            nn.init.zeros_(self.fuse.weight)
            nn.init.zeros_(self.fuse.bias)

    def forward(self, signed, mag):
        return self.fuse(self.signed(signed) + self.mag(mag))


# R4 (G3 revived) fixed alternatives to the Haar high-pass: Sobel-x / Sobel-y /
# Laplacian, 3x3 stride-2. Chosen because they are the Spec §12 comparison set
# ("传统梯度 Sobel/Laplacian"); the low-pass stays a fixed 2x2 average so the
# two-level pyramid structure is identical across kernels.
_SOBEL = torch.tensor([
    [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],   # x-gradient (vertical edges)
    [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],   # y-gradient (horizontal edges)
    [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],     # Laplacian (isotropic)
], dtype=torch.float32).unsqueeze(1) * 0.25  # scale to Haar-like response range


@MODELS.register_module()
class HaarWaveletBypass(BaseModule):
    """Two-level fixed Haar wavelet bypass (Spec §8.2).

    RGB -> luminance -> two levels of Haar DWT. Each level keeps the signed
    ``LH/HL/HH`` coefficients (sign preserves bright/dark polarity) plus their
    ``log1p(|hf|/scale)`` magnitudes -> a 6-ch descriptor, projected (near-zero)
    for injection into the direction router (M0-5) and the decoder gate (M0-6).

    Returns a dict:
    - ``desc1`` ``[B, 6, H/2, W/2]``  (level-1 signed 3 + magnitude 3)
    - ``desc2`` ``[B, 6, H/4, W/4]``  (level-2)
    - ``proj1`` ``[B, proj_channels[0], H/2, W/2]`` (~0 at init)
    - ``proj2`` ``[B, proj_channels[1], H/4, W/4]`` (~0 at init)

    The Haar kernels are fixed (not trained) buffers. Per-subband normalisation
    uses training-set scales (``tools/estimate_wavelet_stats.py``), never per-image
    min-max. Descriptor channel order is ``[LH, HL, HH]`` signed then magnitude.
    """

    def __init__(self, levels=2, luminance_coef=(0.299, 0.587, 0.114),
                 signed_scale=1.0, magnitude_scale=1.0,
                 proj_channels=(48, 96),
                 hf_kernel='haar', zero_init_fuse=True,
                 norm_cfg=dict(type='GN', num_groups=4),
                 act_cfg=dict(type='SiLU'), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert levels == 2, 'v0.5 core uses exactly two Haar levels'
        # hf_kernel (R4 / Spec §12 W3 comparison set):
        #   'haar'      -- fixed 2x2 Haar (v0.5 core, unchanged default)
        #   'sobel'     -- fixed 3x3 Sobel-x/Sobel-y/Laplacian, stride 2
        #   'learnable' -- 3 learnable 2x2 high-pass kernels, Haar-initialised
        #                  (可学习高通; low-pass stays fixed so the pyramid's
        #                  downsampling path is stable)
        # signed/magnitude scales are reused across kernels for the 32k
        # screening tier: all three are first-difference operators in the same
        # response range and the projection's first GN washes constant scale
        # offsets; a 160k confirmation arm must re-measure scales first.
        assert hf_kernel in ('haar', 'sobel', 'learnable'), hf_kernel
        self.hf_kernel = hf_kernel
        self.signed_scale = float(signed_scale)
        self.magnitude_scale = float(magnitude_scale)
        self.register_buffer('haar', _HAAR.clone())
        if hf_kernel == 'sobel':
            self.register_buffer('sobel', _SOBEL.clone())
        elif hf_kernel == 'learnable':
            self.hf_weight = nn.Parameter(_HAAR[1:].clone())
        self.register_buffer(
            'lum', torch.tensor(luminance_coef).view(1, 3, 1, 1))
        self.proj1 = _LevelProj(proj_channels[0], norm_cfg, act_cfg,
                                zero_init_fuse=zero_init_fuse)
        self.proj2 = _LevelProj(proj_channels[1], norm_cfg, act_cfg,
                                zero_init_fuse=zero_init_fuse)

    def _luminance(self, img):
        if img.shape[1] == 1:
            return img
        return (img * self.lum.to(img.dtype)).sum(1, keepdim=True)

    @staticmethod
    def _pad_mult4(x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % 4, (-w) % 4
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        return x

    def _dwt(self, x):
        if self.hf_kernel == 'haar':
            out = F.conv2d(x, self.haar.to(x.dtype), stride=2)  # [B,4,h/2,w/2]
            return out[:, :1], out[:, 1:]  # ll, hf(LH,HL,HH)
        # Alternative kernels keep the fixed Haar LL as the downsampling path
        # (identical pyramid geometry across arms); only the high-pass differs.
        ll = F.conv2d(x, self.haar[:1].to(x.dtype), stride=2)
        if self.hf_kernel == 'sobel':
            hf = F.conv2d(x, self.sobel.to(x.dtype), stride=2, padding=1)
        else:  # learnable
            hf = F.conv2d(x, self.hf_weight.to(x.dtype), stride=2)
        return ll, hf

    def _descriptor(self, hf):
        signed = hf / self.signed_scale
        mag = torch.log1p(hf.abs() / self.magnitude_scale)
        return torch.cat([signed, mag], dim=1)  # [B,6,h,w]

    def forward(self, img):
        y = self._pad_mult4(self._luminance(img))
        ll1, hf1 = self._dwt(y)
        _, hf2 = self._dwt(ll1)
        desc1, desc2 = self._descriptor(hf1), self._descriptor(hf2)
        proj1 = self.proj1(desc1[:, :3], desc1[:, 3:])
        proj2 = self.proj2(desc2[:, :3], desc2[:, 3:])
        return dict(desc1=desc1, desc2=desc2, proj1=proj1, proj2=proj2)
