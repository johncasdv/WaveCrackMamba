"""Axial bidirectional Mamba mixer (Spec §8.4).

Two stacked axial blocks at 1/4 resolution, inserted between trunk stage-1 and
stage-2. One block has a cross-shaped direct receptive field; the second block
propagates the first block's axial aggregates along the other axis -- a single
block must never be presented as full 2D propagation.

"near-zero output projection" (Spec §8.4 ``mixer_output_proj_init``) is
implemented as: EVERY additive branch entering the residual stream has its last
layer zero-initialised (``proj``, the local DW-conv's pointwise, the MLP's
``fc2``). Rationale: the spec's stated goal is "初始前向不变" when inserting
into the pretrained trunk; zeroing only ``proj`` would still let
``β·LocalDWConv(X)`` and the MLP residual perturb the forward. α/β stay
learnable scalars at 0.1 -- they scale branches that grow from zero.
"""
import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmseg.registry import MODELS

from .scan_backend import AxialSSMCore
from .scan_orders import fold_axial, unfold_axial
from .wavelet import SUBBAND_ORDER  # noqa: F401  (router consumes desc2/proj2 per this convention)

ROUTER_MODES = ('context_only', 'hf2', 'context+hf2')


@MODELS.register_module()
class DirectionRouter(BaseModule):
    """``Conv3x3 -> SiLU -> Conv1x1(2) -> softmax(axis)``; last layer zero-init.

    Outputs per-pixel ``(g_row, g_col)`` with ``g_row + g_col = 1``; zero init
    makes it exactly 0.5/0.5 at start. Direction cues from the wavelet must be
    magnitude/statistics based -- never a single HH sign (see wavelet.py; the
    off-phase HH sign flips). The router only expresses a preference; it does
    not hardcode wavelet edge direction as crack direction (Spec §8.4).
    """

    def __init__(self, in_channels, mid_channels=32, zero_init_out=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.out = nn.Conv2d(mid_channels, 2, 1)
        # zero_init_out=False (R1 rewire, 小波设计审查 v0.1 F1): with both this
        # layer AND the bypass fuse zero-initialised, the wavelet->router path
        # has zero gradient at init and never bootstraps (verified causally:
        # router_hf2_zero is lossless on trained C3). A live output layer only
        # perturbs the router's initial preference away from 0.5/0.5; it does
        # NOT break trunk identity-at-init -- the mixer's residual projection
        # (`proj`) stays zero-initialised, so the block output still equals its
        # input at init regardless of what the router says.
        if zero_init_out:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, ctx):
        return torch.softmax(self.out(self.act(self.conv(ctx))), dim=1)


class AxialMambaBlock(BaseModule):
    """One axial block: LN -> shared-core row/col bidirectional scans -> router
    fusion -> ``X + α·Proj(Z) + β·LocalDW(X)`` -> Pre-LN channel MLP residual.
    """

    def __init__(self, d_model=96, core=dict(type='AxialSSMCore'),
                 router_mode='context+hf2', hf2_channels=96,
                 router_zero_init=True,
                 row_column_core_sharing=True, alpha_init=0.1, beta_init=0.1,
                 mlp_ratio=2.0, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert router_mode in ROUTER_MODES, router_mode
        self.router_mode = router_mode
        core = dict(core)
        core.setdefault('d_model', d_model)
        self.core = MODELS.build(core)
        # Row/col share one core by default (same L, half the params, capacity
        # matching); an UNVERIFIED prior -- the independent-core variant is a
        # §14.5 ablation arm, hence the switch.
        self.core_col = None if row_column_core_sharing else MODELS.build(core)

        router_in = {'context_only': d_model, 'hf2': hf2_channels,
                     'context+hf2': d_model + hf2_channels}[router_mode]
        self.router = DirectionRouter(router_in, zero_init_out=router_zero_init)

        self.norm1 = nn.LayerNorm(d_model)
        self.proj = nn.Conv2d(d_model, d_model, 1)
        self.local_dw = nn.Conv2d(d_model, d_model, 3, padding=1,
                                  groups=d_model)
        self.local_pw = nn.Conv2d(d_model, d_model, 1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp_fc1 = nn.Linear(d_model, hidden)
        self.mlp_act = nn.GELU()
        self.mlp_fc2 = nn.Linear(hidden, d_model)
        # Identity-at-init: zero the last layer of every additive branch.
        for m in (self.proj, self.local_pw, self.mlp_fc2):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _bidir(self, x_norm, fwd_key, rev_key, core):
        b, c, h, w = x_norm.shape
        fwd = fold_axial(core(unfold_axial(x_norm, fwd_key)), fwd_key, b, c, h, w)
        rev = fold_axial(core(unfold_axial(x_norm, rev_key)), rev_key, b, c, h, w)
        return 0.5 * (fwd + rev)

    def forward(self, x, hf2=None):
        b, c, h, w = x.shape
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        r = self._bidir(x_norm, 'row_fwd', 'row_rev', self.core)
        col_core = self.core_col if self.core_col is not None else self.core
        cfeat = self._bidir(x_norm, 'col_fwd', 'col_rev', col_core)

        if self.router_mode == 'context_only':
            ctx = x
        else:
            assert hf2 is not None, (
                f"router_mode='{self.router_mode}' needs the wavelet HF2 "
                'projection; use context_only when the bypass is disabled')
            ctx = hf2 if self.router_mode == 'hf2' else torch.cat([x, hf2], 1)
        g = self.router(ctx)
        z = g[:, 0:1] * r + g[:, 1:2] * cfeat

        x = x + self.alpha * self.proj(z) + \
            self.beta * self.local_pw(self.local_dw(x))
        y = self.mlp_fc2(self.mlp_act(self.mlp_fc1(
            self.norm2(x.permute(0, 2, 3, 1)))))
        return x + y.permute(0, 3, 1, 2)


@MODELS.register_module()
class AxialMambaMixer(BaseModule):
    """``n_blocks`` stacked axial blocks; routers are per-block
    (``router_sharing_across_blocks=false``, Spec §8.4 default)."""

    def __init__(self, n_blocks=2, block=dict(), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.blocks = nn.ModuleList(
            AxialMambaBlock(**block) for _ in range(n_blocks))

    @property
    def actual_backend_name(self):
        return self.blocks[0].core.actual_backend_name

    def forward(self, x, hf2=None):
        for blk in self.blocks:
            x = blk(x, hf2=hf2)
        return x
