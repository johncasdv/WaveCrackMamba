"""M0-5 acceptance: scan orders (bijection + negative detection), router,
block identity-at-init, gradients, trunk consistency.

    python -m pytest tests/test_models/test_wcm_axial_scan.py -v
"""
import torch
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS

import projects.wavecrack_mamba  # noqa: F401
from projects.wavecrack_mamba.models.scan_orders import (
    SCAN_KEYS, axial_scan_orders, fold_axial, is_bijective_pair, unfold_axial)

GRU = dict(type='AxialSSMCore', d_model=8, d_state=4, d_conv=2, expand=2,
           backend='gru')


def _block(**kw):
    init_default_scope('mmseg')
    from projects.wavecrack_mamba.models.axial_mixer import AxialMambaBlock
    args = dict(d_model=8, core=GRU, router_mode='context_only')
    args.update(kw)
    return AxialMambaBlock(**args)


def _mixer(**kw):
    init_default_scope('mmseg')
    block = dict(d_model=8, core=GRU, router_mode='context_only')
    block.update(kw.pop('block', {}))
    return MODELS.build(dict(type='AxialMambaMixer', n_blocks=2, block=block))


# ------------------------------------------------------------- scan orders
def test_orders_bijective_all_keys_and_sizes():
    for h, w in ((3, 5), (5, 3), (4, 4), (1, 1), (1, 7), (7, 1)):
        for key, (order, inverse) in axial_scan_orders(h, w).items():
            assert is_bijective_pair(order, inverse), (key, h, w)


def test_scsegamba_constant_inverse_is_detected():
    # Negative detection (Spec §8.4): the official SAVSS_layer builds
    # o3_inverse[idx] = len(o1)-1 -- a constant. Our checker must reject it.
    orders = axial_scan_orders(4, 4)
    order, _ = orders['row_fwd']
    bad_inverse = torch.full_like(order, order.numel() - 1)
    assert not is_bijective_pair(order, bad_inverse)
    # ...and reject a non-permutation "order" too.
    assert not is_bijective_pair(bad_inverse, torch.argsort(order))


def test_direction_line_scan_and_restore():
    # A synthetic direction line survives serialise -> deserialise unchanged
    # for every scan; catches silent index scrambling.
    x = torch.zeros(1, 3, 6, 9)
    x[:, :, 2, :] = torch.arange(9).float()   # marked horizontal line
    x[:, :, :, 5] = -torch.arange(6).float().view(1, 1, 6)
    for key in SCAN_KEYS:
        seq = unfold_axial(x, key)
        back = fold_axial(seq, key, 1, 3, 6, 9)
        torch.testing.assert_close(back, x, rtol=0, atol=0)


def test_reshape_fast_path_matches_explicit_orders():
    # The mixer's reshape/flip serialisation must equal gathering by the
    # explicit permutations -- implementation can never drift from semantics.
    b, c, h, w = 2, 3, 4, 5
    x = torch.arange(b * c * h * w, dtype=torch.float32).reshape(b, c, h, w)
    flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # row-major pixels
    orders = axial_scan_orders(h, w)
    for key in SCAN_KEYS:
        order, _ = orders[key]
        seq = unfold_axial(x, key)          # [b*rows, L, c]
        gathered = flat[:, order]           # [b, h*w, c] in scan visit order
        assert torch.equal(seq.reshape(b, h * w, c), gathered), key


# ----------------------------------------------------------------- router
def test_router_zero_init_is_half_half_and_sums_to_one():
    from projects.wavecrack_mamba.models.axial_mixer import DirectionRouter
    router = DirectionRouter(in_channels=8)
    g = router(torch.rand(2, 8, 6, 6))
    torch.testing.assert_close(g.sum(dim=1), torch.ones(2, 6, 6))
    torch.testing.assert_close(g, torch.full_like(g, 0.5))


def test_router_mode_needs_hf2():
    blk = _block(router_mode='context+hf2', hf2_channels=8)
    x = torch.rand(1, 8, 6, 6)
    try:
        blk(x, hf2=None)
        raise SystemExit('expected assertion')
    except AssertionError:
        pass
    out = blk(x, hf2=torch.rand(1, 8, 6, 6))
    assert out.shape == x.shape


# ---------------------------------------------------- block / mixer semantics
def test_block_is_identity_at_init():
    # Every additive branch is zero-initialised => inserting the mixer between
    # pretrained trunk stages leaves the forward bit-identical (Spec §8.4
    # "初始前向不变" / 硬约束 5).
    blk = _block()
    blk.eval()
    x = torch.rand(2, 8, 6, 7)
    with torch.no_grad():
        torch.testing.assert_close(blk(x), x, rtol=0, atol=0)


def test_alpha_beta_init_registered_value():
    blk = _block()  # fp32 storage of 0.1 -> approximate comparison
    assert abs(float(blk.alpha) - 0.1) < 1e-7
    assert abs(float(blk.beta) - 0.1) < 1e-7


def test_row_column_core_sharing_switch():
    shared = _block(row_column_core_sharing=True)
    indep = _block(row_column_core_sharing=False)
    assert shared.core_col is None
    assert indep.core_col is not None
    n_shared = sum(p.numel() for p in shared.parameters())
    n_indep = sum(p.numel() for p in indep.parameters())
    assert n_indep > n_shared


def test_two_blocks_all_params_get_gradient():
    # Zero-init disconnects branches exactly at 0; perturb the zeroed layers
    # (same rationale as the M0-3 detail-branch test) then check every param.
    mixer = _mixer()
    for blk in mixer.blocks:
        for m in (blk.proj, blk.local_pw, blk.mlp_fc2):
            torch.nn.init.normal_(m.weight, std=1e-3)
        torch.nn.init.normal_(blk.router.out.weight, std=1e-3)
    x = torch.rand(2, 8, 6, 7, requires_grad=True)
    mixer(x).sum().backward()
    for name, p in mixer.named_parameters():
        assert p.grad is not None, f'no grad: {name}'
        assert torch.isfinite(p.grad).all(), f'non-finite grad: {name}'
        assert p.grad.abs().sum() > 0, f'zero grad: {name}'
    assert torch.isfinite(x.grad).all()


def test_mixer_keeps_trunk_forward_identical_at_init():
    # Full-path 硬约束 5: trunk + zero-init mixer == trunk alone, bit-equal.
    init_default_scope('mmseg')
    trunk = MODELS.build(dict(type='WCMTrunk'))
    trunk.eval()
    mixer = MODELS.build(dict(
        type='AxialMambaMixer', n_blocks=2,
        block=dict(d_model=96,
                   core=dict(type='AxialSSMCore', d_model=96, backend='gru'),
                   router_mode='context_only')))
    mixer.eval()
    x = torch.rand(1, 3, 128, 160)
    with torch.no_grad():
        ref = trunk(x)
        got = trunk(x, transform_after_stage0=lambda f: mixer(f))
    for k in ('s1', 's2', 's3'):
        torch.testing.assert_close(got[k], ref[k], rtol=0, atol=0)
