"""G1 R-arm acceptance: inverse-pyramid head (R1a) and DySample variant (R1b).

    python -m pytest tests/test_models/test_wcm_inverse_pyramid.py -v
"""
import pytest
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS

import mmpretrain.models  # noqa: F401
import projects.dinov3_sealing.crack_modules  # noqa: F401
import projects.wavecrack_mamba  # noqa: F401
from projects.wavecrack_mamba.models import DySample

R1A = 'projects/wavecrack_mamba/configs/g1/wcm_s_r1a_crack_v5_512_32k.py'
R1B = 'projects/wavecrack_mamba/configs/g1/wcm_s_r1b_crack_v5_512_32k.py'


def _build(path):
    cfg = Config.fromfile(path, import_custom_modules=False)
    cfg.model.backbone.convnext.init_cfg = None
    init_default_scope('mmseg')
    model = MODELS.build(cfg.model)
    model.init_weights()
    model.eval()
    return model


@pytest.fixture(scope='module')
def r1a():
    return _build(R1A)


@pytest.fixture(scope='module')
def r1b():
    return _build(R1B)


def test_r1a_size_restoration_square_nonsquare_odd(r1a):
    # Same acceptance as the E arms (Spec §15.1-2/3): square, non-square and
    # odd inputs restore exactly; the segmentor's pad-to-16 never leaks out.
    for h, w in ((512, 512), (480, 640), (511, 513)):
        with torch.no_grad():
            out = r1a(torch.rand(1, 3, h, w), mode='tensor')
        assert out.shape == (1, 1, h, w), (h, w)
        assert torch.isfinite(out).all()


def test_r1b_same_contract(r1b):
    with torch.no_grad():
        out = r1b(torch.rand(1, 3, 128, 160), mode='tensor')
    assert out.shape == (1, 1, 128, 160) and torch.isfinite(out).all()


def test_r_arms_are_trunk_only(r1a, r1b):
    assert r1a.detail_branch is None and r1a.wavelet_bypass is None
    assert r1b.detail_branch is None and r1b.wavelet_bypass is None


def test_head_rejects_detail_features(r1a):
    # M0-3 explicit-switch rule, inverted for the R arms: extra features must
    # fail loudly, not be silently ignored.
    feats = {'s1': torch.rand(1, 96, 32, 32), 's2': torch.rand(1, 192, 16, 16),
             's3': torch.rand(1, 384, 8, 8), 'detail': {'skip2': None}}
    with pytest.raises(AssertionError, match='trunk-only'):
        r1a.decode_head(feats)


def test_dysample_near_bilinear_at_init():
    # DySample offsets init to the exact bilinear sub-pixel grid plus
    # N(0, 0.001^2) noise, so at init it must reproduce fixed bilinear
    # (align_corners=False) almost exactly. This is what makes R1b start from
    # R1a behaviour -- the pair stays a clean single-variable contrast.
    torch.manual_seed(0)
    for scale in (2, 4):
        up = DySample(16, scale=scale).eval()
        x = torch.rand(2, 16, 9, 11)
        with torch.no_grad():
            got = up(x)
            ref = F.interpolate(x, scale_factor=scale, mode='bilinear',
                                align_corners=False)
        assert got.shape == ref.shape
        torch.testing.assert_close(got, ref, rtol=0, atol=2e-2)


def test_r1b_matches_r1a_at_init_with_shared_weights(r1a, r1b):
    # Same property at model level: copying R1a's weights into R1b (DySample
    # offsets keep their near-zero init) must give near-identical outputs.
    missing, unexpected = r1b.load_state_dict(r1a.state_dict(), strict=False)
    assert not unexpected
    assert all('up' in k for k in missing)  # only the DySample offset convs
    x = torch.rand(1, 3, 96, 128)
    with torch.no_grad():
        a = r1a(x, mode='tensor')
        b = r1b(x, mode='tensor')
    torch.testing.assert_close(a, b, rtol=0, atol=5e-2)


def test_r1b_grad_flow_reaches_all_head_params(r1b):
    r1b.train()
    x = torch.rand(1, 3, 64, 64)
    out = r1b(x, mode='tensor')
    out.sum().backward()
    for name, p in r1b.decode_head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    offset_grads = [p.grad.abs().sum() for n, p in
                    r1b.decode_head.named_parameters() if 'offset' in n]
    assert offset_grads and sum(offset_grads) > 0
    r1b.zero_grad(set_to_none=True)
    r1b.eval()


def test_r1b_only_adds_dysample_params(r1a, r1b):
    n_a = sum(p.numel() for p in r1a.decode_head.parameters())
    n_b = sum(p.numel() for p in r1b.decode_head.parameters())
    extra = sum(p.numel() for n, p in r1b.decode_head.named_parameters()
                if 'up' in n)
    assert n_b - n_a == extra > 0
    # the head stays the cheap reference: well under WCMDecoderHead's budget
    assert n_b < 5e5
