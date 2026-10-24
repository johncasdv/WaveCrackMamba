"""M0-6 acceptance: Hann/count slide fusion.

    python -m pytest tests/test_models/test_wcm_slide.py -v
"""
import pytest
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS

import mmpretrain.models  # noqa: F401
import projects.dinov3_sealing.crack_modules  # noqa: F401
import projects.wavecrack_mamba  # noqa: F401

C0 = 'projects/wavecrack_mamba/configs/_base_/wcm_s_c0_crack_v5_512.py'


def _model(slide_mode='hann', crop=256, stride=170):
    init_default_scope('mmseg')
    cfg = Config.fromfile(C0, import_custom_modules=False)
    cfg.model.backbone.convnext.init_cfg = None
    cfg.model.slide_mode = slide_mode
    cfg.model.test_cfg = dict(mode='slide', crop_size=(crop, crop),
                              stride=(stride, stride))
    model = MODELS.build(cfg.model)
    model.init_weights()
    model.eval()
    return model


def _metas(h, w):
    return [dict(img_shape=(h, w), ori_shape=(h, w), pad_shape=(h, w),
                 padding_size=[0, 0, 0, 0])]


def test_single_window_slide_equals_whole_for_both_modes():
    # §10-8 resolution basis, verified in code: with exactly one window the
    # per-pixel weights normalise out, so ANY weighting equals whole-image
    # inference bit-for-bit. This is why the 512-patch protocol is untouched.
    x = torch.rand(1, 3, 256, 256)
    for mode in ('hann', 'count'):
        m = _model(mode, crop=256, stride=999)
        with torch.no_grad():
            slid = m.slide_inference(x, _metas(256, 256))
            whole = m.whole_inference(x, _metas(256, 256))
        torch.testing.assert_close(slid, whole, rtol=1e-5, atol=1e-5)


def test_multi_window_both_modes_finite_and_shaped():
    x = torch.rand(1, 3, 384, 320)
    for mode in ('hann', 'count'):
        m = _model(mode)
        with torch.no_grad():
            out = m.slide_inference(x, _metas(384, 320))
        assert out.shape == (1, 1, 384, 320)
        assert torch.isfinite(out).all(), mode


def test_fusion_is_exact_convex_combination():
    # The strict form of "窗权和恒 1": stub every window's logits to a constant
    # c -- a correctly normalised weighted fusion must return exactly c at
    # every pixel, however many windows overlap. Holds for hann AND count.
    c = 3.7
    for mode in ('hann', 'count'):
        m = _model(mode)
        m.encode_decode = lambda inputs, metas: torch.full(
            (inputs.shape[0], 1, *inputs.shape[2:]), c)
        with torch.no_grad():
            out = m.slide_inference(torch.rand(1, 3, 384, 320),
                                    _metas(384, 320))
        torch.testing.assert_close(out, torch.full_like(out, c)), mode


def test_hann_window_properties():
    m = _model('hann')
    win = m._hann_window(256, 256, torch.device('cpu'), torch.float32)
    assert win.shape == (256, 256)
    assert win.min() >= 1e-2                 # clamped: no zero-weight pixels
    # even-length Hann has no grid point exactly at the centre -> max ~0.99992
    assert (win.max() - 1.0).abs() < 1e-3
    assert win[0, 0] == pytest.approx(1e-2)  # corners at the clamp floor


def test_slide_mode_registered_default_is_hann():
    init_default_scope('mmseg')
    cfg = Config.fromfile(C0, import_custom_modules=False)
    cfg.model.backbone.convnext.init_cfg = None
    m = MODELS.build(cfg.model)
    assert m.slide_mode == 'hann'
