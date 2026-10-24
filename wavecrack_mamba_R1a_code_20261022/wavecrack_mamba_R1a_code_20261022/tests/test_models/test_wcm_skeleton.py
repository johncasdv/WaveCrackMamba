"""M0-1 acceptance: config build + random-weight smoke forward.

Run from the repo root:
    python -m pytest tests/test_models/test_wcm_skeleton.py -v
"""
import os.path as osp

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS

# Trigger registration (WCM + shared crack modules + ConvNeXt trunk).
import mmpretrain.models  # noqa: F401
import projects.dinov3_sealing.crack_modules  # noqa: F401
import projects.wavecrack_mamba  # noqa: F401

CFG = osp.join('projects', 'wavecrack_mamba', 'configs', '_base_',
               'wcm_s_c0_crack_v5_512.py')


def _load_cfg():
    # Registration is done by the explicit imports above; disable the config's
    # own custom_imports so parsing never re-imports the package (that would
    # double-register the modules under a second identity).
    cfg = Config.fromfile(CFG, import_custom_modules=False)
    # Skeleton smoke tests use random init -- do not depend on the (large,
    # un-committed) converted trunk checkpoint. Pretrained loading is covered by
    # test_wcm_trunk_consistency.py.
    cfg.model.backbone.convnext.init_cfg = None
    return cfg


def _build(cfg=None):
    init_default_scope('mmseg')
    cfg = cfg or _load_cfg()
    model = MODELS.build(cfg.model)
    # The Runner always calls this before the first forward; ConvNeXt sets up
    # lazily, so a bare build+forward would fail without it.
    model.init_weights()
    return model.eval()


def test_c0_config_builds():
    model = _build()
    assert type(model).__name__ == 'WCMSegmentor'
    assert model.out_channels == 1


def test_smoke_forward_finite():
    model = _build()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 512, 512), mode='tensor')
    assert out.shape == (1, 1, 512, 512)
    assert torch.isfinite(out).all()


def test_forward_nonsquare_small():
    model = _build()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 64, 80), mode='tensor')
    assert out.shape == (1, 1, 64, 80)


def test_in_channels_rgb_and_gray():
    # RGB builds (covered above); gray uses a 1-channel trunk + detail branch
    # + preprocessor.
    cfg = _load_cfg()
    cfg.model.backbone.convnext.in_channels = 1
    if cfg.model.get('detail_branch'):
        cfg.model.detail_branch['in_channels'] = 1
    cfg.model.data_preprocessor.mean = [123.675]
    cfg.model.data_preprocessor.std = [58.395]
    cfg.model.data_preprocessor.bgr_to_rgb = False
    gray = _build(cfg)
    with torch.no_grad():
        out = gray(torch.rand(1, 1, 128, 128), mode='tensor')
    assert out.shape == (1, 1, 128, 128)
