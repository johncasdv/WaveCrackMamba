"""M0-2 acceptance: 3-stage trunk prefix + pretrained-weight loading.

    python -m pytest tests/test_models/test_wcm_trunk_consistency.py -v

The converted checkpoint is a build artifact (not committed); the pretrained
tests skip when it is absent. Regenerate it with:
    python projects/wavecrack_mamba/tools/convert_trunk_weights.py \\
        --src pretrain/DINOv3_weights/backbones/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \\
        --out pretrain/wcm_convnext_tiny_dinov3_mmpretrain.pth
"""
import os.path as osp

import pytest
import torch
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS

import mmpretrain.models  # noqa: F401
import projects.wavecrack_mamba  # noqa: F401

CKPT = 'pretrain/wcm_convnext_tiny_dinov3_mmpretrain.pth'
TRUNK_ARCH = dict(depths=[3, 3, 9], channels=[96, 192, 384])
ALLOWED_MISSING = {f'norm{i}.{s}' for i in (0, 1, 2) for s in ('weight', 'bias')}


def _convnext(**kw):
    init_default_scope('mmseg')
    return MODELS.build(dict(
        type='mmpretrain.ConvNeXt', arch=TRUNK_ARCH, out_indices=(0, 1, 2),
        gap_before_final_norm=False, **kw))


def test_trunk_is_three_stage_prefix():
    init_default_scope('mmseg')
    trunk = MODELS.build(dict(type='WCMTrunk'))
    # Only stages 0-2 are built: no stage-3 (768@1/32), no norm3.
    assert len(trunk.backbone.stages) == 3
    assert len(trunk.backbone.downsample_layers) == 3
    assert not hasattr(trunk.backbone, 'norm3')


def test_trunk_wraps_convnext_identically():
    # Injection-zeroed trunk (M0-2 has no injections yet) must forward exactly
    # like the reference ConvNeXt prefix (Spec §8.3 / 硬约束 5). Weights shared.
    # NB: mmpretrain ConvNeXt.train() returns None (no `return self`), so eval()
    # must be called as a statement, never chained.
    ref = _convnext()
    ref.eval()
    trunk = MODELS.build(dict(type='WCMTrunk'))
    trunk.eval()
    trunk.backbone.load_state_dict(ref.state_dict())
    x = torch.rand(1, 3, 256, 320)
    with torch.no_grad():
        r, t = ref(x), trunk(x)
    for i, key in enumerate(('s1', 's2', 's3')):
        torch.testing.assert_close(t[key], r[i], rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not osp.exists(CKPT), reason='converted trunk ckpt absent')
def test_converted_weights_key_whitelist():
    ck = torch.load(CKPT, map_location='cpu')['state_dict']
    res = _convnext().load_state_dict(ck, strict=False)
    assert set(res.missing_keys) == ALLOWED_MISSING  # DINOv3 lacks norm0/1/2
    assert not res.unexpected_keys


@pytest.mark.skipif(not osp.exists(CKPT), reason='converted trunk ckpt absent')
def test_trunk_loads_pretrained_and_forwards():
    trunk = MODELS.build(dict(
        type='WCMTrunk',
        convnext=dict(type='mmpretrain.ConvNeXt', arch=TRUNK_ARCH,
                      out_indices=(0, 1, 2), gap_before_final_norm=False,
                      init_cfg=dict(type='Pretrained', checkpoint=CKPT))))
    trunk.init_weights()
    trunk.eval()
    with torch.no_grad():
        out = trunk(torch.rand(1, 3, 512, 512))
    assert out['s1'].shape == (1, 96, 128, 128)
    assert out['s2'].shape == (1, 192, 64, 64)
    assert out['s3'].shape == (1, 384, 32, 32)
    assert all(torch.isfinite(v).all() for v in out.values())
