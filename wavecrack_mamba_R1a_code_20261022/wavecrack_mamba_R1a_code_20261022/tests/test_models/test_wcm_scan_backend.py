"""M0-5 acceptance: scan backends (gru debug / auto fallback / mamba CUDA).

    python -m pytest tests/test_models/test_wcm_scan_backend.py -v

The mamba tests run in the WSL ``mmseg_env`` (see the §8-3 environment report);
on the Windows ``.venv`` they skip.
"""
import warnings

import pytest
import torch
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS

import projects.wavecrack_mamba  # noqa: F401
from projects.wavecrack_mamba.models.scan_backend import mamba_available


def _core(**kw):
    init_default_scope('mmseg')
    args = dict(type='AxialSSMCore', d_model=8, d_state=4, d_conv=2, expand=2)
    args.update(kw)
    return MODELS.build(args)


def test_gru_backend_cpu_forward_backward_finite():
    core = _core(backend='gru')
    assert core.actual_backend_name.startswith('torch.GRU')
    seq = torch.rand(4, 16, 8, requires_grad=True)
    out = core(seq)
    assert out.shape == seq.shape
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(seq.grad).all()


def test_auto_backend_resolves_and_warns_only_on_fallback():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        core = _core(backend='auto')
    if mamba_available():
        assert core.backend == 'mamba'
        assert not any('falling back' in str(w.message) for w in caught)
    else:
        assert core.backend == 'gru'
        assert any('falling back' in str(w.message) for w in caught)
    assert core.requested_backend == 'auto'


@pytest.mark.skipif(mamba_available(), reason='mamba present; error path n/a')
def test_mamba_backend_fails_loudly_when_missing():
    # Formal configs pin backend='mamba'; silent gru degradation would produce
    # results that must not be reported as Mamba (Spec §8.4).
    with pytest.raises(ImportError):
        _core(backend='mamba')


@pytest.mark.skipif(not mamba_available() or not torch.cuda.is_available(),
                    reason='needs mamba_ssm + CUDA (run in WSL mmseg_env)')
def test_mamba_cuda_smoke():
    core = _core(d_model=96, d_state=16, d_conv=4, expand=2,
                 backend='mamba').cuda()
    assert core.actual_backend_name.startswith('mamba_ssm.Mamba')
    seq = torch.rand(4, 128, 96, device='cuda', requires_grad=True)
    out = core(seq)
    out.float().sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(seq.grad).all()


def test_full_model_all_modules_gru_forward():
    # Whole WCM-S (trunk+detail+wavelet+mixer) builds from the base config with
    # the backend overridden to gru, and forwards 512 -> 512.
    from mmengine.config import Config
    init_default_scope('mmseg')
    cfg = Config.fromfile(
        'projects/wavecrack_mamba/configs/_base_/wcm_s_c0_crack_v5_512.py',
        import_custom_modules=False)
    cfg.model.backbone.convnext.init_cfg = None
    # restore the full-module defaults that c0 nulls
    base_model = Config.fromfile(
        'projects/wavecrack_mamba/configs/_base_/wcm_s_model.py').model
    cfg.model.wavelet_bypass = base_model['wavelet_bypass']
    cfg.model.axial_mixer = base_model['axial_mixer']
    cfg.model.axial_mixer['block']['core']['backend'] = 'gru'
    model = MODELS.build(cfg.model)
    model.init_weights()
    model.eval()
    assert model.axial_mixer.actual_backend_name.startswith('torch.GRU')
    with torch.no_grad():
        out = model(torch.rand(1, 3, 512, 512), mode='tensor')
    assert out.shape == (1, 1, 512, 512) and torch.isfinite(out).all()
