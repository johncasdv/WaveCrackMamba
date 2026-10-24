"""Contracts for shared refinement, checkpoint admission and frozen pilots."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.logging import HistoryBuffer

import mmpretrain.models  # noqa: F401
import projects.dinov3_sealing.crack_modules  # noqa: F401
import projects.wavecrack_mamba  # noqa: F401
from mmseg.registry import MODELS
from projects.wavecrack_mamba.models import WCMRecurrentRefiner
from projects.wavecrack_mamba.models.recurrent_training import (
    WCMFrozenRefinerHook, load_refiner_base_checkpoint,
    validate_refiner_checkpoint)
from projects.wavecrack_mamba.transforms import WCMFixedCenterCrop


def _refiner(mode='transformer', steps=3, **kwargs):
    return WCMRecurrentRefiner(
        channels=16,
        heads=4,
        window_size=4,
        conv_channels=20,
        mode=mode,
        steps=steps,
        **kwargs)


def _activate(refiner):
    for updater in refiner.updaters:
        nn.init.normal_(updater.output.weight, std=0.03)
        nn.init.normal_(updater.ffn[-1].weight, std=0.03)


@pytest.mark.parametrize('mode', ['transformer', 'conv'])
@pytest.mark.parametrize('shape', [(2, 16, 7, 11), (1, 16, 1, 1),
                                   (1, 16, 8, 8)])
def test_zero_init_and_zero_steps_are_identity(mode, shape):
    m = _refiner(mode)
    x = torch.randn(shape)
    classifier = nn.Conv2d(16, 1, 1)
    assert m(x, classifier, steps=0) is x
    for steps in (1, 3):
        torch.testing.assert_close(
            m(x, classifier, steps=steps), x, rtol=0, atol=0)
    assert not any('classifier' in key for key in m.state_dict())


def test_shared_parameters_do_not_scale_with_iterations():
    torch.manual_seed(27)
    one = _refiner(steps=1)
    torch.manual_seed(27)
    three = _refiner(steps=3)
    assert one.state_dict().keys() == three.state_dict().keys()
    for key, value in one.state_dict().items():
        torch.testing.assert_close(
            value, three.state_dict()[key], rtol=0, atol=0)
    independent = _refiner(share_weights=False)
    assert sum(p.numel() for p in independent.parameters()) > sum(
        p.numel() for p in three.parameters())


def test_memory_is_encoded_once_per_forward_and_gradients_remain_live():
    m = _refiner()
    _activate(m)
    counts = dict(key=0, value=0)

    def count(name):

        def hook(*args):
            counts[name] += 1

        return hook

    handles = [
        getattr(m, name).register_forward_hook(count(name)) for name in counts
    ]
    classifier = nn.Conv2d(16, 1, 1).requires_grad_(False)
    x = torch.randn(2, 16, 7, 9, requires_grad=True)
    m(x, classifier).square().mean().backward()
    assert counts == dict(key=1, value=1)
    assert x.grad is not None and x.grad.abs().sum() > 0
    for name in ('key', 'value'):
        assert getattr(m, name).weight.grad.abs().sum() > 0
    assert m.updaters[0].mask_embed.weight.grad.abs().sum() > 0
    assert classifier.weight.grad is None
    m(x.detach(), classifier)
    assert counts == dict(key=2, value=2)
    for h in handles:
        h.remove()


def _uniform_attention(shift=2):
    m = _refiner(steps=1, shifts=(shift, ), use_mask_feedback=False)
    m.memory_norm = nn.Identity()
    u = m.updaters[0]
    with torch.no_grad():
        m.key.weight.zero_()
        m.key.bias.zero_()
        m.value.weight.copy_(torch.eye(16))
        m.value.bias.zero_()
        u.query.weight.zero_()
        u.query.bias.zero_()
        u.output.weight.copy_(torch.eye(16))
    return m


def test_shifted_windows_do_not_connect_opposite_image_edges():
    m = _uniform_attention()
    x = torch.zeros(1, 16, 8, 8)
    x[0, 0, 0, 0] = 1
    out = m(x, None)
    assert out[0, 0, 0, 1] > 0  # a real neighbour receives context
    assert out[0, 0, 0, -1] == 0
    assert out[0, 0, -1, 0] == 0


@pytest.mark.parametrize('hw', [(7, 5), (1, 1), (3, 2), (8, 8)])
def test_padding_never_dilutes_valid_memory(hw):
    # Uniform valid values must remain 1, even in partial/shifted windows.
    m = _uniform_attention()
    x = torch.ones(2, 16, *hw)
    torch.testing.assert_close(m(x, None), torch.full_like(x, 1.1))


def test_batch_members_cannot_exchange_memory():
    m = _uniform_attention()
    x = torch.randn(2, 16, 7, 11)
    torch.testing.assert_close(m(x, None)[0:1], m(x[0:1], None))


def test_diagnostics_include_initial_state_and_do_not_change_output():
    m = _refiner()
    _activate(m)
    x = torch.randn(1, 16, 5, 7)
    classifier = nn.Conv2d(16, 1, 1)
    final, coarse = m(x, classifier, return_intermediates=True)
    assert len(coarse) == 4
    torch.testing.assert_close(coarse[0], classifier(x))
    torch.testing.assert_close(coarse[-1], classifier(final))
    torch.testing.assert_close(final, m(x, classifier))


def test_invalid_dimensions_and_steps_fail_explicitly():
    with pytest.raises(ValueError, match='divisible'):
        WCMRecurrentRefiner(channels=15, heads=4)
    with pytest.raises(ValueError, match='shifts'):
        _refiner(shifts=(0, 4, 0))
    m = _refiner()
    for step in (-1, 4, 1.5, True):
        with pytest.raises(ValueError, match='steps'):
            m(torch.ones(1, 16, 5, 7), None, steps=step)
    with pytest.raises(ValueError, match='one logit'):
        m(torch.ones(1, 16, 5, 7), nn.Conv2d(16, 2, 1))


def _model(refiner=True, frozen=False):
    cfg = Config.fromfile(
        'projects/wavecrack_mamba/configs/g1/wcm_s_r1a_crack_v5_512_32k.py',
        import_custom_modules=False)
    cfg.model.backbone.convnext.init_cfg = None
    cfg.model.freeze_base_for_refiner = frozen
    if refiner:
        cfg.model.decode_head.refiner = dict(type='WCMRecurrentRefiner')
    init_default_scope('mmseg')
    model = MODELS.build(cfg.model)
    model.init_weights()
    return model


@pytest.fixture(scope='module')
def model_pair():
    base = _model(refiner=False).eval()
    new = _model(frozen=True).eval()
    weights = validate_refiner_checkpoint(
        new, base.state_dict(), base_only=True)
    missing, unexpected = new.load_state_dict(weights, strict=False)
    assert not unexpected and all(
        k.startswith('decode_head.refiner.') for k in missing)
    return base, new


def test_frozen_model_identity_and_cached_feature_interface(model_pair):
    base, new = model_pair
    x = torch.rand(1, 3, 65, 83)
    with torch.no_grad():
        expected = base(x, mode='tensor')
        result = new(x, mode='tensor')
        feats = new.extract_feat(x)
        fused = new.decode_head.fuse_features(feats)
        cached = new.decode_head.predict_from_fused(fused)
        online = new.decode_head(feats)
    torch.testing.assert_close(expected, result, rtol=0, atol=0)
    torch.testing.assert_close(cached, online, rtol=0, atol=0)
    assert result.shape == (1, 1, 65, 83)


def test_whole_tensor_and_slide_crop_odd_inputs(model_pair):
    _, model = model_pair
    x = torch.rand(1, 3, 65, 83)
    model.test_cfg = Config(
        dict(mode='slide', crop_size=(48, 64), stride=(32, 48)))
    metas = [
        dict(
            img_shape=(65, 83),
            ori_shape=(65, 83),
            pad_shape=(65, 83),
            padding_size=[0, 0, 0, 0])
    ]
    with torch.no_grad():
        for fn in (lambda: model(x, mode='tensor'),
                   lambda: model.whole_inference(x, copy.deepcopy(metas)),
                   lambda: model.slide_inference(x, copy.deepcopy(metas))):
            out = fn()
            assert out.shape == (1, 1, 65, 83) and torch.isfinite(out).all()


def test_train_keeps_old_modules_eval_and_classifier_transmits_gradients(
        model_pair):
    _, model = model_pair
    model.train()
    assert not model.backbone.training
    assert not model.decode_head.training
    assert model.decode_head.refiner.training
    seen = []
    handle = model.decode_head.dropout.register_forward_hook(
        lambda *args: seen.append(1))
    out = model(torch.rand(1, 3, 64, 80), mode='tensor')
    out.square().mean().backward()
    assert len(seen) == 1  # cls_seg only at the final readout
    assert model.decode_head.refiner.updaters[0].output.weight.grad.abs().sum(
    ) > 0
    for name, p in model.named_parameters():
        if not name.startswith('decode_head.refiner.'):
            assert not p.requires_grad and p.grad is None, name
    handle.remove()
    model.zero_grad(set_to_none=True)
    model.eval()


def test_strict_loader_rejects_bad_base_before_mutating(model_pair, tmp_path):
    base, model = model_pair
    weights = dict(base.state_dict())
    first = next(iter(weights))
    before = model.state_dict()[first].clone()
    weights.pop(first)
    with pytest.raises(ValueError, match='missing'):
        validate_refiner_checkpoint(model, weights, base_only=True)
    weights = dict(base.state_dict())
    weights[first] = torch.zeros(1)
    with pytest.raises(ValueError, match='shape_mismatch'):
        validate_refiner_checkpoint(model, weights, base_only=True)
    torch.testing.assert_close(
        before, model.state_dict()[first], rtol=0, atol=0)
    checkpoint = tmp_path / 'base.pth'
    torch.save(dict(state_dict=base.state_dict()), checkpoint)
    with pytest.raises(ValueError, match='SHA-256'):
        load_refiner_base_checkpoint(model, checkpoint, '0' * 64)
    digest = load_refiner_base_checkpoint(model, checkpoint)
    assert len(digest) == 64
    hook = WCMFrozenRefinerHook(str(checkpoint))
    runner = SimpleNamespace(model=model, _resume=True, _load_from='full.pth')
    hook.before_run(runner)
    hook.after_load_checkpoint(runner, dict(state_dict=model.state_dict()))
    with pytest.raises(ValueError, match='missing'):
        hook.after_load_checkpoint(runner, dict(state_dict=base.state_dict()))


def test_legacy_mmengine_base_requires_explicit_pinned_digest(
        model_pair, tmp_path):
    base, model = model_pair
    path = tmp_path / 'legacy.pth'
    torch.save(
        dict(state_dict=base.state_dict(), history=HistoryBuffer()), path)
    with pytest.raises(ValueError, match='trusted expected_sha256'):
        load_refiner_base_checkpoint(model, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_refiner_base_checkpoint(model, path, digest) == digest


def test_standalone_evaluation_accepts_full_checkpoint_without_old_base(
        model_pair):
    _, model = model_pair
    hook = WCMFrozenRefinerHook('not-needed-for-evaluation.pth')
    runner = SimpleNamespace(
        model=model,
        _resume=False,
        _load_from='trained.pth',
        _has_loaded=False)
    hook.before_run(runner)
    with pytest.raises(ValueError, match='evaluation requires'):
        hook.before_test(runner)
    hook.after_load_checkpoint(runner, dict(state_dict=model.state_dict()))
    runner._has_loaded = True
    hook.before_test(runner)
    hook.before_val(runner)
    with pytest.raises(ValueError, match='fresh pilot'):
        hook.before_train(runner)


def test_resume_without_a_loaded_checkpoint_fails_closed(model_pair):
    _, model = model_pair
    hook = WCMFrozenRefinerHook('unused.pth')
    runner = SimpleNamespace(
        model=model, _resume=True, _load_from=None, _has_loaded=False)
    hook.before_run(runner)
    with pytest.raises(ValueError, match='no complete refiner checkpoint'):
        hook.before_train(runner)


@pytest.mark.parametrize('arm,mode,steps', [('t1', 'transformer', 1),
                                            ('t3', 'transformer', 3),
                                            ('c3', 'conv', 3)])
def test_pilot_configs_are_fixed_view_frozen_r1a(arm, mode, steps):
    cfg = Config.fromfile(
        f'projects/wavecrack_mamba/configs/recurrent/wcm_fz_{arm}_crack_v5.py')
    assert cfg.model.decode_head.type == 'WCMInversePyramidHead'
    assert cfg.model.detail_branch is None and cfg.model.wavelet_bypass is None
    assert cfg.model.axial_mixer.block.core.backend == 'mamba'
    assert cfg.model.axial_mixer.block.router_mode == 'context_only'
    assert cfg.model.freeze_base_for_refiner
    assert cfg.model.decode_head.refiner.mode == mode
    assert cfg.model.decode_head.refiner.steps == steps
    assert cfg.model.backbone.convnext.init_cfg is None and cfg.load_from is None
    assert cfg.train_cfg.max_iters == 4000
    assert cfg.train_dataloader.batch_size == 8
    assert cfg.default_hooks.checkpoint.save_best is None
    assert not any('Random' in p.type
                   for p in cfg.train_dataloader.dataset.pipeline)
    assert cfg.test_dataloader.dataset.ann_file == cfg.val_dataloader.dataset.ann_file


def test_fixed_crop_preserves_alignment_and_small_images():
    image = np.arange(7 * 9).reshape(7, 9)
    transform = WCMFixedCenterCrop((3, 5))
    data = dict(
        img=image[..., None].repeat(3, 2),
        gt_seg_map=image.copy(),
        seg_fields=['gt_seg_map'])
    a, b = transform(copy.deepcopy(data)), transform(copy.deepcopy(data))
    np.testing.assert_array_equal(a['img'], b['img'])
    np.testing.assert_array_equal(a['img'][..., 0], image[2:5, 2:7])
    np.testing.assert_array_equal(a['gt_seg_map'], image[2:5, 2:7])
    assert a['wcm_crop_bbox'] == (2, 2, 7, 5)
    small = WCMFixedCenterCrop((20, 20))(copy.deepcopy(data))
    np.testing.assert_array_equal(small['img'], data['img'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('mode', ['transformer', 'conv'])
def test_cuda_bfloat16_forward_backward(mode):
    m = _refiner(mode).cuda()
    _activate(m)
    classifier = nn.Conv2d(16, 1, 1).cuda().requires_grad_(False)
    x = torch.randn(2, 16, 7, 11, device='cuda', requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = m(x, classifier)
    out.float().square().mean().backward()
    assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


@pytest.mark.skipif(
    shutil.which('powershell') is None, reason='Windows PowerShell required')
def test_pilot_wrapper_exports_gpu_and_delegates_without_launching_training(
        tmp_path):
    wrapper = tmp_path / 'run_refiner_pilot.ps1'
    shutil.copyfile('projects/wavecrack_mamba/scripts/run_refiner_pilot.ps1',
                    wrapper)
    (tmp_path / 'run_train.ps1').write_text(
        "param([string]$Config,[switch]$Wsl,[string]$CudaVisibleDevices='')\n"
        "[pscustomobject]@{Config=$Config;Wsl=[bool]$Wsl;"
        "Gpu=$env:CUDA_VISIBLE_DEVICES;GpuArgument=$CudaVisibleDevices;"
        "Exports=$env:WSLENV;Sha=$env:WCM_REFINER_BASE_SHA256} | ConvertTo-Json -Compress\n"
        "exit 0\n",
        encoding='utf-8')
    checkpoint = tmp_path / 'base.pth'
    checkpoint.write_bytes(b'wrapper contract; never deserialized')
    args = [
        'powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        str(wrapper), '-Arm', 'T3', '-BaseCheckpoint',
        str(checkpoint), '-DataRoot',
        str(tmp_path), '-CudaVisibleDevices', '1'
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout.strip())
    assert record['Config'].endswith('wcm_fz_t3_crack_v5.py')
    assert record['Wsl'] and record['Gpu'] == '1' and record[
        'GpuArgument'] == ''
    assert 'CUDA_VISIBLE_DEVICES' in record['Exports'].split(':')
    assert 'WCM_REFINER_BASE_CHECKPOINT/p' in record['Exports'].split(':')
    assert record['Sha'].lower() == hashlib.sha256(
        checkpoint.read_bytes()).hexdigest()
    bad = subprocess.run(
        args + ['-BaseSha256', '0' * 64],
        capture_output=True,
        text=True,
        check=False)
    assert bad.returncode != 0 and 'SHA-256 mismatch' in bad.stderr
