"""Current HardCrack entrypoints, provenance rejection and paired initial states."""
import copy
import hashlib

import pytest
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

import mmpretrain.models  # noqa: F401
import projects.wavecrack_mamba  # noqa: F401
from mmseg.registry import MODELS
from projects.wavecrack_mamba.tools.hardcrack_protocol import (
    initialize_common, initialize_refiner, release_identity, validate_base,
    validate_config)
from projects.wavecrack_mamba.tools.prepare_hardcrack_run import CONFIGS


def config(arm):
    return Config.fromfile(
        'projects/wavecrack_mamba/configs/hardcrack/' + CONFIGS[arm],
        import_custom_modules=False)


@pytest.mark.parametrize('arm', CONFIGS)
def test_resolved_entrypoints_preserve_current_data_and_comparison(arm):
    cfg = config(arm)
    validate_config(cfg)
    assert cfg.model.decode_head.upsample == 'fixed'
    assert cfg.model.decode_head.channels == 128
    assert cfg.train_dataloader.dataset.ann_file == 'splits/train_tiles.txt'
    assert cfg.train_dataloader.batch_size == 8
    assert cfg.randomness.seed == 42
    assert 'crack_v5' not in cfg.pretty_text
    if arm in ('nm', 'm'):
        assert cfg.train_cfg.max_epochs == 64
        assert 'max_iters' not in cfg.train_cfg
    else:
        assert cfg.train_cfg.max_iters == 4000
        assert 'max_epochs' not in cfg.train_cfg
        assert cfg.train_dataloader.sampler.type == 'InfiniteSampler'
        assert cfg.test_dataloader.dataset.ann_file == 'splits/val.txt'
        assert cfg.default_hooks.checkpoint.save_best == 'val/ODS'


@pytest.mark.parametrize('change',
                         ['whole-list', 'resize', 'subset', 'mixed-root'])
def test_rejects_silent_training_view_changes(change):
    cfg = config('nm')
    if change == 'whole-list':
        cfg.train_dataloader.dataset.ann_file = 'splits/train.txt'
    elif change == 'resize':
        cfg.train_dataloader.dataset.pipeline.insert(3, dict(type='Resize'))
    elif change == 'subset':
        cfg.train_dataloader.dataset.indices = [0]
    else:
        cfg.val_dataloader.dataset.data_root = 'data/another_dataset'
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_release_hash_rejects_changed_split(tmp_path):
    (tmp_path / 'split.txt').write_bytes(b'a b\n')
    expected = {'split.txt': hashlib.sha256(b'a b\n').hexdigest()}
    assert release_identity(tmp_path, expected)['files'] == expected
    (tmp_path / 'split.txt').write_bytes(b'c d\n')
    with pytest.raises(ValueError, match='snapshot mismatch'):
        release_identity(tmp_path, expected)


def checkpoint(tmp_path, info):
    path = tmp_path / 'base.pth'
    torch.save(dict(meta=dict(hardcrack_run=info)), path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_old_final_flag_without_completion_receipt_is_not_admitted(tmp_path):
    identity = dict(dataset='HardCrack-AC', files={'train': 'current'})
    info = dict(arm='m', data=identity, seed=42, is_final=True, smoke=False)
    path, digest = checkpoint(tmp_path, info)
    with pytest.raises(ValueError, match='completed_run.json'):
        validate_base(path, digest, identity)


def test_common_head_initialization_is_independent_of_construction_rng():
    init_default_scope('mmseg')
    cfg = config('nm').model
    cfg.backbone.convnext.init_cfg = None
    torch.manual_seed(123)
    first = MODELS.build(copy.deepcopy(cfg))
    torch.manual_seed(456)
    second = MODELS.build(copy.deepcopy(cfg))
    a = initialize_common(first, cfg, 42)
    b = initialize_common(second, cfg, 42)
    assert a == b
    for name, value in first.state_dict().items():
        torch.testing.assert_close(
            value, second.state_dict()[name], rtol=0, atol=0)


def test_t1_t3_receive_identical_initial_tensors():
    from types import SimpleNamespace
    init_default_scope('mmseg')
    configs = [config(arm).model.decode_head.refiner for arm in ('t1', 't3')]
    models = [
        SimpleNamespace(decode_head=SimpleNamespace(refiner=MODELS.build(c)))
        for c in configs
    ]
    a = initialize_refiner(models[0], configs[0], 42)
    b = initialize_refiner(models[1], configs[1], 42)
    assert a == b


@pytest.mark.parametrize('arm', ['nm', 't3'])
def test_standalone_evaluation_does_not_require_original_base(
        arm, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from projects.wavecrack_mamba.tools import hardcrack_protocol as protocol
    identity = dict(dataset='HardCrack-AC', files={'train': 'current'})
    monkeypatch.setattr(protocol, 'release_identity', lambda root: identity)
    runner = SimpleNamespace(
        cfg=config(arm),
        _load_from='full.pth',
        _resume=False,
        model=object(),
        work_dir=tmp_path)
    hook = protocol.WCMHardCrackProtocolHook()
    hook.before_run(runner)
    with pytest.raises(ValueError, match='matching loaded checkpoint'):
        hook.before_test(runner)
    saved = dict(
        arm=arm, seed=42, smoke=False, data=identity, slide_mode='count')
    hook.after_load_checkpoint(runner, dict(meta=dict(hardcrack_run=saved)))
    hook.before_test(runner)
    hook.before_val(runner)
    with pytest.raises(ValueError, match='Fresh training'):
        hook.before_train(runner)


def test_evaluation_rejects_unrecorded_old_checkpoint(monkeypatch):
    from types import SimpleNamespace
    from projects.wavecrack_mamba.tools import hardcrack_protocol as protocol
    monkeypatch.setattr(protocol, 'release_identity', lambda root: {})
    hook = protocol.WCMHardCrackProtocolHook()
    runner = SimpleNamespace(cfg=config('nm'))
    hook.before_run(runner)
    with pytest.raises(ValueError, match='metadata mismatch'):
        hook.after_load_checkpoint(runner, dict(meta={}))


def test_pinned_evaluation_removes_training_pickle_objects(tmp_path):
    from mmengine.logging import HistoryBuffer
    from projects.wavecrack_mamba.tools.evaluate_hardcrack import prepare_eval_checkpoint
    source = tmp_path / 'training.pth'
    destination = tmp_path / 'evaluation.pth'
    torch.save(
        dict(
            state_dict={'weight': torch.ones(2)},
            meta=dict(hardcrack_run={
                'arm': 'm',
                'smoke': True
            }),
            message_hub=HistoryBuffer()), source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='SHA-256'):
        prepare_eval_checkpoint(source, '0' * 64, destination)
    assert not destination.exists()
    prepare_eval_checkpoint(source, digest, destination)
    loaded = torch.load(destination, weights_only=True)
    assert set(loaded) == {'meta', 'state_dict'}
    torch.testing.assert_close(loaded['state_dict']['weight'], torch.ones(2))
