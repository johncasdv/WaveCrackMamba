"""ODS parity, a complete 64-epoch schedule, and selected-base provenance."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmengine.optim.scheduler import LinearLR, CosineAnnealingLR

from projects.wavecrack_mamba.tools.hardcrack_ods import (
    BINS, HardCrackODSMetric, probability_histograms, summarize_histograms)
from projects.wavecrack_mamba.tools.hardcrack_protocol import (
    WCMHardCrackProtocolHook, planned_updates, validate_config)
from projects.wavecrack_mamba.tools.hardcrack_selection import (
    METRIC, RECEIPT, RULE, best_validation, complete_run,
    validate_selected_base, write_json)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cfg(arm='m'):
    return Config.fromfile(
        f'projects/wavecrack_mamba/configs/hardcrack/wcm_r1a_{arm}_512_b8_64ep.py',
        import_custom_modules=False)


@pytest.mark.parametrize('arm', ['nm', 'm'])
def test_64_epochs_have_one_consistent_iteration_clock(arm):
    config = cfg(arm)
    validate_config(config)
    assert config.train_cfg.max_epochs == 64
    assert config.train_cfg.val_interval == 2
    assert planned_updates(config) == 167936
    assert config.default_hooks.checkpoint.interval == 2
    assert config.default_hooks.checkpoint.save_best == METRIC
    assert config.val_evaluator[1].prefix == 'val'
    assert config.test_evaluator[1].prefix == 'test'
    assert config.param_scheduler[0].end == 1000
    assert config.param_scheduler[-1].T_max == 166936


def test_actual_lr_trace_does_not_end_at_the_old_five_epoch_boundary():
    config = cfg()
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=5e-4)
    schedulers = []
    for item in config.param_scheduler:
        args = dict(item)
        kind = args.pop('type')
        cls = LinearLR if kind == 'LinearLR' else CosineAnnealingLR
        schedulers.append(cls(optimizer, **args))
    observed = {}
    for iteration in range(1, 167937):
        if iteration in (1, 1000, 1001, 13120, 83968, 167936):
            observed[iteration] = optimizer.param_groups[0]['lr']
        for scheduler in schedulers:
            scheduler.step()
    assert observed[1] == pytest.approx(5e-7)
    assert observed[1000] == pytest.approx(5e-4)
    assert observed[1001] == pytest.approx(5e-4)
    assert observed[13120] > 4e-4
    assert observed[83968] > observed[167936]
    assert observed[167936] == pytest.approx(1e-6, abs=1e-12)
    assert parameter.item() == 0


def test_batch_or_accumulation_changes_require_schedule_recalibration():
    config = cfg()
    config.train_dataloader.batch_size = 4
    with pytest.raises(ValueError, match='disagree'):
        validate_config(config)
    config = cfg()
    config.optim_wrapper.accumulative_counts = 4
    with pytest.raises(ValueError, match='gradient accumulation'):
        validate_config(config)


def test_ods_matches_existing_offline_histogram_definition():
    source = Path('projects/crack_baselines/eval_crack_v5_domain_metrics.py')
    spec = importlib.util.spec_from_file_location('offline_ods_reference',
                                                  source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    assert reference.BINS == BINS
    p = np.array([0, 0.0009, 0.5, 0.5001, 0.99, 1], dtype=np.float32)
    gt = np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8)
    pos, neg = probability_histograms(p, gt)
    f1, _, _ = reference.f1_curve(pos, neg)
    result = summarize_histograms(pos, neg)
    assert result['ODS'] == pytest.approx(float(f1.max() * 100), abs=1e-12)
    assert result['ODS_threshold'] == np.argmax(f1) / BINS


def sample(p, gt):
    return dict(
        seg_logits=dict(data=torch.tensor(p).float()[None, None]),
        gt_sem_seg=dict(data=torch.tensor(gt)[None, None]))


def test_metric_sums_pixels_before_sweep_instead_of_averaging_image_f1():
    metric = HardCrackODSMetric(prefix='val')
    metric.process(
        None,
        [sample([0.9, 0.1], [1, 0]),
         sample([0.1] * 100, [1] * 50 + [0] * 50)])
    result = metric.compute_metrics(metric.results)
    assert result['num_images'] == 2
    combined_p = np.array([0.9, 0.1] + [0.1] * 100, np.float32)
    combined_gt = np.array([1, 0] + [1] * 50 + [0] * 50)
    expected = summarize_histograms(
        *probability_histograms(combined_p, combined_gt))
    assert result['ODS'] == pytest.approx(expected['ODS'])
    per_image = [summarize_histograms(*x)['ODS'] for x in metric.results]
    assert result['ODS'] != pytest.approx(np.mean(per_image))


def test_ods_ignores_padding_and_rejects_raw_logits():
    pos, neg = probability_histograms([0.9, 0.8, 0.1], [1, 255, 0])
    assert pos.sum() == neg.sum() == 1
    assert summarize_histograms(pos, neg)['ODS'] == 100
    with pytest.raises(ValueError, match='probabilities'):
        probability_histograms([2.0], [1])
    with pytest.raises(ValueError, match='binary'):
        probability_histograms([0.5], [2])


def completed_fixture(tmp_path, smoke=False):
    identity = dict(dataset='HardCrack-AC', files={'split': 'registered'})
    history = [
        dict(updates=2, score=80.0, threshold=0.4),
        dict(updates=4, score=75.0, threshold=0.5)
    ]
    info = dict(
        run_id='one-run',
        arm='m',
        seed=42,
        smoke=smoke,
        data=identity,
        planned_updates=4,
        checkpoint_rule=RULE,
        validation_history=history)
    selected, final = tmp_path / 'best_val_ODS_epoch_1.pth', tmp_path / 'epoch_2.pth'
    for path, row, is_final in [(selected, history[0], False),
                                (final, history[1], True)]:
        cp_info = dict(
            info,
            updates=row['updates'],
            last_validation=row,
            is_final=is_final)
        torch.save(
            dict(
                state_dict={'weight': torch.ones(1)},
                meta=dict(hardcrack_run=cp_info)), path)
    receipt = complete_run(info, selected, final, 4)
    write_json(tmp_path / RECEIPT, receipt)
    return identity, selected, final, receipt


def test_early_best_is_admitted_only_after_the_run_finishes(tmp_path):
    identity, selected, final, receipt = completed_fixture(tmp_path)
    info = validate_selected_base(selected, sha(selected), identity)
    assert info['updates'] == 2 and not info['is_final']
    assert info['selected_metric'] == METRIC
    with pytest.raises(ValueError, match='validation-best'):
        validate_selected_base(final, sha(final), identity)
    receipt['status'] = 'running'
    write_json(tmp_path / RECEIPT, receipt)
    with pytest.raises(ValueError, match='completed matching'):
        validate_selected_base(selected, sha(selected), identity)


@pytest.mark.parametrize('field,value', [
    ('split', 'test'),
    ('metric', 'fgIoU'),
    ('sha256', '0' * 64),
    ('score', 90.0),
    ('updates', 4),
    ('threshold', 0.7),
])
def test_receipt_cannot_substitute_selection_identity(tmp_path, field, value):
    identity, selected, _, receipt = completed_fixture(tmp_path)
    receipt['selection'][field] = value
    write_json(tmp_path / RECEIPT, receipt)
    with pytest.raises(ValueError, match='validation-best'):
        validate_selected_base(selected, sha(selected), identity)


def test_final_checkpoint_and_run_identity_are_also_checked(tmp_path):
    identity, selected, final, _ = completed_fixture(tmp_path)
    final.write_bytes(b'changed final file')
    with pytest.raises(ValueError, match='SHA-256'):
        validate_selected_base(selected, sha(selected), identity)


def test_smoke_completion_does_not_qualify_for_formal_training(tmp_path):
    identity, selected, _, _ = completed_fixture(tmp_path, smoke=True)
    with pytest.raises(ValueError, match='smoke checkpoint'):
        validate_selected_base(selected, sha(selected), identity)
    assert validate_selected_base(
        selected, sha(selected), identity, allow_smoke=True)['smoke']


def test_tied_validation_scores_keep_the_earlier_checkpoint():
    history = [
        dict(updates=2, score=80.0, threshold=0.5),
        dict(updates=4, score=80.0, threshold=0.5)
    ]
    assert best_validation(history)['updates'] == 2


@pytest.mark.parametrize('iteration', [6, 12])
def test_epoch_checkpoint_metadata_uses_completed_iterations(iteration):
    hook = WCMHardCrackProtocolHook()
    hook.info = dict(run_id='test', planned_updates=12)
    runner = SimpleNamespace(iter=iteration, max_iters=12)
    checkpoint = dict(meta=dict(iter=iteration))
    hook.before_save_checkpoint(runner, checkpoint)
    info = checkpoint['meta']['hardcrack_run']
    assert info['updates'] == iteration
    assert info['is_final'] == (iteration == 12)


def test_interrupted_run_never_receives_complete_status(tmp_path):
    hook = WCMHardCrackProtocolHook()
    hook.info = dict(checkpoint_rule=RULE, run_id='interrupted')
    hook.after_train(SimpleNamespace(iter=3, max_iters=4, work_dir=tmp_path))
    assert json.loads(
        (tmp_path / RECEIPT).read_text())['status'] == 'incomplete'


def test_completion_writer_rejects_wrong_saved_selection(tmp_path):
    _, selected, final, receipt = completed_fixture(tmp_path)
    raw = torch.load(selected, weights_only=False)
    raw['meta']['hardcrack_run']['last_validation']['score'] = 1.0
    torch.save(raw, selected)
    info = {
        k: receipt[k]
        for k in ('run_id', 'arm', 'seed', 'smoke', 'data', 'planned_updates',
                  'checkpoint_rule', 'validation_history')
    }
    with pytest.raises(ValueError, match='do not prove'):
        complete_run(info, selected, final, 4)


def test_pure_evaluation_can_disable_checkpoint_writes(monkeypatch, tmp_path):
    from projects.wavecrack_mamba.tools import hardcrack_protocol as protocol
    config = cfg()
    config.hardcrack_evaluation_only = True
    config.default_hooks.checkpoint = None
    monkeypatch.setattr(protocol, 'release_identity', lambda root: {})
    runner = SimpleNamespace(
        cfg=config,
        model=object(),
        _resume=False,
        _load_from='evaluation.pth',
        work_dir=tmp_path)
    hook = WCMHardCrackProtocolHook()
    hook.before_run(runner)
    with pytest.raises(ValueError, match='evaluation-only'):
        hook.before_train(runner)
