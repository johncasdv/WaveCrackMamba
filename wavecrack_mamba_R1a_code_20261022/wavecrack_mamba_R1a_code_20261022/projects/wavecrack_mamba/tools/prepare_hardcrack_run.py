"""Resolve the five HardCrack configurations and their bounded smoke phase."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path

from mmengine.config import Config

from projects.wavecrack_mamba.tools.hardcrack_protocol import (
    planned_updates, release_identity, validate_base, validate_config)

CONFIGS = {
    'nm': 'wcm_r1a_nm_512_b8_64ep.py',
    'm': 'wcm_r1a_m_512_b8_64ep.py',
    't1': 'wcm_r1a_fz_t1_512_b8_4k.py',
    't3': 'wcm_r1a_fz_t3_512_b8_4k.py',
    'conv3': 'wcm_r1a_fz_conv3_512_b8_4k.py',
}


def smoke_indices(root):
    tiles = [
        json.loads(line)
        for line in (root /
                     'metadata/train_manifest.jsonl').read_text().splitlines()
    ]
    selected = {}
    for i, row in enumerate(tiles):
        key = (row['source'], int(row['fg_px']) == 0)
        bucket = selected.setdefault(key, [])
        if len(bucket) < 8:
            bucket.append(i)
    indices = sorted(i for bucket in selected.values() for i in bucket)
    with (root /
          'splits/image_split.csv').open(encoding='utf-8-sig') as handle:
        rows = {r['id']: r for r in csv.DictReader(handle)}
    val = (root / 'splits/val.txt').read_text().splitlines()
    picked = {}
    for i, line in enumerate(val):
        r = rows[Path(line.split()[0]).stem]
        picked.setdefault((r['source'], int(r['fg_px']) == 0), i)
    return indices, sorted(picked.values())


def prepare(arm, seed, tag, smoke_only=False, dry_run=False):
    root = Path(os.environ['WCM_HARDCRACK_DATA_ROOT']).resolve()
    cfg_path = Path(
        'projects/wavecrack_mamba/configs/hardcrack') / CONFIGS[arm]
    cfg = Config.fromfile(str(cfg_path), import_custom_modules=False)
    cfg.randomness.seed = seed
    work = Path(
        'work_dirs/wavecrack_mamba/hardcrack_r1a') / f'{arm}_seed{seed}_{tag}'
    cfg.work_dir = str(work)
    validate_config(cfg)
    identity = release_identity(root)
    if arm in ('t1', 't3', 'conv3'):
        hook = next(h for h in cfg.custom_hooks
                    if h.type == 'WCMFrozenRefinerHook')
        validate_base(
            hook.checkpoint,
            hook.expected_sha256,
            identity,
            allow_smoke=smoke_only)
    if dry_run:
        return dict(
            arm=arm,
            seed=seed,
            data=identity,
            config=str(cfg_path),
            planned_updates=planned_updates(cfg),
            checkpoint_rule=cfg.hardcrack_checkpoint_rule,
            training_started=False)
    if list(work.glob('*.pth')):
        raise FileExistsError(
            'Run directory has checkpoints; choose a new RunTag')
    train_ids, val_ids = smoke_indices(root)
    smoke = copy.deepcopy(cfg)
    smoke.work_dir = str(work / '_smoke')
    if list(Path(smoke.work_dir).glob('*.pth')):
        raise FileExistsError(
            'Smoke already has checkpoints; choose a new RunTag')
    smoke.hardcrack_smoke = True
    epoch_smoke = arm in ('nm', 'm')
    smoke.train_cfg = (
        dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=1)
        if epoch_smoke else dict(
            type='IterBasedTrainLoop', max_iters=20, val_interval=10))
    smoke.train_dataloader.sampler = dict(
        type='DefaultSampler' if epoch_smoke else 'InfiniteSampler',
        shuffle=True)
    smoke.train_dataloader.batch_size = 8 if epoch_smoke else 1
    smoke.train_dataloader.dataset.indices = train_ids
    smoke.val_dataloader.dataset.indices = val_ids
    smoke.test_dataloader = copy.deepcopy(smoke.val_dataloader)
    # Baseline test must retain its declared test role, though smoke never calls it.
    if arm in ('nm', 'm'):
        smoke.test_dataloader = copy.deepcopy(cfg.test_dataloader)
    for phase in ('train', 'val', 'test'):
        smoke[f'{phase}_dataloader'].num_workers = 0
        smoke[f'{phase}_dataloader'].persistent_workers = False
    smoke.param_scheduler = []
    smoke.default_hooks.checkpoint = dict(
        type='CheckpointHook',
        by_epoch=epoch_smoke,
        interval=1 if epoch_smoke else 10,
        save_best='val/ODS',
        rule='greater',
        save_last=True,
        max_keep_ckpts=2)
    smoke.log_processor.by_epoch = epoch_smoke
    for path, item in [(work / 'resolved.py', cfg),
                       (work / '_smoke/resolved.py', smoke)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        item.dump(str(path))
    (work / 'preflight.json').write_text(
        json.dumps(
            dict(
                data=identity,
                smoke_train_indices=train_ids,
                smoke_val_indices=val_ids,
                training_started=False),
            indent=2),
        encoding='utf8')
    return dict(
        full=str(work / 'resolved.py'),
        smoke=str(work / '_smoke/resolved.py'),
        training_started=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=CONFIGS, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tag', default='v1')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.tag.replace('-', '').replace('_', '').isalnum():
        parser.error(
            'tag must contain only letters, numbers, hyphens and underscores')
    print(
        json.dumps(
            prepare(args.arm, args.seed, args.tag, args.smoke_only,
                    args.dry_run)))
