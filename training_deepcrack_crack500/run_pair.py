"""Train the formal R1a m arm on one public crack dataset."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODEL_ROOT = (HERE.parent / 'wavecrack_mamba_R1a_code_20261022'
              / 'wavecrack_mamba_R1a_code_20261022')
DATASETS = ('deepcrack', 'crack500')
EXPECTED_SPLITS = json.loads(
    (HERE / 'split_audit.json').read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_data(data_root):
    report = {}
    split_images = {}
    for split in ('train', 'val', 'test'):
        index = data_root / 'splits' / f'{split}.txt'
        if not index.is_file():
            raise FileNotFoundError(index)
        images = set()
        lines = index.read_text(encoding='utf-8').splitlines()
        for number, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f'{index}:{number}: expected image and mask')
            image, mask = (data_root / part for part in parts)
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f'{index}:{number}: {image} / {mask}')
            if image in images:
                raise ValueError(f'{index}:{number}: duplicate image {image}')
            images.add(image)
        report[split] = dict(images=len(images), index_sha256=sha256(index))
        split_images[split] = images
    for first, second in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = split_images[first] & split_images[second]
        if overlap:
            raise ValueError(f'{first}/{second} image overlap: {next(iter(overlap))}')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--initialization', choices=('dinov3', 'imagenet'),
                        required=True)
    parser.add_argument('--checkpoint-prefix', default='')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--run-root', type=Path,
                        default=Path('runs'))
    parser.add_argument('--microbatch', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()

    if not MODEL_ROOT.is_dir():
        raise FileNotFoundError(MODEL_ROOT)
    if not args.pretrained.is_file():
        raise FileNotFoundError(args.pretrained)
    if args.initialization == 'dinov3' and args.checkpoint_prefix:
        raise ValueError('Converted DINOv3 weights do not use a prefix')
    if args.microbatch < 1 or args.accumulation < 1:
        raise ValueError('microbatch and accumulation must be positive')
    if args.microbatch * args.accumulation != 8:
        raise ValueError('this experiment requires effective batch size 8')

    sys.path.insert(0, str(MODEL_ROOT))
    from mmengine import Config
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings

    cfg = Config.fromfile(str(HERE / 'configs' / f'{args.dataset}.py'))
    import_modules_from_strings(**cfg.custom_imports)

    data_root = args.data_root or Path(cfg.train_dataloader.dataset.data_root)
    for split in ('train', 'val', 'test'):
        cfg[f'{split}_dataloader'].dataset.data_root = str(data_root)
    cfg.train_dataloader.batch_size = args.microbatch
    cfg.optim_wrapper.accumulative_counts = args.accumulation
    data_audit = audit_data(data_root)
    if data_audit != EXPECTED_SPLITS[args.dataset]:
        raise ValueError(
            f'{args.dataset} split counts or SHA-256 checksums do not match '
            'the recorded experiment; see split_audit.json')
    weight_hash = sha256(args.pretrained)
    cfg.model.backbone.convnext.init_cfg = dict(
        type='Pretrained', checkpoint=str(args.pretrained.resolve()))
    if args.checkpoint_prefix:
        cfg.model.backbone.convnext.init_cfg.prefix = args.checkpoint_prefix

    suffix = '_smoke' if args.smoke else ''
    run_dir = args.run_root / f'{args.dataset}_seed42{suffix}'
    cfg.work_dir = str(run_dir)
    if args.smoke:
        cfg.train_dataloader.dataset.indices = list(range(8))
        cfg.val_dataloader.dataset.indices = list(range(2))
        cfg.train_dataloader.num_workers = 0
        cfg.val_dataloader.num_workers = 0
        cfg.train_cfg.max_epochs = 1
        cfg.train_cfg.val_interval = 1
        cfg.param_scheduler = []
        cfg.default_hooks.checkpoint = None
    if args.resume:
        cfg.resume = True
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'{run_dir} already contains a run; use --resume')

    manifest = dict(
        dataset=args.dataset,
        model='WaveCrackMamba R1a m',
        backend='mamba_ssm.Mamba',
        initialization=args.initialization,
        pretrained=str(args.pretrained.resolve()),
        pretrained_sha256=weight_hash,
        checkpoint_prefix=args.checkpoint_prefix,
        data_root=str(data_root),
        splits=data_audit,
        config=str(HERE / 'configs' / f'{args.dataset}.py'),
        microbatch=args.microbatch,
        accumulation=args.accumulation,
        effective_batch_size=args.microbatch * args.accumulation,
        smoke=args.smoke,
        created_utc=datetime.now(timezone.utc).isoformat(),
        status='preflight' if args.preflight_only else 'started')
    print(json.dumps(manifest, indent=2), flush=True)
    if args.preflight_only:
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = run_dir / 'run_manifest.json'
    manifest_file.write_text(json.dumps(manifest, indent=2) + '\n',
                             encoding='utf-8')
    try:
        Runner.from_cfg(cfg).train()
    except BaseException:
        manifest['status'] = 'failed_or_interrupted'
        manifest_file.write_text(json.dumps(manifest, indent=2) + '\n',
                                 encoding='utf-8')
        raise
    manifest['status'] = 'completed'
    manifest_file.write_text(json.dumps(manifest, indent=2) + '\n',
                             encoding='utf-8')


if __name__ == '__main__':
    main()
