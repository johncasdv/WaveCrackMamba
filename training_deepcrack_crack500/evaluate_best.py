"""Evaluate the validation-selected checkpoint on the held-out test split."""

import argparse
import json
import sys
from pathlib import Path

from run_pair import HERE, MODEL_ROOT, EXPECTED_SPLITS, audit_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('deepcrack', 'crack500'), required=True)
    parser.add_argument('--data-parent', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()

    directory = ('DeepCrack-MixerCSeg' if args.dataset == 'deepcrack'
                 else 'Crack500-MixerCSeg')
    data_root = args.data_parent / directory
    run_dir = args.run_root / f'{args.dataset}_seed42'
    manifest = json.loads((run_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed' or manifest['dataset'] != args.dataset:
        raise ValueError('This run has not completed for the requested dataset')
    if manifest['initialization'] != 'imagenet':
        raise ValueError('Unexpected initialization in run manifest')
    current_splits = audit_data(data_root)
    if current_splits != EXPECTED_SPLITS[args.dataset]:
        raise ValueError('Dataset split checksums do not match split_audit.json')
    if manifest['splits'] != current_splits:
        raise ValueError('Dataset split checksums changed since training')
    checkpoints = list(run_dir.glob('best_val_ODS_epoch_*.pth'))
    if len(checkpoints) != 1:
        raise ValueError(f'Expected one best validation ODS checkpoint: {checkpoints}')

    sys.path.insert(0, str(MODEL_ROOT))
    from mmengine import Config
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings

    cfg = Config.fromfile(str(HERE / 'configs' / f'{args.dataset}.py'))
    import_modules_from_strings(**cfg.custom_imports)
    cfg.test_dataloader.dataset.data_root = str(data_root)
    cfg.model.backbone.convnext.init_cfg = None
    cfg.load_from = str(checkpoints[0])
    cfg.work_dir = str(run_dir)
    metrics = Runner.from_cfg(cfg).test()
    result = dict(dataset=args.dataset, checkpoint=str(checkpoints[0]),
                  initialization=manifest['initialization'], metrics=metrics)
    output = run_dir / 'test_best_ods.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
