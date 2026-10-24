"""Run DeepCrack and Crack500 sequentially on one GPU."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--initialization', choices=('dinov3', 'imagenet'),
                        required=True)
    parser.add_argument('--checkpoint-prefix', default='')
    parser.add_argument('--data-parent', type=Path)
    parser.add_argument('--run-root', type=Path,
                        default=Path('runs'))
    parser.add_argument('--microbatch', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    script = Path(__file__).with_name('run_pair.py')
    common = [
        '--pretrained', str(args.pretrained),
        '--initialization', args.initialization,
        '--run-root', str(args.run_root),
        '--microbatch', str(args.microbatch),
        '--accumulation', str(args.accumulation),
    ]
    if args.checkpoint_prefix:
        common += ['--checkpoint-prefix', args.checkpoint_prefix]
    if args.smoke:
        common += ['--smoke']
    for dataset in ('deepcrack', 'crack500'):
        command = [sys.executable, '-u', str(script),
                   '--dataset', dataset, *common]
        if args.data_parent:
            directory = ('DeepCrack-MixerCSeg' if dataset == 'deepcrack'
                         else 'Crack500-MixerCSeg')
            command += ['--data-root', str(args.data_parent / directory)]
        print(f'Starting {dataset}', flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
