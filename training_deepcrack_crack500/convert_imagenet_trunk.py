"""Convert the official MMPretrain ConvNeXt-tiny ImageNet checkpoint for R1a."""

import argparse
import hashlib
from pathlib import Path

import torch
from mmpretrain.models import ConvNeXt


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.src, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    trunk = {}
    for name, value in state.items():
        if name.startswith('module.'):
            name = name[len('module.'):]
        if name.startswith('backbone.'):
            name = name[len('backbone.'):]
        elif not name.startswith(('downsample_layers.', 'stages.', 'norm')):
            continue
        if name.startswith(('downsample_layers.3.', 'stages.3.', 'norm3.')):
            continue
        if name in trunk:
            raise ValueError(f'duplicate trunk key: {name}')
        trunk[name] = value

    reference = ConvNeXt(
        arch=dict(depths=[3, 3, 9], channels=[96, 192, 384]),
        out_indices=(0, 1, 2), gap_before_final_norm=False)
    result = reference.load_state_dict(trunk, strict=False)
    allowed_missing = {
        f'norm{index}.{part}'
        for index in range(3) for part in ('weight', 'bias')
    }
    missing = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    if missing - allowed_missing or unexpected:
        raise ValueError(
            f'incompatible checkpoint: missing={sorted(missing)} '
            f'unexpected={sorted(unexpected)}')
    if len(trunk) < 100:
        raise ValueError(f'only {len(trunk)} tensors matched the trunk')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=trunk), args.out)
    print(f'source_sha256={digest(args.src)}')
    print(f'converted_sha256={digest(args.out)}')
    print(f'loaded_tensors={len(trunk)} expected_missing={sorted(missing)}')
    print(f'output={args.out}')


if __name__ == '__main__':
    main()
