"""M0-2: convert DINOv3 ConvNeXt-tiny SSL weights into a 3-stage WCM trunk
checkpoint (mmpretrain naming, stage-3 dropped).

The WCM trunk is the ConvNeXt-tiny stage-1..3 prefix (96/192/384 @ 1/4·1/8·1/16,
Spec §8.3); stage-3 (768@1/32) is not part of the trunk, so its weights and the
stage-3 output norm are dropped. The result loads into
``ConvNeXt(arch=dict(depths=[3,3,9], channels=[96,192,384]), out_indices=(0,1,2))``
with only ``norm0/1/2`` missing (DINOv3 has no per-stage output norms) and nothing
unexpected -- exactly the registered white-list of Spec §8.3 / 实施计划 M0-2.

Usage (from repo root):
    python projects/wavecrack_mamba/tools/convert_trunk_weights.py \
        --src pretrain/DINOv3_weights/backbones/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth \
        --out pretrain/wcm_convnext_tiny_dinov3_mmpretrain.pth
"""
import argparse
import hashlib
import os.path as osp
import sys

import torch

# Allow running as a plain script (`python projects/.../convert_trunk_weights.py`):
# put the repo root (three levels up) on sys.path so `projects.*` resolves.
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..')))

from projects.dinov3_sealing.convert_dinov3_convnext import convert  # noqa: E402

# 3-stage prefixes @ 1/4,1/8,1/16 (Spec §8.3). stage-3 is dropped in every case.
# small shares tiny's channels and only deepens stage-2, so it swaps into the WCM
# decoder unchanged; base/large also widen, and their arms must additionally set
# decode_head.trunk_channels.
TRUNK_ARCH = {
    'tiny': dict(depths=[3, 3, 9], channels=[96, 192, 384]),
    'small': dict(depths=[3, 3, 27], channels=[96, 192, 384]),
    'base': dict(depths=[3, 3, 27], channels=[128, 256, 512]),
    'large': dict(depths=[3, 3, 27], channels=[192, 384, 768]),
}
ALLOWED_MISSING = {f'norm{i}.{s}' for i in (0, 1, 2) for s in ('weight', 'bias')}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _drop_stage3(state):
    kept, dropped = {}, []
    for k, v in state.items():
        if (k.startswith('downsample_layers.3.') or k.startswith('stages.3.')
                or k.startswith('norm3.')):
            dropped.append(k)
        else:
            kept[k] = v
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--arch', default='tiny', choices=sorted(TRUNK_ARCH),
                    help='which ConvNeXt the source weights are (default tiny)')
    args = ap.parse_args()
    arch = TRUNK_ARCH[args.arch]

    full, dropped_cls = convert(args.src)
    trunk, dropped_s3 = _drop_stage3(full)
    print(f'converted tensors  : {len(full)} ; kept 3-stage prefix: {len(trunk)}')
    print(f'dropped by convert : {dropped_cls}')
    print(f'dropped stage-3    : {len(dropped_s3)} keys')

    from mmpretrain.models import ConvNeXt
    ref = ConvNeXt(arch=arch, out_indices=(0, 1, 2),
                   gap_before_final_norm=False)
    res = ref.load_state_dict(trunk, strict=False)
    missing = set(res.missing_keys)
    assert missing.issubset(ALLOWED_MISSING), \
        f'unexpected MISSING beyond norm0/1/2: {missing - ALLOWED_MISSING}'
    assert not res.unexpected_keys, f'unexpected keys: {res.unexpected_keys}'
    print(f'missing (expected norm0/1/2): {sorted(missing)}')

    torch.save({'state_dict': trunk}, args.out)
    print(f'src sha256: {_sha256(args.src)}')
    print(f'out sha256: {_sha256(args.out)}')
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
