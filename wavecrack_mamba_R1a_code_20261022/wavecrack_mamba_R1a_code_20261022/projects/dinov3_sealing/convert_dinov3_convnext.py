"""Convert official DINOv3 ConvNeXt SSL weights to mmpretrain.ConvNeXt format.

DINOv3 ConvNeXt checkpoints use the original Facebook ConvNeXt naming
(``dwconv``/``pwconv1``/``pwconv2`` inside each block, a single ``norms.3`` output
LayerNorm and a final classification ``norm``).  mmpretrain's ``ConvNeXt`` renames
the block sub-modules (``depthwise_conv``/``pointwise_conv1``/``pointwise_conv2``)
and, with ``gap_before_final_norm=False`` + ``out_indices=[0,1,2,3]``, expects one
output norm per stage (``norm0..norm3``).

Mapping (verified by tensor count, see module docstring in the project):
  downsample_layers.*            -> identical
  stages.X.Y.dwconv.*            -> stages.X.Y.depthwise_conv.*
  stages.X.Y.pwconv1.*           -> stages.X.Y.pointwise_conv1.*
  stages.X.Y.pwconv2.*           -> stages.X.Y.pointwise_conv2.*
  stages.X.Y.{norm,gamma}        -> identical
  norms.3.*                      -> norm3.*   (stage-3 output norm)
  norm.*                         -> dropped   (final cls norm, unused for features)
  norm0/1/2                      -> absent in DINOv3, left at mmpretrain init and
                                    learned during fine-tuning.
The converted file is saved as ``{'state_dict': <backbone params, no prefix>}`` so it
loads via ``init_cfg=dict(type='Pretrained', checkpoint=...)``.
"""
import argparse
import re
from collections import OrderedDict

import torch


def _load_state_dict(src_path):
    """Load either an original Facebook .pth or a Hugging Face .safetensors."""
    if src_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(src_path), 'hf'
    ck = torch.load(src_path, map_location='cpu')
    if isinstance(ck, dict) and 'state_dict' in ck:
        ck = ck['state_dict']
    if isinstance(ck, dict) and 'model' in ck:
        ck = ck['model']
    return ck, 'fb'


def _convert_fb(ck):
    """Original facebookresearch DINOv3 ConvNeXt naming -> mmpretrain."""
    out, dropped = OrderedDict(), []
    for k, v in ck.items():
        if k.startswith('downsample_layers.'):
            out[k] = v
        elif k.startswith('stages.'):
            nk = (k.replace('.dwconv.', '.depthwise_conv.')
                   .replace('.pwconv1.', '.pointwise_conv1.')
                   .replace('.pwconv2.', '.pointwise_conv2.'))
            out[nk] = v
        elif re.match(r'norms\.\d+\.(weight|bias)$', k):
            out[re.sub(r'norms\.(\d+)', r'norm\1', k)] = v
        elif k in ('norm.weight', 'norm.bias'):
            dropped.append(k)  # final classification norm, unused for features
        else:
            dropped.append(k)
    return out, dropped


def _convert_hf(sd):
    """Hugging Face DINOv3ConvNextModel naming -> mmpretrain.

    HF nests downsampling inside each stage (``stages.S.downsample_layers.*``),
    names blocks ``stages.S.layers.B.*`` (with ``layer_norm``), and keeps one
    top-level final ``layer_norm``.
    """
    out, dropped = OrderedDict(), []
    for k, v in sd.items():
        m = re.match(r'stages\.(\d+)\.downsample_layers\.(\d+)\.(weight|bias)$', k)
        if m:
            s, x, p = m.groups()
            out[f'downsample_layers.{s}.{x}.{p}'] = v
            continue
        m = re.match(r'stages\.(\d+)\.layers\.(\d+)\.(.+)$', k)
        if m:
            s, b, rest = m.groups()
            rest = rest.replace('layer_norm', 'norm')
            out[f'stages.{s}.{b}.{rest}'] = v
            continue
        if k in ('layer_norm.weight', 'layer_norm.bias'):
            out[k.replace('layer_norm', 'norm3')] = v
            continue
        dropped.append(k)
    return out, dropped


def convert(src_path):
    sd, fmt = _load_state_dict(src_path)
    out, dropped = _convert_hf(sd) if fmt == 'hf' else _convert_fb(sd)
    print(f'detected format    : {fmt}')
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--arch', default='base')
    args = ap.parse_args()

    out, dropped = convert(args.src)
    print(f'converted tensors : {len(out)}')
    print(f'dropped (unused)  : {dropped}')

    # Strict-ish verification against a fresh mmpretrain backbone.
    from mmpretrain.models import ConvNeXt
    m = ConvNeXt(arch=args.arch, out_indices=[0, 1, 2, 3],
                 drop_path_rate=0.0, layer_scale_init_value=1.0,
                 gap_before_final_norm=False)
    res = m.load_state_dict(out, strict=False)
    missing = list(res.missing_keys)
    unexpected = list(res.unexpected_keys)
    print(f'missing keys      : {missing}')
    print(f'unexpected keys   : {unexpected}')

    allowed_missing = {f'norm{i}.{s}' for i in (0, 1, 2) for s in ('weight', 'bias')}
    assert set(missing).issubset(allowed_missing), \
        f'unexpected MISSING keys beyond norm0/1/2: {set(missing) - allowed_missing}'
    assert not unexpected, f'unexpected keys present: {unexpected}'

    # sanity forward
    m.eval()
    with torch.no_grad():
        feats = m(torch.randn(1, 3, 512, 512))
    print('feature pyramid shapes:', [tuple(f.shape) for f in feats])

    torch.save({'state_dict': out}, args.dst)
    print(f'saved -> {args.dst}')


if __name__ == '__main__':
    main()
