"""Custom dataset + transform for the binary crack/sealing segmentation task.

The on-disk masks are single-channel uint8 with mixed encodings: legacy sources
use ``{0, 1}``, while newer augmented sources use ``{0, 255}``. ``BinarizeSeg``
maps any non-zero foreground value to label ``1`` before it reaches the loss;
``OpenCrackseg`` is a thin ``BaseSegDataset`` with the 2-class metainfo and the
``.jpg`` image / ``.png`` mask suffixes used by this dataset.
"""
import re

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmengine.optim import DefaultOptimWrapperConstructor

from mmseg.datasets import BaseSegDataset
from mmseg.evaluation import IoUMetric
from mmseg.registry import (DATASETS, METRICS, MODELS,
                            OPTIM_WRAPPER_CONSTRUCTORS, TRANSFORMS)


@DATASETS.register_module()
class OpenCrackseg(BaseSegDataset):
    """Binary crack/sealing dataset (images/*.jpg, masks/*.png, splits/*.txt)."""

    METAINFO = dict(
        classes=('background', 'sealing'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self, img_suffix='.jpg', seg_map_suffix='.png',
                 reduce_zero_label=False, **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@TRANSFORMS.register_module()
class BinarizeSeg(BaseTransform):
    """Map a ``{0, fg}`` mask to ``{0, 1}`` labels by thresholding.

    Args:
        thr (int): pixels with value > ``thr`` become foreground label 1.
            Default 0 supports both legacy {0,1} and augmented {0,255} masks.
    """

    def __init__(self, thr=0):
        self.thr = thr

    def transform(self, results):
        if 'gt_seg_map' in results:
            gt = np.asarray(results['gt_seg_map'])
            # CrackMap ships RGBA masks (alpha constant 255, RGB channels
            # equal). Collapse to a single channel before thresholding for
            # binary CE/Dice heads; drop alpha first so the opaque background
            # does not become FG. Plain RGB masks take the same path.
            if gt.ndim == 3:
                if gt.shape[-1] >= 4:
                    gt = gt[..., :3]
                gt = gt.max(axis=-1)
            results['gt_seg_map'] = (gt > self.thr).astype(np.uint8)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(thr={self.thr})'


_MANIFEST_CACHE = {}


@DATASETS.register_module()
class ManifestSegDataset(BaseSegDataset):
    """Binary segmentation dataset for the v5/v4 'combined' datasets.

    Their split files list opaque ``sample_id``s (e.g. ``legacy__block_9753``);
    a ``combined_manifest.jsonl`` maps each id to its real absolute image/mask
    paths, which live across several source datasets. We override
    ``load_data_list`` to resolve ids through the manifest. Masks may be {0,1}
    or {0,255}; use ``BinarizeSeg`` in the pipeline. The manifest is cached per
    path so the train/val/test dataloaders don't each re-parse the (large) jsonl.
    """

    METAINFO = dict(
        classes=('background', 'foreground'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self, manifest='metadata/combined_manifest.jsonl',
                 ann_file='splits/train.txt', data_root=None,
                 path_mapping=None, reduce_zero_label=False, **kwargs):
        self.manifest = manifest
        self.path_mapping = path_mapping or {}
        self._active_path_mapping = self._filter_existing_path_mapping(
            self.path_mapping)
        kwargs['data_prefix'] = dict(img_path='', seg_map_path='')
        super().__init__(ann_file=ann_file, data_root=data_root,
                         reduce_zero_label=reduce_zero_label, **kwargs)

    def _abspath(self, p):
        import os.path as osp
        if self.data_root and not osp.isabs(p):
            return osp.join(self.data_root, p)
        return p

    def _load_manifest(self):
        import json
        mpath = self._abspath(self.manifest)
        if mpath not in _MANIFEST_CACHE:
            id2 = {}
            with open(mpath, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    id2[r['sample_id']] = (r['image_path'], r['mask_path'])
            _MANIFEST_CACHE[mpath] = id2
        return _MANIFEST_CACHE[mpath]

    @staticmethod
    def _filter_existing_path_mapping(path_mapping):
        import os.path as osp
        active = []
        for src, dst in path_mapping.items():
            if osp.exists(dst):
                active.append((src.replace('\\', '/'), dst.replace('\\', '/')))
        return active

    def _map_manifest_path(self, path):
        normalized = path.replace('\\', '/')
        for src, dst in self._active_path_mapping:
            if normalized.startswith(src):
                return dst + normalized[len(src):]
        return path

    def load_data_list(self):
        id2 = self._load_manifest()
        data_list = []
        with open(self._abspath(self.ann_file), encoding='utf-8') as f:
            for line in f:
                sid = line.strip()
                if not sid:
                    continue
                rec = id2.get(sid)
                if rec is None:
                    continue
                img_path = self._map_manifest_path(rec[0])
                seg_map_path = self._map_manifest_path(rec[1])
                data_list.append(dict(
                    img_path=img_path, seg_map_path=seg_map_path,
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label,
                    seg_fields=[]))
        return data_list


@MODELS.register_module()
class IgnoreSafeCELoss(nn.Module):
    """Weighted cross-entropy that defers ignore_index + class-weight handling
    entirely to ``torch.nn.functional.cross_entropy``.

    Why this exists: mmseg's stock ``CrossEntropyLoss`` crashes on CUDA when a
    ``class_weight`` is set *and* the label tensor contains the ignore value
    (255). Its mean-reduction path computes
    ``torch.stack([class_weight[cls] for cls in label])`` over *every* pixel,
    including ignored ones, so it indexes ``class_weight[255]`` (out of bounds
    for a length-2 weight) -> device-side assert. PyTorch's own
    ``cross_entropy(weight=..., ignore_index=255, reduction='mean')`` handles
    the ignore mask correctly, which is all we need here.
    """

    def __init__(self, class_weight=None, loss_weight=1.0, ignore_index=255,
                 loss_name='loss_ce'):
        super().__init__()
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, cls_score, label, weight=None, avg_factor=None,
                reduction_override=None, ignore_index=None, **kwargs):
        ii = self.ignore_index if ignore_index is None else ignore_index
        cw = (cls_score.new_tensor(self.class_weight)
              if self.class_weight is not None else None)
        loss = F.cross_entropy(cls_score, label.long(), weight=cw,
                               ignore_index=ii, reduction='mean')
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name


@METRICS.register_module()
class BinaryFgIoUMetric(IoUMetric):
    """IoUMetric that also exposes the foreground (class-1) scores as scalar
    keys: ``fgIoU`` / ``fgDice`` / ``fgFscore`` / ``fgPrecision`` / ``fgRecall``.

    For binary crack segmentation the 2-class means (mIoU/mFscore) are dominated
    by the ~99% background class, so ``save_best='mIoU'`` tends to pick a
    precision-biased, background-good checkpoint. Selecting on ``fgIoU`` (or
    ``fgRecall`` for the recall-biased deployment target) tracks the crack class
    directly. Everything IoUMetric reports is still emitted unchanged.
    """

    fg_index = 1  # METAINFO classes = ('background', 'foreground') -> crack = 1

    def compute_metrics(self, results):
        metrics = super().compute_metrics(results)  # means + per-class table
        if not results:
            return metrics
        res = tuple(zip(*results))
        ret = self.total_area_to_metrics(
            sum(res[0]), sum(res[1]), sum(res[2]), sum(res[3]),
            self.metrics, self.nan_to_num, self.beta)
        for k in ('IoU', 'Dice', 'Fscore', 'Precision', 'Recall'):
            vals = ret.get(k)
            if vals is not None and len(vals) > self.fg_index:
                metrics[f'fg{k}'] = float(np.round(float(vals[self.fg_index]) * 100, 2))
        # specificity = background recall = TN/(TN+FP); fgFPR = FP/(FP+TN) =
        # crack pixels hallucinated on background. The primary signal on the
        # all-background negative split (where fgIoU/fgRecall are undefined).
        rec = ret.get('Recall')   # present when 'mFscore' in iou_metrics
        if rec is not None and len(rec) > 0:
            spec = float(np.round(float(rec[0]) * 100, 2))
            metrics['specificity'] = spec
            metrics['fgFPR'] = float(np.round(100.0 - spec, 2))
        return metrics


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class DINOv3LayerDecayConstructor(DefaultOptimWrapperConstructor):
    """Layer-wise LR decay for the DINOv3 ViT-Adapter backbone.

    Conservative by design: the ViT trunk is *capped* at ``backbone_lr_scale``
    (default 0.1, == the previous uniform backbone lr_mult) at its TOP block and
    decays by ``decay_rate`` for shallower layers, so no ViT param ever exceeds
    0.1x base -- this can only be gentler on the pretrained features than the
    uniform 0.1x. From-scratch parts (SPM/injector/extractor/level_embed/up/output
    norms + decode/aux heads) and LoRA adapters train at the full base lr. Frozen
    params (LoRA mode) are skipped. Norms/biases/1-D params/tokens/embeds get no
    weight decay.

    paramwise_cfg keys: ``num_layers`` (ViT depth, 24 for ViT-L), ``decay_rate``
    (0.9), ``backbone_lr_scale`` (0.1).
    """

    def add_params(self, params, module, **kwargs):
        pcfg = self.paramwise_cfg or {}
        num_layers = pcfg.get('num_layers', 24)
        decay_rate = pcfg.get('decay_rate', 0.9)
        backbone_scale = pcfg.get('backbone_lr_scale', 0.1)
        base_lr = self.base_lr
        base_wd = self.base_wd if self.base_wd is not None else 0.0

        groups = {}
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            scale = self._lr_scale(name, num_layers, decay_rate, backbone_scale)
            no_decay = (p.ndim == 1 or name.endswith('.bias') or any(
                t in name for t in ('gamma', 'level_embed', 'cls_token',
                                    'storage_tokens', 'mask_token', 'periods',
                                    'query_embed', 'query_feat')))
            wd = 0.0 if no_decay else base_wd
            key = (round(scale, 8), wd)
            grp = groups.setdefault(
                key, dict(params=[], lr=base_lr * scale, weight_decay=wd))
            grp['params'].append(p)
        params.extend(groups.values())

    @staticmethod
    def _lr_scale(name, num_layers, decay_rate, backbone_scale):
        if 'lora_' in name:                 # new low-rank params -> full lr
            return 1.0
        if name.startswith('backbone.vit.'):
            if 'blocks.' in name:
                m = re.search(r'blocks\.(\d+)\.', name)
                blk = int(m.group(1)) if m else 0
                depth_from_top = (num_layers - 1) - blk      # 0 at the last block
            elif any(t in name for t in ('patch_embed', 'cls_token',
                                         'storage_tokens', 'mask_token',
                                         'rope_embed')):
                depth_from_top = num_layers                  # earliest -> smallest lr
            else:                                            # vit.norm (final)
                depth_from_top = 0
            return backbone_scale * (decay_rate ** depth_from_top)
        return 1.0                          # adapter + decode/aux heads -> full lr
