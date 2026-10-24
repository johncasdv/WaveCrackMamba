import os.path as osp
from types import SimpleNamespace
from typing import List, Optional, Sequence

import cv2
import mmengine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.datasets import BaseSegDataset
from mmseg.evaluation import IoUMetric
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.utils import resize
from mmseg.registry import DATASETS, METRICS, MODELS, TRANSFORMS
from mmseg.structures import SegDataSample
from mmseg.utils import OptConfigType, OptMultiConfig, OptSampleList, SampleList


def _conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1)


class _ConvRelu(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = _conv3x3(in_channels, out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.conv(x))


class _Down(nn.Module):

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs: Tensor):
        down = self.module(inputs)
        unpooled_shape = down.size()
        outputs, indices = self.pool(down)
        return outputs, down, indices, unpooled_shape


class _Up(nn.Module):

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, inputs: Tensor, indices: Tensor, output_shape) -> Tensor:
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        return self.module(outputs)


class _Fuse(nn.Module):

    def __init__(self, module: nn.Module, scale: int) -> None:
        super().__init__()
        self.module = module
        self.scale = scale
        self.conv = _conv3x3(64, 1)

    def forward(self, down_inp: Tensor, up_inp: Tensor) -> Tensor:
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = F.interpolate(
            outputs, scale_factor=self.scale, mode='bilinear',
            align_corners=False)
        outputs = self.module(outputs)
        return self.conv(outputs)


class _FastFuse(nn.Module):
    """Fuse side outputs at native resolution before the final upsample.

    ``_Fuse`` upsamples the concatenated features FIRST and runs the conv at the
    enlarged resolution; this variant convolves at the native side-output
    resolution and upsamples the single-channel result instead -- same fusion
    topology, a fraction of the compute at the deep (scale 8/16) sides.
    """

    def __init__(self, module: nn.Module, scale: int) -> None:
        super().__init__()
        self.module = module
        self.scale = scale
        self.conv = _conv3x3(64, 1)

    def forward(self, down_inp: Tensor, up_inp: Tensor) -> Tensor:
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = self.module(outputs)
        outputs = self.conv(outputs)
        if self.scale != 1:
            outputs = F.interpolate(
                outputs, scale_factor=self.scale, mode='bilinear',
                align_corners=False)
        return outputs


class _DeepCrackNet(nn.Module):
    """DeepCrack network from Zou et al., TIP 2019."""

    def __init__(self, fuse_cls=_Fuse) -> None:
        super().__init__()
        self.down1 = _Down(nn.Sequential(_ConvRelu(3, 64), _ConvRelu(64, 64)))
        self.down2 = _Down(nn.Sequential(_ConvRelu(64, 128),
                                         _ConvRelu(128, 128)))
        self.down3 = _Down(nn.Sequential(_ConvRelu(128, 256),
                                         _ConvRelu(256, 256),
                                         _ConvRelu(256, 256)))
        self.down4 = _Down(nn.Sequential(_ConvRelu(256, 512),
                                         _ConvRelu(512, 512),
                                         _ConvRelu(512, 512)))
        self.down5 = _Down(nn.Sequential(_ConvRelu(512, 512),
                                         _ConvRelu(512, 512),
                                         _ConvRelu(512, 512)))

        self.up1 = _Up(nn.Sequential(_ConvRelu(64, 64), _ConvRelu(64, 64)))
        self.up2 = _Up(nn.Sequential(_ConvRelu(128, 128), _ConvRelu(128, 64)))
        self.up3 = _Up(nn.Sequential(_ConvRelu(256, 256),
                                     _ConvRelu(256, 256),
                                     _ConvRelu(256, 128)))
        self.up4 = _Up(nn.Sequential(_ConvRelu(512, 512),
                                     _ConvRelu(512, 512),
                                     _ConvRelu(512, 256)))
        self.up5 = _Up(nn.Sequential(_ConvRelu(512, 512),
                                     _ConvRelu(512, 512),
                                     _ConvRelu(512, 512)))

        self.fuse5 = fuse_cls(_ConvRelu(512 + 512, 64), scale=16)
        self.fuse4 = fuse_cls(_ConvRelu(512 + 256, 64), scale=8)
        self.fuse3 = fuse_cls(_ConvRelu(256 + 128, 64), scale=4)
        self.fuse2 = fuse_cls(_ConvRelu(128 + 64, 64), scale=2)
        self.fuse1 = fuse_cls(_ConvRelu(64 + 64, 64), scale=1)
        self.final = _conv3x3(5, 1)

    def forward(self, inputs: Tensor):
        out, down1, indices1, shape1 = self.down1(inputs)
        out, down2, indices2, shape2 = self.down2(out)
        out, down3, indices3, shape3 = self.down3(out)
        out, down4, indices4, shape4 = self.down4(out)
        out, down5, indices5, shape5 = self.down5(out)

        up5 = self.up5(out, indices=indices5, output_shape=shape5)
        up4 = self.up4(up5, indices=indices4, output_shape=shape4)
        up3 = self.up3(up4, indices=indices3, output_shape=shape3)
        up2 = self.up2(up3, indices=indices2, output_shape=shape2)
        up1 = self.up1(up2, indices=indices1, output_shape=shape1)

        fuse5 = self.fuse5(down_inp=down5, up_inp=up5)
        fuse4 = self.fuse4(down_inp=down4, up_inp=up4)
        fuse3 = self.fuse3(down_inp=down3, up_inp=up3)
        fuse2 = self.fuse2(down_inp=down2, up_inp=up2)
        fuse1 = self.fuse1(down_inp=down1, up_inp=up1)
        output = self.final(torch.cat([fuse5, fuse4, fuse3, fuse2, fuse1], 1))
        return output, fuse5, fuse4, fuse3, fuse2, fuse1


class _DeepCrackFastFuseNet(_DeepCrackNet):

    def __init__(self) -> None:
        super().__init__(fuse_cls=_FastFuse)


@MODELS.register_module()
class DeepCrackSegmentor(BaseSegmentor):
    """MMSegmentation wrapper for DeepCrack.

    The original implementation supervises the fused output and all five side
    outputs with BCEWithLogitsLoss. This wrapper keeps that behavior while using
    MMSegmentation runners, visualizers, checkpoint hooks and evaluators.
    """

    def __init__(self,
                 pos_weight: float = 1.0,
                 loss_weight: float = 1.0,
                 divide_loss_by_batch: bool = True,
                 threshold: float = 0.5,
                 ignore_index: int = 255,
                 align_corners: bool = False,
                 test_cfg: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 net_cls=_DeepCrackNet) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.net = net_cls()
        self.loss_weight = loss_weight
        self.divide_loss_by_batch = divide_loss_by_batch
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.test_cfg = SimpleNamespace(**(test_cfg or dict(mode='whole')))
        self.num_classes = 2
        self.out_channels = 1
        self.decode_head = SimpleNamespace(threshold=threshold)
        self.register_buffer('pos_weight', torch.tensor([pos_weight],
                                                        dtype=torch.float32))

    def extract_feat(self, inputs: Tensor):
        return self.net(inputs)

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        return self.net(inputs)[0]

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None):
        return self.net(inputs)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load native or original DeepCrack checkpoints.

        The original project saves the bare ``DeepCrack`` module with keys such
        as ``down1.nn.0.conv.weight``. This MMSegmentation wrapper stores that
        network under ``net`` and names the child block ``module``.
        """
        if any(k.startswith(('down', 'up', 'fuse', 'final', 'module.down',
                             'module.up', 'module.fuse', 'module.final'))
               for k in state_dict):
            remapped = state_dict.__class__()
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    key = key[len('module.'):]
                if key.startswith(('down', 'up', 'fuse', 'final')):
                    key = 'net.' + key.replace('.nn.', '.module.')
                remapped[key] = value
            remapped.setdefault('pos_weight', self.pos_weight)
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        preds = self.net(inputs)
        gt = torch.stack(
            [sample.gt_sem_seg.data.squeeze(0) for sample in data_samples],
            dim=0).to(device=inputs.device)
        valid = gt != self.ignore_index
        target = gt.float()

        names = ('output', 'fuse5', 'fuse4', 'fuse3', 'fuse2', 'fuse1')
        losses = {}
        total = preds[0].new_tensor(0.0)
        for name, pred in zip(names, preds):
            pred = pred.squeeze(1)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = resize(
                    pred.unsqueeze(1),
                    size=gt.shape[-2:],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(1)
            if valid.any():
                loss = F.binary_cross_entropy_with_logits(
                    pred[valid],
                    target[valid],
                    pos_weight=self.pos_weight.to(pred.device),
                    reduction='mean')
            else:
                loss = pred.sum() * 0
            loss = loss * self.loss_weight
            if self.divide_loss_by_batch:
                loss = loss / max(inputs.shape[0], 1)
            losses[f'loss_{name}'] = loss
            total = total + loss
        losses['total'] = total.detach()

        with torch.no_grad():
            probs = preds[0].sigmoid().squeeze(1)
            labels = gt == 1
            pred_labels = probs > self.threshold
            denom = valid.sum().clamp_min(1)
            losses['acc_seg'] = (
                (pred_labels.eq(labels) & valid).sum().float() / denom * 100)
            pos = valid & labels
            neg = valid & ~labels
            losses['acc_pos'] = (
                (pred_labels.eq(labels) & pos).sum().float() /
                pos.sum().clamp_min(1) * 100)
            losses['acc_neg'] = (
                (pred_labels.eq(labels) & neg).sum().float() /
                neg.sum().clamp_min(1) * 100)
        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        if data_samples is not None:
            batch_img_metas = [sample.metainfo for sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        seg_logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        mode = getattr(self.test_cfg, 'mode', 'whole')
        if mode == 'slide':
            return self.slide_inference(inputs, batch_img_metas)
        if mode != 'whole':
            raise ValueError(f'Unsupported DeepCrack inference mode: {mode}')
        return self.encode_decode(inputs, batch_img_metas)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.shape
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros(
            (batch_size, self.out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop = inputs[:, :, y1:y2, x1:x2]
                crop_h, crop_w = crop.shape[-2:]
                crop = F.pad(
                    crop, (0, w_crop - crop_w, 0, h_crop - crop_h))
                crop_logits = self.encode_decode(crop, batch_img_metas)
                crop_logits = crop_logits[:, :, :crop_h, :crop_w]
                preds[:, :, y1:y2, x1:x2] += crop_logits
                count_mat[:, :, y1:y2, x1:x2] += 1

        assert not torch.any(count_mat == 0)
        return preds / count_mat

    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        batch_size, _, height, width = seg_logits.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            i_seg_logits = seg_logits[i:i + 1]
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                padding_size = img_meta.get(
                    'img_padding_size',
                    img_meta.get('padding_size', [0, 0, 0, 0]))
                pad_left, pad_right, pad_top, pad_bottom = padding_size
                i_seg_logits = i_seg_logits[:, :,
                                            pad_top:height - pad_bottom,
                                            pad_left:width - pad_right]

                if img_meta.get('flip', False):
                    direction = img_meta.get('flip_direction', None)
                    if direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    elif direction == 'vertical':
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)

            i_seg_logits = i_seg_logits.squeeze(0)
            i_seg_score = i_seg_logits.sigmoid()
            i_seg_pred = (i_seg_score > self.threshold).to(i_seg_score)
            data_samples[i].set_data({
                'seg_logits': PixelData(data=i_seg_score),
                'pred_sem_seg': PixelData(data=i_seg_pred)
            })
        return data_samples


@MODELS.register_module()
class DeepCrackFastFuseSegmentor(DeepCrackSegmentor):
    """DeepCrack wrapper with cheaper native-resolution side fusion.

    Ported from the .80 host's ``projects/crack_baselines/deepcrack`` copy
    (2026-08-29): the public-set baseline configs
    (``_base/models/deepcrack_fast_fuse.py``) have always referenced this type,
    but the implementation only ever existed on that host, so every
    ``deepcrack_fast_fuse`` cell outside DeepCrack itself failed with a
    misleading "not in the mmseg::model registry". Only the FastFuse additions
    were taken; this checkout's own ``slide_inference`` support (absent there)
    is preserved.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(net_cls=_DeepCrackFastFuseNet, **kwargs)


@DATASETS.register_module()
class DeepCrackPairDataset(BaseSegDataset):
    """Dataset for DeepCrack two-column index files.

    Each non-empty annotation line must contain ``image_path mask_path``. Paths
    may be absolute or relative to ``data_root``.
    """

    METAINFO = dict(
        classes=('background', 'crack'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix: str = '',
                 seg_map_suffix: str = '',
                 reduce_zero_label: bool = False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            data_prefix=dict(img_path='', seg_map_path=''),
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def _resolve_path(self, filename: str) -> str:
        if osp.isabs(filename) or self.data_root is None:
            return filename
        return osp.join(self.data_root, filename)

    def load_data_list(self) -> List[dict]:
        if not self.ann_file or not osp.isfile(self.ann_file):
            raise FileNotFoundError(f'Invalid ann_file: {self.ann_file}')

        data_list = []
        lines = mmengine.list_from_file(
            self.ann_file, backend_args=self.backend_args)
        for line_no, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f'{self.ann_file}:{line_no} must contain image and mask '
                    'paths separated by whitespace.')
            data_list.append(
                dict(
                    img_path=self._resolve_path(parts[0]),
                    seg_map_path=self._resolve_path(parts[1]),
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label,
                    seg_fields=[]))
        return data_list


@TRANSFORMS.register_module()
class DeepCrackMaskBinarize(BaseTransform):
    """Convert crack masks from ``{0, foreground}`` to ``{0, 1}``."""

    def __init__(self, threshold: int = 0) -> None:
        self.threshold = threshold

    def transform(self, results: dict) -> dict:
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = (
                results['gt_seg_map'] > self.threshold).astype(np.uint8)
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(threshold={self.threshold})'


@TRANSFORMS.register_module()
class DeepCrackRandomBlur(BaseTransform):
    """Random blur augmentation matching the original DeepCrack script."""

    def __init__(self, prob: float = 0.2, kernel_size: int = 5) -> None:
        self.prob = prob
        self.kernel_size = kernel_size

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results
        op = np.random.choice(('gaussian', 'mean', 'median'))
        if op == 'gaussian':
            results['img'] = cv2.GaussianBlur(
                results['img'], (self.kernel_size, self.kernel_size), 0)
        elif op == 'mean':
            results['img'] = cv2.blur(
                results['img'], (self.kernel_size, self.kernel_size))
        else:
            results['img'] = cv2.medianBlur(results['img'], self.kernel_size)
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(prob={self.prob}, '
                f'kernel_size={self.kernel_size})')


@METRICS.register_module()
class DeepCrackFgMetric(IoUMetric):
    """IoUMetric with foreground crack scores exposed as scalar keys."""

    fg_index = 1

    def compute_metrics(self, results):
        metrics = super().compute_metrics(results)
        if not results:
            return metrics
        totals = tuple(zip(*results))
        ret = self.total_area_to_metrics(
            sum(totals[0]), sum(totals[1]), sum(totals[2]), sum(totals[3]),
            self.metrics, self.nan_to_num, self.beta)
        for key in ('IoU', 'Dice', 'Fscore', 'Precision', 'Recall'):
            values = ret.get(key)
            if values is not None and len(values) > self.fg_index:
                metrics[f'fg{key}'] = float(
                    np.round(float(values[self.fg_index]) * 100, 2))
        return metrics
