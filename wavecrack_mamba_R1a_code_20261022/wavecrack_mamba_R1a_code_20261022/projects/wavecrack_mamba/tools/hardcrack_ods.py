"""Streaming pixel-micro ODS matching the current HardCrack evaluator."""
import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS

BINS = 1000


def probability_histograms(probability, target, ignore_index=255):
    """Quantize probabilities exactly as eval_hardcrack.py does."""
    p = np.asarray(probability, dtype=np.float32)
    gt = np.asarray(target)
    if p.shape != gt.shape:
        raise ValueError('ODS probability/target shapes differ')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('ODS expects postprocessed probabilities in [0, 1]')
    valid = gt != ignore_index
    if not np.isin(gt[valid], [0, 1]).all():
        raise ValueError('ODS requires binary targets')
    quantized = np.clip((p * BINS).astype(np.int32), 0, BINS)
    positive = np.bincount(quantized[valid & (gt == 1)], minlength=BINS + 1)
    negative = np.bincount(quantized[valid & (gt == 0)], minlength=BINS + 1)
    return positive, negative


def summarize_histograms(positive, negative):
    tp = np.cumsum(positive[::-1], dtype=np.float64)[::-1]
    fp = np.cumsum(negative[::-1], dtype=np.float64)[::-1]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(positive.sum(), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best = int(np.argmax(f1))  # Lowest threshold wins a tie, as offline.
    return dict(ODS=float(f1[best] * 100), ODS_threshold=best / BINS)


@METRICS.register_module()
class HardCrackODSMetric(BaseMetric):
    """Use WCM's one-channel, already-sigmoid seg_logits without re-sigmoid.

    Each result stores only two 1001-bin histograms, not an entire image.
    Prefix is explicit: validation selection uses val/ODS; test uses test/ODS.
    """
    default_prefix = 'val'

    def __init__(self, collect_device='cpu', prefix=None, ignore_index=255):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            if hasattr(sample, 'to_dict'):
                sample = sample.to_dict()
            prediction = sample['seg_logits']['data']
            if prediction.ndim != 3 or prediction.shape[0] != 1:
                raise ValueError(
                    'HardCrack ODS requires one probability channel')
            probability = prediction[0].detach().float().cpu().numpy()
            target = sample['gt_sem_seg']['data'][0].detach().cpu().numpy()
            self.results.append(
                probability_histograms(probability, target, self.ignore_index))

    def compute_metrics(self, results):
        if not results:
            raise ValueError(
                'Cannot select a model from empty validation results')
        positive = np.sum([r[0] for r in results], axis=0, dtype=np.int64)
        negative = np.sum([r[1] for r in results], axis=0, dtype=np.int64)
        return dict(
            **summarize_histograms(positive, negative),
            num_images=len(results))
