"""Deterministic image views for the frozen-feature diagnostic pilot."""
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class WCMFixedCenterCrop(BaseTransform):
    """Crop one repeatable view; smaller images are padded by the preprocessor.

    This does not resize full images, sample by foreground labels, or apply
    random augmentations. Training candidates therefore see identical views.
    """

    def __init__(self, crop_size=(512, 512)):
        if len(crop_size) != 2 or any(not isinstance(x, int) or x < 1
                                      for x in crop_size):
            raise ValueError('crop_size must be two positive integers')
        self.crop_size = tuple(crop_size)

    def transform(self, results):
        h, w = results['img'].shape[:2]
        ch, cw = min(h, self.crop_size[0]), min(w, self.crop_size[1])
        y, x = (h - ch) // 2, (w - cw) // 2
        results['img'] = results['img'][y:y + ch, x:x + cw].copy()
        for field in results.get('seg_fields', []):
            results[field] = results[field][y:y + ch, x:x + cw].copy()
        results['img_shape'] = (ch, cw)
        results['wcm_crop_bbox'] = (x, y, x + cw, y + ch)
        return results
