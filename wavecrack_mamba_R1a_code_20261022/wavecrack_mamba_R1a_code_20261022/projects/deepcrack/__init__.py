from mmseg.utils import register_all_modules

register_all_modules(init_default_scope=True)

from .deepcrack import (DeepCrackFastFuseSegmentor, DeepCrackFgMetric,
                        DeepCrackMaskBinarize, DeepCrackPairDataset,
                        DeepCrackRandomBlur, DeepCrackSegmentor)

__all__ = [
    'DeepCrackSegmentor', 'DeepCrackFastFuseSegmentor', 'DeepCrackPairDataset',
    'DeepCrackMaskBinarize', 'DeepCrackRandomBlur', 'DeepCrackFgMetric'
]
