"""WaveCrackMamba reference implementation (``projects/wavecrack_mamba``).

Importing this package registers all WCM modules with the mmseg registry.
See ``docs/design/WaveCrackMamba_模型设计Spec_v0.5.md`` and
``docs/design/WaveCrackMamba_实施计划_v1.0.md``.

Current scope: trunk -> decoder path (M0-1) + pretrained 3-stage trunk (M0-2)
+ high-res detail branch with near-zero injection (M0-3) + two-level Haar
wavelet bypass (M0-4) + axial bidirectional Mamba mixer with direction router
(M0-5). Later tasks add the full decoder/Hann slide (M0-6) and loss suite
(M0-7).
"""
from .models import (AxialMambaBlock, AxialMambaMixer, AxialSSMCore,
                     DirectionRouter, HaarWaveletBypass, WCMDecoderHead,
                     WCMDetailBranch, WCMSegmentor, WCMTrunk)

from .transforms import WCMFixedCenterCrop

__all__ = ['WCMSegmentor', 'WCMTrunk', 'WCMDetailBranch', 'WCMDecoderHead',
           'HaarWaveletBypass', 'AxialSSMCore', 'AxialMambaMixer',
           'AxialMambaBlock', 'DirectionRouter', 'WCMFixedCenterCrop']
