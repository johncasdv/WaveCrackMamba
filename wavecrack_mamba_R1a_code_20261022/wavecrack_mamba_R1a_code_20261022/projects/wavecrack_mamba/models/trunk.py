import torch.nn.functional as F
from mmengine.model import BaseModule

from mmseg.registry import MODELS


@MODELS.register_module()
class WCMTrunk(BaseModule):
    """Pretrained spatial trunk: a mappable prefix of a published backbone.

    Spec §8.3: the trunk maps a ``>=1/4``-scale prefix of a released backbone
    and keeps its *native* normalisation/activation (ConvNeXt: LayerNorm/GELU).
    The norm/act are companion variables of the loaded weights, not free design
    axes, so they are never rewritten to GroupNorm/SiLU. The reference prefix is
    ConvNeXt-tiny stage-1..3 (``96@1/4, 192@1/8, 384@1/16``).

    The trunk is a genuine 3-stage prefix (``arch`` carries only
    ``depths/channels`` for stages 0-2), so ConvNeXt-tiny's stage-3
    (``768@1/32``) is neither built nor run. Pretrained weights are loaded by
    the ConvNeXt sub-config's own ``init_cfg`` (produced by
    ``tools/convert_trunk_weights.py``); ``norm0/1/2`` are absent in DINOv3 and
    stay at their init.

    The forward runs a manual stage loop that replicates ``ConvNeXt.forward``
    but exposes an injection hook after stage-0 at 1/4 resolution: the detail
    branch adds its near-zero ``inject`` here (M0-3) and the axial Mamba mixer
    will transform the same features (M0-5). When the injection is ``None`` or
    zero, ``s1/s2/s3`` match the reference ConvNeXt prefix exactly (硬约束 5).
    """

    def __init__(self,
                 convnext=dict(
                     type='mmpretrain.ConvNeXt',
                     arch=dict(depths=[3, 3, 9], channels=[96, 192, 384]),
                     out_indices=(0, 1, 2), gap_before_final_norm=False),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.backbone = MODELS.build(convnext)
        assert list(self.backbone.out_indices) == [0, 1, 2], \
            'WCM trunk expects a 3-stage prefix with out_indices=(0, 1, 2), ' \
            f'got {self.backbone.out_indices}'
        assert not self.backbone.gap_before_final_norm, \
            'WCM trunk needs spatial (not GAP) stage outputs'

    def forward(self, x, inject_after_stage0=None, transform_after_stage0=None):
        # After-stage-0 hooks, applied in exec-plan order: additive detail
        # inject first, then the axial-mixer transform (both identity at init,
        # so s1/s2/s3 stay bit-equal to the reference prefix -- 硬约束 5).
        bb = self.backbone
        outs = []
        for i, stage in enumerate(bb.stages):
            x = bb.downsample_layers[i](x)
            x = stage(x)
            if i == 0:
                if inject_after_stage0 is not None:
                    # Shapes match under the mult-of-16 padding contract
                    # (Spec §6.1); resize guards direct odd-size forwards.
                    if inject_after_stage0.shape[-2:] != x.shape[-2:]:
                        inject_after_stage0 = F.interpolate(
                            inject_after_stage0, size=x.shape[-2:],
                            mode='bilinear', align_corners=False)
                    x = x + inject_after_stage0
                if transform_after_stage0 is not None:
                    x = transform_after_stage0(x)
            outs.append(getattr(bb, f'norm{i}')(x))
        return dict(s1=outs[0], s2=outs[1], s3=outs[2])
