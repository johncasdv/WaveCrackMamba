"""Frozen refinement of a provenance-checked HardCrack Mamba R1a base."""
_base_ = ['../hardcrack/wcm_r1a_m_512_b8_5ep.py']

import os

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='BinarizeSeg', thr=0),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    sampler=dict(_delete_=True, type='InfiniteSampler', shuffle=True),
    dataset=dict(pipeline=train_pipeline))
# Pilot test commands also use validation; final test remains untouched.
test_dataloader = dict(dataset=dict(ann_file='splits/val.txt'))
model = dict(
    backbone=dict(convnext=dict(init_cfg=None)),
    freeze_base_for_refiner=True,
    decode_head=dict(
        refiner=dict(
            type='WCMRecurrentRefiner',
            channels=128,
            steps=3,
            mode='transformer',
            heads=4,
            window_size=8,
            share_weights=True,
            use_mask_feedback=True,
            conv_channels=160,
            ffn_ratio=2,
            residual_scale=0.1)))
custom_hooks = [
    dict(type='WCMHardCrackProtocolHook'),
    dict(
        type='WCMFrozenRefinerHook',
        checkpoint=os.environ.get('WCM_HARDCRACK_BASE_CHECKPOINT', ''),
        expected_sha256=os.environ.get('WCM_HARDCRACK_BASE_SHA256', ''))
]
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=4000,
    val_interval=1000)
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=200, by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        T_max=3800,
        eta_min=0,
        begin=200,
        end=4000,
        by_epoch=False)
]
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=4,
        save_best='val/ODS',
        rule='greater'))
log_processor = dict(by_epoch=False)

# Module-training length remains a separate, not-yet-calibrated budget.
hardcrack_checkpoint_rule = 'val_ODS_best'
val_evaluator = [
    dict(type='BinaryFgIoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']),
    dict(type='HardCrackODSMetric', prefix='val'),
]
test_evaluator = val_evaluator
