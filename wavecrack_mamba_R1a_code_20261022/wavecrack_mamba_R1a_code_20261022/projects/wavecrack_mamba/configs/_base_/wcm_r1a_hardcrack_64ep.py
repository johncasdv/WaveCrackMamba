"""64 epochs, 167936 updates at batch 8, with validation-ODS selection."""
_base_ = ['./wcm_r1a_hardcrack.py']

max_epochs = 64
max_updates = 167936
warmup_updates = 1000
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=warmup_updates),
    dict(
        type='CosineAnnealingLR',
        by_epoch=False,
        eta_min=1e-6,
        begin=warmup_updates,
        end=max_updates,
        T_max=max_updates - warmup_updates),
]
hardcrack_checkpoint_rule = 'val_ODS_best'
hardcrack_expected_updates = max_updates
val_evaluator = [
    dict(type='BinaryFgIoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']),
    dict(type='HardCrackODSMetric', prefix='val'),
]
test_evaluator = [
    dict(type='BinaryFgIoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']),
    dict(type='HardCrackODSMetric', prefix='test'),
]
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=2,
        max_keep_ckpts=3,
        save_best='val/ODS',
        rule='greater',
        save_last=True))
