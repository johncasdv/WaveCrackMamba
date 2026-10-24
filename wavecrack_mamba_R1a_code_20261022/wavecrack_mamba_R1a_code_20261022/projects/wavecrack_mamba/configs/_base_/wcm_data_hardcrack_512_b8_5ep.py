# WCM-S HardCrack data / eval / schedule base -- the b8_5ep SAME-TRACK protocol.
#
# Split-file roles (verified 2026-09-11):
#   train_tiles.txt: official input, 20,996 tile image/mask pairs.
#   train.txt: inventory of 3,412 train originals, not this loader's input.
#   unique train_manifest.jsonl image_id: 3,163 contributing parents;
#       249 inventory originals contribute no official tiles.
#   An ID-only parent list is provenance, not a two-column dataset input.
# Re-sampling full images changes the protocol, even with the same parents.
# Keep ann_file=train_tiles.txt; record actual split/manifest/data version hashes.
# See paper/wavecrack_mamba/data/HardCrack_DATASHEET.md, section 2.4.1.
#
# Why this file exists (公开集实验计划 v0.3 D2/D4/D7): HardCrack is the paper's
# only in-domain set, and WCM-S must sit on the same track as the baseline
# lineup there. This base mirrors projects/mixercseg_baselines/_base/
# {runtime, schedule_b8_5ep, datasets/hardcrack}.py:
#   data      = HardCrack official tiles (train) / untouched full-res images
#               scored by sliding window (val/test), BinarizeSeg(thr=0);
#   schedule  = EpochBased 5 epochs, batch 8, AdamW lr 5e-4 wd 0.01,
#               PolyLR -> 1e-6 by epoch, validate every epoch;
#   selection = BinaryFgIoUMetric, save_best fgIoU (same as the baselines).
# Kept from the WCM Track-A base (numerics, not budget): bf16 AmpOptimWrapper,
# clip_grad 1.0, norm/bias no-decay, and a warmup (LinearLR 1e-3 over the first
# epoch; end=2 for the reason documented in mixercseg_improved_512x512_b8_5ep.py)
# because the zero-initialised mixer/wavelet branches need it -- MixerCSeg-improved
# carries the same per-model warmup on this track.
# Pipeline = the baseline track's (horizontal flip only; tiles are already
# 512x512). The Track-A heavy augmentation (RandomResize/Crop/Rotate/
# PhotoMetric) is deliberately NOT used: same track means same data view.
# Leaf configs must add model.test_cfg = slide (512/384) -- see the hardcrack/
# arm files -- because mmengine forbids a second base overriding `model`.
# WSL runs: override data_root to the /mnt/d/... path in a *_localwsl leaf
# (the data/HardCrack junction is a Windows-side convenience).
custom_imports = dict(
    imports=[
        'projects.wavecrack_mamba',                # WCM registrations
        'projects.dinov3_sealing.crack_modules',   # BinarizeSeg / BinaryFgIoUMetric
        'projects.deepcrack',                      # DeepCrackPairDataset (two-column split files)
        'mmpretrain.models',                       # ConvNeXt trunk prefix
    ],
    allow_failed_imports=False)

crop_size = (512, 512)
data_root = 'data/HardCrack'  # junction -> D:/Data/hardcrack/HardCrack_v1.0
dataset_type = 'DeepCrackPairDataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='BinarizeSeg', thr=0),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='BinarizeSeg', thr=0),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=8, num_workers=2, persistent_workers=False,
    # Keep the established full-batch training protocol; discard any partial
    # final batch. The current manifest count is documented above.
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type=dataset_type, data_root=data_root,
                 ann_file='splits/train_tiles.txt', pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root,
                 ann_file='splits/val.txt', pipeline=test_pipeline,
                 test_mode=True))
test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root,
                 ann_file='splits/test.txt', pipeline=test_pipeline,
                 test_mode=True))

val_evaluator = dict(
    type='BinaryFgIoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'))
launcher = 'none'

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=42)  # baseline runtime default; seeds vary per run via --cfg-options

max_epochs = 5
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper', dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=2, by_epoch=True),
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=2, end=max_epochs,
         by_epoch=True),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1,
                    max_keep_ckpts=3, save_best='fgIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
