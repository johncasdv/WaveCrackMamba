# WaveCrackMamba data + evaluation + schedule base (Track A: crack_v5, b8/160k).
#
# Derived in spirit from projects/crack_baselines/_base_crack_v5_512.py. Three
# deliberate differences from that file are marked (diff N) inline; everything
# else (ManifestSegDataset, BinarizeSeg(thr=0), BinaryFgIoUMetric, pipeline)
# is kept identical so the protocol stays comparable to the crack baselines.
custom_imports = dict(
    imports=[
        'projects.wavecrack_mamba',                # (diff 1) WCM registrations
        'projects.dinov3_sealing.crack_modules',   # ManifestSegDataset / BinarizeSeg / IgnoreSafeCELoss / BinaryFgIoUMetric
        'mmpretrain.models',                       # ConvNeXt trunk prefix
    ],
    allow_failed_imports=False)

crop_size = (512, 512)

# (diff 2) data_root points at the real on-disk location. _base_crack_v5_512's
# 'H:/DataAug/...' drive is dead on this machine; the manifest holds absolute
# D: paths that exist, and path_mapping self-filters dead destinations
# (crack_modules._filter_existing_path_mapping), so keeping the same mapping is
# safe -- on this box it filters to empty and the D: paths are used directly.
data_root = (
    'D:/Data/to_extract_mask/seg_dataset_aug/crack_combined/'
    'crack_v5_mmseg/')
dataset_type = 'ManifestSegDataset'
manifest = 'metadata/combined_manifest.jsonl'
path_mapping = {
    'D:/Data/to_extract_mask/seg_dataset_aug/crack_combined/':
    'H:/DataAug/seg_dataset_aug/crack_combined/',
    'D:/Data/to_extract_mask/seg_dataset_aug/crack_sealing_combined/':
    'H:/DataAug/seg_dataset_aug/crack_sealing_combined/',
}

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # legacy fg=1 / new fg=255 -> {0,1}; thr=0 is mandatory (Spec §8.0/§10.2).
    dict(type='BinarizeSeg', thr=0),
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=20, pad_val=0, seg_pad_val=255),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='BinarizeSeg', thr=0),
    dict(type='PackSegInputs'),
]

# Worker settings follow the b8_5ep precedent (num_workers<=2, no persistent
# workers): the Windows-stability setting all fair-protocol baselines use; kept
# identical under WSL for one less cross-environment variable.
train_dataloader = dict(
    batch_size=8, num_workers=2, persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type=dataset_type, data_root=data_root, manifest=manifest,
                 path_mapping=path_mapping, ann_file='splits/train.txt',
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root, manifest=manifest,
                 path_mapping=path_mapping,
                 # (diff 3) small val split for a b8_5ep-style validation cadence
                 ann_file='splits/new_val_small.txt',
                 pipeline=test_pipeline, test_mode=True))
test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root, manifest=manifest,
                 path_mapping=path_mapping, ann_file='splits/new_test.txt',
                 pipeline=test_pipeline, test_mode=True))

val_evaluator = dict(
    type='BinaryFgIoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    # single-model single-GPU: gloo CUDA collectives segfault on this box, so
    # there is no single-model multi-GPU DDP path (two GPUs = two runs).
    dist_cfg=dict(backend='gloo'))
launcher = 'none'

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=20260718)

# Track A protocol: global batch 8, 160k iters, validate every 16k.
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# bf16 AMP: the trunk is pretrained, but the axial Mamba recurrence + decoder
# train from scratch. bf16 keeps fp32 range and avoids the fp16 underflow that
# froze the recurrent TANGO run; no GradScaler / loss_scale needed.
optim_wrapper = dict(
    type='AmpOptimWrapper', dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=4e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=1500, by_epoch=False),
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=1500, end=160000,
         by_epoch=False),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000,
                    max_keep_ckpts=3, save_best='fgIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
