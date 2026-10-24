"""WaveCrackMamba R1a m arm on native-resolution public crack images."""

_base_ = [
    '../../wavecrack_mamba_R1a_code_20261022/'
    'wavecrack_mamba_R1a_code_20261022/projects/wavecrack_mamba/'
    'configs/_base_/wcm_s_model.py'
]

custom_imports = dict(
    imports=[
        'projects.wavecrack_mamba',
        'projects.dinov3_sealing.crack_modules',
        'projects.deepcrack',
        'projects.wavecrack_mamba.tools.hardcrack_ods',
        'mmpretrain.models',
    ],
    allow_failed_imports=False)

model = dict(
    data_preprocessor=dict(
        _delete_=True,
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size_divisor=16),
    detail_branch=None,
    wavelet_bypass=None,
    slide_mode='count',
    axial_mixer=dict(
        _delete_=True,
        type='AxialMambaMixer',
        n_blocks=2,
        block=dict(
            d_model=96,
            router_mode='context_only',
            row_column_core_sharing=True,
            alpha_init=0.1,
            beta_init=0.1,
            mlp_ratio=2.0,
            core=dict(
                type='AxialSSMCore', d_model=96, d_state=16,
                d_conv=4, expand=2, backend='mamba'))),
    decode_head=dict(
        _delete_=True,
        type='WCMInversePyramidHead',
        trunk_channels=(96, 192, 384),
        channels=128,
        upsample='fixed',
        num_classes=2,
        out_channels=1,
        threshold=0.5,
        norm_cfg=dict(type='GN', num_groups=16),
        act_cfg=dict(type='SiLU'),
        loss_decode=dict(type='WCMRegionLoss', loss_weight=1.0)),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='BinarizeSeg', thr=0),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackSegInputs'),
]
eval_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='BinarizeSeg', thr=0),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='DeepCrackPairDataset',
        data_root='data',
        ann_file='splits/train.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DeepCrackPairDataset',
        data_root='data',
        ann_file='splits/val.txt',
        pipeline=eval_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DeepCrackPairDataset',
        data_root='data',
        ann_file='splits/test.txt',
        pipeline=eval_pipeline,
        test_mode=True))

val_evaluator = dict(type='HardCrackODSMetric', prefix='val')
test_evaluator = dict(type='HardCrackODSMetric', prefix='test')

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'))
launcher = 'none'
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=42)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    accumulative_counts=8,
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=2, by_epoch=True),
    dict(
        type='CosineAnnealingLR', eta_min=1e-6,
        begin=1, end=50, T_max=49, by_epoch=True),
]
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=2,
        max_keep_ckpts=2, save_best='val/ODS', rule='greater',
        save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
