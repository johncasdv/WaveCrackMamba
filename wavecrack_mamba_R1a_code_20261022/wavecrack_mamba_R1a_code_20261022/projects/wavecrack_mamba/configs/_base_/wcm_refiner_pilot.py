"""Frozen crack_v5 diagnostic pilot, not the formal HardCrack recipe.

Set WCM_REFINER_DATA_ROOT and WCM_REFINER_BASE_CHECKPOINT explicitly.
Under WSL, manifest D:/ paths are remapped to /mnt/d/ without changing files.
"""
# isort: skip_file
# MMEngine must encounter _base_ before imports in this legacy config.
_base_ = ['../g1/wcm_s_r1a_crack_v5_512_32k.py']

import os  # noqa: E402

data_root = os.environ.get('WCM_REFINER_DATA_ROOT', '')
path_mapping = {} if os.name == 'nt' else {'D:/': '/mnt/d/'}
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='BinarizeSeg', thr=0),
    dict(type='WCMFixedCenterCrop', crop_size=(512, 512)),
    dict(type='PackSegInputs'),
]
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        path_mapping=path_mapping,
        ann_file=os.environ.get('WCM_REFINER_TRAIN_SPLIT',
                                '_diag/train_diag3k.txt'),
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        path_mapping=path_mapping,
        ann_file=os.environ.get('WCM_REFINER_VAL_SPLIT',
                                'splits/new_val_small.txt')))
# Deliberately use the validation view for pilot test commands, never test
# data for candidate choice. Formal HardCrack runs need separate configs.
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        path_mapping=path_mapping,
        ann_file=os.environ.get('WCM_REFINER_VAL_SPLIT',
                                'splits/new_val_small.txt')))

model = dict(
    backbone=dict(convnext=dict(init_cfg=None)),
    freeze_base_for_refiner=True,
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
                type='AxialSSMCore',
                d_model=96,
                d_state=16,
                d_conv=4,
                expand=2,
                backend='mamba'))),
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

load_from = None
resume = False
custom_hooks = [
    dict(
        type='WCMFrozenRefinerHook',
        checkpoint=os.environ.get('WCM_REFINER_BASE_CHECKPOINT', ''),
        expected_sha256=os.environ.get('WCM_REFINER_BASE_SHA256')),
    dict(type='WCMRunInfoHook'),
]
randomness = dict(seed=20260728)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=4000, val_interval=1000)
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
        by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=4,
        save_best=None))
work_dir = './work_dirs/wavecrack_mamba/recurrent/fz_base'
