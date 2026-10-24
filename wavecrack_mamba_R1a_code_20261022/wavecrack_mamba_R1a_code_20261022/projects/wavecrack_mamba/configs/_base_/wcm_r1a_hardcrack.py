"""Current HardCrack tiles + R1a, without detail or wavelet branches."""
_base_ = ['./wcm_data_hardcrack_512_b8_5ep.py', './wcm_s_model.py']

import os

data_root = os.environ.get('WCM_HARDCRACK_DATA_ROOT', 'data/HardCrack')
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))
custom_imports = dict(
    imports=[
        'projects.wavecrack_mamba', 'projects.dinov3_sealing.crack_modules',
        'projects.deepcrack', 'mmpretrain.models',
        'projects.wavecrack_mamba.tools.hardcrack_protocol'
    ],
    allow_failed_imports=False)
model = dict(
    detail_branch=None,
    wavelet_bypass=None,
    axial_mixer=None,
    slide_mode='count',
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
custom_hooks = [dict(type='WCMHardCrackProtocolHook')]
hardcrack_arm = 'nm'
hardcrack_smoke = False
randomness = dict(seed=42)
work_dir = './work_dirs/wavecrack_mamba/hardcrack_r1a/nm_seed42'
