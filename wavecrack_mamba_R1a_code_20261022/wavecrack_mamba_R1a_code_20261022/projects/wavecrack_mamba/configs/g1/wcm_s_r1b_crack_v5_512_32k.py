# G1 arm R1b: low-resolution entry + inverse-pyramid fusion head, 32k initial
# screening -- DySample at the same three lift sites -- the ONLY delta vs R1a.
# Trunk-only entry (detail_branch=None, like E0); resolution is recovered at the
# decode end by WCMInversePyramidHead (SCSegamba-form reference, Spec 982/983/447).
# decode_head is _delete_-replaced: the base head's detail/wavelet switches do not
# exist on the inverse-pyramid head. Loss/num_classes/threshold restated verbatim
# from wcm_s_model so the mask-head protocol stays identical to the E arms.
_base_ = ['../_base_/wcm_s_c0_crack_v5_512.py']

model = dict(
    detail_branch=None,
    decode_head=dict(
        _delete_=True,
        type='WCMInversePyramidHead',
        trunk_channels=(96, 192, 384),
        channels=128,
        upsample='dysample',
        num_classes=2,
        out_channels=1,
        threshold=0.5,
        norm_cfg=dict(type='GN', num_groups=16),
        act_cfg=dict(type='SiLU'),
        loss_decode=dict(type='WCMRegionLoss', loss_weight=1.0)))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=32000, val_interval=3200)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=1500, by_epoch=False),
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=1500, end=32000,
         by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=3200,
                    max_keep_ckpts=3, save_best='fgIoU', rule='greater'))
work_dir = './work_dirs/wavecrack_mamba/g1_r1b_32k'
