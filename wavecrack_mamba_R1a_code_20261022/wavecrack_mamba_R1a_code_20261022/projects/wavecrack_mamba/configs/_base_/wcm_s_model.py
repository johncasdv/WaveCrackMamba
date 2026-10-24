# WCM-S model base: FULL-module defaults (exec-plan §6.1) -- pretrained
# ConvNeXt-tiny trunk prefix + E3 detail branch + Haar wavelet bypass + two
# axial Mamba blocks + simplified U-shaped decoder + binary sigmoid mask head.
# Ablation arms null modules (C0: wavelet+mixer None; C1: mixer None; C2:
# wavelet None; G1 E-arms vary detail_branch).
num_classes = 2

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512))

model = dict(
    type='WCMSegmentor',
    # §10-8 resolution: hann is the default slide fusion (smooth weights per
    # Spec §12.2); 'count' stays available as the comparison mode. The 512
    # patch protocol never slides, so this only affects large-image inference.
    slide_mode='hann',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='WCMTrunk',
        # 3-stage ConvNeXt-tiny prefix (96/192/384 @ 1/4·1/8·1/16); stage-3 not
        # built. Pretrained weights loaded on the ConvNeXt sub-config; produce
        # the file with projects/wavecrack_mamba/tools/convert_trunk_weights.py.
        # Source: DINOv3 ConvNeXt-tiny LVD1689m
        # (sha256 21b726bb...); converted (sha256 4c5aa94e...). G0 decides
        # whether this pretrained init is kept (exec-plan §5); a from-scratch
        # arm sets init_cfg=None.
        convnext=dict(
            type='mmpretrain.ConvNeXt',
            arch=dict(depths=[3, 3, 9], channels=[96, 192, 384]),
            out_indices=(0, 1, 2), gap_before_final_norm=False,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='pretrain/wcm_convnext_tiny_dinov3_mmpretrain.pth'))),
    # E3 high-res detail branch is the WCM-S default (exec-plan §8.3); the G1
    # arm configs vary it (E0=None / E1 / E2). Its near-zero inject leaves the
    # trunk forward unchanged at init. Detail is orthogonal to the C-axis, so C0
    # keeps E3 and only nulls wavelet+mamba.
    detail_branch=dict(type='WCMDetailBranch', form='stride1'),
    # signed/magnitude scales are training-set statistics (Spec §8.2: never
    # per-image min-max). Measured 2026-07-26 by tools/estimate_wavelet_stats.py
    # over 2000 crack_v5 train images ON NORMALISED inputs -- the bypass sits
    # after the data preprocessor, and measuring on raw 0-255 pixels gives scales
    # ~58x too large. level-1 |hf| median 0.0261 / p90 0.1391; level-2 median
    # 0.0782 / p90 0.4042. Both scales take the level-1 median so the log1p
    # compression knee sits at typical edge energy; the placeholder 1.0 would
    # have put the knee ~38x above the actual signal.
    wavelet_bypass=dict(type='HaarWaveletBypass', proj_channels=(48, 96),
                        signed_scale=0.0261, magnitude_scale=0.0261),
    # Spec §8.4 registered defaults. backend='mamba' is deliberate for formal
    # runs (missing mamba_ssm must fail loudly, not silently degrade to gru --
    # gru results may never be reported as Mamba); unit tests override to gru.
    axial_mixer=dict(
        type='AxialMambaMixer',
        n_blocks=2,
        block=dict(
            d_model=96,
            core=dict(type='AxialSSMCore', d_model=96, d_state=16, d_conv=4,
                      expand=2, backend='mamba'),
            router_mode='context+hf2',
            hf2_channels=96,
            row_column_core_sharing=True,
            alpha_init=0.1,
            beta_init=0.1,
            mlp_ratio=2.0)),
    decode_head=dict(
        type='WCMDecoderHead',
        trunk_channels=(96, 192, 384),
        detail_channels=(24, 48),
        wavelet_proj_channels=(48, 96),
        channels=128,
        # Explicit switches (no silent adaptation): E0 arms set
        # detail_skips=False, E1/E2 set full_res_skip=False, C0/C1 set
        # wavelet_gate=False.
        detail_skips=True,
        full_res_skip=True,
        wavelet_gate=True,
        num_classes=num_classes,
        out_channels=1,
        threshold=0.5,
        norm_cfg=dict(type='GN', num_groups=16),
        act_cfg=dict(type='SiLU'),
        # Spec §9.6: L_core = L_region = 0.5·focal(α_fg=0.75,γ=2) +
        # 0.5·tversky(fp=0.3,fn=0.7). Structure supervision (T1-T3) derives
        # per-arm: append WCMSoftClDiceLoss(loss_weight=0.20) + a
        # LossWeightWarmupHook schedule (custom_hooks) per exec-plan §6.
        loss_decode=dict(type='WCMRegionLoss', loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
