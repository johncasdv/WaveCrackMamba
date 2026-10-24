# WCM-S C0 main base = data/eval/schedule base + WCM-S model base, with the
# C-axis modules nulled (exec-plan §6.1): C0 = no wavelet bypass + no Mamba
# mixer; the E3 detail branch stays (detail is the G1 axis, orthogonal to C).
# Stage/arm configs derive from this file and only override the deltas.
_base_ = [
    './wcm_data_crack_v5_512.py',
    './wcm_s_model.py',
]

model = dict(wavelet_bypass=None, axial_mixer=None,
             decode_head=dict(wavelet_gate=False))

work_dir = './work_dirs/wavecrack_mamba/wcm_s_c0_crack_v5_512'
