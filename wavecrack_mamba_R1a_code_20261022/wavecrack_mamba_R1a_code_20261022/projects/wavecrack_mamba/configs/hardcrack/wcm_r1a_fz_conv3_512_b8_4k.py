"""HardCrack frozen conv3 pilot; requires a qualified HC-M-B0 checkpoint."""
_base_ = ['../_base_/wcm_frozen_hardcrack.py']
model = dict(decode_head=dict(refiner=dict(mode='conv', steps=3)))
hardcrack_arm = 'conv3'
work_dir = './work_dirs/wavecrack_mamba/hardcrack_r1a/fz_conv3_seed42'
