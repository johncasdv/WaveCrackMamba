"""HardCrack frozen t3 pilot; requires a qualified HC-M-B0 checkpoint."""
_base_ = ['../_base_/wcm_frozen_hardcrack.py']
model = dict(decode_head=dict(refiner=dict(mode='transformer', steps=3)))
hardcrack_arm = 't3'
work_dir = './work_dirs/wavecrack_mamba/hardcrack_r1a/fz_t3_seed42'
