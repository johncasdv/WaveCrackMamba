"""One Transformer update; launch through run_train.ps1 -Wsl + schtasks."""
_base_ = ['../_base_/wcm_refiner_pilot.py']
model = dict(decode_head=dict(refiner=dict(steps=1)))
work_dir = './work_dirs/wavecrack_mamba/recurrent/fz_t1_seed28'
