"""Three shared Transformer updates, with shifts 0/4/0."""
_base_ = ['../_base_/wcm_refiner_pilot.py']
model = dict(decode_head=dict(refiner=dict(steps=3)))
work_dir = './work_dirs/wavecrack_mamba/recurrent/fz_t3_seed28'
