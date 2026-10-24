"""Three shared convolution updates, with a 160-channel spatial interior.

State and FFN stay 128-wide. Analytic main MACs are about 3.5% below T3 at
512 squared, excluding norms, biases and elementwise ops; measure runtime.
"""
_base_ = ['../_base_/wcm_refiner_pilot.py']
model = dict(decode_head=dict(refiner=dict(mode='conv', steps=3)))
work_dir = './work_dirs/wavecrack_mamba/recurrent/fz_c3_seed28'
