_base_ = ['./common.py']

data_root = 'datasets/Crack500-MixerCSeg'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))
work_dir = 'runs/crack500_seed42'
