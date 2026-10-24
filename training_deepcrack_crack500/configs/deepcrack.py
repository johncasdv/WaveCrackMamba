_base_ = ['./common.py']

data_root = 'datasets/DeepCrack-MixerCSeg'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))
work_dir = 'runs/deepcrack_seed42'
