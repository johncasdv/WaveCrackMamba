# WaveCrackMamba

Official code release for the WaveCrackMamba R1a m crack-segmentation model and the DeepCrack/Crack500 experiments reported with this implementation.

## Model

The reported R1a m configuration uses a three-stage ConvNeXt-Tiny encoder (96/192/384 channels at strides 4/8/16), two context-routed axial Mamba blocks with shared row/column state-space cores, and a three-scale inverse-pyramid decoder. The decoder projects each scale to 128 channels, resizes features with fixed bilinear interpolation, fuses them at stride 4, and predicts a single crack logit map. The training loss is the model's focal/Tversky region loss. The experiment configuration disables the separate detail and wavelet branches.

The implementation is in `wavecrack_mamba_R1a_code_20261022/wavecrack_mamba_R1a_code_20261022/projects/wavecrack_mamba`. Dataset-specific training configs and scripts are in `training_deepcrack_crack500`.

## Reported checkpoints

| Dataset | Test images | Selected epoch | Test ODS | Test ODS threshold | Checkpoint |
| --- | ---: | ---: | ---: | ---: | --- |
| DeepCrack | 237 | 20 | 83.4413% | 0.543 | [Download checkpoint](https://github.com/johncasdv/WaveCrackMamba/releases/latest/download/deepcrack_best_val_ods_epoch_20.pth) |
| Crack500 | 674 | 10 | 79.2586% | 0.751 | [Download checkpoint](https://github.com/johncasdv/WaveCrackMamba/releases/latest/download/crack500_best_val_ods_epoch_10.pth) |

Each checkpoint was selected using validation ODS. The test ODS threshold is computed on the corresponding test split as required by the reported ODS protocol. The reported values are benchmark results and do not constitute field deployment validation.

Checkpoint SHA-256:

- DeepCrack: `904c593c95d98d32400fe649787ee40e6507229e3b4d197707dbf95994150eda`
- Crack500: `c81be1157e4147d9bf1b52255d3c8d01624c8f5a0a8dd23b3d4ea2abb05bcf8f`

## Environment

The runs were completed on Ubuntu 22.04 with Python 3.10.12, PyTorch 2.1.0+cu118, MMCV 2.1.0, MMSegmentation 1.2.2, MMPretrain 1.2.0, and Mamba SSM 2.2.2 on a CUDA GPU. Install the compatible OpenMMLab packages following their official installation instructions; the compiled Mamba SSM package must match the PyTorch/CUDA environment.

The model was initialized from a converted public ImageNet-1K ConvNeXt-Tiny checkpoint. The converted file used for the reported runs had SHA-256 `b2ad0058499a3ea31634fe4fba49fe67a183750b10225b6874a1eeb4a16a31ec`. Place it at `weights/wcm_convnext_tiny_imagenet_mmpretrain.pth`. The conversion utility is `training_deepcrack_crack500/convert_imagenet_trunk.py`.

## Dataset layout

Obtain DeepCrack and Crack500 from their respective sources and arrange them as:

```text
datasets/
  DeepCrack-MixerCSeg/
    splits/train.txt
    splits/val.txt
    splits/test.txt
  Crack500-MixerCSeg/
    splits/train.txt
    splits/val.txt
    splits/test.txt
```

Each split file contains one image path and one mask path per line, relative to that dataset directory. The expected split sizes are DeepCrack 240/60/237 and Crack500 2357/337/674 for train/validation/test. The scripts verify files, split disjointness, and split-index SHA-256 values before training. Dataset images, masks, and split files are not included in this repository.

DeepCrack source: <https://github.com/yhlleo/DeepCrack>

Crack500 source paper: Yang et al., *Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection*, IEEE Transactions on Intelligent Transportation Systems, 2020. Obtain the dataset from its authorized source.

## Training

From the repository root, with the environment and datasets prepared:

```bash
export PYTHONPATH="$PWD/wavecrack_mamba_R1a_code_20261022/wavecrack_mamba_R1a_code_20261022${PYTHONPATH:+:$PYTHONPATH}"
python training_deepcrack_crack500/run_all.py \
  --pretrained weights/wcm_convnext_tiny_imagenet_mmpretrain.pth \
  --initialization imagenet \
  --data-parent datasets \
  --run-root runs \
  --microbatch 2 \
  --accumulation 4
```

This runs DeepCrack and then Crack500 with seed 42, effective batch size 8, bf16, 50 epochs, and validation ODS checkpoint selection every two epochs. To check the data and configuration without starting a full run, add `--smoke` for a one-epoch, eight-image smoke run. To evaluate a completed run's validation-selected checkpoint on its test set:

```bash
python training_deepcrack_crack500/evaluate_best.py \
  --dataset deepcrack \
  --data-parent datasets \
  --run-root runs
```

Use `--dataset crack500` for Crack500. A run writes its configuration, split hashes, and completion status in `run_manifest.json`.

Download the two checkpoint files from the latest GitHub Release and place them in a local `checkpoints/` directory if you need them for inference or evaluation. The checkpoint files are attached to the release rather than stored in Git history.

## Repository contents

- `training_deepcrack_crack500/`: training, evaluation, environment-check, and initialization-conversion scripts with dataset configs.
- `wavecrack_mamba_R1a_code_20261022/`: model modules, dependencies specific to the project, and model tests.
- GitHub Releases: validation-selected DeepCrack and Crack500 PyTorch checkpoints.

No benchmark images or masks are redistributed here. Checkpoint files are provided for research reproducibility. Third-party software and datasets remain subject to their own licenses and terms.

## License

The author has not yet selected a software license. Until a `LICENSE` file is added, no license to reuse or redistribute this code is granted.

## Citation

See `CITATION.cff`. Please cite the associated WaveCrackMamba paper when available.
