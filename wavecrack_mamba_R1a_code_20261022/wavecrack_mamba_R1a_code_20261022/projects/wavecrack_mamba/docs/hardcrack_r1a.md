# HardCrack R1a training entrypoints

These entrypoints implement the 2026-09-11 HardCrack plan. They are separate
from the earlier U-decoder arms and the retired crack_v5 diagnostic pilots.

## Arms

- `nm`: ConvNeXt-tiny + R1a; no Mamba, detail branch or wavelet.
- `m`: the same backbone/head with two context-only axial Mamba blocks.
- `t1`: one frozen-base Transformer update.
- `t3`: three shared frozen-base Transformer updates.
- `conv3`: three shared frozen-base convolution updates. This is not the old
  wavelet-enabled C3 model.

The baseline configurations are `configs/hardcrack/wcm_r1a_{nm,m}_512_b8_64ep.py`.
The frozen configurations are `configs/hardcrack/wcm_r1a_fz_{t1,t3,conv3}_512_b8_4k.py`.
Baselines now use 64 epochs at batch 8 (167,936 updates with the current
20,996 tiles and drop_last=True), with 1,000 update warm-up and cosine decay
through the complete budget. Validate and save regular checkpoints every two
epochs. Old five-epoch leaf files remain historical configurations. Frozen pilots
use 4,000 updates, batch 8, AdamW 1e-4, 200 warm-up updates, a cosine schedule
and deterministic official-tile views. All arms use a fixed R1a upsampler.

## Data and checkpoint contract

Use the current `HardCrack_v1.0` release through `-DataRoot`. The launcher
checks the registered split and tile-manifest SHA-256 values before execution.
`train_tiles.txt` is the training input; `train.txt` remains an original-image
inventory. A changed release must be audited and the registered hashes
updated explicitly. This check covers split/manifest identity, not an
exhaustive rehash of all image and mask bytes.

Validation uses the native 455-image split with count-averaged 512/384 sliding
windows. Baseline test configuration uses the 873-image test split. Frozen
pilot test commands use validation to keep candidate selection off test.
Primary selection is the highest validation pixel-micro ODS, with the
earliest checkpoint retained on ties. A 1,000-bin probability histogram
matches the offline HardCrack evaluator; it is aggregated over the full
validation split before F1 is maximized. Diagnostic fixed-threshold fgIoU
remains available. Validation reports val/ODS; baseline test reports test/ODS.
Final checkpoints remain available for budget and convergence diagnostics.

Common baseline tensors are initialized from the same deterministic
no-Mamba construction; the shared-state digest is saved in `hardcrack_run.json`.
T1/T3 copy identical initial refiner tensors. Conv3 copies compatible common
submodules while keeping its convolutional replacement separate.

Frozen runs require a pinned checkpoint from a completed `m` run on the same
registered release. Model keys/shapes are checked by the existing strict
refiner loader. Checkpoints store `meta.hardcrack_run` with arm, data hashes,
seeds, run identity, smoke status and completed iterations. After successful
training and final validation, completed_run.json binds the selected file,
its SHA-256, validation score/threshold/step, full validation history, planned
and completed updates, and the final checkpoint's SHA-256. Keep this receipt
and the final checkpoint beside the selected base, including when copying it.
Admission accepts an earlier validation-best checkpoint only when these
records match and the final checkpoint proves that its run finished. A bare
is_final flag, unselected final file, old-domain run, interrupted run or smoke
base cannot qualify for a formal frozen pilot. Smoke bases are smoke-only.

## Launcher

All arms use WSL, so the paired baselines share the same software environment.
The wrapper exports the data/checkpoint paths and `PYTHONPATH` through WSLENV;
it exports GPU selection to Python rather than attaching it only to `cd`.

Read-only preflight (can be run interactively):

```powershell
./projects/wavecrack_mamba/scripts/run_hardcrack_r1a.ps1 `
  -Arm m -DataRoot D:/Data/hardcrack/HardCrack_v1.0 `
  -CudaVisibleDevices 1 -Seed 42 -RunTag pilot1 -DryRun
```

Training must be invoked through **schtasks + PowerShell**, as required by
`scripts/README.md`. For example, register a task whose action is:

```powershell
powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass `
  -File D:/mmsegmentation/projects/wavecrack_mamba/scripts/run_hardcrack_r1a.ps1 `
  -Arm m -DataRoot D:/Data/hardcrack/HardCrack_v1.0 `
  -CudaVisibleDevices 1 -Seed 42 -RunTag pilot1
```

Every full run first executes bounded smoke, with two validation points:
baselines use two epochs over 48 official tiles at batch 8 (12 updates);
frozen pilots use 20 updates at batch 1 over the same subset. Four native
validation images cover all three sources. Any nonzero exit aborts the full
phase. -SmokeOnly runs just this check. It verifies integration, selection
and completion records; it does not establish convergence or accuracy.

For `t1`, `t3` or `conv3`, also supply `-BaseCheckpoint` pointing to the completed
HardCrack m checkpoint named by completed_run.json's selection field.
The wrapper computes its digest and checks the completion receipt before
training. The selected file can precede the final epoch. A smoke base cannot
qualify a full run. Frozen lengths remain a separate calibration budget;
they are not automatically expanded to 64 epochs.

Outputs are isolated under
`work_dirs/wavecrack_mamba/hardcrack_r1a/<arm>_seed<seed>_<RunTag>/`.
The `_smoke/` subdirectory contains smoke checkpoints and logs; `resolved.py`
records the exact full configuration. Existing checkpoint directories are
not overwritten: use a new run tag. This wrapper does not implement resume;
the strict hook supports a separately prepared explicit Runner resume with
matching provenance.

## Checks

For standalone evaluation, use `tools/evaluate_hardcrack.py` with `--config`,
`--checkpoint`, `--sha256`, `--work-dir` and `--phase val` (or `test` for a
configuration declaring the formal test split). This verifies the selected
local checkpoint's digest and writes a separate weights-only evaluation copy.
It avoids the PyTorch 2.6+ / MMEngine `HistoryBuffer` pickle incompatibility
without globally disabling weights-only loading. Frozen evaluation does not
require the original base checkpoint. An example under the repository root:

```bash
PYTHONPATH=. python projects/wavecrack_mamba/tools/evaluate_hardcrack.py \
  --config RUN/resolved.py --checkpoint RUN/SELECTED_CHECKPOINT.pth \
  --sha256 ACTUAL_CHECKPOINT_SHA256 --work-dir EVAL_OUTPUT --phase val
```

```powershell
python -m pytest tests/test_models/test_wcm_hardcrack.py tests/test_models/test_wcm_hardcrack_64ods.py -q
```

Tests cover current configuration inheritance, rejection of original-image
training inputs and silent resizing/subsets, release drift, checkpoint
origin/completion/selection/smoke rules, equal common initialization tensors,
64-epoch learning-rate boundaries, and streaming ODS parity.
Smoke hooks additionally assert binary tile labels, finite training outputs
and prediction sizes matching native validation images.
