#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
training="$root/training_deepcrack_crack500"
python="${WCM_PYTHON:-python}"
weights="${WCM_PRETRAINED:-$root/weights/wcm_convnext_tiny_imagenet_mmpretrain.pth}"
data="${WCM_DATA_PARENT:-$root/datasets}"
runs="${WCM_RUN_ROOT:-$root/runs}"
export PYTHONPATH="$root/wavecrack_mamba_R1a_code_20261022/wavecrack_mamba_R1a_code_20261022${PYTHONPATH:+:$PYTHONPATH}"

mode="${1:-check}"
case "$mode" in
  check)
    "$python" "$training/mentor_check.py"
    for dataset in deepcrack crack500; do
      name=DeepCrack-MixerCSeg
      [[ "$dataset" == crack500 ]] && name=Crack500-MixerCSeg
      "$python" "$training/run_pair.py" --dataset "$dataset" \
        --data-root "$data/$name" --pretrained "$weights" \
        --initialization imagenet --preflight-only
    done
    ;;
  smoke)
    "$python" "$training/run_all.py" --pretrained "$weights" \
      --initialization imagenet --data-parent "$data" --run-root "$runs" \
      --microbatch 1 --accumulation 8 --smoke
    ;;
  train)
    "$python" "$training/run_all.py" --pretrained "$weights" \
      --initialization imagenet --data-parent "$data" --run-root "$runs" \
      --microbatch 2 --accumulation 4
    ;;
  deepcrack|crack500|resume-deepcrack|resume-crack500)
    dataset="${mode#resume-}"
    name=DeepCrack-MixerCSeg
    [[ "$dataset" == crack500 ]] && name=Crack500-MixerCSeg
    extra=()
    [[ "$mode" == resume-* ]] && extra+=(--resume)
    "$python" "$training/run_pair.py" --dataset "$dataset" \
      --data-root "$data/$name" --pretrained "$weights" \
      --initialization imagenet --run-root "$runs" \
      --microbatch 2 --accumulation 4 "${extra[@]}"
    ;;
  test-deepcrack|test-crack500)
    "$python" "$training/evaluate_best.py" --dataset "${mode#test-}" \
      --data-parent "$data" --run-root "$runs"
    ;;
  *)
    echo "Usage: $0 {check|smoke|train|deepcrack|crack500|resume-deepcrack|resume-crack500|test-deepcrack|test-crack500}" >&2
    exit 2
    ;;
esac
