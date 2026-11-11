#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL="$1"
STAGE="$2"
case "$MODEL" in DF|VarCNN|TikTok|RoFiRe);; *) echo "model must be DF, VarCNN, TikTok or RoFiRe"; exit 2;; esac
PY="/data/ltd-sota-bsierra/venv-keras-authors/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "BLOCKED_KERAS_ENV_MISSING: $PY"
  echo "Create an isolated compatible Keras/TensorFlow venv; do not replace the PyTorch CUDA environment."
  exit 5
fi
MAIN="scripts/sota/s3_s4_s8_authors_keras_v1.py"
mkdir -p /data/ltd-attack/sota/authors_keras_logs
log="/data/ltd-attack/sota/authors_keras_logs/$(date -u +%Y%m%dT%H%M%SZ)_$MODEL-$STAGE.log"
case "$STAGE" in
  smoke) CUDA_VISIBLE_DEVICES=-1 "$PY" -u "$MAIN" --model "$MODEL" --stage smoke 2>&1 | tee "$log" ;;
  B|C)
    active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
    if [[ -n "$active" ]]; then echo "BLOCKED_GPU_OCCUPIED $active"; exit 5; fi
    "$PY" -u "$MAIN" --model "$MODEL" --stage "$STAGE" 2>&1 | tee "$log"
    ;;
  09B|Future-B)
    cohort="09B"
    if [[ "$STAGE" == "Future-B" ]]; then cohort="10B_DEV_FROZEN"; fi
    CUDA_VISIBLE_DEVICES=-1 "$PY" -u "$MAIN" --model "$MODEL" --stage evaluate --cohort "$cohort" 2>&1 | tee "$log"
    ;;
  all)
    bash "$0" "$MODEL" smoke
    bash "$0" "$MODEL" B
    bash "$0" "$MODEL" C
    bash "$0" "$MODEL" 09B
    bash "$0" "$MODEL" Future-B ;;
  *) echo "stage must be smoke|B|C|09B|Future-B|all"; exit 2 ;;
esac
