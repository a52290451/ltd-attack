#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${LTD_SOTA_PYTORCH_PY:-/data/ltd-sota-bsierra/venv-cu128/bin/python}"
MAIN="scripts/sota/s11_swallow_authors_v1.py"
STAGE="$1"
LOGS="${LTD_SOTA_ARTIFACT_ROOT:-/data/ltd-attack/sota}/s11_swallow_authors_seed42/logs"
mkdir -p "$LOGS"
log="$LOGS/$(date -u +%Y%m%dT%H%M%SZ)_$STAGE.log"
case "$STAGE" in
  smoke|prepare)
    CUDA_VISIBLE_DEVICES=-1 "$PY" -u "$MAIN" --stage "$STAGE" 2>&1 | tee "$log" ;;
  pretrain-B|supervised-B|C)
    active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
    if [[ -n "$active" ]]; then echo "BLOCKED_GPU_OCCUPIED $active"; exit 5; fi
    "$PY" -u "$MAIN" --stage "$STAGE" 2>&1 | tee "$log" ;;
  09B|Future-B)
    cohort="09B";if [[ "$STAGE" == "Future-B" ]]; then cohort="10B_DEV_FROZEN";fi
    CUDA_VISIBLE_DEVICES=-1 "$PY" -u "$MAIN" --stage evaluate --cohort "$cohort" 2>&1 | tee "$log" ;;
  all)
    bash "$0" smoke
    bash "$0" prepare
    bash "$0" pretrain-B
    bash "$0" supervised-B
    bash "$0" C
    bash "$0" 09B
    bash "$0" Future-B ;;
  *) echo "Usage: $0 {smoke|prepare|pretrain-B|supervised-B|C|09B|Future-B|all}";exit 2 ;;
esac
