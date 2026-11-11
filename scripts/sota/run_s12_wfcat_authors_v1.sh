#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${LTD_SOTA_PYTORCH_PY:-/data/ltd-sota-bsierra/venv-cu128/bin/python}"
STAGE="$1"
MAIN="scripts/sota/s12_wfcat_authors_v1.py"
mkdir -p "${LTD_SOTA_ARTIFACT_ROOT:-/data/ltd-attack/sota}/s12_wfcat_authors_iat_seed42/logs
log="${LTD_SOTA_ARTIFACT_ROOT:-/data/ltd-attack/sota}/s12_wfcat_authors_iat_seed42/logs/$(date -u +%Y%m%dT%H%M%SZ)_$STAGE.log"
case "$STAGE" in
  smoke|integration-smoke|prepare)
    CUDA_VISIBLE_DEVICES=-1 "$PY" -u "$MAIN" --stage "$STAGE" 2>&1 | tee "$log" ;;
  B|C)
    active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
    if [[ -n "$active" ]]; then echo "BLOCKED_GPU_OCCUPIED $active"; exit 5; fi
    "$PY" -u "$MAIN" --stage "$STAGE" 2>&1 | tee "$log" ;;
  09B|Future-B)
    cohort="09B";if [[ "$STAGE" == "Future-B" ]]; then cohort="10B_DEV_FROZEN"; fi
    CUDA_VISIBLE_DEVICES=-1 "$PY" -u "$MAIN" --stage evaluate --cohort "$cohort" 2>&1 | tee "$log" ;;
  all)
    bash "$0" smoke
    bash "$0" prepare
    bash "$0" B
    bash "$0" C
    bash "$0" 09B
    bash "$0" Future-B ;;
  *) echo "Usage: $0 {smoke|prepare|B|C|09B|Future-B|all}"; exit 2 ;;
esac
