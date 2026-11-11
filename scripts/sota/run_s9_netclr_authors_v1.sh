#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${LTD_SOTA_PYTORCH_PY:-/data/ltd-sota-bsierra/venv-cu128/bin/python}"
STAGE="$1"
LOGS="${LTD_SOTA_ARTIFACT_ROOT:-/data/ltd-attack/sota}/s9_netclr_author_original_v1/logs"
case "$STAGE" in
  preflight)
    "$PY" -m py_compile scripts/sota/s9_netclr_authors_pretrain_b_v1.py \
      scripts/sota/s9_netclr_authors_supervised_b_v1.py \
      scripts/sota/s9_netclr_authors_frozen_c_eval_v1.py \
      scripts/sota/s9_netclr_authors_results_gate_v1.py
    "$PY" -u scripts/sota/s9_netclr_authors_pretrain_b_v1.py --stage smoke ;;
  pretrain-B|supervised-B|C|09B|Future-B|results)
    if [[ "$STAGE" == "pretrain-B" || "$STAGE" == "supervised-B" || "$STAGE" == "C" ]]; then
      active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
      if [[ -n "$active" ]]; then echo "BLOCKED_GPU_OCCUPIED $active"; exit 5; fi
    fi
    mkdir -p "$LOGS"
    log="$LOGS/$(date -u +%Y%m%dT%H%M%SZ)_$STAGE.log"
    echo "S9_AUTHOR_STAGE=$STAGE LOG=$log"
    case "$STAGE" in
      pretrain-B) "$PY" -u scripts/sota/s9_netclr_authors_pretrain_b_v1.py --stage B 2>&1 | tee "$log" ;;
      supervised-B) "$PY" -u scripts/sota/s9_netclr_authors_supervised_b_v1.py --stage train-B 2>&1 | tee "$log" ;;
      C) "$PY" -u scripts/sota/s9_netclr_authors_frozen_c_eval_v1.py --stage C 2>&1 | tee "$log" ;;
      09B) "$PY" -u scripts/sota/s9_netclr_authors_frozen_c_eval_v1.py --stage evaluate --cohort 09B 2>&1 | tee "$log" ;;
      Future-B) "$PY" -u scripts/sota/s9_netclr_authors_frozen_c_eval_v1.py --stage evaluate --cohort 10B_DEV_FROZEN 2>&1 | tee "$log" ;;
      results) "$PY" -u scripts/sota/s9_netclr_authors_results_gate_v1.py 2>&1 | tee "$log" ;;
    esac ;;
  all)
    bash "$0" preflight
    bash "$0" pretrain-B
    bash "$0" supervised-B
    bash "$0" C
    bash "$0" 09B
    bash "$0" Future-B
    bash "$0" results ;;
  *) echo "Usage: bash $0 {preflight|all|pretrain-B|supervised-B|C|09B|Future-B|results}"; exit 2 ;;
esac
