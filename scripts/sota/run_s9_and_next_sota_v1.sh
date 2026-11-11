#!/usr/bin/env bash
# LTD S9 NetCLR: one stage at a time, evidence-first and no auto-overwrite.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="/data/ltd-sota-bsierra/venv-cu128/bin/python"
OUT="/data/ltd-attack/sota/s9_netclr_wflib_adapted_seed42"
LOGDIR="/data/ltd-attack/sota/s9_netclr_orchestration_logs"
STAGE="$1"
# PI policy: author-source-first. Current S9 stages are WFlib-adapted only,
# NEVER silently treat them as original NetCLR authors' reproduction.
case "$STAGE" in
  all|B|C|09B|Future-B|results)
    if [[ "${LTD_ALLOW_WFLIB_ADAPTED_S9:-NO}" != "YES" ]]; then
      echo "BLOCKED_AUTHOR_SOURCE_FIRST: This runner trains/evaluates WFlib-adapted NetCLR, NOT the authors' NetCLR."
      echo "Original NetCLR exists at SPIN-UMass; prioritize original notebook adapter."
      echo "To run an expressly labeled ADAPTED secondary benchmark, export LTD_ALLOW_WFLIB_ADAPTED_S9=YES."
      exit 6
    fi ;;
esac
case "$STAGE" in
  preflight)
    "$PY" -m py_compile \
      scripts/sota/s9_netclr_pretrain_b_v1.py \
      scripts/sota/s9_netclr_supervised_b_v1.py \
      scripts/sota/s9_netclr_frozen_c_eval_v1.py \
      scripts/sota/s10_s13_upstream_source_gate_v1.py \
      scripts/sota/s9_netclr_results_gate_v1.py
    "$PY" -u scripts/sota/s9_netclr_supervised_b_v1.py --stage smoke
    "$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
m=json.loads((p/"pretrain_B_v2/manifest.json").read_text())
assert m["n"]==39258
assert m["train_splits"]==["DEV_EARLY","DEV_MIDDLE"]
assert m["external_data_opened"] is False
assert m["dataset_c_opened"] is False
print("NETCLR_PRETRAIN_SOURCE_CONTRACT_PASS",m["checkpoint_sha256"])
PY
    ;;
  all)
    # Strict end-to-end S9 sequence. Fail fast; never skip or overwrite stages.
    bash "$0" preflight
    bash "$0" B
    bash "$0" C
    bash "$0" 09B
    bash "$0" Future-B
    bash "$0" results
    ;;
  B|C|09B|Future-B)
    if [[ "$STAGE" == "B" || "$STAGE" == "C" ]]; then
      command -v nvidia-smi >/dev/null || { echo "NVIDIA_SMI_MISSING"; exit 5; }
      active="$(nvidia-smi --query-compute-apps=pid \
          --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
      [[ -z "$active" ]] || {
        echo "BLOCKED_GPU_OCCUPIED: $active — coordinate before training"
        exit 5
      }
    fi
    mkdir -p "$LOGDIR"
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    log="$LOGDIR/s9_${STAGE}_${stamp}.log"
    echo "LTD_S9_STAGE=$STAGE LOG=$log"
    case "$STAGE" in
      B) "$PY" -u scripts/sota/s9_netclr_supervised_b_v1.py \
              --stage train-B 2>&1 | tee "$log" ;;
      C) "$PY" -u scripts/sota/s9_netclr_frozen_c_eval_v1.py \
              --stage C 2>&1 | tee "$log" ;;
      09B) "$PY" -u scripts/sota/s9_netclr_frozen_c_eval_v1.py \
              --stage evaluate --cohort 09B 2>&1 | tee "$log" ;;
      Future-B) "$PY" -u scripts/sota/s9_netclr_frozen_c_eval_v1.py \
              --stage evaluate --cohort 10B_DEV_FROZEN 2>&1 | tee "$log" ;;
    esac
    ;;
  results)
    "$PY" -u scripts/sota/s9_netclr_results_gate_v1.py ;;
  audit-upstream)
    "$PY" -u scripts/sota/s10_s13_upstream_source_gate_v1.py --stage inspect ;;
  fetch-upstream)
    "$PY" -u scripts/sota/s10_s13_upstream_source_gate_v1.py --stage fetch ;;
  *)
    echo "USAGE: bash scripts/sota/run_s9_and_next_sota_v1.sh {preflight|all|B|C|09B|Future-B|results|audit-upstream|fetch-upstream}"
    exit 2 ;;
esac
