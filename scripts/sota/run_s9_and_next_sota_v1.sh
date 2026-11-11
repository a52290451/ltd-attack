#!/usr/bin/env bash
# LTD S9 NetCLR: one stage at a time, evidence-first and no auto-overwrite.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="/data/ltd-sota-bsierra/venv-cu128/bin/python"
OUT="/data/ltd-attack/sota/s9_netclr_wflib_adapted_seed42"
LOGDIR="/data/ltd-attack/sota/s9_netclr_orchestration_logs"
STAGE="$1"
case "$STAGE" in
  preflight)
    "$PY" -m py_compile \
      scripts/sota/s9_netclr_pretrain_b_v1.py \
      scripts/sota/s9_netclr_supervised_b_v1.py \
      scripts/sota/s9_netclr_frozen_c_eval_v1.py \
      scripts/sota/s10_s13_upstream_source_gate_v1.py
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
    log="$LOGDIR/s9_$STAGE_$stamp.log"
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
  audit-upstream)
    "$PY" -u scripts/sota/s10_s13_upstream_source_gate_v1.py --stage inspect ;;
  fetch-upstream)
    "$PY" -u scripts/sota/s10_s13_upstream_source_gate_v1.py --stage fetch ;;
  *)
    echo "USAGE: bash scripts/sota/run_s9_and_next_sota_v1.sh {preflight|B|C|09B|Future-B|audit-upstream|fetch-upstream}"
    exit 2 ;;
esac
