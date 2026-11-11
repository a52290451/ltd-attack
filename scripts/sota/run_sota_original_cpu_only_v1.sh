#!/usr/bin/env bash
# CPU-only original source/data readiness. No CUDA training or external data.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="/data/ltd-sota-bsierra/venv-cu128/bin/python"
STAGE="$1"

echo "===== PYTHON SYNTAX ====="
"$PY" -m py_compile \
  scripts/sota/s3_s9_original_sources_gate_v1.py \
  scripts/sota/s3_s9_cpu_source_readiness_v1.py \
  scripts/sota/s9_netclr_authors_cpu_gate_v1.py \
  scripts/sota/s9_netclr_authors_prepare_cpu_v1.py \
  scripts/sota/s9_netclr_authors_augment_historical_audit_v1.py

echo "===== SOURCE PINS ====="
"$PY" -u scripts/sota/s3_s9_original_sources_gate_v1.py --stage inspect

echo "===== ORIGINAL SOURCE INVENTORY ====="
"$PY" -u scripts/sota/s3_s9_cpu_source_readiness_v1.py

echo "===== AUTHORS' NETCLR MODEL CPU SMOKE ====="
"$PY" -u scripts/sota/s9_netclr_authors_cpu_gate_v1.py --stage smoke

case "$STAGE" in
  quick)
    echo "SOTA_ORIGINAL_CPU_QUICK_PASS" ;;
  prepare-dev)
    echo "===== HISTORICAL DEV CPU CACHE ====="
    "$PY" -u scripts/sota/s9_netclr_authors_prepare_cpu_v1.py
    echo "SOTA_ORIGINAL_CPU_PREPARE_DEV_PASS" ;;
  audit-augment)
    echo "===== ORIGINAL NETCLR HISTORICAL AUGMENTATION AUDIT ====="
    "$PY" -u scripts/sota/s9_netclr_authors_augment_historical_audit_v1.py --write
    echo "SOTA_ORIGINAL_CPU_AUGMENT_DIAGNOSTIC_COMPLETED" ;;
  *)
    echo "USAGE: bash scripts/sota/run_sota_original_cpu_only_v1.sh {quick|prepare-dev|audit-augment}"
    exit 2 ;;
esac
echo "NO_CUDA_TRAINING_STARTED"
echo "NO_EXTERNAL_COHORTS_OR_DATASET_C_OPENED"
