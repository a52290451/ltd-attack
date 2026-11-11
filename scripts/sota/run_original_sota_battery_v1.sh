#!/usr/bin/env bash
# LTD-Attack author-source SOTA battery; explicit stages and GPU-safe orchestration.
# CUMUL S7 original-source reproduction is NOT included: public source incomplete.
# VarCNN is the authors' DIRECTION BRANCH only, not full original ensemble.
# S13 Proteus future adaptation not comparable under frozen 09B/Future-B protocol.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE="$1"
PY="/data/ltd-sota-bsierra/venv-cu128/bin/python"
KPY="/data/ltd-sota-bsierra/venv-keras-authors/bin/python"
SOTA="scripts/sota"
check_gpu() {
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -n "$active" ]]; then
    echo "BLOCKED_GPU_OCCUPIED $active"; exit 5
  fi
}
case "$MODE" in
  cpu-preflight)
    echo "===== COMPILE NEW SOTA PIPELINES ====="
    "$PY" -m py_compile \
      "$SOTA/s9_netclr_authors_pretrain_b_v1.py" \
      "$SOTA/s9_netclr_authors_supervised_b_v1.py" \
      "$SOTA/s9_netclr_authors_frozen_c_eval_v1.py" \
      "$SOTA/s9_netclr_authors_results_gate_v1.py" \
      "$SOTA/s3_s4_s8_authors_keras_v1.py" \
      "$SOTA/s11_swallow_authors_v1.py" \
      "$SOTA/s12_wfcat_authors_v1.py"
    echo "===== VERIFY PINNED AUTHORS REPOSITORIES ====="
    python3 "$SOTA/s3_s9_original_sources_gate_v1.py" --stage inspect
    "$PY" "$SOTA/s10_s13_upstream_source_gate_v1.py" --stage inspect
    echo "===== ORIGINAL NETCLR CPU MODEL + CONTRASTIVE ====="
    CUDA_VISIBLE_DEVICES=-1 bash "$SOTA/run_s9_netclr_authors_v1.sh" preflight
    echo "===== ORIGINAL SWALLOW CPU ====="
    CUDA_VISIBLE_DEVICES=-1 bash "$SOTA/run_s11_swallow_authors_v1.sh" smoke
    echo "===== ORIGINAL WFCAT CPU ====="
    CUDA_VISIBLE_DEVICES=-1 bash "$SOTA/run_s12_wfcat_authors_v1.sh" smoke
    if [[ -x "$KPY" ]]; then
      bash "$SOTA/setup_isolated_authors_keras_v1.sh" check
      for model in DF VarCNN TikTok RoFiRe; do
        bash "$SOTA/run_authors_keras_s3_s4_s8_v1.sh" "$model" smoke
      done
    else
      echo "KERAS_VENV_NOT_INSTALLED: run setup_isolated_authors_keras_v1.sh install"
    fi
    echo "ORIGINAL_SOTA_CPU_PREFLIGHT_PASS_NO_CUDA_TRAINING" ;;
  cpu-prepare)
    CUDA_VISIBLE_DEVICES=-1 bash "$SOTA/run_s9_netclr_authors_v1.sh" preflight
    CUDA_VISIBLE_DEVICES=-1 bash "$SOTA/run_s11_swallow_authors_v1.sh" prepare
    CUDA_VISIBLE_DEVICES=-1 bash "$SOTA/run_s12_wfcat_authors_v1.sh" prepare
    echo "ORIGINAL_SOTA_HISTORICAL_CPU_CACHE_PASS" ;;
  gpu-all)
    check_gpu
    [[ -x "$KPY" ]] || {
      echo "BLOCKED: isolated original Keras environment missing";exit 5;
    }
    # Original-author models; separate output paths, fail-fast on any stage error.
    bash "$SOTA/run_s9_netclr_authors_v1.sh" all
    bash "$SOTA/run_s11_swallow_authors_v1.sh" all
    bash "$SOTA/run_s12_wfcat_authors_v1.sh" all
    for model in DF VarCNN TikTok RoFiRe; do
      bash "$SOTA/run_authors_keras_s3_s4_s8_v1.sh" "$model" all
    done
    echo "SOTA_BATTERY_AUTHORS_EXECUTED_VERIFY_FROZEN_EVIDENCE" ;;
  *)
    echo "Usage: bash $0 {cpu-preflight|cpu-prepare|gpu-all}"
    exit 2 ;;
esac
