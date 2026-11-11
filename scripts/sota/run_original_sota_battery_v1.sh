#!/usr/bin/env bash
# LTD-Attack author-source SOTA battery; explicit stages and GPU-safe orchestration.
# CUMUL S7 original-source reproduction is NOT included: public source incomplete.
# VarCNN is the authors' DIRECTION BRANCH only, not full original ensemble.
# S13 Proteus future adaptation not comparable under frozen 09B/Future-B protocol.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE="$1"
PY="${LTD_SOTA_PYTORCH_PY:-/data/ltd-sota-bsierra/venv-cu128/bin/python}"
KPY="${LTD_SOTA_KERAS_PY:-/data/ltd-sota-bsierra/venv-keras-authors/bin/python}"
SOTA="scripts/sota"
check_gpu() {
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -n "$active" ]]; then
    echo "BLOCKED_GPU_OCCUPIED $active"; exit 5
  fi
}
case "$MODE" in
  cpu-preflight)
    echo "===== SHELL SYNTAX ====="
    for script in "$SOTA"/run_*authors*.sh "$SOTA"/run_original_sota_battery_v1.sh "$SOTA"/setup_isolated_authors_keras_v1.sh; do
      bash -n "$script"
    done
    echo "SHELL_SYNTAX_PASS"
    echo "===== COMPILE NEW SOTA PIPELINES ====="
    "$PY" -m py_compile \
      "$SOTA/s9_netclr_authors_pretrain_b_v1.py" \
      "$SOTA/s9_netclr_authors_supervised_b_v1.py" \
      "$SOTA/s9_netclr_authors_frozen_c_eval_v1.py" \
      "$SOTA/s9_netclr_authors_results_gate_v1.py" \
      "$SOTA/s3_s4_s8_authors_keras_v1.py" \
      "$SOTA/s11_swallow_authors_v1.py" \
      "$SOTA/s12_wfcat_authors_v1.py" \
      "$SOTA/original_authors_results_integrity_v1.py" \
      "$SOTA/original_sota_orchestrator_v2.py" \
      "$SOTA/sota_zeus_evidence_bootstrap_v1.py" \
      "$SOTA/original_sota_paired_date_stats_v1.py" \
      "$SOTA/sota_paths_v1.py" \
      "$SOTA/test_original_sota_contract_v1.py" \
      "$SOTA/run_original_sota_paired_comparisons_v1.py" \
      "$SOTA/sota_host_readiness_certificate_v1.py"
    echo "===== OFFLINE END-TO-END STAGE CONTRACT TESTS ====="
    CUDA_VISIBLE_DEVICES=-1 "$PY" -u "$SOTA/test_original_sota_contract_v1.py"
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
    CUDA_VISIBLE_DEVICES=-1 "$PY" "$SOTA/sota_host_readiness_certificate_v1.py" --stage record --name cpu-preflight
    echo "ORIGINAL_SOTA_CPU_PREFLIGHT_PASS_NO_CUDA_TRAINING" ;;
  cpu-integration)
    echo "===== CPU ORIGINAL-AUTHORS GRADIENT/SERIALIZATION INTEGRATION ====="
    CUDA_VISIBLE_DEVICES=-1 bash "$SOTA/run_s11_swallow_authors_v1.sh" integration-smoke
    CUDA_VISIBLE_DEVICES=-1 bash "$SOTA/run_s12_wfcat_authors_v1.sh" integration-smoke
    [[ -x "$KPY" ]] || { echo "BLOCKED_KERAS_ENV_MISSING"; exit 5; }
    for model in DF VarCNN TikTok RoFiRe; do
      CUDA_VISIBLE_DEVICES=-1 bash "$SOTA/run_authors_keras_s3_s4_s8_v1.sh" "$model" integration-smoke
    done
    CUDA_VISIBLE_DEVICES=-1 "$PY" "$SOTA/sota_host_readiness_certificate_v1.py" --stage record --name cpu-integration
    echo "ORIGINAL_SOTA_CPU_TRAIN_SAVE_RELOAD_INTEGRATION_PASS" ;;
  host-check)
    CUDA_VISIBLE_DEVICES=-1 "$PY" -u "$SOTA/sota_zeus_evidence_bootstrap_v1.py" --stage inspect ;;
  statistics)
    CUDA_VISIBLE_DEVICES=-1 "$PY" -u "$SOTA/run_original_sota_paired_comparisons_v1.py" ;;
  cpu-prepare)
    CUDA_VISIBLE_DEVICES=-1 bash "$SOTA/run_s9_netclr_authors_v1.sh" preflight
    echo "===== Historical TikTok DT cache (shared data only; NO WFlib fit) ====="
    CUDA_VISIBLE_DEVICES=-1 "$PY" -c "from scripts.sota.s8_tiktok_train_b_v1 import load_data; a=load_data(); print('S8_TIKTOK_AUTHOR_INPUT_CACHE_PASS', len(a[1]))"
    CUDA_VISIBLE_DEVICES=-1 bash "$SOTA/run_s11_swallow_authors_v1.sh" prepare
    CUDA_VISIBLE_DEVICES=-1 bash "$SOTA/run_s12_wfcat_authors_v1.sh" prepare
    CUDA_VISIBLE_DEVICES=-1 "$PY" "$SOTA/sota_host_readiness_certificate_v1.py" --stage record --name cpu-prepare
    echo "ORIGINAL_SOTA_HISTORICAL_CPU_CACHE_PASS" ;;
  gpu-all)
    echo "Use guarded/resumable v2 orchestrator; old one-shot is deprecated."
    "$PY" "$SOTA/original_sota_orchestrator_v2.py" --stage all --run ;;
  *)
    echo "Usage: bash $0 {cpu-preflight|cpu-integration|cpu-prepare|host-check|statistics|gpu-all}"
    exit 2 ;;
esac
