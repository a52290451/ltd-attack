#!/usr/bin/env bash
set -euo pipefail

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  echo "[$(timestamp)] $*"
}

if [[ "${LTD_ENV:-}" != "zeus" ]]; then
  echo "[$(timestamp)] ERROR: LTD_ENV=zeus es obligatorio" >&2
  exit 2
fi

export PYTHONPATH=.
SKIP_14A=false
if [[ "${1:-}" == "--skip-14a" ]]; then
  SKIP_14A=true
elif [[ "$#" -gt 0 ]]; then
  echo "[$(timestamp)] ERROR: opción desconocida: $1" >&2
  exit 2
fi
export P13_B1_SKIP_14A="$SKIP_14A"

status=FAILED
on_exit() {
  local code=$?
  if [[ "$code" -eq 0 ]]; then
    status=SUCCESS
  fi
  log "P13-B1 STATUS ${status}"
  exit "$code"
}
trap on_exit EXIT

log "Inicio Phase 13B + 13C con --resume"
python3 scripts/train/run_phase13_battery.py \
  --config configs/experiments/PHASE13_BATTERY_V1.yaml \
  --resume

if [[ "$SKIP_14A" == "false" ]]; then
  log "Inicio diagnóstico 14A Future-B"
  python3 scripts/diagnostics/run_future_rank_gap_battery.py \
    --config configs/experiments/FUTURE_RANK_GAP_V1.yaml
else
  log "14A omitido por --skip-14a"
fi

log "Validación final de outputs"
python3 - <<'PY'
from pathlib import Path

from src.utils.paths import result_path

phase13 = Path(result_path("micro", "MICRO-ROBUSTNESS-PHASE13-BATTERY"))
phase13c = Path(result_path("hybrid", "LTD-ROBUSTNESS-PHASE13-BATTERY"))
required_13b = ["13B_model_summary.csv", "13B_pairwise_deltas.csv", "13B_day_block_bootstrap.csv", "13B_expert_diversity.csv", "13B_training_summary.csv", "13B_rank_gap_historical.csv", "13B_manifest.json"]
required_13c = ["13C_factorial_summary.csv", "13C_factorial_effects.csv", "13C_pairwise_deltas.csv", "13C_day_block_bootstrap.csv", "13C_rank_gap_historical.csv", "13C_manifest.json"]
for directory, names in ((phase13, required_13b), (phase13c, required_13c)):
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise SystemExit(f"Faltan outputs en {directory}: {missing}")
import os
if os.environ.get("P13_B1_SKIP_14A") != "true" and not Path(result_path("final", "FUTURE-RANK-GAP-PHASE14", "14A_manifest.json")).is_file():
    raise SystemExit("Falta 14A_manifest.json")
print("Outputs 13B/13C validados")
PY

log "Fin correcto de P13-B1"
