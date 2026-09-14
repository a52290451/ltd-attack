#!/bin/bash
# ==========================================================================
#  run_baselines.sh  —  Entrenamiento y evaluación de baselines de vectores
# ==========================================================================
#
#  Uso:
#    chmod +x run_baselines.sh
#    ./run_baselines.sh [train|test|all]
#
#  Descripción:
#    - train : Entrena el modelo multimodal (Dir + Peso) con secuencias de 3000.
#    - test  : Evalúa todos los modelos (.pth) contra el hold‑out set.
#    - all   : Ejecuta train y después test (por defecto).
# ==========================================================================

set -euo pipefail

MODE="${1:-all}"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

echo "Directorio del proyecto: $PROJECT_ROOT"
echo "Modo: $MODE"

cd "$PROJECT_ROOT"

case "$MODE" in
  train)
    echo ""
    echo "=============================================="
    echo "  ENTRENANDO BASELINE MULTIMODAL 3000 (VP1)"
    echo "=============================================="
    "$PYTHON" -m src.models.micro.VP1_Transformers_dir_size
    ;;
  test)
    echo ""
    echo "=============================================="
    echo "  EVALUANDO CONCEPT DRIFT + AUTOPSIA (VP_Eval)"
    echo "=============================================="
    "$PYTHON" -m src.evaluation.VP_Evaluar_ConceptDrift_Vectores
    ;;
  all)
    echo ""
    echo "=============================================="
    echo "  PIPELINE COMPLETO: TRAIN + TEST + DRIFT"
    echo "=============================================="
    "$PYTHON" -m src.models.micro.VP1_Transformers_dir_size
    "$PYTHON" -m src.evaluation.VP_Evaluar_ConceptDrift_Vectores
    ;;
  *)
    echo "Uso: $0 [train|test|all]"
    exit 1
    ;;
esac

echo ""
echo "Listo."
