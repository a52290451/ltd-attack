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
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"

echo "Directorio de trabajo: $SCRIPT_DIR"
echo "Modo: $MODE"

cd "$SCRIPT_DIR"

case "$MODE" in
  train)
    echo ""
    echo "=============================================="
    echo "  ENTRENANDO BASELINE MULTIMODAL 3000 (VP1)"
    echo "=============================================="
    $PYTHON VP1_Transformers_dir_size.py
    ;;
  test)
    echo ""
    echo "=============================================="
    echo "  EVALUANDO CONCEPT DRIFT + AUTOPSIA (VP_Eval)"
    echo "=============================================="
    $PYTHON VP_Evaluar_ConceptDrift_Vectores.py
    ;;
  all)
    echo ""
    echo "=============================================="
    echo "  PIPELINE COMPLETO: TRAIN + TEST + DRIFT"
    echo "=============================================="
    $PYTHON VP1_Transformers_dir_size.py
    $PYTHON VP_Evaluar_ConceptDrift_Vectores.py
    ;;
  *)
    echo "Uso: $0 [train|test|all]"
    exit 1
    ;;
esac

echo ""
echo "Listo."