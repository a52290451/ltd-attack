#!/bin/bash
# ==============================================================================
# 🚀 PIPELINE MAESTRO: EVALUACIÓN DE FEATURES Y DIAGNÓSTICO DE CONCEPT DRIFT
# ==============================================================================
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
cd "$PROJECT_ROOT"

echo "================================================================="
echo "⚙️ INICIANDO ENTRENAMIENTO Y EVALUACIÓN DE MODELOS BASE (MACRO)"
echo "================================================================="

echo -e "\n▶️ [1/10] Entrenando Hourly Encoder Original (EXP1)..."
"$PYTHON" -m src.models.macro.EXP1_Train_Invariantes

echo -e "\n▶️ [2/10] Evaluando Resiliencia del Hourly Encoder vs Drift (EXP1)..."
"$PYTHON" -m src.evaluation.EXP1_Evaluar_Invariantes

echo -e "\n▶️ [3/10] Entrenando y Evaluando XGBoost sobre Meta-Features 24h (EXP2)..."
"$PYTHON" -m src.models.macro.EXP2_XGBoost_MetaFeatures_24h

echo "================================================================="
echo "🎨 GENERANDO EVIDENCIA VISUAL Y DIAGNÓSTICO (ESTUDIO DE ABLACIÓN)"
echo "================================================================="

echo -e "\n▶️ [4/10] Generando Gráficas 3 y 4 (Ranking de Invariantes y PCA Base)..."
"$PYTHON" -m src.features.GEN_01_Ranking_Outliers

echo -e "\n▶️ [5/10] Generando Gráfica 2 (Degradación Predictiva)..."
"$PYTHON" -m src.features.GEN_01B_Degradacion_Drift

echo -e "\n▶️ [6/10] Generando Gráfica 5 (PCA de Meta-Features 752D)..."
"$PYTHON" -m src.features.GEN_02_PCA

echo -e "\n▶️ [7/10] Generando Gráfica 6 (Matriz de Ortogonalidad de Pearson)..."
"$PYTHON" -m src.features.GEN_03_Ortogonalidad

echo -e "\n▶️ [8/10] Generando Gráficas 8A y 8B (Z-Shift KDE - Mejor y Peor Caso)..."
"$PYTHON" -m src.features.GEN_04_PSI_KDE

echo -e "\n▶️ [9/10] Generando Gráfica 7 (Autopsia de Traslación Latente)..."
"$PYTHON" -m src.features.GEN_05_PCA_Traslacion

echo -e "\n▶️ [10/10] Generando Gráficas 9 y 10 (Similitud Coseno e Intra/Inter Clase)..."
"$PYTHON" -m src.features.GEN_06_Similitud_Metric

echo "================================================================="
echo "✅ BATERÍA DE EXPERIMENTOS 'FEATURES' FINALIZADA CON ÉXITO."
echo "================================================================="
