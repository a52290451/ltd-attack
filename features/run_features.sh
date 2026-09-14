#!/bin/bash
# ==============================================================================
# 🚀 PIPELINE MAESTRO: EVALUACIÓN DE FEATURES Y DIAGNÓSTICO DE CONCEPT DRIFT
# ==============================================================================
set -e

echo "================================================================="
echo "⚙️ INICIANDO ENTRENAMIENTO Y EVALUACIÓN DE MODELOS BASE (MACRO)"
echo "================================================================="

echo -e "\n▶️ [1/10] Entrenando Hourly Encoder Original (EXP1)..."
python3 EXP1_Train_Invariantes.py

echo -e "\n▶️ [2/10] Evaluando Resiliencia del Hourly Encoder vs Drift (EXP1)..."
python3 EXP1_Evaluar_Invariantes.py

echo -e "\n▶️ [3/10] Entrenando y Evaluando XGBoost sobre Meta-Features 24h (EXP2)..."
python3 EXP2_XGBoost_MetaFeatures_24h.py

echo "================================================================="
echo "🎨 GENERANDO EVIDENCIA VISUAL Y DIAGNÓSTICO (ESTUDIO DE ABLACIÓN)"
echo "================================================================="

echo -e "\n▶️ [4/10] Generando Gráficas 3 y 4 (Ranking de Invariantes y PCA Base)..."
python3 GEN_01_Ranking_Outliers.py

echo -e "\n▶️ [5/10] Generando Gráfica 2 (Degradación Predictiva)..."
python3 GEN_01B_Degradacion_Drift.py

echo -e "\n▶️ [6/10] Generando Gráfica 5 (PCA de Meta-Features 752D)..."
python3 GEN_02_PCA.py

echo -e "\n▶️ [7/10] Generando Gráfica 6 (Matriz de Ortogonalidad de Pearson)..."
python3 GEN_03_Ortogonalidad.py

echo -e "\n▶️ [8/10] Generando Gráficas 8A y 8B (Z-Shift KDE - Mejor y Peor Caso)..."
python3 GEN_04_PSI_KDE.py

echo -e "\n▶️ [9/10] Generando Gráfica 7 (Autopsia de Traslación Latente)..."
python3 GEN_05_PCA_Traslacion.py

echo -e "\n▶️ [10/10] Generando Gráficas 9 y 10 (Similitud Coseno e Intra/Inter Clase)..."
python3 GEN_06_Similitud_Metric.py

echo "================================================================="
echo "✅ BATERÍA DE EXPERIMENTOS 'FEATURES' FINALIZADA CON ÉXITO."
echo "================================================================="