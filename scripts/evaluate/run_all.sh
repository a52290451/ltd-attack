#!/bin/bash
# ==============================================================================
# 🚀 SCRIPT DE ORQUESTACIÓN FINAL: ESTUDIO DE ABLACIÓN Y CONCEPT DRIFT
# ==============================================================================
# Detener el script inmediatamente si ocurre un error crítico
set -e

# Resolver el proyecto independientemente del cwd del llamador.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
cd "$PROJECT_ROOT"

echo "================================================================="
echo "⚙️ INICIANDO SECUENCIA AUTOMÁTICA (POST-EXP4 TRAINING)"
echo "================================================================="

# ---------------------------------------------------------
# 1. EVALUAR EXP4 (El Padre)
# ---------------------------------------------------------
echo -e "\n▶️ [1/9] Evaluando EXP4 (Híbrido Inclinado a Vectores) frente al Drift..."
"$PYTHON" -m src.evaluation.EXP4_Evaluar_94F3000V_InclinadoV

# ---------------------------------------------------------
# 2. ENTRENAR Y EVALUAR EXP7 (El Hijo - DML)
# ---------------------------------------------------------
echo -e "\n▶️ [2/9] Entrenando EXP7 (Transfer Learning + Fine-Tuning DML)..."
"$PYTHON" -m src.models.dml.EXP7_Train_Transfer_DML

echo -e "\n▶️ [3/9] Evaluando EXP7 (Two-Stage DML) frente al Drift..."
"$PYTHON" -m src.evaluation.EXP7_Evaluar_Transfer_DML

# ---------------------------------------------------------
# 3. ENTRENAR Y EVALUAR EXP2 (Híbrido Neutro)
# ---------------------------------------------------------
echo -e "\n▶️ [4/9] Entrenando EXP2 (Híbrido Neutro Optimizado)..."
"$PYTHON" -m src.models.hybrid.EXP2_Train_94F3000V_Neutro

echo -e "\n▶️ [5/9] Evaluando EXP2 frente al Drift..."
"$PYTHON" -m src.evaluation.EXP2_Evaluar_94F3000V_Neutro

# ---------------------------------------------------------
# 4. ENTRENAR Y EVALUAR EXP3 (Inclinado a Features)
# ---------------------------------------------------------
echo -e "\n▶️ [6/9] Entrenando EXP3 (Inclinado a Features Optimizado)..."
"$PYTHON" -m src.models.hybrid.EXP3_Train_94F3000V_InclinadoF

echo -e "\n▶️ [7/9] Evaluando EXP3 frente al Drift..."
"$PYTHON" -m src.evaluation.EXP3_Evaluar_94F3000V_InclinadoF

# ---------------------------------------------------------
# 5. ENTRENAR Y EVALUAR EXP5 (Cross-Attention)
# ---------------------------------------------------------
echo -e "\n▶️ [8/9] Entrenando EXP5 (Gated Cross-Attention Optimizado)..."
"$PYTHON" -m src.models.hybrid.EXP5_Train_94F3000V_CrossAttention

echo -e "\n▶️ [9/9] Evaluando EXP5 frente al Drift..."
"$PYTHON" -m src.evaluation.EXP5_Evaluar_94F3000V_CrossAttention

echo "================================================================="
echo "✅ LA BATERÍA DE EXPERIMENTOS HA FINALIZADO CON ÉXITO."
echo "================================================================="
