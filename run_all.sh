#!/bin/bash
# ==============================================================================
# 🚀 SCRIPT DE ORQUESTACIÓN FINAL: ESTUDIO DE ABLACIÓN Y CONCEPT DRIFT
# ==============================================================================
# Detener el script inmediatamente si ocurre un error crítico
set -e

# Definir rutas absolutas
BASE_DIR="$HOME/modelos/predicciones"
DIR_TRAIN="$BASE_DIR/vectores_features_v2"
DIR_EVAL="$BASE_DIR/concept_drift"

echo "================================================================="
echo "⚙️ INICIANDO SECUENCIA AUTOMÁTICA (POST-EXP4 TRAINING)"
echo "================================================================="

# ---------------------------------------------------------
# 1. EVALUAR EXP4 (El Padre)
# ---------------------------------------------------------
echo -e "\n▶️ [1/9] Evaluando EXP4 (Híbrido Inclinado a Vectores) frente al Drift..."
cd "$DIR_EVAL"
python3 EXP4_Evaluar_94F3000V_InclinadoV.py

# ---------------------------------------------------------
# 2. ENTRENAR Y EVALUAR EXP7 (El Hijo - DML)
# ---------------------------------------------------------
echo -e "\n▶️ [2/9] Entrenando EXP7 (Transfer Learning + Fine-Tuning DML)..."
cd "$DIR_TRAIN"
python3 EXP7_Train_Transfer_DML.py

echo -e "\n▶️ [3/9] Evaluando EXP7 (Two-Stage DML) frente al Drift..."
cd "$DIR_EVAL"
python3 EXP7_Evaluar_Transfer_DML.py

# ---------------------------------------------------------
# 3. ENTRENAR Y EVALUAR EXP2 (Híbrido Neutro)
# ---------------------------------------------------------
echo -e "\n▶️ [4/9] Entrenando EXP2 (Híbrido Neutro Optimizado)..."
cd "$DIR_TRAIN"
python3 EXP2_Train_94F3000V_Neutro.py

echo -e "\n▶️ [5/9] Evaluando EXP2 frente al Drift..."
cd "$DIR_EVAL"
python3 EXP2_Evaluar_94F3000V_Neutro.py

# ---------------------------------------------------------
# 4. ENTRENAR Y EVALUAR EXP3 (Inclinado a Features)
# ---------------------------------------------------------
echo -e "\n▶️ [6/9] Entrenando EXP3 (Inclinado a Features Optimizado)..."
cd "$DIR_TRAIN"
python3 EXP3_Train_94F3000V_InclinadoF.py

echo -e "\n▶️ [7/9] Evaluando EXP3 frente al Drift..."
cd "$DIR_EVAL"
python3 EXP3_Evaluar_94F3000V_InclinadoF.py

# ---------------------------------------------------------
# 5. ENTRENAR Y EVALUAR EXP5 (Cross-Attention)
# ---------------------------------------------------------
echo -e "\n▶️ [8/9] Entrenando EXP5 (Gated Cross-Attention Optimizado)..."
cd "$DIR_TRAIN"
python3 EXP5_Train_94F3000V_CrossAttention.py

echo -e "\n▶️ [9/9] Evaluando EXP5 frente al Drift..."
cd "$DIR_EVAL"
python3 EXP5_Evaluar_94F3000V_CrossAttention.py

echo "================================================================="
echo "✅ LA BATERÍA DE EXPERIMENTOS HA FINALIZADO CON ÉXITO."
echo "================================================================="