#!/bin/bash
# =============================================================================
# 🛡️ Estado del Arte — Script de Ejecución
# =============================================================================
# 📂 DIRECTORIO: src/models/sota/
# 📄 ARCHIVO:    run_estado_del_arte.sh
# 📅 FECHA:      20 de Julio, 2026
#
# 📝 DESCRIPCIÓN:
#    Este script ejecuta los baselines SOTA_01_CUMUL_SVM.py, SOTA_02_DeepFingerprinting_CNN.py
#    y SOTA_03_Rimmer_LSTM.py, que implementan los enfoques clásico (SVM), Deep Learning
#    espacial (CNN) y Deep Learning temporal (LSTM) para evaluar el impacto del Concept
#    Drift en Website Fingerprinting.
#
# 🚀 USO:
#    1. Primero, otorga permisos de ejecución:
#       chmod +x run_estado_del_arte.sh
#
#    2. Luego, ejecuta el script:
#       ./run_estado_del_arte.sh
#
# 📋 SALIDA:
#    Los resultados se guardarán bajo results/sota/
#    - SVM: log_SOTA_01_CUMUL_[timestamp].txt, G_SOTA_01_CUMUL_Drift_[timestamp].png
#    - CNN: log_SOTA_02_DF_[timestamp].txt, best_df_cnn.pth, G_SOTA_02_DF_Curvas_[timestamp].png
#    - LSTM: log_SOTA_03_LSTM_[timestamp].txt, best_rimmer_lstm.pth, G_SOTA_03_LSTM_Curvas_[timestamp].png
# =============================================================================

echo "═══════════════════════════════════════════════════════════════════"
echo "🛡️  Estado del Arte — Baselines SOTA 01 (SVM) + SOTA 02 (CNN) + SOTA 03 (LSTM)"
echo "═══════════════════════════════════════════════════════════════════"

# Resolver el proyecto independientemente del cwd del llamador.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Verificar dependencias
echo ""
echo "📦 Verificando dependencias..."

python3 -c "import pandas, numpy, sklearn, matplotlib, seaborn, joblib, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Advertencia: Algunas dependencias no están instaladas."
    echo "   Instala con: pip install pandas numpy scikit-learn matplotlib seaborn joblib torch"
    echo ""
    read -p "¿Deseas continuar de todas formas? (s/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo "Ejecución cancelada."
        exit 1
    fi
fi

echo "   ✅ Dependencias verificadas."

# =============================================================================
# SOTA 01: CUMUL SVM
# =============================================================================
echo ""
echo "🚀 [1/3] Iniciando SOTA_01_CUMUL_SVM.py (Baseline Clásico — SVM)..."
echo "═══════════════════════════════════════════════════════════════════"

python3 -m src.models.sota.SOTA_01_CUMUL_SVM

EXIT_CODE_1=$?

echo "═══════════════════════════════════════════════════════════════════"

if [ $EXIT_CODE_1 -eq 0 ]; then
    echo "✅ SOTA_01 (SVM) finalizada exitosamente."
else
    echo "❌ SOTA_01 (SVM) falló con código de salida: $EXIT_CODE_1"
    echo "   Revisa los logs para más detalles."
fi

# =============================================================================
# SOTA 02: Deep Fingerprinting CNN
# =============================================================================
echo ""
echo "🚀 [2/3] Iniciando SOTA_02_DeepFingerprinting_CNN.py (Baseline DL — CNN)..."
echo "═══════════════════════════════════════════════════════════════════"

python3 -m src.models.sota.SOTA_02_DeepFingerprinting_CNN

EXIT_CODE_2=$?

echo "═══════════════════════════════════════════════════════════════════"

if [ $EXIT_CODE_2 -eq 0 ]; then
    echo "✅ SOTA_02 (CNN) finalizada exitosamente."
else
    echo "❌ SOTA_02 (CNN) falló con código de salida: $EXIT_CODE_2"
    echo "   Revisa los logs para más detalles."
fi

# =============================================================================
# SOTA 03: Rimmer LSTM (Baseline Temporal)
# =============================================================================
echo ""
echo "🚀 [3/3] Iniciando SOTA_03_Rimmer_LSTM.py (Baseline Temporal — LSTM)..."
echo "═══════════════════════════════════════════════════════════════════"

python3 -m src.models.sota.SOTA_03_Rimmer_LSTM

EXIT_CODE_3=$?

echo "═══════════════════════════════════════════════════════════════════"

if [ $EXIT_CODE_3 -eq 0 ]; then
    echo "✅ SOTA_03 (LSTM) finalizada exitosamente."
else
    echo "❌ SOTA_03 (LSTM) falló con código de salida: $EXIT_CODE_3"
    echo "   Revisa los logs para más detalles."
fi

# =============================================================================
# Resumen Final
# =============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "📊 Resumen de Ejecución"
echo "═══════════════════════════════════════════════════════════════════"

if [ $EXIT_CODE_1 -eq 0 ] && [ $EXIT_CODE_2 -eq 0 ] && [ $EXIT_CODE_3 -eq 0 ]; then
    echo "✅ Los tres baselines (SVM, CNN, LSTM) ejecutados exitosamente."
else
    echo "⚠️  Uno o más baselines fallaron. Revisa los logs."
fi

echo ""
echo "📁 Resultados guardados bajo: results/sota/"
ls -lh results/sota/ 2>/dev/null

echo "═══════════════════════════════════════════════════════════════════"
echo "🛡️  Estado del Arte — Ejecución Completada"
echo "═══════════════════════════════════════════════════════════════════"
