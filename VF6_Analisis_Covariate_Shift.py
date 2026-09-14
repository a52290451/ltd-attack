"""
=========================================================================================
                        VF6_Analisis_Covariate_Shift.py
=========================================================================================
DESCRIPCIÓN ACADÉMICA:
    Analizador de "Covariate Shift" mediante el Test de Kolmogorov-Smirnov (KS-Test).
    Compara empíricamente la distribución estadística de cada Feature durante los 60 días
    de entrenamiento contra su distribución 2 meses en el futuro (Concept Drift).

OBJETIVO:
    Identificar cuáles variables macro-temporales son "Invariantes" (resisten al paso
    del tiempo en Tor) y cuáles son "Volátiles" (causantes del colapso del modelo al 3%).
=========================================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os
import datetime

# --- 1. RUTAS DE LOS DATOS ---
# Usamos los CSV preprocesados donde ya rellenaste los datos de 24h
HISTORIC_CSV = '../../output/preprocessed/02_features_robust_.csv'
FUTURE_CSV = '../../output/preprocessed/02_features_robust_drift.csv'
SAVE_DIR = '../vectores_features/resultados_analisis'

os.makedirs(SAVE_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

print("\n" + "═"*60)
print("🔍 INICIANDO AUTOPSIA DE FEATURES: COVARIATE SHIFT (KS-TEST)")
print("═"*60)

# --- 2. CARGA DE DATOS ---
print("\n[1/3] Cargando conjuntos de datos (Pasado vs Futuro)...")
try:
    df_hist = pd.read_csv(HISTORIC_CSV)
    df_fut = pd.read_csv(FUTURE_CSV)
except Exception as e:
    print(f"❌ Error al cargar los CSV: {e}")
    exit()

excluded = ['sample_uid', 'site_label', 'date_id', 'hour_bin', 'direction_vector', 'size_vector', 'pcap_uid', 'target', 'target_id']
features = [col for col in df_hist.columns if col not in excluded and col in df_fut.columns]

print(f"   ✅ Total de Features a analizar: {len(features)}")

# --- 3. ANÁLISIS DE KOLMOGOROV-SMIRNOV ---
print("\n[2/3] Calculando distancias de distribución (KS Statistic)...")
results = []

for feat in features:
    hist_vals = df_hist[feat].dropna().values
    fut_vals = df_fut[feat].dropna().values
    
    # Si la feature está vacía o es constante, la ignoramos
    if len(hist_vals) < 2 or len(fut_vals) < 2:
        continue
        
    # KS-Test: Devuelve el "statistic" (distancia máxima entre las dos distribuciones, 0 a 1)
    # y el p-value.
    stat, p_value = ks_2samp(hist_vals, fut_vals)
    
    results.append({
        'Feature': feat,
        'KS_Distance': stat,  # Más cerca de 0 = Mejor (Invariante). Más cerca de 1 = Peor (Volátil).
        'P_Value': p_value
    })

# --- 4. CLASIFICACIÓN Y GUARDADO ---
print("\n[3/3] Clasificando Supervivientes...")
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='KS_Distance', ascending=True)

# Definimos un umbral empírico: Si la distancia es menor a 0.15, la consideramos "Estable"
umbral_estabilidad = 0.15
df_invariantes = df_results[df_results['KS_Distance'] <= umbral_estabilidad]

print("\n" + "🏆"*20)
print(f"RESUMEN DE SUPERVIVENCIA DE FEATURES")
print("🏆"*20)
print(f"Total Features Evaluadas: {len(df_results)}")
print(f"Features Invariantes (KS <= {umbral_estabilidad}): {len(df_invariantes)}")
print(f"Features Volátiles (Basura temporal): {len(df_results) - len(df_invariantes)}")

print("\n💎 TOP 10 FEATURES MÁS ESTABLES (Las que debemos usar):")
print(df_results.head(10)[['Feature', 'KS_Distance']].to_string(index=False))

print("\n☠️ TOP 10 FEATURES MÁS VOLÁTILES (Las que destruyeron el modelo):")
print(df_results.tail(10)[['Feature', 'KS_Distance']].to_string(index=False))

# Guardar listas
df_results.to_csv(os.path.join(SAVE_DIR, f"reporte_ks_drift_{timestamp}.csv"), index=False)
invariantes_list = df_invariantes['Feature'].tolist()

with open(os.path.join(SAVE_DIR, "features_invariantes_seguras.txt"), "w") as f:
    for item in invariantes_list:
        f.write("%s\n" % item)

print(f"\n📄 Reporte completo guardado en: {SAVE_DIR}")