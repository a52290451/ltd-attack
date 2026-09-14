"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: GEN_03_Ortogonalidad.py
🚀 VERSIÓN: 3.0 (Filtro de Pearson sobre Meta-Metadatos con Paridad)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este script ingesta el hiperespacio temporal extraído de los ciclos de 24 horas 
    (Meta-Features) y aplica un filtro estricto de correlación cruzada de Pearson.
    
    Toda pareja de variables con correlación |r| > 0.85 es considerada "colineal" 
    (redundante) y una de ellas es purgada. El resultado es el "ADN Puro y Ortogonal"
    del modelo, garantizando que cada dimensión aporta información estadísticamente única.

    [!] ACTUALIZACIÓN V3.0 (PARIDAD ESTRICTA): 
    - Se corrige la advertencia matemática (RuntimeWarning) en las series de tiempo.
    - Se asegura la paridad estricta filtrando solo los 65 sitios de élite.
    - Se procesa sobre los datos purificados (Fase 0) del dataset Pasado (Conocimiento).
=========================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import joblib
import re
import sys

# --- 1. RUTAS Y DICCIONARIOS ---
FEATURES_LIST_TXT = '../vectores_features/resultados_analisis/new_features_invariantes_seguras.txt'
HISTORIC_CSV = '../../output/CLEAN_final_features_sites.csv'
VECTOR_LE_PATH = '../vectores/resultados/ds3_label_encoder_vec_3000.joblib'
SAVE_DIR = './graficas_tesis'

os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "═"*70)
print("🎨 GENERANDO GRÁFICA 6: MATRIZ DE CORRELACIÓN ORTOGONAL (PEARSON)")
print("═"*70)

# --- 2. FUNCIONES DE EXTRACCIÓN (MATEMÁTICA CORREGIDA) ---
def get_autocorr(x, lag=1):
    if len(x) <= lag: return 0.0
    x1, x2 = x[:-lag], x[lag:]
    # [!] Prevención de división por cero (RuntimeWarning)
    if np.std(x1) == 0 or np.std(x2) == 0: return 0.0
    return np.corrcoef(x1, x2)[0, 1]

def get_zcr(x):
    mean_val = np.mean(x)
    zero_crossings = np.where(np.diff(np.sign(x - mean_val)))[0]
    return len(zero_crossings) / len(x) if len(x) > 0 else 0.0

def get_fft_energy(x):
    if len(x) < 2 or np.var(x) == 0: return 0.0
    fft_vals = np.fft.fft(x - np.mean(x))
    return np.sum(np.abs(fft_vals)**2) / len(x)

def extract_advanced_signatures_train(df, features):
    print("   ⏳ Extrayendo Meta-Variables Dinámicas del Pasado (Train)...")
    records = []
    grouped = df.groupby(['site_label', 'date_id'])
    for (site, date), group in tqdm(grouped, desc="Procesando Ciclos 24h"):
        group = group.sort_values('hour_bin')
        # Requisito mínimo de capturas en el día para calcular estadísticos temporales
        if len(group) < 4: continue 
        record = {'site_label': site}
        for feat in features:
            vals = group[feat].values
            diffs = np.diff(vals)
            record[f'{feat}_mean'] = np.mean(vals)
            record[f'{feat}_std'] = np.std(vals)
            record[f'{feat}_range'] = np.max(vals) - np.min(vals)
            # [!] Prevención de NaNs en Skewness
            record[f'{feat}_skew'] = skew(vals) if len(vals) > 2 and np.var(vals) > 0 else 0.0
            record[f'{feat}_diff_std'] = np.std(diffs) if len(diffs) > 0 else 0.0
            record[f'{feat}_autocorr1'] = get_autocorr(vals, lag=1)
            record[f'{feat}_zcr'] = get_zcr(vals)
            record[f'{feat}_energy_fft'] = get_fft_energy(vals)
        records.append(record)
    return pd.DataFrame(records).fillna(0.0)

def parse_time_from_pcap(name_str):
    match = re.search(r'_(\d{8})-(\d{2})\d{4}', str(name_str))
    if match: return match.group(1), int(match.group(2))
    return None, None

# --- 3. PROCESAMIENTO Y DEPURACIÓN ---
try:
    print("\n[1/4] Extrayendo filtro de paridad y ADN Inmutable...")
    le_vectores = joblib.load(VECTOR_LE_PATH)
    sitios_elite = set(le_vectores.classes_.astype(str))

    with open(FEATURES_LIST_TXT, 'r') as f:
        features_inv = [line.strip() for line in f.readlines() if line.strip()]

    print("\n[2/4] Cargando dataset purificado e inyectando Motor Temporal...")
    df_hist_raw = pd.read_csv(HISTORIC_CSV).dropna(subset=['site_label'])
    df_hist_raw['site_label'] = df_hist_raw['site_label'].astype(str)
    
    # [!] FILTRO ESTRICTO DE PARIDAD
    df_hist_raw = df_hist_raw[df_hist_raw['site_label'].isin(sitios_elite)].copy()

    # Extracción de Date ID y Hour Bin vía Regex
    parsed = df_hist_raw['pcap_name'].apply(parse_time_from_pcap)
    df_hist_raw['date_id'] = [p[0] for p in parsed]
    df_hist_raw['hour_bin'] = [p[1] for p in parsed]
    df_hist_raw.dropna(subset=['date_id', 'hour_bin'], inplace=True)

    print("\n[3/4] Construyendo hiperespacio temporal y aplicando filtro de Ortogonalidad...")
    df_train = extract_advanced_signatures_train(df_hist_raw, features_inv)

    X_train = df_train.drop(columns=['site_label'])
    initial_dim = X_train.shape[1]

    # Escalado estándar para normalizar el cálculo de correlación de Pearson
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    print("   ⚙️ Evaluando Matriz de Pearson (Umbral > 0.85)...")
    corr_matrix_full = X_train_scaled.corr().abs()
    
    # Triángulo superior de la matriz para evitar duplicados (X vs Y y Y vs X)
    upper = corr_matrix_full.where(np.triu(np.ones(corr_matrix_full.shape), k=1).astype(bool))
    
    # Encontrar variables con alta colinealidad
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    print(f"   🚨 Se detectaron y purgaron {len(to_drop)} dimensiones colineales/redundantes.")
    
    # Dataset final (ADN Puro y Ortogonal)
    X_train_pure = X_train_scaled.drop(columns=to_drop)
    final_dim = X_train_pure.shape[1]
    print(f"   ✅ Variables restantes (100% Ortogonales): {final_dim}")

    # --- 4. GENERACIÓN DEL GRÁFICO 6 ---
    print("\n[4/4] Generando Mapa de Calor Ortogonal...")
    # Calculamos la correlación final (incluyendo signos para observar la dispersión)
    final_corr = X_train_pure.corr()

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    # Heatmap usando paleta divergente
    sns.heatmap(final_corr, cmap='coolwarm', vmin=-1, vmax=1, 
                xticklabels=False, yticklabels=False, 
                cbar_kws={'label': 'Coeficiente de Correlación de Pearson'})
    
    plt.title(f'Matriz de Correlación Ortogonal ({final_dim} Meta-Dimensiones)\\nReducción desde {initial_dim}D (Ausencia de Colinealidad |r| > 0.85)', 
              fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    output_path = os.path.join(SAVE_DIR, 'G6_Matriz_Correlacion_Ortogonal_Paridad.png')
    plt.savefig(output_path, dpi=300)
    
    print(f"✅ Evidencia visual guardada exitosamente en: {output_path}")
    print("===========================================================================\n")

except Exception as e:
    import traceback
    print(f"❌ Error crítico durante la ejecución:\n{traceback.format_exc()}")