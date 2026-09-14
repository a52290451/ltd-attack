"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: GEN_02_PCA.py
🚀 VERSIÓN: 3.0 (Meta-Metadatos 752D, Paridad 65 Clases y Datos Purificados)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este script eleva el nivel de abstracción del análisis. En lugar de graficar el 
    espacio latente de las features individuales, extrae "Firmas Dinámicas" (Meta-Features)
    evaluando cómo fluctúa cada variable dentro de su ciclo de vida de 24 horas.
    
    Operaciones aplicadas por variable (x8): Mean, Std, Range, Skew, Diff_Std, 
    Auto-Correlación, Tasa de Cruce por Cero (ZCR) y Energía FFT.
    
    [!] ACTUALIZACIÓN V3.0: 
    - Aplica paridad estricta (65 sitios de élite de la rama Micro).
    - Ingesta dinámicamente las 94 Invariantes, generando un hiperespacio de 752D.
    - Utiliza expresiones regulares para la cuantización horaria de los datos CLEAN.
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
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import joblib
import re
import sys
from src.utils.paths import data_path, artifact_path, result_path

# --- 1. RUTAS Y DICCIONARIOS ---
FEATURES_LIST_TXT = result_path('macro', 'resultados_analisis', 'new_features_invariantes_seguras.txt')
HISTORIC_CSV = data_path('historical', 'CLEAN_final_features_sites.csv')
DRIFT_CSV = data_path('historical', 'CLEAN_final_features_sites_concept_drift.csv')
VECTOR_LE_PATH = artifact_path('ds3_label_encoder_vec_3000.joblib')
SAVE_DIR = result_path('features', 'graficas_tesis')

os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "═"*70)
print("🎨 GENERANDO GRÁFICO 5: PCA DE META-CARACTERÍSTICAS (HIPERESPACIO 24H)")
print("═"*70)

# --- 2. FUNCIONES MATEMÁTICAS DE EXTRACCIÓN TEMPORAL ---
def get_autocorr(x, lag=1):
    if len(x) <= lag or np.var(x) == 0: return 0.0
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]

def get_zcr(x):
    mean_val = np.mean(x)
    zero_crossings = np.where(np.diff(np.sign(x - mean_val)))[0]
    return len(zero_crossings) / len(x) if len(x) > 0 else 0.0

def get_fft_energy(x):
    if len(x) < 2 or np.var(x) == 0: return 0.0
    fft_vals = np.fft.fft(x - np.mean(x))
    return np.sum(np.abs(fft_vals)**2) / len(x)

def extract_advanced_signatures_raw(df, features, dataset_name):
    print(f"   ⏳ Extrayendo Firmas Dinámicas de {dataset_name}...")
    records = []
    grouped = df.groupby(['site_label', 'date_id'])
    
    for (site, date), group in tqdm(grouped, desc=f"Procesando Ciclos 24h ({dataset_name})"):
        group = group.sort_values('hour_bin')
        if len(group) < 4: continue # Requisito mínimo de puntos para la FFT/Skew
        
        record = {'site_label': site}
        for feat in features:
            vals = group[feat].values
            diffs = np.diff(vals)
            record[f'{feat}_mean'] = np.mean(vals)
            record[f'{feat}_std'] = np.std(vals)
            record[f'{feat}_range'] = np.max(vals) - np.min(vals)
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

# --- 3. PIPELINE DE PREPROCESAMIENTO Y PARIDAD ---
try:
    print("\n[1/4] Extrayendo filtro de paridad (65 Clases de Élite)...")
    le_vectores = joblib.load(VECTOR_LE_PATH)
    sitios_elite = set(le_vectores.classes_.astype(str))
    
    with open(FEATURES_LIST_TXT, 'r') as f:
        features_inv = [line.strip() for line in f.readlines() if line.strip()]

    print("\n[2/4] Cargando datasets purificados y aplicando motor temporal...")
    df_hist_raw = pd.read_csv(HISTORIC_CSV)
    df_drift_raw = pd.read_csv(DRIFT_CSV)

    # Forzar string para el cruce de paridad
    df_hist_raw['site_label'] = df_hist_raw['site_label'].astype(str)
    df_drift_raw['site_label'] = df_drift_raw['site_label'].astype(str)

    # [!] FILTRO ESTRICTO DE PARIDAD
    df_hist_raw = df_hist_raw[df_hist_raw['site_label'].isin(sitios_elite)].copy()
    df_drift_raw = df_drift_raw[df_drift_raw['site_label'].isin(sitios_elite)].copy()

    # Extraer Date ID y Hour Bin
    for df in [df_hist_raw, df_drift_raw]:
        parsed = df['pcap_name'].apply(parse_time_from_pcap)
        df['date_id'] = [p[0] for p in parsed]
        df['hour_bin'] = [p[1] for p in parsed]
        df.dropna(subset=['date_id', 'hour_bin', 'site_label'], inplace=True)

    print("\n[3/4] Construyendo el Espacio de Meta-Características...")
    # Extracción de dimensiones múltiples
    df_train = extract_advanced_signatures_raw(df_hist_raw, features_inv, "Pasado")
    df_test = extract_advanced_signatures_raw(df_drift_raw, features_inv, "Futuro")

    n_dimensions = len(df_train.columns) - 1 # Restamos 'site_label'
    print(f"   ✅ Meta-Espacio construido: {n_dimensions} Dimensiones dinámicas.")

    # Escalar para PCA (vital para dimensiones heterogéneas como FFT vs ZCR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train.drop(columns=['site_label']))
    X_test_scaled = scaler.transform(df_test.drop(columns=['site_label']))

    # --- 4. GENERACIÓN DEL GRÁFICO (PCA MULTISITIO) ---
    print("\n[4/4] Entrenando PCA y proyectando Autopsia Visual...")
    pca = PCA(n_components=2)
    pca.fit(X_train_scaled) # El espacio base es estrictamente el pasado

    X_tr_pca = pca.transform(X_train_scaled)
    X_te_pca = pca.transform(X_test_scaled)

    # Seleccionar 3 sitios de élite aleatorios pero reproducibles
    np.random.seed(42)
    top_sites = np.random.choice(df_train['site_label'].unique(), 3, replace=False).tolist()
    colors = ['#3498db', '#2ecc71', '#9b59b6'] # Azul, Verde, Morado

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Fondo: Contexto de los otros 62 sitios
    mask_others = ~df_train['site_label'].isin(top_sites)
    plt.scatter(X_tr_pca[mask_others, 0], X_tr_pca[mask_others, 1], 
                c='lightgrey', alpha=0.3, s=20, label='Contexto Global (62 Sitios)')

    for i, site in enumerate(top_sites):
        # Nubes del Pasado
        m_tr = df_train['site_label'] == site
        plt.scatter(X_tr_pca[m_tr, 0], X_tr_pca[m_tr, 1], 
                    c=colors[i], marker='o', s=80, edgecolors='white', 
                    alpha=0.8, label=f'Sitio {site} (Pasado)')
        
        # Dispersión del Futuro (Concept Drift afectando las series de tiempo)
        m_te = df_test['site_label'] == site
        if sum(m_te) > 0:
            plt.scatter(X_te_pca[m_te, 0], X_te_pca[m_te, 1], 
                        c=colors[i], marker='X', s=150, edgecolors='black', 
                        linewidth=1.5, alpha=0.9, label=f'Sitio {site} (Drift 2 Meses)')

    var_1 = pca.explained_variance_ratio_[0] * 100
    var_2 = pca.explained_variance_ratio_[1] * 100

    plt.title(f'Colapso del Espacio Latente Dinámico (Meta-Features {n_dimensions}D)\\nImpacto del Drift sobre Flujos Temporales de 24h', 
              fontsize=15, fontweight='bold', pad=15)
    plt.xlabel(f'Componente Principal 1 ({var_1:.1f}%)', fontsize=12, fontweight='bold')
    plt.ylabel(f'Componente Principal 2 ({var_2:.1f}%)', fontsize=12, fontweight='bold')

    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Ajuste de encuadre para omitir ruido extremo
    x_min, x_max = np.percentile(X_tr_pca[:, 0], [1, 99])
    y_min, y_max = np.percentile(X_tr_pca[:, 1], [1, 99])
    plt.xlim(x_min - 5, x_max + 5)
    plt.ylim(y_min - 5, y_max + 5)

    plt.tight_layout()

    output_path = os.path.join(SAVE_DIR, 'G5_PCA_MetaFeatures_752D_Paridad.png')
    plt.savefig(output_path, dpi=300)
    print(f"✅ Gráfico guardado exitosamente en: {output_path}")
    print("===========================================================================\n")

except Exception as e:
    import traceback
    print(f"❌ Error crítico:\n{traceback.format_exc()}")
