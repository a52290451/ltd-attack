"""
=========================================================================================
                    EXP6_02_Matriz_Similitud_Capturas.py
=========================================================================================
OBJETIVO:
    Generar la Matriz de Similitud Coseno entre capturas (instancias) del Pasado y Futuro.
    Demostrar visualmente al director de tesis que la "distancia relativa" entre 
    capturas del mismo sitio se mantiene, justificando el uso de Metric Learning.
=========================================================================================
"""

import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
from scipy.stats import skew
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from tqdm import tqdm

# --- 1. RUTAS ---
FEATURES_LIST_TXT = '../vectores_features/resultados_analisis/features_invariantes_seguras.txt'
HISTORIC_CSV = '../../output/preprocessed/02_features_robust_.csv'
DRIFT_CSV = '../../output/preprocessed/02_features_robust_drift.csv'
SAVE_DIR = '../features/resultados/resultados_exp6'

os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "═"*60)
print("🚀 EXP 6.2: MATRIZ DE SIMILITUD ENTRE CAPTURAS (PASADO VS FUTURO)")
print("═"*60)

# --- 2. CARGA DE FEATURES Y EXTRACCIÓN (Versión Simplificada Rápida) ---
try:
    with open(FEATURES_LIST_TXT, 'r') as f:
        features_45 = [line.strip() for line in f.readlines() if line.strip()]
except Exception as e:
    print(f"❌ Error al cargar features: {e}")
    exit()

def get_fft_energy(x):
    if len(x) < 2 or np.var(x) == 0: return 0.0
    return np.sum(np.abs(np.fft.fft(x - np.mean(x)))**2) / len(x)

def extract_advanced_signatures(df, features, dataset_name):
    print(f"⏳ Extrayendo datos de {dataset_name}...")
    counts = df.groupby(['site_label', 'date_id']).size().groupby('site_label').size()
    valid_sites = counts[counts >= 10].index 
    df = df[df['site_label'].isin(valid_sites)].copy()
    
    records = []
    grouped = df.groupby(['site_label', 'date_id'])
    for (site, date), group in tqdm(grouped, desc="Procesando", leave=False):
        group = group.sort_values('hour_bin')
        if len(group) < 8: continue 
        record = {'site_label': site, 'date_id': date}
        for feat in features:
            vals = group[feat].values
            record[f'{feat}_mean'] = np.mean(vals)
            record[f'{feat}_std'] = np.std(vals)
            record[f'{feat}_integral'] = np.trapezoid(vals)
            record[f'{feat}_energy_fft'] = get_fft_energy(vals)
        records.append(record)
    return pd.DataFrame(records).fillna(0.0)

# --- 3. PREPARACIÓN DE DATOS ---
df_hist = pd.read_csv(HISTORIC_CSV).dropna(subset=['site_label'])
df_drift = pd.read_csv(DRIFT_CSV)

if 'site_label' not in df_drift.columns and 'site' in df_drift.columns:
    df_bridge = pd.read_csv('../../output/final_features_sites.csv', usecols=['site', 'site_label']).drop_duplicates()
    site_to_id = dict(zip(df_bridge['site'], df_bridge['site_label']))
    df_drift['site_label'] = df_drift['site'].map(site_to_id)
df_drift = df_drift.dropna(subset=['site_label'])

df_train_full = extract_advanced_signatures(df_hist, features_45, "Pasado")
df_test_full = extract_advanced_signatures(df_drift, features_45, "Futuro")

common_sites = set(df_train_full['site_label']).intersection(set(df_test_full['site_label']))
df_train_full = df_train_full[df_train_full['site_label'].isin(common_sites)]
df_test_full = df_test_full[df_test_full['site_label'].isin(common_sites)]

# Limpieza rápida de variables colineales
drop_cols = ['site_label', 'date_id']
X_train_raw = df_train_full.drop(columns=drop_cols)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
corr_matrix = X_train_scaled.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

df_train_clean = df_train_full.drop(columns=to_drop)
df_test_clean = df_test_full.drop(columns=to_drop)

# --- 4. CREACIÓN DE LA MATRIZ DE SIMILITUD ---
print("\n[!] Construyendo Matriz de Similitud Coseno...")
# Seleccionamos los 4 sitios con más datos para que la gráfica sea legible
top_4_sites = list(df_train_clean['site_label'].value_counts().index[:4])
n_samples = 15 # Capturas por sitio y por época

ordered_data = []
labels_for_plot = []
ticks_positions = []
current_pos = 0

for site in top_4_sites:
    # Muestrear Pasado
    train_samples = df_train_clean[df_train_clean['site_label'] == site].head(n_samples)
    ordered_data.append(train_samples.drop(columns=drop_cols))
    labels_for_plot.append(f"Sitio {site}\n(Pasado)")
    ticks_positions.append(current_pos + len(train_samples)/2)
    current_pos += len(train_samples)
    
    # Muestrear Futuro
    test_samples = df_test_clean[df_test_clean['site_label'] == site].head(n_samples)
    ordered_data.append(test_samples.drop(columns=drop_cols))
    labels_for_plot.append(f"Sitio {site}\n(Futuro)")
    ticks_positions.append(current_pos + len(test_samples)/2)
    current_pos += len(test_samples)

# Unimos todo en un solo tensor en orden: S1_Pasado, S1_Futuro, S2_Pasado, S2_Futuro...
final_df = pd.concat(ordered_data)

# SOLUCIÓN: Crear un nuevo escalador ajustado exclusivamente a las variables limpias
final_scaler = StandardScaler()
final_matrix_scaled = final_scaler.fit_transform(final_df)

# Calcular Similitud Coseno entre todas las capturas (N x N)
similarity_matrix = cosine_similarity(final_matrix_scaled)

# --- 5. VISUALIZACIÓN ---
print("🎨 Generando el Mapa de Calor de Correlación de Instancias...")
plt.figure(figsize=(14, 12))

# Usamos un colormap divergente: Rojo (muy similar), Azul (muy diferente)
ax = sns.heatmap(similarity_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1, 
                 xticklabels=False, yticklabels=False, cbar_kws={'label': 'Similitud Coseno (1 = Idéntico)'})

# Dibujar rectángulos para agrupar visualmente cada sitio web (Pasado + Futuro)
for i in range(len(top_4_sites)):
    start_idx = i * (n_samples * 2)
    size = n_samples * 2
    rect = patches.Rectangle((start_idx, start_idx), size, size, linewidth=3, edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(rect)

# Añadir las etiquetas a los ejes
plt.xticks(ticks_positions, labels_for_plot, rotation=45, ha='right', fontsize=10)
plt.yticks(ticks_positions, labels_for_plot, rotation=0, fontsize=10)

plt.title("Matriz de Correlación entre Capturas Individuales (Pasado y Futuro)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Capturas (Instancias) Agrupadas por Sitio y Época", fontsize=12, labelpad=15)
plt.ylabel("Capturas (Instancias) Agrupadas por Sitio y Época", fontsize=12, labelpad=15)

plt.tight_layout()
file_path = os.path.join(SAVE_DIR, '05_Matriz_Correlacion_Capturas.png')
plt.savefig(file_path, dpi=300)
plt.close()

print(f"\n✅ PROCESO FINALIZADO. Archivo guardado en: {file_path}")
print("   -> Llévale esta imagen a tu director. Es la evidencia de que las firmas relativas existen.")