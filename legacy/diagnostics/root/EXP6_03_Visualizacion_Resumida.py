"""
=========================================================================================
                    EXP6_03_Visualizacion_Resumida.py
=========================================================================================
OBJETIVO:
    Generar visualizaciones ejecutivas para la tesis:
    1. Matriz de Similitud Agregada (Centroides).
    2. Curvas KDE de Separabilidad (Intra-clase vs Inter-clase).
=========================================================================================
"""

import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 1. CONFIGURACIÓN Y RUTAS ---
HISTORIC_CSV = '../../output/preprocessed/02_features_robust_.csv'
DRIFT_CSV = '../../output/preprocessed/02_features_robust_drift.csv'
FEATURES_LIST_TXT = '../vectores_features/resultados_analisis/features_invariantes_seguras.txt'
SAVE_DIR = '../features/resultados/resultados_exp6'
os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "═"*60)
print("🚀 EXP 6.3: VISUALIZACIÓN EJECUTIVA - SEPARABILIDAD DE FIRMAS")
print("═"*60)

# --- 2. EXTRACCIÓN DE DATOS (ADN PURO) ---
def quick_extract(df, features):
    records = []
    grouped = df.groupby(['site_label', 'date_id'])
    for (site, date), group in grouped:
        if len(group) < 8: continue
        record = {'site_label': site, 'date_id': date}
        for feat in features:
            vals = group[feat].values
            record[f'{feat}_mean'] = np.mean(vals)
            record[f'{feat}_std'] = np.std(vals)
            record[f'{feat}_integral'] = np.trapezoid(vals)
        records.append(record)
    return pd.DataFrame(records).fillna(0.0)

# Carga de features base
with open(FEATURES_LIST_TXT, 'r') as f:
    features_45 = [line.strip() for line in f.readlines() if line.strip()]

# Carga de datasets
df_hist = pd.read_csv(HISTORIC_CSV).dropna(subset=['site_label'])
df_drift = pd.read_csv(DRIFT_CSV)
if 'site_label' not in df_drift.columns:
    df_bridge = pd.read_csv('../../output/final_features_sites.csv', usecols=['site', 'site_label']).drop_duplicates()
    df_drift['site_label'] = df_drift['site'].map(dict(zip(df_bridge['site'], df_bridge['site_label'])))
df_drift = df_drift.dropna(subset=['site_label'])

# Extracción de firmas
df_train = quick_extract(df_hist, features_45)
df_test = quick_extract(df_drift, features_45)

# Sincronizar sitios y REINICIAR ÍNDICES (Corrección del error)
common_sites = list(set(df_train['site_label']).intersection(set(df_test['site_label'])))
df_train = df_train[df_train['site_label'].isin(common_sites)].reset_index(drop=True)
df_test = df_test[df_test['site_label'].isin(common_sites)].reset_index(drop=True)

# Limpieza de colinealidad (ADN Puro)
X_train_raw = df_train.drop(columns=['site_label', 'date_id'])
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
to_drop = [column for column in X_train_scaled.corr().abs().where(np.triu(np.ones(X_train_scaled.corr().shape), k=1).astype(bool)).columns if any(X_train_scaled.corr().abs().where(np.triu(np.ones(X_train_scaled.corr().shape), k=1).astype(bool))[column] > 0.85)]

X_train_clean = X_train_scaled.drop(columns=to_drop)
X_test_clean = pd.DataFrame(scaler.transform(df_test.drop(columns=['site_label', 'date_id'])), columns=X_train_raw.columns).drop(columns=to_drop)

# --- 3. CÁLCULO DE SIMILITUDES AGREGADAS ---
# Tomamos los top 10 sitios para la matriz 
top_10 = pd.Series(df_train['site_label']).value_counts().index[:10]

centroids_past = []
centroids_future = []
labels = []

for site in top_10:
    centroids_past.append(X_train_clean[df_train['site_label'] == site].mean().values)
    centroids_future.append(X_test_clean[df_test['site_label'] == site].mean().values)
    labels.append(f"Sitio {int(site)}")

# Matriz combinada para Heatmap (Past0, Future0, Past1, Future1...)
combined_centroids = []
combined_labels = []
for i in range(len(labels)):
    combined_centroids.append(centroids_past[i])
    combined_centroids.append(centroids_future[i])
    combined_labels.append(f"{labels[i]}\n(Pasado)")
    combined_labels.append(f"{labels[i]}\n(Futuro)")

sim_matrix = cosine_similarity(combined_centroids)

# --- 4. GRÁFICA 1: MATRIZ DE CENTROIDES ---
print("🎨 Generando Matriz de Similitud de Centroides...")
plt.figure(figsize=(12, 10))
sns.heatmap(sim_matrix, xticklabels=combined_labels, yticklabels=combined_labels, 
            annot=True, fmt=".2f", cmap="RdYlBu_r", vmin=0, vmax=1, cbar_kws={'label': 'Similitud Coseno'})
plt.title("Firma Temporal: Similitud de Centroides (Pasado vs Futuro)\nTop 10 Sitios Web", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '06_Matriz_Centroides_Similitud.png'), dpi=300)

# --- 5. GRÁFICA 2: CURVAS DE SEPARABILIDAD (KDE) ---
print("🎨 Generando Curvas KDE de Separabilidad...")

intra_class_sim = [] # Mismo sitio (Pasado vs Futuro)
inter_class_sim = [] # Sitios diferentes

# Usamos todos los sitios comunes para mayor potencia estadística
for i, site_a in enumerate(common_sites):
    vec_past_a = X_train_clean[df_train['site_label'] == site_a].mean().values.reshape(1, -1)
    vec_futr_a = X_test_clean[df_test['site_label'] == site_a].mean().values.reshape(1, -1)
    
    # Similitud del sitio consigo mismo en el futuro
    intra_class_sim.append(cosine_similarity(vec_past_a, vec_futr_a)[0,0])
    
    # Similitud con otros sitios en el futuro (muestreo aleatorio para no saturar)
    other_sites = [s for s in common_sites if s != site_a]
    for site_b in other_sites:
        vec_futr_b = X_test_clean[df_test['site_label'] == site_b].mean().values.reshape(1, -1)
        inter_class_sim.append(cosine_similarity(vec_past_a, vec_futr_b)[0,0])

plt.figure(figsize=(10, 6))
sns.kdeplot(intra_class_sim, fill=True, color="green", label="Mismo Sitio (Pasado vs Futuro)", bw_adjust=1.5)
sns.kdeplot(inter_class_sim, fill=True, color="red", label="Sitios Diferentes", bw_adjust=1.5)

plt.axvline(np.mean(intra_class_sim), color='darkgreen', linestyle='--')
plt.axvline(np.mean(inter_class_sim), color='darkred', linestyle='--')

plt.title("Capacidad Discriminativa del ADN Temporal\nDistribución de Similitud Coseno", fontsize=14, fontweight='bold')
plt.xlabel("Similitud Coseno (Cercanía en el Espacio de Embeddings)")
plt.ylabel("Densidad de Probabilidad")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '07_KDE_Separabilidad_Similitud.png'), dpi=300)

print(f"\n✅ PROCESO FINALIZADO. Revisa las imágenes 06 y 07 en {SAVE_DIR}")