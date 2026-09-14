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

# --- 1. RUTAS ---
FEATURES_LIST_TXT = '../vectores_features/resultados_analisis/features_invariantes_seguras.txt'
HISTORIC_CSV = '../../output/preprocessed/02_features_robust_.csv'
DRIFT_CSV = '../../output/preprocessed/02_features_robust_drift.csv'
SAVE_DIR = './graficas_tesis'
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. FUNCIONES DE EXTRACCIÓN ---
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
    print(f"⏳ Extrayendo Meta-Variables de {dataset_name}...")
    records = []
    grouped = df.groupby(['site_label', 'date_id'])
    for (site, date), group in tqdm(grouped, desc=f"Procesando {dataset_name}"):
        group = group.sort_values('hour_bin')
        if len(group) < 8: continue 
        record = {'site_label': site}
        for feat in features:
            vals = group[feat].values
            diffs = np.diff(vals)
            record[f'{feat}_mean'] = np.mean(vals)
            record[f'{feat}_std'] = np.std(vals)
            record[f'{feat}_range'] = np.max(vals) - np.min(vals)
            record[f'{feat}_skew'] = skew(vals)
            record[f'{feat}_diff_std'] = np.std(diffs) if len(diffs) > 0 else 0.0
            record[f'{feat}_autocorr1'] = get_autocorr(vals, lag=1)
            record[f'{feat}_zcr'] = get_zcr(vals)
            record[f'{feat}_energy_fft'] = get_fft_energy(vals)
        records.append(record)
    return pd.DataFrame(records).fillna(0.0)

# --- 3. PROCESAMIENTO ---
with open(FEATURES_LIST_TXT, 'r') as f:
    features_45 = [line.strip() for line in f.readlines() if line.strip()]

df_hist_raw = pd.read_csv(HISTORIC_CSV)
df_drift_raw = pd.read_csv(DRIFT_CSV)

# ---> CORRECCIÓN: Mapeo de 'site' a 'site_label' para el dataset del futuro <---
if 'site_label' not in df_drift_raw.columns and 'site' in df_drift_raw.columns:
    print("🔄 Sincronizando etiquetas 'site_label' para el dataset del Futuro...")
    df_bridge = pd.read_csv('../../output/final_features_sites.csv', usecols=['site', 'site_label']).drop_duplicates()
    site_to_id = dict(zip(df_bridge['site'], df_bridge['site_label']))
    df_drift_raw['site_label'] = df_drift_raw['site'].map(site_to_id)

df_hist_raw = df_hist_raw.dropna(subset=['site_label'])
df_drift_raw = df_drift_raw.dropna(subset=['site_label'])

# Extracción de las 540 dimensiones SIN LIMPIEZA
df_train = extract_advanced_signatures_raw(df_hist_raw, features_45, "Pasado")
df_test = extract_advanced_signatures_raw(df_drift_raw, features_45, "Futuro")

# Escalar para PCA (necesario), pero SIN CLIPPING de outliers
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train.drop(columns=['site_label']))
X_test_scaled = scaler.transform(df_test.drop(columns=['site_label']))

# --- 4. GENERACIÓN DEL GRÁFICO 4 (PCA MULTISITIO CON OUTLIERS) ---
print("⚙️ Entrenando PCA y proyectando...")
pca = PCA(n_components=2)
pca.fit(X_train_scaled) # El espacio lo define el pasado

X_tr_pca = pca.transform(X_train_scaled)
X_te_pca = pca.transform(X_test_scaled)

# Seleccionar sitios para mostrar el colapso (ej. los que tengan más datos)
top_sites = df_train['site_label'].value_counts().head(3).index.tolist()
colors = ['#3498db', '#2ecc71', '#9b59b6'] # Azul, Verde, Morado

plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

# Fondo: Otros sitios en gris
mask_others = ~df_train['site_label'].isin(top_sites)
plt.scatter(X_tr_pca[mask_others, 0], X_tr_pca[mask_others, 1], c='lightgrey', alpha=0.3, s=20, label='Otros Sitios (Contexto Histórico)')

for i, site in enumerate(top_sites):
    # Pasado
    m_tr = df_train['site_label'] == site
    plt.scatter(X_tr_pca[m_tr, 0], X_tr_pca[m_tr, 1], c=colors[i], marker='o', s=80, edgecolors='white', label=f'Sitio {int(site)} (Pasado)')
    
    # Futuro (Aquí se verán los Outliers disparados)
    m_te = df_test['site_label'] == site
    if sum(m_te) > 0:
        plt.scatter(X_te_pca[m_te, 0], X_te_pca[m_te, 1], c=colors[i], marker='X', s=150, edgecolors='black', linewidth=1.5, label=f'Sitio {int(site)} (Drift/Outliers)')

var_1 = pca.explained_variance_ratio_[0] * 100
var_2 = pca.explained_variance_ratio_[1] * 100

plt.title('Colapso del Espacio Latente (Meta-Metadatos 540D)\nEvidencia de Anomalías antes de Limpieza Z-Score', fontsize=14, fontweight='bold', pad=15)
plt.xlabel(f'Componente Principal 1 ({var_1:.1f}%)', fontsize=12, fontweight='bold')
plt.ylabel(f'Componente Principal 2 ({var_2:.1f}%)', fontsize=12, fontweight='bold')

plt.legend(loc='best', bbox_to_anchor=(1, 1), fontsize=9)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

output_path = os.path.join(SAVE_DIR, 'G4_PCA_540D_Outliers.png')
plt.savefig(output_path, dpi=300)
print(f"✅ Gráfico 4 generado en: {output_path}")