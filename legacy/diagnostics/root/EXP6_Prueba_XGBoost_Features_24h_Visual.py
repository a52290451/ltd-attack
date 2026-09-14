"""
=========================================================================================
                    EXP6_03_Prueba_XGBoost_Visual_Corregido.py
=========================================================================================
OBJETIVO:
    Validación Empírica frente a Concept Drift extremo.
    - Limpieza Global de Outliers (Z-Score Absoluto).
    - Filtro de Colinealidad (>0.85).
    - FEATURE SELECTION POR DRIFT: Cálculo de Population Stability Index (PSI < 0.2).
    - Validación Adversaria para comprobar estabilidad temporal.
=========================================================================================
"""

import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
from scipy.stats import skew, zscore
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 1. RUTAS ---
FEATURES_LIST_TXT = '../vectores_features/resultados_analisis/features_invariantes_seguras.txt'
HISTORIC_CSV = '../../output/preprocessed/02_features_robust_.csv'
DRIFT_CSV = '../../output/preprocessed/02_features_robust_drift.csv'
SAVE_DIR = '../features/resultados/resultados_exp6'

os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "═"*60)
print("🚀 EXP 6.3: XGBOOST + LIMPIEZA GLOBAL + PSI DRIFT FILTER")
print("═"*60)

# --- 2. CARGA DE LAS FEATURES ---
try:
    with open(FEATURES_LIST_TXT, 'r') as f:
        features_45 = [line.strip() for line in f.readlines() if line.strip()]
except Exception as e:
    print(f"❌ Error al cargar features: {e}")
    exit()

# --- 3. FUNCIONES MATEMÁTICAS (PSI INCLUIDO) ---
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

def calculate_psi(expected, actual, bins=10):
    """Calcula el Population Stability Index entre dos distribuciones (Pasado vs Futuro)"""
    if np.min(expected) == np.max(expected): return 0.0
    breakpoints = np.linspace(np.min(expected), np.max(expected), bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Evitar división por cero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_value

# --- 4. EXTRACCIÓN INICIAL ---
def extract_advanced_signatures(df, features, dataset_name):
    print(f"\n⏳ Extrayendo Meta-Variables de {dataset_name}...")
    
    counts = df.groupby(['site_label', 'date_id']).size().groupby('site_label').size()
    valid_sites = counts[counts >= 10].index 
    df = df[df['site_label'].isin(valid_sites)].copy()
    
    records = []
    grouped = df.groupby(['site_label', 'date_id'])
    
    for (site, date), group in tqdm(grouped, desc=f"Generando Perfiles ({dataset_name})"):
        group = group.sort_values('hour_bin')
        if len(group) < 8: continue 
        
        record = {'site_label': site}
        for feat in features:
            vals = group[feat].values
            diffs = np.diff(vals)
            
            record[f'{feat}_mean'] = np.mean(vals)
            record[f'{feat}_std'] = np.std(vals)
            record[f'{feat}_range'] = np.max(vals) - np.min(vals)
            s_val = skew(vals)
            record[f'{feat}_skew'] = s_val if not np.isnan(s_val) else 0.0
            
            record[f'{feat}_diff_std'] = np.std(diffs) if len(diffs) > 0 else 0.0
            record[f'{feat}_diff_max'] = np.max(diffs) if len(diffs) > 0 else 0.0
            record[f'{feat}_diff_min'] = np.min(diffs) if len(diffs) > 0 else 0.0
            
            record[f'{feat}_autocorr1'] = get_autocorr(vals, lag=1)
            record[f'{feat}_autocorr2'] = get_autocorr(vals, lag=2)
            record[f'{feat}_zcr'] = get_zcr(vals)
            record[f'{feat}_integral'] = np.trapezoid(vals)
            record[f'{feat}_energy_fft'] = get_fft_energy(vals)
            
        records.append(record)
        
    return pd.DataFrame(records).fillna(0.0)

# --- 5. PREPARACIÓN Y ESCALADO ---
print("\n[1/5] Cargando y alineando datasets...")
df_hist = pd.read_csv(HISTORIC_CSV)
df_drift_raw = pd.read_csv(DRIFT_CSV)

if 'site_label' not in df_drift_raw.columns and 'site' in df_drift_raw.columns:
    df_bridge = pd.read_csv('../../output/final_features_sites.csv', usecols=['site', 'site_label']).drop_duplicates()
    site_to_id = dict(zip(df_bridge['site'], df_bridge['site_label']))
    df_drift_raw['site_label'] = df_drift_raw['site'].map(site_to_id)

df_hist = df_hist.dropna(subset=['site_label'])
df_drift_raw = df_drift_raw.dropna(subset=['site_label'])

df_train_full = extract_advanced_signatures(df_hist, features_45, "Pasado (Train)")
df_test_full = extract_advanced_signatures(df_drift_raw, features_45, "Futuro (Drift)")

common_sites = set(df_train_full['site_label']).intersection(set(df_test_full['site_label']))
df_train_full = df_train_full[df_train_full['site_label'].isin(common_sites)]
df_test_full = df_test_full[df_test_full['site_label'].isin(common_sites)]

# --- 5.1 ELIMINACIÓN GLOBAL DE OUTLIERS (Corrección de error metodológico) ---
print("\n[2/5] Eliminando trazas anómalas (Fallas Globales de Red Tor)...")

def remove_global_outliers(df, threshold=4.0):
    numeric_cols = df.drop(columns=['site_label'])
    z_scores = np.abs(zscore(numeric_cols))
    z_scores = np.nan_to_num(z_scores) 
    mask = (z_scores < threshold).all(axis=1) 
    return df[mask]

original_len_train = len(df_train_full)
original_len_test = len(df_test_full)

df_train_clean = remove_global_outliers(df_train_full)
df_test_clean = remove_global_outliers(df_test_full)

# --- CORRECCIÓN: Re-sincronizar sitios tras la limpieza masiva ---
common_sites = set(df_train_clean['site_label']).intersection(set(df_test_clean['site_label']))
df_train_clean = df_train_clean[df_train_clean['site_label'].isin(common_sites)]
df_test_clean = df_test_clean[df_test_clean['site_label'].isin(common_sites)]

print(f"   -> Pasado: Eliminadas {original_len_train - len(df_train_clean)} trazas rotas.")
print(f"   -> Futuro: Eliminadas {original_len_test - len(df_test_clean)} trazas rotas.")
print(f"   -> Sitios robustos que sobrevivieron a la limpieza: {len(common_sites)}")

X_train_raw = df_train_clean.drop(columns=['site_label'])
y_train = df_train_clean['site_label']
X_test_raw = df_test_clean.drop(columns=['site_label'])
y_test = df_test_clean['site_label']

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

print("-> Aplicando StandardScaler para normalizar dimensiones...")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns)

# --- 5.2 DEPURACIÓN DE COLINEALIDAD ---
print("\n[3/5] Depuración por Colinealidad...")
corr_matrix = X_train_scaled.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
umbral_corr = 0.85
to_drop = [column for column in upper.columns if any(upper[column] > umbral_corr)]

X_train_no_corr = X_train_scaled.drop(columns=to_drop)
X_test_no_corr = X_test_scaled.drop(columns=to_drop)
print(f"   -> Eliminadas {len(to_drop)} variables. Quedan {X_train_no_corr.shape[1]}.")

# --- 5.3 FILTRO PSI (Cruce Pasado vs Futuro) ---
print("\n[4/5] Análisis de Estabilidad Temporal (Cálculo PSI)...")
psi_scores = {}
for col in X_train_no_corr.columns:
    psi_val = calculate_psi(X_train_no_corr[col].values, X_test_no_corr[col].values)
    psi_scores[col] = psi_val

# Filtrar variables donde el Drift las destruyó (PSI > 0.20)
stable_features = [col for col, psi in psi_scores.items() if psi < 0.20]
print(f"   🚨 Variables destruidas por el Concept Drift (PSI >= 0.2): {X_train_no_corr.shape[1] - len(stable_features)}")
print(f"   ✅ 'ADN Inmutable' (PSI < 0.2): {len(stable_features)} variables.")

X_train_stable = X_train_no_corr[stable_features]
X_test_stable = X_test_no_corr[stable_features]

# --- 5.4 VALIDACIÓN ADVERSARIA ---
print("\n[5/5] Ejecutando Validación Adversaria...")
# Mezclamos train y test asignándoles etiquetas 0 y 1 respectivamente
X_adv = pd.concat([X_train_stable, X_test_stable])
y_adv = np.concatenate([np.zeros(len(X_train_stable)), np.ones(len(X_test_stable))])

# Modelo para adivinar si el dato es del pasado o del futuro
adv_model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=4)
adv_model.fit(X_adv, y_adv)
adv_preds = adv_model.predict_proba(X_adv)[:, 1]
adv_auc = roc_auc_score(y_adv, adv_preds)

print(f"   -> ROC-AUC Adversario: {adv_auc:.4f} (El ideal es 0.5000)")
if adv_auc > 0.80:
    print("   ⚠️ ADVERTENCIA: Los datasets aún son matemáticamente separables. El Drift es extremadamente fuerte.")
else:
    print("   ✅ ÉXITO: Los datasets son estadísticamente superpuestos. Hemos limpiado el Drift de los metadatos.")

# Asignar dataset final
X_train = X_train_stable
X_test = X_test_stable

# --- 6. ENTRENAMIENTO XGBOOST FINAL ---
print("\n" + "="*40)
print(f"⚙️ FASE 6: ENTRENAMIENTO SOBRE ADN INMUTABLE ({X_train.shape[1]} VARIABLES)")
print("="*40)

xgb_model = xgb.XGBClassifier(
    n_estimators=300,        
    max_depth=5,             
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,    
    reg_alpha=2.0,           
    reg_lambda=5.0,          
    min_child_weight=3,      
    objective='multi:softmax',
    num_class=len(common_sites),
    tree_method='hist',
    n_jobs=4,
    random_state=42
)

evals = [(X_train, y_train_enc), (X_test, y_test_enc)]
xgb_model.fit(X_train, y_train_enc, eval_set=evals, verbose=50)

# --- 7. EVALUACIÓN FINAL ---
print("\n" + "="*40)
print("🔍 FASE 7: RESULTADOS ABSOLUTOS")
print("="*40)
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

acc_train = accuracy_score(y_train_enc, y_pred_train)
acc_test = accuracy_score(y_test_enc, y_pred_test)

print("\n" + "🏆"*20)
print(f"   ACCURACY PASADO (Train/Val): {acc_train*100:.2f}%")
print(f"   ACCURACY FUTURO (Drift):     {acc_test*100:.2f}%")
print("🏆"*20)

# --- 8. VISUALIZACIÓN GRÁFICA ---
print("\n" + "="*40)
print("🎨 FASE 8: GENERACIÓN DE GRÁFICOS")
print("="*40)

importances_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=importances_df.head(20), x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
plt.title('Top 20 Variables del ADN Inmutable', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '01_Importancia_Variables_Inmutables.png'), dpi=300)
plt.close()

print("-> Realizando reducción PCA y generando gráficos limpios...")
top_10_sites = list(pd.Series(y_train).value_counts().index[:10])

mask_train_all = y_train.isin(top_10_sites).values
mask_test_all = y_test.isin(top_10_sites).values

X_pca_train_all = X_train[mask_train_all]
X_pca_test_all = X_test[mask_test_all]

pca = PCA(n_components=2)
X_combined_pca = pd.concat([X_pca_train_all, X_pca_test_all])
pca.fit(X_combined_pca)

pca_train_res = pca.transform(X_pca_train_all)
pca_test_res = pca.transform(X_pca_test_all)

for site in top_10_sites:
    plt.figure(figsize=(10, 8))
    
    plt.scatter(pca_train_res[:, 0], pca_train_res[:, 1], 
                color='lightgray', alpha=0.3, s=30, label='Otros Sitios')
    
    idx_tr = (y_train[mask_train_all] == site).values
    plt.scatter(pca_train_res[idx_tr, 0], pca_train_res[idx_tr, 1], 
                color='royalblue', alpha=0.8, s=120, edgecolor='white', label=f'Sitio {site} (Pasado)')
    
    idx_te = (y_test[mask_test_all] == site).values
    if sum(idx_te) > 0:
        plt.scatter(pca_test_res[idx_te, 0], pca_test_res[idx_te, 1], 
                    color='crimson', marker='X', s=150, edgecolor='black', linewidth=1.5, label=f'Sitio {site} (Drift)')

    plt.title(f'Espacio Latente (ADN Inmutable) - Sitio Web {site}', fontsize=14, fontweight='bold')
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'02_Plano_PCA_Sitio_{site}.png'), dpi=300)
    plt.close()

top_1_feature = importances_df.iloc[0]['Feature']
plt.figure(figsize=(10, 6))
sns.kdeplot(X_train[top_1_feature], fill=True, color='blue', label='Pasado (Train)', alpha=0.5)
sns.kdeplot(X_test[top_1_feature], fill=True, color='red', label='Futuro (Drift)', alpha=0.5)
plt.title(f'Validación de Superposición PSI:\n{top_1_feature}', fontweight='bold')
plt.xlabel('Valor Escalado')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '04_Distribucion_Drift.png'), dpi=300)
plt.close()

print("\n🚀 PROCESO FINALIZADO.")