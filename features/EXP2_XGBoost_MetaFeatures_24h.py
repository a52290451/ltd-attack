"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: EXP2_XGBoost_MetaFeatures_24h.py
🚀 VERSIÓN: 3.0 (Evaluación Predictiva del Hiperespacio 24h con Paridad Estricta)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este experimento constituye la validación predictiva final de la rama Macro.
    Tras descubrir que las características crudas caen al 28.05% (EXP1), se eleva 
    el nivel de abstracción extrayendo "Firmas Dinámicas" (Meta-Features) sobre 
    ciclos de 24 horas.
    
    Se aplica un filtro de ortogonalidad (Pearson < 0.85) sobre el set de entrenamiento
    para destilar el "ADN Puro" y se entrena un modelo XGBoost hiper-regularizado 
    (L1/L2) para medir si la evaluación de flujos temporales resiste el Concept Drift.

    [!] PARIDAD DE ABLACIÓN: Solo se evalúan los 65 sitios de élite (rama Micro).
=========================================================================================
"""

import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
from scipy.stats import skew
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import re
import sys

# --- 1. RUTAS Y DICCIONARIOS ---
FEATURES_LIST_TXT = '../vectores_features/resultados_analisis/new_features_invariantes_seguras.txt'
HISTORIC_CSV = '../../output/CLEAN_final_features_sites.csv'
DRIFT_CSV = '../../output/CLEAN_final_features_sites_concept_drift.csv'
VECTOR_LE_PATH = '../vectores/resultados/ds3_label_encoder_vec_3000.joblib'
SAVE_DIR = './resultados'

os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "═"*70)
print("🚀 EXP 2: PREDICCIÓN XGBOOST SOBRE META-FEATURES ORTOGONALES (24H)")
print("═"*70)

# --- 2. FUNCIONES MATEMÁTICAS PROTEGIDAS ---
def get_autocorr(x, lag=1):
    if len(x) <= lag: return 0.0
    x1, x2 = x[:-lag], x[lag:]
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

def extract_advanced_signatures(df, features, dataset_name):
    print(f"   ⏳ Extrayendo Meta-Variables de {dataset_name}...")
    records = []
    grouped = df.groupby(['site_label', 'date_id'])
    
    for (site, date), group in tqdm(grouped, desc=f"Generando Perfiles ({dataset_name})"):
        group = group.sort_values('hour_bin')
        if len(group) < 4: continue # Filtro mínimo de muestras diarias
        
        record = {'site_label': site}
        for feat in features:
            vals = group[feat].values
            diffs = np.diff(vals)
            
            record[f'{feat}_mean'] = np.mean(vals)
            record[f'{feat}_std'] = np.std(vals)
            record[f'{feat}_range'] = np.max(vals) - np.min(vals)
            s_val = skew(vals) if len(vals) > 2 and np.var(vals) > 0 else 0.0
            record[f'{feat}_skew'] = s_val if not np.isnan(s_val) else 0.0
            
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

# --- 3. PIPELINE DE CARGA Y PARIDAD ---
try:
    print("\n[1/5] Extrayendo filtro de paridad y ADN Inmutable...")
    le_vectores = joblib.load(VECTOR_LE_PATH)
    sitios_elite = set(le_vectores.classes_.astype(str))

    with open(FEATURES_LIST_TXT, 'r') as f:
        features_inv = [line.strip() for line in f.readlines() if line.strip()]

    print("\n[2/5] Cargando datasets purificados y aplicando Paridad Estricta...")
    df_hist_raw = pd.read_csv(HISTORIC_CSV).dropna(subset=['site_label'])
    df_drift_raw = pd.read_csv(DRIFT_CSV).dropna(subset=['site_label'])

    df_hist_raw['site_label'] = df_hist_raw['site_label'].astype(str)
    df_drift_raw['site_label'] = df_drift_raw['site_label'].astype(str)

    # Paridad
    df_hist_raw = df_hist_raw[df_hist_raw['site_label'].isin(sitios_elite)].copy()
    df_drift_raw = df_drift_raw[df_drift_raw['site_label'].isin(sitios_elite)].copy()

    # Motor Temporal Regex
    for df in [df_hist_raw, df_drift_raw]:
        parsed = df['pcap_name'].apply(parse_time_from_pcap)
        df['date_id'] = [p[0] for p in parsed]
        df['hour_bin'] = [p[1] for p in parsed]
        df.dropna(subset=['date_id', 'hour_bin'], inplace=True)

    # Extracción de Meta-Features
    df_train = extract_advanced_signatures(df_hist_raw, features_inv, "Pasado (Train)")
    df_test = extract_advanced_signatures(df_drift_raw, features_inv, "Futuro (Drift)")

    # Separación de X e Y
    X_train_raw = df_train.drop(columns=['site_label'])
    y_train = df_train['site_label']
    X_test_raw = df_test.drop(columns=['site_label'])
    y_test = df_test['site_label']

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # --- 4. CLIPPING Y ESCALADO (SIN DATA LEAKAGE) ---
    print("\n[3/5] Aplicando Clipping y Z-Score (Ajustado solo en el Pasado)...")
    for col in X_train_raw.columns:
        if col in X_test_raw.columns:
            lower = np.percentile(X_train_raw[col], 1)
            upper = np.percentile(X_train_raw[col], 99)
            X_train_raw[col] = np.clip(X_train_raw[col], lower, upper)
            X_test_raw[col] = np.clip(X_test_raw[col], lower, upper)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns)

    # --- 5. DEPURACIÓN DE ORTOGONALIDAD (PEARSON) ---
    print("\n[4/5] 🧬 Destilando ADN Puro (Purgando Colinealidad > 0.85)...")
    corr_matrix = X_train_scaled.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
    
    print(f"   🚨 Se purgaron {len(to_drop)} dimensiones redundantes.")
    
    X_train_pure = X_train_scaled.drop(columns=to_drop)
    X_test_pure = X_test_scaled.drop(columns=to_drop)
    print(f"   ✅ Meta-Variables Ortogonales finales: {X_train_pure.shape[1]}")

    # --- 6. ENTRENAMIENTO XGBOOST ---
    print("\n[5/5] ⚙️ Entrenando XGBoost Hiper-Regularizado sobre ADN Puro...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,        
        max_depth=5,             
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,    
        reg_alpha=2.0,  # Regularización L1 (Lasso)
        reg_lambda=5.0, # Regularización L2 (Ridge)         
        min_child_weight=3,      
        objective='multi:softmax',
        num_class=len(sitios_elite),
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )

    # Entrenamos
    xgb_model.fit(X_train_pure, y_train_enc)

    # --- 7. EVALUACIÓN DE CONCEPT DRIFT ---
    y_pred_train = xgb_model.predict(X_train_pure)
    y_pred_test = xgb_model.predict(X_test_pure)

    acc_train = accuracy_score(y_train_enc, y_pred_train)
    acc_test = accuracy_score(y_test_enc, y_pred_test)

    print("\n" + "🏆"*25)
    print(f"   ACCURACY PASADO (Validación): {acc_train*100:.2f}%")
    print(f"   ACCURACY FUTURO (Drift 24h):  {acc_test*100:.2f}%")
    print("🏆"*25 + "\n")

    # --- 8. GENERACIÓN DE EVIDENCIA VISUAL ---
    print("🎨 Generando Gráficas Finales (G11 y G12)...")
    
    # G11: Importancia de Variables
    importances = pd.DataFrame({
        'Feature': X_train_pure.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importances.head(15), x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title('Top 15 Meta-Características del ADN Puro (Impacto XGBoost)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Ganancia de Información (F-Score)', fontsize=12)
    plt.ylabel('Firma Dinámica (24h)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'G11_Importancia_MetaFeatures.png'), dpi=300)
    plt.close()

    # G12: Desplazamiento de la Mejor Variable
    top_feature = importances.iloc[0]['Feature']
    plt.figure(figsize=(10, 6))
    sns.kdeplot(X_train_pure[top_feature], fill=True, color='#3498db', label='Pasado (Entrenamiento)', alpha=0.5)
    sns.kdeplot(X_test_pure[top_feature], fill=True, color='#e74c3c', label='Futuro (Concept Drift)', alpha=0.5)
    
    plt.axvline(X_train_pure[top_feature].mean(), color='#2980b9', linestyle='--')
    plt.axvline(X_test_pure[top_feature].mean(), color='#c0392b', linestyle='--')

    plt.title(f'Desplazamiento Estructural en la Mejor Meta-Variable\n{top_feature}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Valor Escalado (Z-Score)', fontsize=12)
    plt.ylabel('Densidad de Población', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(-4, 4)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'G12_KDE_Top_MetaFeature.png'), dpi=300)
    plt.close()

    print(f"✅ Gráficas G11 y G12 guardadas en: {SAVE_DIR}")
    print("===========================================================================\n")

except Exception as e:
    import traceback
    print(f"❌ Error crítico durante la ejecución:\n{traceback.format_exc()}")