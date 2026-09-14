import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, zscore
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os

# --- 1. RUTAS ---
FEATURES_LIST_TXT = '../vectores_features/resultados_analisis/features_invariantes_seguras.txt'
HISTORIC_CSV = '../../output/preprocessed/02_features_robust_.csv'
SAVE_DIR = './graficas_tesis'
os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "="*60)
print("🎨 GENERANDO GRÁFICO 5: MATRIZ DE CORRELACIÓN (ADN PURO)")
print("="*60)

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

def extract_advanced_signatures_train(df, features):
    print("⏳ Extrayendo Meta-Variables de Pasado (Train)...")
    records = []
    grouped = df.groupby(['site_label', 'date_id'])
    for (site, date), group in tqdm(grouped, desc="Procesando"):
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

# --- 3. PROCESAMIENTO Y DEPURACIÓN ---
try:
    with open(FEATURES_LIST_TXT, 'r') as f:
        features_45 = [line.strip() for line in f.readlines() if line.strip()]

    df_hist_raw = pd.read_csv(HISTORIC_CSV).dropna(subset=['site_label'])
    df_train = extract_advanced_signatures_train(df_hist_raw, features_45)

    X_train = df_train.drop(columns=['site_label'])

    print("⚙️ Aplicando Limpieza y Escalado...")
    # Escalado estándar para normalizar el cálculo de correlación
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    print("⚙️ Aplicando Filtro de Ortogonalidad (Pearson > 0.85)...")
    corr_matrix_full = X_train_scaled.corr().abs()
    
    # Triángulo superior de la matriz
    upper = corr_matrix_full.where(np.triu(np.ones(corr_matrix_full.shape), k=1).astype(bool))
    
    # Encontrar variables con correlación > 0.85
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    print(f"   🚨 Se detectaron y eliminaron {len(to_drop)} variables colineales.")
    
    # Dataset final (ADN Puro)
    X_train_pure = X_train_scaled.drop(columns=to_drop)
    print(f"   ✅ Variables restantes (Ortogonales): {X_train_pure.shape[1]}")

    # --- 4. GENERACIÓN DEL GRÁFICO 5 ---
    print("🎨 Generando Mapa de Calor...")
    # Calculamos la correlación final (ahora incluyendo signos para ver variabilidad)
    final_corr = X_train_pure.corr()

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    # Heatmap usando paleta divergente
    sns.heatmap(final_corr, cmap='coolwarm', vmin=-1, vmax=1, 
                xticklabels=False, yticklabels=False, 
                cbar_kws={'label': 'Coeficiente de Correlación de Pearson'})
    
    plt.title(f'Matriz de Correlación de Variables Finales ({X_train_pure.shape[1]} Dimensiones)\nEvidencia de Ausencia de Colinealidad (>0.85)', 
              fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    output_path = os.path.join(SAVE_DIR, 'G5_Matriz_Correlacion_Ortogonal.png')
    plt.savefig(output_path, dpi=300)
    
    print(f"✅ Gráfico 5 generado en: {output_path}")

except Exception as e:
    print(f"❌ Error durante la ejecución: {e}")