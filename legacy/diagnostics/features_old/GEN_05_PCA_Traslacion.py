import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# --- 1. RUTAS ---
CLEAN_PAST_CSV = '../../output/preprocessed/02_features_robust_.csv'
CLEAN_DRIFT_CSV = '../../output/preprocessed/02_features_robust_drift.csv'
SAVE_DIR = './graficas_tesis'

os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "="*60)
print("🎨 GENERANDO GRÁFICO 7: PCA DE TRASLACIÓN LATENTE (FASE 5)")
print("="*60)

try:
    print("⏳ Cargando datasets depurados...")
    df_past = pd.read_csv(CLEAN_PAST_CSV)
    df_drift = pd.read_csv(CLEAN_DRIFT_CSV)

    # --- CORRECCIÓN: MAPEO DE 'SITE' A 'SITE_LABEL' ---
    if 'site_label' not in df_drift.columns and 'site' in df_drift.columns:
        print("🔄 Sincronizando etiquetas 'site_label' para el dataset del Futuro...")
        df_bridge = pd.read_csv('../../output/final_features_sites.csv', usecols=['site', 'site_label']).drop_duplicates()
        site_to_id = dict(zip(df_bridge['site'], df_bridge['site_label']))
        df_drift['site_label'] = df_drift['site'].map(site_to_id)
        
    if 'site_label' not in df_past.columns and 'site' in df_past.columns:
        df_past['site_label'] = df_past['site'].map(site_to_id)

    # Asegurarnos de tener el site_label sin nulos
    df_past = df_past.dropna(subset=['site_label'])
    df_drift = df_drift.dropna(subset=['site_label'])

    # Identificar las variables numéricas (las 79 ortogonales)
    cols_features = df_past.select_dtypes(include=[np.number]).columns.tolist()
    for meta in ['site_label', 'pcap_uid', 'hour_bin', 'date_id', 'site', 'Unnamed: 0']:
        if meta in cols_features: cols_features.remove(meta)

    # --- 2. CLIPPING PARA VER EL NÚCLEO DEL CLÚSTER (SIN COLAS LARGAS) ---
    print("🧹 Aplicando Clipping de Percentiles (1-99)...")
    for col in cols_features:
        if col in df_drift.columns:
            lower = np.percentile(df_past[col].dropna(), 1)
            upper = np.percentile(df_past[col].dropna(), 99)
            df_past[col] = np.clip(df_past[col], lower, upper)
            df_drift[col] = np.clip(df_drift[col], lower, upper)

    # --- 3. PREPARACIÓN Y PCA ---
    print("⚙️ Estandarizando y Entrenando PCA sobre el Pasado...")
    scaler = StandardScaler()
    
    # Extraemos solo las columnas comunes
    common_cols = [c for c in cols_features if c in df_drift.columns]
    
    X_past_raw = df_past[common_cols].values
    X_drift_raw = df_drift[common_cols].values
    
    # El espacio latente se define con las coordenadas del Pasado
    X_past_scaled = scaler.fit_transform(X_past_raw)
    X_drift_scaled = scaler.transform(X_drift_raw)

    pca = PCA(n_components=2)
    X_past_pca = pca.fit_transform(X_past_scaled)
    X_drift_pca = pca.transform(X_drift_scaled)

    # --- 4. SELECCIÓN DEL SITIO A VISUALIZAR ---
    # Tomaremos el sitio que tenga suficientes datos en ambos tiempos para que el clúster se vea denso
    sites_past = set(df_past['site_label'].unique())
    sites_drift = set(df_drift['site_label'].unique())
    valid_sites = list(sites_past.intersection(sites_drift))
    
    # Elegiremos el primer sitio válido (puedes cambiar este ID si quieres mostrar otro)
    target_site = valid_sites[0] 
    print(f"📌 Seleccionando Sitio {int(target_site)} para ilustrar la traslación...")

    # Máscaras booleanas
    mask_past_target = (df_past['site_label'] == target_site).values
    mask_drift_target = (df_drift['site_label'] == target_site).values
    
    # --- 5. GENERACIÓN DEL GRÁFICO 7 ---
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")

    # 5.1. Fondo Gris (Contexto del Espacio Latente)
    plt.scatter(X_past_pca[:, 0], X_past_pca[:, 1], 
                c='#bdc3c7', alpha=0.3, s=25, label='Contexto (Otros Sitios - Pasado)')

    # 5.2. El Clúster del Pasado (Azul)
    plt.scatter(X_past_pca[mask_past_target, 0], X_past_pca[mask_past_target, 1], 
                c='#3498db', marker='o', s=90, edgecolors='white', linewidth=1, 
                alpha=0.8, label=f'Sitio {int(target_site)} (Clúster Histórico)')

    # 5.3. El Clúster del Futuro trasladado (Rojo oscuro)
    if sum(mask_drift_target) > 0:
        plt.scatter(X_drift_pca[mask_drift_target, 0], X_drift_pca[mask_drift_target, 1], 
                    c='#c0392b', marker='X', s=120, edgecolors='black', linewidth=1.2, 
                    alpha=0.9, label=f'Sitio {int(target_site)} (Concept Drift / Trasladado)')

    var_1 = pca.explained_variance_ratio_[0] * 100
    var_2 = pca.explained_variance_ratio_[1] * 100

    plt.title(f'Traslación del Espacio Latente por Concept Drift (Sitio {int(target_site)})\nEvidencia de Fracaso de Clasificación por Fronteras Estáticas', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(f'Componente Principal 1 ({var_1:.1f}%)', fontsize=12, fontweight='bold')
    plt.ylabel(f'Componente Principal 2 ({var_2:.1f}%)', fontsize=12, fontweight='bold')
    
    plt.legend(loc='best', framealpha=0.9, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Limitamos los ejes para enfocar el clúster (ignorando posibles picos extremos sueltos)
    x_min, x_max = np.percentile(X_past_pca[:, 0], [1, 99])
    y_min, y_max = np.percentile(X_past_pca[:, 1], [1, 99])
    plt.xlim(x_min - 2, x_max + 2)
    plt.ylim(y_min - 2, y_max + 2)
    
    plt.tight_layout()
    
    output_path = os.path.join(SAVE_DIR, f'G7_PCA_Traslacion_Sitio_{int(target_site)}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"🚀 Gráfico 7 generado exitosamente en: {output_path}")

except Exception as e:
    import traceback
    print(f"❌ Error crítico durante la generación:\n{traceback.format_exc()}")