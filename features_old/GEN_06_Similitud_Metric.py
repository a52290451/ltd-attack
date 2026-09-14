import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- 1. RUTAS ---
CLEAN_PAST_CSV = '../../output/preprocessed/02_features_robust_.csv'
CLEAN_DRIFT_CSV = '../../output/preprocessed/02_features_robust_drift.csv'
SAVE_DIR = './graficas_tesis'

os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "="*60)
print("🎨 FASE 5.3: GENERANDO MATRIZ DE SIMILITUD Y CURVAS KDE")
print("="*60)

try:
    print("⏳ Cargando y preparando datasets...")
    df_past = pd.read_csv(CLEAN_PAST_CSV)
    df_drift = pd.read_csv(CLEAN_DRIFT_CSV)

    # --- PUENTE DE ETIQUETAS ---
    if 'site_label' not in df_drift.columns and 'site' in df_drift.columns:
        df_bridge = pd.read_csv('../../output/final_features_sites.csv', usecols=['site', 'site_label']).drop_duplicates()
        site_to_id = dict(zip(df_bridge['site'], df_bridge['site_label']))
        df_drift['site_label'] = df_drift['site'].map(site_to_id)
        if 'site_label' not in df_past.columns:
            df_past['site_label'] = df_past['site'].map(site_to_id)

    df_past = df_past.dropna(subset=['site_label'])
    df_drift = df_drift.dropna(subset=['site_label'])

    # Identificar variables numéricas (las 79 ortogonales)
    cols_feat = df_past.select_dtypes(include=[np.number]).columns.tolist()
    for meta in ['site_label', 'pcap_uid', 'hour_bin', 'date_id', 'site', 'Unnamed: 0']:
        if meta in cols_feat: cols_feat.remove(meta)

    # --- LIMPIEZA Y ESCALADO ---
    for col in cols_feat:
        low, upp = np.percentile(df_past[col], [1, 99])
        df_past[col] = np.clip(df_past[col], low, upp)
        df_drift[col] = np.clip(df_drift[col], low, upp)

    scaler = StandardScaler()
    X_past = scaler.fit_transform(df_past[cols_feat])
    X_drift = scaler.transform(df_drift[cols_feat])

    # --- 2. CÁLCULO DE CENTROIDES (Top 10 sitios para claridad visual) ---
    top_sites = df_past['site_label'].value_counts().head(10).index.tolist()
    
    past_centroids = []
    drift_centroids = []
    
    for s in top_sites:
        past_centroids.append(X_past[df_past['site_label'] == s].mean(axis=0))
        drift_centroids.append(X_drift[df_drift['site_label'] == s].mean(axis=0))

    past_centroids = np.array(past_centroids)
    drift_centroids = np.array(drift_centroids)

    # --- GRÁFICO 8: MATRIZ DE SIMILITUD COSENO ---
    print("📊 Generando Gráfico 8: Matriz de Similitud de Centroides...")
    sim_matrix = cosine_similarity(drift_centroids, past_centroids)

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap='YlGnBu',
                xticklabels=[f'S{int(s)} P' for s in top_sites],
                yticklabels=[f'S{int(s)} D' for s in top_sites])
    
    plt.title('Matriz de Similitud Coseno de Centroides\n(Pasado vs. Drift - Top 10 Sitios)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Centroides del Pasado (Train)', fontweight='bold')
    plt.ylabel('Centroides del Futuro (Drift)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'G8_Matriz_Similitud_Coseno.png'), dpi=300)

    # --- GRÁFICO 9: KDE INTRA-CLASE VS INTER-CLASE ---
    print("📊 Generando Gráfico 9: Curvas KDE de Proximidad...")
    
    intra_sims = []
    inter_sims = []

    for i, s in enumerate(top_sites):
        # Muestras reales del sitio en el futuro
        samples_drift = X_drift[df_drift['site_label'] == s]
        
        # Similitud Intra: Muestras del Sitio X (Drift) vs Centroide del Sitio X (Pasado)
        intra = cosine_similarity(samples_drift, past_centroids[i].reshape(1, -1)).flatten()
        intra_sims.extend(intra)
        
        # Similitud Inter: Muestras del Sitio X (Drift) vs Otros Centroides (Pasado)
        other_centroids = np.delete(past_centroids, i, axis=0)
        inter = cosine_similarity(samples_drift, other_centroids).flatten()
        inter_sims.extend(inter)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(intra_sims, fill=True, color='green', label='Similitud Intra-clase (Mismo Sitio)', alpha=0.5, linewidth=2)
    sns.kdeplot(inter_sims, fill=True, color='red', label='Similitud Inter-clase (Sitios Diferentes)', alpha=0.5, linewidth=2)
    
    plt.axvline(np.mean(intra_sims), color='darkgreen', linestyle='--')
    plt.axvline(np.mean(inter_sims), color='darkred', linestyle='--')

    plt.title('Distribución de Similitud Coseno Post-Drift\nEvidencia de Separabilidad Relativa', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Similitud Coseno (-1 a 1)', fontsize=12)
    plt.ylabel('Densidad', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'G9_KDE_Similitud_Intra_Inter.png'), dpi=300)

    print(f"🚀 Gráficos 8 y 9 generados en {SAVE_DIR}")

except Exception as e:
    print(f"❌ Error: {e}")