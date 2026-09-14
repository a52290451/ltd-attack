import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Para servidores sin entorno gráfico
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Rutas de entrada
PAST_CSV = '../../output/preprocessed/01_all_features.csv'
PAST_CSV_ROBUST = '../../output/preprocessed/02_features_robust_.csv'
RAW_FUTURO_CSV = '../../output/final_features_sites_concept_drift.csv'
CLEAN_PASADO_CSV = '../../output/preprocessed/01_all_features.csv'
RAW_PASADO_CSV = '../../output/final_features_sites.csv'
FEATURES_TXT = '../vectores_features/resultados_analisis/features_invariantes_seguras.txt'
# Futuro (Crudo, con las anomalías que queremos evidenciar)
FUTR_CSV = '../../output/final_features_sites_concept_drift.csv'

# Ruta de salida
SAVE_DIR = './graficas_tesis'
os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "="*60)
print("🎨 GENERANDO GRÁFICAS: BLOQUE 1 (RANKING 45 INVARIANTES Y OUTLIERS)")
print("="*60)

# ---------------------------------------------------------
# GRÁFICA 3: Ranking de Importancia (45 Invariantes)
# ---------------------------------------------------------
print("1. Generando Gráfica 3: Recalculando Ranking de las 45 Invariantes...")
try:
    # 1. Leer la lista de 45 variables
    with open(FEATURES_TXT, 'r') as f:
        features_45 = [line.strip() for line in f.readlines() if line.strip()]
    
    # 2. Cargar datos del pasado solo con esas 45 variables
    print("   - Entrenando Random Forest rápido para extraer pesos...")
    df_past = pd.read_csv(PAST_CSV_ROBUST).dropna(subset=['site_label'])
    
    X = df_past[features_45]
    y = df_past['site_label']
    
    # 3. Entrenar modelo para obtener Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # 4. Crear DataFrame de ranking
    df_rank = pd.DataFrame({
        'feature': features_45,
        'importance': rf.feature_importances_
    })
    
    # Ordenar y tomar el Top 20
    df_rank = df_rank.sort_values(by='importance', ascending=False).head(20)

    # 5. Graficar
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(x='importance', y='feature', data=df_rank, palette='viridis')
    
    plt.title('Top 20 Características Más Importantes (Subconjunto 45 Invariantes)', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Importancia Relativa (Gini Importance)', fontsize=12)
    plt.ylabel('Característica (Feature)', fontsize=12)
    plt.tight_layout()
    
    path_g3 = os.path.join(SAVE_DIR, 'G3_Ranking_45_Invariantes.png')
    plt.savefig(path_g3, dpi=300)
    plt.close()
    print(f"   ✅ Gráfica 3 guardada en: {path_g3}")
except Exception as e:
    print(f"   ❌ Error en Gráfica 3: {e}")

# ---------------------------------------------------------
# GRÁFICA 4: Evidencia de Outliers (Pre-Limpieza)
# ---------------------------------------------------------
print("\n2. Generando Gráfica 4: Evidencia de Outliers...")
try:
    print("⏳ Cargando datasets...")
    df_past = pd.read_csv(PAST_CSV_ROBUST)
    df_futr = pd.read_csv(FUTR_CSV)
    
    # Asegurar que ambos tengan 'site_label'
    if 'site_label' not in df_past.columns or 'site_label' not in df_futr.columns:
        raise ValueError("No se encontró la columna 'site_label'.")

    # Identificar columnas numéricas comunes (features)
    num_cols_past = set(df_past.select_dtypes(include=[np.number]).columns)
    num_cols_futr = set(df_futr.select_dtypes(include=[np.number]).columns)
    features = list(num_cols_past.intersection(num_cols_futr))
    
    # Remover metadatos de las features
    for meta in ['site_label', 'pcap_uid', 'hour_bin', 'date_id', 'site', 'Unnamed: 0']:
        if meta in features:
            features.remove(meta)
            
    # Limpiar NaNs rápidos
    df_past = df_past.dropna(subset=features + ['site_label'])
    df_futr = df_futr.dropna(subset=features + ['site_label'])

    print("⚙️ Entrenando PCA sobre el Pasado...")
    X_past = df_past[features].values
    y_past = df_past['site_label'].values
    
    # Escalado estándar vital para PCA
    scaler = StandardScaler()
    X_past_scaled = scaler.fit_transform(X_past)
    
    # Entrenar PCA
    pca = PCA(n_components=2)
    X_past_pca = pca.fit_transform(X_past_scaled)
    
    # Proyectar el futuro (usando el mismo scaler y pca del pasado)
    X_futr = df_futr[features].values
    y_futr = df_futr['site_label'].values
    X_futr_scaled = scaler.transform(X_futr)
    X_futr_pca = pca.transform(X_futr_scaled)
    
    # Seleccionamos 3 sitios para destacar (Asegúrate de que existan en tu dataset)
    # Por defecto tomaré los 3 sitios más frecuentes en el pasado
    sitios_top = df_past['site_label'].value_counts().head(3).index.tolist()
    
    print(f"📌 Sitios seleccionados para destacar: {sitios_top}")
    
    # Paleta de colores para los 3 sitios
    colores = ['#3498db', '#2ecc71', '#9b59b6'] # Azul, Verde, Morado
    
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # 1. Dibujar el "Contexto" (Fondo gris con el resto de los sitios del pasado)
    mask_bg = ~np.isin(y_past, sitios_top)
    plt.scatter(X_past_pca[mask_bg, 0], X_past_pca[mask_bg, 1], 
                c='#bdc3c7', alpha=0.3, label='Otros Sitios (Contexto Histórico)', s=30)
    
    # 2. Dibujar los Sitios Destacados
    for idx, sitio in enumerate(sitios_top):
        color = colores[idx]
        
        # Pasado (Círculos limpios)
        mask_past = (y_past == sitio)
        plt.scatter(X_past_pca[mask_past, 0], X_past_pca[mask_past, 1], 
                    c=color, marker='o', s=80, edgecolors='white', linewidth=1, 
                    alpha=0.8, label=f'Sitio {sitio} (Pasado)')
        
        # Futuro (Cruces rojas/anómalas grandes con borde negro)
        mask_futr = (y_futr == sitio)
        plt.scatter(X_futr_pca[mask_futr, 0], X_futr_pca[mask_futr, 1], 
                    c=color, marker='X', s=150, edgecolors='black', linewidth=1.5, 
                    alpha=0.9, label=f'Sitio {sitio} (Futuro/Anomalías)')

    # Varianzas explicadas
    var_1 = pca.explained_variance_ratio_[0] * 100
    var_2 = pca.explained_variance_ratio_[1] * 100
    
    plt.title('Colapso Estructural Multisitio en el Espacio Latente (Pre-Limpieza Z-Score)', 
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel(f'Componente Principal 1 ({var_1:.1f}%)', fontsize=12, fontweight='bold')
    plt.ylabel(f'Componente Principal 2 ({var_2:.1f}%)', fontsize=12, fontweight='bold')
    
    # Ajustar grilla y leyenda
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    path_g4 = os.path.join(SAVE_DIR, 'G4_PCA_Multisitio_Outliers_Robust.png')
    plt.savefig(path_g4, dpi=300)
    plt.close()
    
    print(f"✅ Gráfico 4 (PCA Multisitio) guardado en: {path_g4}")

except Exception as e:
    print(f"❌ Error al generar el PCA: {e}")

print("="*60)
print("🚀 Bloque 1 finalizado.")