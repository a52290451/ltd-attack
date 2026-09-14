"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: GEN_01_Ranking_Outliers.py
🚀 VERSIÓN: 3.0 (Integración con Datos CLEAN, Paridad 65 Clases y 94 Invariantes)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este script genera las primeras evidencias visuales (Gráficas 3 y 4) de la tesis:
    
    1. G3_Ranking_94_Invariantes: Entrena un Random Forest sobre el "ADN Inmutable"
       purificado para extraer el peso relativo (Gini Importance) de cada característica.
    2. G4_PCA_Multisitio_ConceptDrift: Proyecta el espacio latente global para 
       evidenciar el colapso estructural provocado por el paso del tiempo.
       
    [!] PARIDAD DE ABLACIÓN: Se aplica un filtro estricto cargando el LabelEncoder 
    de la rama Micro, garantizando que todo el análisis visual se circunscriba a 
    los 65 sitios de élite con integridad secuencial comprobada.
=========================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Para servidores sin entorno gráfico
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import joblib
from src.utils.paths import data_path, artifact_path, result_path

# --- 1. RUTAS DE ENTRADA Y DICCIONARIOS ---
CLEAN_PAST_CSV = data_path('historical', 'CLEAN_final_features_sites.csv')
CLEAN_DRIFT_CSV = data_path('historical', 'CLEAN_final_features_sites_concept_drift.csv')
FEATURES_TXT = result_path('macro', 'resultados_analisis', 'new_features_invariantes_seguras.txt')
VECTOR_LE_PATH = artifact_path('ds3_label_encoder_vec_3000.joblib')

SAVE_DIR = result_path('features', 'graficas_tesis')
os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "═"*70)
print("🎨 GENERANDO GRÁFICAS: BLOQUE 1 (RANKING DE ADN INMUTABLE Y PCA LATENTE)")
print("═"*70)

# --- 2. CARGA DEL FILTRO MAESTRO DE PARIDAD ---
print("⏳ Extrayendo filtro de paridad desde la rama de Vectores...")
try:
    le_vectores = joblib.load(VECTOR_LE_PATH)
    sitios_elite = set(le_vectores.classes_.astype(str))
    print(f"   ✅ Filtro cargado: {len(sitios_elite)} sitios de élite detectados.")
except Exception as e:
    print(f"   ❌ Error crítico al cargar el LabelEncoder de Vectores: {e}")
    exit()

# ---------------------------------------------------------
# GRÁFICA 3: Ranking de Importancia (94 Invariantes)
# ---------------------------------------------------------
print("\n1. Generando Gráfica 3: Recalculando Ranking de las 94 Invariantes...")
try:
    # 1. Leer la lista del nuevo ADN inmutable
    with open(FEATURES_TXT, 'r') as f:
        features_inv = [line.strip() for line in f.readlines() if line.strip()]
    
    # 2. Cargar datos del pasado limpios
    df_past = pd.read_csv(CLEAN_PAST_CSV).dropna(subset=['site_label'])
    df_past['site_label'] = df_past['site_label'].astype(str)
    
    # [!] APLICAR PARIDAD
    df_past = df_past[df_past['site_label'].isin(sitios_elite)].copy()
    
    print("   - Entrenando Random Forest para extraer pesos (Gini Importance)...")
    X = df_past[features_inv]
    y = df_past['site_label']
    
    # 3. Entrenar modelo para obtener Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # 4. Crear DataFrame de ranking
    df_rank = pd.DataFrame({
        'feature': features_inv,
        'importance': rf.feature_importances_
    })
    
    # Ordenar y tomar el Top 20
    df_rank = df_rank.sort_values(by='importance', ascending=False).head(20)

    # 5. Graficar
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(x='importance', y='feature', data=df_rank, palette='viridis')
    
    plt.title(f'Top 20 Características Más Importantes (ADN Inmutable: {len(features_inv)} Features)', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Importancia Relativa (Gini Importance)', fontsize=12)
    plt.ylabel('Característica (Feature)', fontsize=12)
    plt.tight_layout()
    
    path_g3 = os.path.join(SAVE_DIR, 'G3_Ranking_94_Invariantes_Paridad.png')
    plt.savefig(path_g3, dpi=300)
    plt.close()
    print(f"   ✅ Gráfica 3 guardada en: {path_g3}")

except Exception as e:
    print(f"   ❌ Error en Gráfica 3: {e}")

# ---------------------------------------------------------
# GRÁFICA 4: Evidencia de Colapso en Espacio Latente (PCA)
# ---------------------------------------------------------
print("\n2. Generando Gráfica 4: Evidencia de Colapso por Concept Drift...")
try:
    print("⏳ Cargando datasets del Futuro y sincronizando...")
    df_futr = pd.read_csv(CLEAN_DRIFT_CSV).dropna(subset=['site_label'])
    df_futr['site_label'] = df_futr['site_label'].astype(str)
    
    # [!] APLICAR PARIDAD AL FUTURO
    df_futr = df_futr[df_futr['site_label'].isin(sitios_elite)].copy()

    # Identificar columnas numéricas comunes (features)
    num_cols_past = set(df_past.select_dtypes(include=[np.number]).columns)
    num_cols_futr = set(df_futr.select_dtypes(include=[np.number]).columns)
    features_comunes = list(num_cols_past.intersection(num_cols_futr))
    
    # Remover metadatos de las features
    for meta in ['site_label', 'pcap_uid', 'hour_bin', 'date_id', 'site', 'Unnamed: 0']:
        if meta in features_comunes:
            features_comunes.remove(meta)
            
    # Limpiar NaNs rápidos
    df_past = df_past.dropna(subset=features_comunes + ['site_label'])
    df_futr = df_futr.dropna(subset=features_comunes + ['site_label'])

    print("⚙️ Entrenando PCA sobre el Pasado...")
    X_past = df_past[features_comunes].values
    y_past = df_past['site_label'].values
    
    # Escalado estándar vital para PCA
    scaler = StandardScaler()
    X_past_scaled = scaler.fit_transform(X_past)
    
    # Entrenar PCA
    pca = PCA(n_components=2)
    X_past_pca = pca.fit_transform(X_past_scaled)
    
    # Proyectar el futuro (usando el mismo scaler y pca del pasado)
    X_futr = df_futr[features_comunes].values
    y_futr = df_futr['site_label'].values
    X_futr_scaled = scaler.transform(X_futr)
    X_futr_pca = pca.transform(X_futr_scaled)
    
    # Seleccionamos 3 sitios de élite para destacar
    sitios_top = df_past['site_label'].value_counts().head(3).index.tolist()
    
    print(f"📌 Sitios de élite seleccionados para destacar: {sitios_top}")
    
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
        if mask_futr.sum() > 0:
            plt.scatter(X_futr_pca[mask_futr, 0], X_futr_pca[mask_futr, 1], 
                        c=color, marker='X', s=150, edgecolors='black', linewidth=1.5, 
                        alpha=0.9, label=f'Sitio {sitio} (Futuro/Drift)')

    # Varianzas explicadas
    var_1 = pca.explained_variance_ratio_[0] * 100
    var_2 = pca.explained_variance_ratio_[1] * 100
    
    plt.title('Colapso Estructural Multisitio en el Espacio Latente (Paridad 65 Clases)', 
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel(f'Componente Principal 1 ({var_1:.1f}%)', fontsize=12, fontweight='bold')
    plt.ylabel(f'Componente Principal 2 ({var_2:.1f}%)', fontsize=12, fontweight='bold')
    
    # Ajustar grilla y leyenda
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Limitamos los ejes para enfocar el clúster (ignorando posibles picos extremos de ruido)
    x_min, x_max = np.percentile(X_past_pca[:, 0], [1, 99])
    y_min, y_max = np.percentile(X_past_pca[:, 1], [1, 99])
    plt.xlim(x_min - 5, x_max + 5)
    plt.ylim(y_min - 5, y_max + 5)

    plt.tight_layout()
    
    path_g4 = os.path.join(SAVE_DIR, 'G4_PCA_Multisitio_Drift_Paridad.png')
    plt.savefig(path_g4, dpi=300)
    plt.close()
    
    print(f"✅ Gráfico 4 (PCA Multisitio) guardado en: {path_g4}")

except Exception as e:
    import traceback
    print(f"❌ Error al generar el PCA:\n{traceback.format_exc()}")

print("="*70)
print("🚀 Bloque 1 finalizado con éxito.")
print("===========================================================================\n")
