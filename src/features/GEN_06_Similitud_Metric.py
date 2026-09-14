"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: GEN_06_Similitud_Metric.py
🚀 VERSIÓN: 3.0 (Justificación Matemática del Deep Metric Learning con Paridad)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este script cierra la fase de diagnóstico (Ablation Study). Tras demostrar el 
    fracaso de la clasificación por fronteras estáticas (coordenadas absolutas), este 
    análisis evalúa la "Similitud Angular" (Cosine Similarity) en el hiperespacio.

    HIPÓTESIS DE TRANSICIÓN: 
    Aunque el Concept Drift traslada geográficamente el clúster de un sitio web, 
    su estructura interna y su distancia relativa frente a otros clústeres se conserva.
    Si la curva Intra-clase (verde) es separable de la Inter-clase (roja), se 
    demuestra empíricamente que el Deep Metric Learning es la solución definitiva.

    [!] ACTUALIZACIÓN V3.0 (RIGOR ESTRICTO):
    - Paridad de Ablación: 65 Clases de Élite de la rama Micro.
    - Proyección exclusiva sobre el ADN Inmutable (94 Invariantes Reales).
    - Prevención estricta de Fuga de Datos (Clipping y Z-Score ajustados solo en Pasado).
=========================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
import sys
from src.utils.paths import data_path, artifact_path, result_path

# --- 1. RUTAS Y DICCIONARIOS ---
CLEAN_PAST_CSV = data_path('historical', 'CLEAN_final_features_sites.csv')
CLEAN_DRIFT_CSV = data_path('historical', 'CLEAN_final_features_sites_concept_drift.csv')
FEATURES_LIST_TXT = result_path('macro', 'resultados_analisis', 'new_features_invariantes_seguras.txt')
VECTOR_LE_PATH = artifact_path('ds3_label_encoder_vec_3000.joblib')
SAVE_DIR = result_path('features', 'graficas_tesis')

os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "═"*70)
print("🎨 FASE FINAL: GENERANDO G9 Y G10 (JUSTIFICACIÓN DE DEEP METRIC LEARNING)")
print("═"*70)

try:
    print("\n[1/4] Extrayendo filtro de paridad y ADN Inmutable...")
    le_vectores = joblib.load(VECTOR_LE_PATH)
    sitios_elite = set(le_vectores.classes_.astype(str))

    with open(FEATURES_LIST_TXT, 'r') as f:
        features_inv = [line.strip() for line in f.readlines() if line.strip()]

    print("[2/4] Cargando datasets purificados y aplicando Paridad Estricta...")
    df_past = pd.read_csv(CLEAN_PAST_CSV).dropna(subset=['site_label'])
    df_drift = pd.read_csv(CLEAN_DRIFT_CSV).dropna(subset=['site_label'])

    df_past['site_label'] = df_past['site_label'].astype(str)
    df_drift['site_label'] = df_drift['site_label'].astype(str)

    # Filtro Estricto de Paridad
    df_past = df_past[df_past['site_label'].isin(sitios_elite)].copy()
    df_drift = df_drift[df_drift['site_label'].isin(sitios_elite)].copy()

    df_past = df_past.dropna(subset=features_inv)
    df_drift = df_drift.dropna(subset=features_inv)

    # --- 3. LIMPIEZA SIN DATA LEAKAGE (CLIPPING) ---
    print("[3/4] Aplicando Clipping (1% - 99%) basado EXCLUSIVAMENTE en el Pasado...")
    for col in features_inv:
        if col in df_drift.columns:
            lower = np.percentile(df_past[col], 1)
            upper = np.percentile(df_past[col], 99)
            df_past[col] = np.clip(df_past[col], lower, upper)
            df_drift[col] = np.clip(df_drift[col], lower, upper)

    # --- 4. PREPARACIÓN Y CÁLCULO DE CENTROIDES ---
    print("[4/4] Escalando Hiperespacio y Calculando Centroides Relativos...")
    scaler = StandardScaler()
    X_past = scaler.fit_transform(df_past[features_inv])
    X_drift = scaler.transform(df_drift[features_inv])

    # Seleccionamos el Top 10 de sitios de élite con más muestras para claridad visual en el Heatmap
    top_sites = df_past['site_label'].value_counts().head(10).index.tolist()
    
    past_centroids = []
    drift_centroids = []
    
    for s in top_sites:
        past_centroids.append(X_past[df_past['site_label'] == s].mean(axis=0))
        drift_centroids.append(X_drift[df_drift['site_label'] == s].mean(axis=0))

    past_centroids = np.array(past_centroids)
    drift_centroids = np.array(drift_centroids)

    # --- 5. GRÁFICA 9: MATRIZ DE SIMILITUD COSENO ---
    print("\n📊 Generando Gráfica 9: Matriz de Similitud Coseno de Centroides...")
    sim_matrix = cosine_similarity(drift_centroids, past_centroids)

    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap='YlGnBu',
                xticklabels=[f'Sitio {s} (Pasado)' for s in top_sites],
                yticklabels=[f'Sitio {s} (Futuro)' for s in top_sites])
    
    plt.title('Matriz de Similitud Coseno de Centroides Post-Drift\n(Top 10 Sitios de Élite evaluados sobre 94 Invariantes)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Base de Conocimiento (Centroides del Pasado)', fontsize=12, fontweight='bold')
    plt.ylabel('Concept Drift (Centroides del Futuro)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    path_g9 = os.path.join(SAVE_DIR, 'G9_Matriz_Similitud_Coseno_Paridad.png')
    plt.savefig(path_g9, dpi=300)
    plt.close()

    # --- 6. GRÁFICA 10: KDE INTRA-CLASE VS INTER-CLASE ---
    print("📊 Generando Gráfica 10: Curvas KDE de Separabilidad (Intra vs Inter)...")
    
    intra_sims = []
    inter_sims = []

    for i, s in enumerate(top_sites):
        # Tomamos todas las muestras capturadas en el Futuro para el sitio 's'
        samples_drift = X_drift[df_drift['site_label'] == s]
        
        # Si no hay muestras futuras, saltamos
        if len(samples_drift) == 0:
            continue
            
        # Similitud Intra-Clase: Distancia angular entre las muestras futuras de S y el centroide pasado de S
        intra = cosine_similarity(samples_drift, past_centroids[i].reshape(1, -1)).flatten()
        intra_sims.extend(intra)
        
        # Similitud Inter-Clase: Distancia angular entre las muestras futuras de S y TODOS los demás centroides
        other_centroids = np.delete(past_centroids, i, axis=0)
        inter = cosine_similarity(samples_drift, other_centroids).flatten()
        inter_sims.extend(inter)

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    sns.kdeplot(intra_sims, fill=True, color='#2ecc71', label='Similitud Intra-clase (Mismo Sitio)', alpha=0.6, linewidth=2)
    sns.kdeplot(inter_sims, fill=True, color='#e74c3c', label='Similitud Inter-clase (Distinto Sitio)', alpha=0.6, linewidth=2)
    
    plt.axvline(np.mean(intra_sims), color='darkgreen', linestyle='--', linewidth=2, label=f'Media Intra: {np.mean(intra_sims):.2f}')
    plt.axvline(np.mean(inter_sims), color='darkred', linestyle='--', linewidth=2, label=f'Media Inter: {np.mean(inter_sims):.2f}')

    plt.title('Separabilidad Angular Post-Drift (Justificación del Metric Learning)\n¿Puede la distancia relativa distinguir los sitios en el futuro?', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Similitud Coseno (-1 a 1: Más cerca de 1 es más idéntico)', fontsize=12, fontweight='bold')
    plt.ylabel('Densidad de Muestras', fontsize=12, fontweight='bold')
    plt.legend(loc='upper left', frameon=True, fontsize=11)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(-1, 1)
    
    plt.tight_layout()
    path_g10 = os.path.join(SAVE_DIR, 'G10_KDE_Separabilidad_Metric_Paridad.png')
    plt.savefig(path_g10, dpi=300)
    plt.close()

    print(f"✅ Gráficas 9 y 10 guardadas exitosamente en {SAVE_DIR}")
    print("===========================================================================\n")

except Exception as e:
    import traceback
    print(f"❌ Error crítico durante la ejecución:\n{traceback.format_exc()}")
