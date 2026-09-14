"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: GEN_05_PCA_Traslacion.py
🚀 VERSIÓN: 3.0 (Autopsia Visual de la Traslación Latente con Paridad Estricta)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este script proyecta el hiperespacio de características (ADN Inmutable) a 2 
    Dimensiones (PCA) para evidenciar geométricamente el fenómeno del Concept Drift.
    
    Demuestra que, aunque la identidad estructural de un sitio se mantenga, el ruteo 
    dinámico de Tor traslada sus coordenadas absolutas en el espacio latente. Esto 
    explica visualmente el fracaso de las funciones de pérdida basadas en fronteras 
    estáticas (Cross-Entropy).

    [!] ACTUALIZACIÓN V3.0 (RIGOR DE ABLACIÓN):
    - Filtrado de Paridad Estricta: Proyecta solo los 65 sitios de élite.
    - Proyección pura sobre las 94 Invariantes Reales post KS-Test.
    - Zoom adaptativo para evidenciar la salida del clúster de su "Caja de Entrenamiento".
=========================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import joblib
from src.utils.paths import data_path, artifact_path, result_path

# --- 1. RUTAS Y DICCIONARIOS ---
CLEAN_PAST_CSV = data_path('historical', 'CLEAN_final_features_sites.csv')
CLEAN_DRIFT_CSV = data_path('historical', 'CLEAN_final_features_sites_concept_drift.csv')
FEATURES_LIST_TXT = result_path('macro', 'resultados_analisis', 'new_features_invariantes_seguras.txt')
VECTOR_LE_PATH = artifact_path('ds3_label_encoder_vec_3000.joblib')
SAVE_DIR = result_path('features', 'graficas_tesis')

os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "═"*70)
print("🎨 GENERANDO GRÁFICO 7: PCA DE TRASLACIÓN LATENTE (CON PARIDAD)")
print("═"*70)

try:
    print("\n[1/4] Cargando filtro de paridad y ADN Inmutable...")
    le_vectores = joblib.load(VECTOR_LE_PATH)
    sitios_elite = set(le_vectores.classes_.astype(str))
    
    with open(FEATURES_LIST_TXT, 'r') as f:
        features_inv = [line.strip() for line in f.readlines() if line.strip()]

    print("[2/4] Cargando datasets purificados y aplicando Paridad Estricta...")
    df_past = pd.read_csv(CLEAN_PAST_CSV).dropna(subset=['site_label'])
    df_drift = pd.read_csv(CLEAN_DRIFT_CSV).dropna(subset=['site_label'])

    df_past['site_label'] = df_past['site_label'].astype(str)
    df_drift['site_label'] = df_drift['site_label'].astype(str)

    # [!] FILTRO DE PARIDAD ESTRICTA
    df_past = df_past[df_past['site_label'].isin(sitios_elite)].copy()
    df_drift = df_drift[df_drift['site_label'].isin(sitios_elite)].copy()

    # Prevenir NaNs en las features de interés
    df_past = df_past.dropna(subset=features_inv)
    df_drift = df_drift.dropna(subset=features_inv)

    # --- 3. CLIPPING DE PERCENTILES (SIN DATA LEAKAGE) ---
    print("[3/4] Aplicando Clipping (1% - 99%) basado EXCLUSIVAMENTE en el Pasado...")
    for col in features_inv:
        if col in df_drift.columns:
            lower = np.percentile(df_past[col], 1)
            upper = np.percentile(df_past[col], 99)
            df_past[col] = np.clip(df_past[col], lower, upper)
            df_drift[col] = np.clip(df_drift[col], lower, upper)

    # --- 4. PREPARACIÓN Y PCA ---
    print("[4/4] Estandarizando y Entrenando PCA sobre el Pasado...")
    scaler = StandardScaler()
    
    X_past_raw = df_past[features_inv].values
    X_drift_raw = df_drift[features_inv].values
    
    # El espacio latente se define matemáticamente con las coordenadas del Pasado
    X_past_scaled = scaler.fit_transform(X_past_raw)
    X_drift_scaled = scaler.transform(X_drift_raw)

    pca = PCA(n_components=2, random_state=42)
    X_past_pca = pca.fit_transform(X_past_scaled)
    X_drift_pca = pca.transform(X_drift_scaled)

    # --- 5. SELECCIÓN DEL SITIO A VISUALIZAR ---
    # Tomaremos un sitio de élite que ilustre de forma clara el desplazamiento
    np.random.seed(42) # Semilla para reproducibilidad en la tesis
    target_site = str(np.random.choice(list(sitios_elite)))
    print(f"\n📌 Dibujando traslación latente para el Sitio de Élite ID: {target_site}...")

    # Máscaras booleanas
    mask_past_target = (df_past['site_label'] == target_site).values
    mask_drift_target = (df_drift['site_label'] == target_site).values
    
    # --- 6. GENERACIÓN DEL GRÁFICO 7 ---
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    # 6.1. Fondo Gris (Contexto del Espacio Latente Restante)
    mask_past_others = ~mask_past_target
    plt.scatter(X_past_pca[mask_past_others, 0], X_past_pca[mask_past_others, 1], 
                c='#bdc3c7', alpha=0.3, s=20, label='Resto del Universo (Pasado)')

    # 6.2. El Clúster del Pasado (Azul - Caja de Entrenamiento)
    plt.scatter(X_past_pca[mask_past_target, 0], X_past_pca[mask_past_target, 1], 
                c='#3498db', marker='o', s=100, edgecolors='white', linewidth=1, 
                alpha=0.9, label=f'Sitio {target_site} (Caja de Entrenamiento)')

    # 6.3. El Clúster del Futuro trasladado (Rojo oscuro - Fuera de la Caja)
    if sum(mask_drift_target) > 0:
        plt.scatter(X_drift_pca[mask_drift_target, 0], X_drift_pca[mask_drift_target, 1], 
                    c='#c0392b', marker='X', s=120, edgecolors='black', linewidth=1.2, 
                    alpha=0.9, label=f'Sitio {target_site} (Drift: Fuera de la Caja)')

    var_1 = pca.explained_variance_ratio_[0] * 100
    var_2 = pca.explained_variance_ratio_[1] * 100

    plt.title(f'Traslación del Espacio Latente por Concept Drift (Sitio {target_site})\\nEvidencia de Fracaso de Fronteras Estáticas (Accuracy: 28.05%)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(f'Componente Principal 1 ({var_1:.1f}%)', fontsize=12, fontweight='bold')
    plt.ylabel(f'Componente Principal 2 ({var_2:.1f}%)', fontsize=12, fontweight='bold')
    
    plt.legend(loc='best', framealpha=0.9, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # [!] ZOOM ADAPTATIVO: Enfocar la cámara estrictamente en el sitio objetivo
    all_target_x = np.concatenate([X_past_pca[mask_past_target, 0], X_drift_pca[mask_drift_target, 0]])
    all_target_y = np.concatenate([X_past_pca[mask_past_target, 1], X_drift_pca[mask_drift_target, 1]])
    
    margin_x = (all_target_x.max() - all_target_x.min()) * 0.4
    margin_y = (all_target_y.max() - all_target_y.min()) * 0.4
    
    plt.xlim(all_target_x.min() - margin_x, all_target_x.max() + margin_x)
    plt.ylim(all_target_y.min() - margin_y, all_target_y.max() + margin_y)
    
    plt.tight_layout()
    
    output_path = os.path.join(SAVE_DIR, f'G8_PCA_Traslacion_Paridad_Sitio_{target_site}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Evidencia visual guardada exitosamente en: {output_path}")
    print("===========================================================================\n")

except Exception as e:
    import traceback
    print(f"❌ Error crítico durante la generación:\n{traceback.format_exc()}")
