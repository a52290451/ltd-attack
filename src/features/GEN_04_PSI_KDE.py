"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: GEN_04_PSI_KDE.py
🚀 VERSIÓN: 3.1 (KDE de Estabilidad con Paridad de 65 Clases y Protección de Varianza)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este script cuantifica la mutación de las distribuciones de probabilidad 
    (Concept Drift) midiendo el desplazamiento espacial (Z-Shift) entre las medias del 
    Pasado y el Futuro para las 94 Invariantes.

    Para fines de exposición académica, se grafican exclusivamente los dos extremos 
    (el Mejor y el Peor Caso), representando los límites superior e inferior de la 
    degradación estructural del modelo.
=========================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import joblib
import sys

# --- 1. RUTAS Y DICCIONARIOS ---
CLEAN_PAST_CSV = '../../output/CLEAN_final_features_sites.csv'
CLEAN_DRIFT_CSV = '../../output/CLEAN_final_features_sites_concept_drift.csv'
FEATURES_LIST_TXT = '../vectores_features/resultados_analisis/new_features_invariantes_seguras.txt'
VECTOR_LE_PATH = '../vectores/resultados/ds3_label_encoder_vec_3000.joblib'
SAVE_DIR = './graficas_tesis'

os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "═"*70)
print("🎨 GENERANDO GRÁFICAS 8A/8B: KDE DE ESTABILIDAD (MEJOR Y PEOR CASO)")
print("═"*70)

try:
    print("\n[1/4] Extrayendo filtro de paridad y ADN Inmutable...")
    le_vectores = joblib.load(VECTOR_LE_PATH)
    sitios_elite = set(le_vectores.classes_.astype(str))

    with open(FEATURES_LIST_TXT, 'r') as f:
        features_inv = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"   ✅ Se evaluará la mutación espacial sobre {len(features_inv)} variables Invariantes.")

    print("\n[2/4] Cargando datasets purificados y aplicando Paridad Estricta...")
    df_past = pd.read_csv(CLEAN_PAST_CSV).dropna(subset=['site_label'])
    df_drift = pd.read_csv(CLEAN_DRIFT_CSV).dropna(subset=['site_label'])

    df_past['site_label'] = df_past['site_label'].astype(str)
    df_drift['site_label'] = df_drift['site_label'].astype(str)

    # Filtro Estricto de Paridad
    df_past = df_past[df_past['site_label'].isin(sitios_elite)].copy()
    df_drift = df_drift[df_drift['site_label'].isin(sitios_elite)].copy()

    # Prevenir NaNs en las features de interés
    df_past = df_past.dropna(subset=features_inv)
    df_drift = df_drift.dropna(subset=features_inv)

    # --- 3. PASO CRÍTICO: CLIPPING DE PERCENTILES (LIMPIEZA SIN DATA LEAKAGE) ---
    print("\n[3/4] Aplicando Clipping (1% - 99%) basado EXCLUSIVAMENTE en el Pasado...")
    for col in features_inv:
        if col in df_drift.columns:
            # Límites basados en el conocimiento previo (Pasado) para no contaminar
            lower_limit = np.percentile(df_past[col], 1)
            upper_limit = np.percentile(df_past[col], 99)
            
            # Aplicar recorte a ambos para aislar el núcleo distributivo
            df_past[col] = np.clip(df_past[col], lower_limit, upper_limit)
            df_drift[col] = np.clip(df_drift[col], lower_limit, upper_limit)

    # --- 4. CÁLCULO DEL Z-SHIFT (DESPLAZAMIENTO LATENTE) ---
    print("\n[4/4] Escalando Z-Score y buscando los extremos de mutación...")
    scaler_measure = StandardScaler()
    
    # El scaler se ajusta solo en el pasado
    X_past_scaled = pd.DataFrame(scaler_measure.fit_transform(df_past[features_inv]), columns=features_inv)
    X_drift_scaled = pd.DataFrame(scaler_measure.transform(df_drift[features_inv]), columns=features_inv)

    max_shift = -1
    min_shift = float('inf')
    worst_feat = ""
    best_feat = ""

    # Evaluamos el desplazamiento absoluto de la media en el espacio Z para las 94 variables
    for col in features_inv:
        
        # [!] PROTECCIÓN: Si el clipping colapsó la variable a una constante (Varianza = 0),
        # la ignoramos, ya que es matemáticamente imposible trazar su curva KDE.
        if X_past_scaled[col].std() == 0 or X_drift_scaled[col].std() == 0:
            continue
            
        shift = abs(X_past_scaled[col].mean() - X_drift_scaled[col].mean())
        
        # Guardamos el Peor caso (Mayor Mutación)
        if shift > max_shift:
            max_shift = shift
            worst_feat = col
            
        # Guardamos el Mejor caso (Mayor Estabilidad)
        if shift < min_shift:
            min_shift = shift
            best_feat = col

    print(f"\n" + "☠️"*15)
    print(f"   PEOR CASO (Mutación Extrema): {worst_feat}")
    print(f"   Desplazamiento Z-Shift:       {max_shift:.4f}")
    print("☠️"*15)
    
    print(f"\n" + "💎"*15)
    print(f"   MEJOR CASO (ADN Inmutable):   {best_feat}")
    print(f"   Desplazamiento Z-Shift:       {min_shift:.4f}")
    print("💎"*15 + "\n")

    # --- 5. GENERACIÓN DE EVIDENCIA VISUAL ---
    def generate_kde_plot(feature_name, shift_value, title_desc, filename):
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        past_vals_plot = X_past_scaled[feature_name]
        drift_vals_plot = X_drift_scaled[feature_name]

        sns.kdeplot(past_vals_plot, fill=True, color='#3498db', label='Pasado (Entrenamiento)', alpha=0.5, linewidth=2)
        sns.kdeplot(drift_vals_plot, fill=True, color='#e74c3c', label='Futuro (Concept Drift)', alpha=0.5, linewidth=2)

        # Líneas de las medias
        plt.axvline(past_vals_plot.mean(), color='#2980b9', linestyle='--', alpha=0.8, linewidth=1.5)
        plt.axvline(drift_vals_plot.mean(), color='#c0392b', linestyle='--', alpha=0.8, linewidth=1.5)

        plt.title(f'{title_desc} (Z-Shift: {shift_value:.3f})\\nCaracterística Analizada: {feature_name}', 
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Valor Escalado (Z-Score Espacial)', fontsize=12, fontweight='bold')
        plt.ylabel('Densidad de Población', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', frameon=True, fontsize=10)
        
        # Fijar límites para una comparativa estandarizada visualmente
        plt.xlim(-4, 4) 
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        output_path = os.path.join(SAVE_DIR, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✅ Evidencia guardada en: {output_path}")

    # Generamos ambas gráficas
    generate_kde_plot(worst_feat, max_shift, "Mutación Distributiva [PEOR CASO]", "G7A_KDE_Peor_Caso_Paridad.png")
    generate_kde_plot(best_feat, min_shift, "Estabilidad Conservada [MEJOR CASO]", "G7B_KDE_Mejor_Caso_Paridad.png")
    print("===========================================================================\n")

except Exception as e:
    import traceback
    print(f"❌ Error crítico durante la generación:\n{traceback.format_exc()}")