import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# --- 1. RUTAS ---
CLEAN_PAST_CSV = '../../output/preprocessed/02_features_robust_.csv'
CLEAN_DRIFT_CSV = '../../output/preprocessed/02_features_robust_drift.csv'
SAVE_DIR = './graficas_tesis'

os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "="*60)
print("🎨 GENERANDO GRÁFICO 6: KDE DE ESTABILIDAD (MEJOR Y PEOR CASO)")
print("="*60)

try:
    print(f"⏳ Leyendo datasets reales (79 variables)...")
    df_past = pd.read_csv(CLEAN_PAST_CSV)
    df_drift = pd.read_csv(CLEAN_DRIFT_CSV)

    # Identificar columnas numéricas de features (excluyendo metadatos)
    cols_features = df_past.select_dtypes(include=[np.number]).columns.tolist()
    for meta in ['site_label', 'pcap_uid', 'hour_bin', 'date_id']:
        if meta in cols_features: cols_features.remove(meta)

    # --- 2. PASO CRÍTICO: CLIPPING DE PERCENTILES (LIMPIEZA DE OUTLIERS) ---
    print("🧹 Aplicando Clipping de Anomalías (Percentil 1 y 99) basado en el Pasado...")
    for col in cols_features:
        if col in df_drift.columns:
            # Calculamos límites basados EXCLUSIVAMENTE en el Pasado (Entrenamiento)
            lower_limit = np.percentile(df_past[col].dropna(), 1)
            upper_limit = np.percentile(df_past[col].dropna(), 99)
            
            # Aplicamos el recorte a AMBOS datasets
            df_past[col] = np.clip(df_past[col], lower_limit, upper_limit)
            df_drift[col] = np.clip(df_drift[col], lower_limit, upper_limit)

    # --- 3. SELECCIÓN DE LAS VARIABLES (PEOR Y MEJOR CASO) ---
    scaler_measure = StandardScaler()
    X_past_scaled = pd.DataFrame(scaler_measure.fit_transform(df_past[cols_features]), columns=cols_features)
    X_drift_scaled = pd.DataFrame(scaler_measure.transform(df_drift[cols_features]), columns=cols_features)

    max_shift = -1
    min_shift = float('inf')
    worst_feat = ""
    best_feat = ""

    # Buscamos los extremos del desplazamiento en el espacio Z
    for col in cols_features:
        shift = abs(X_past_scaled[col].mean() - X_drift_scaled[col].mean())
        
        # Peor caso (Mayor Mutación)
        if shift > max_shift:
            max_shift = shift
            worst_feat = col
            
        # Mejor caso (Mayor Estabilidad)
        if shift < min_shift:
            min_shift = shift
            best_feat = col

    print(f"\n❌ Peor Caso (Mutación Extrema): {worst_feat} | Z-Shift: {max_shift:.2f}")
    print(f"✅ Mejor Caso (ADN Inmutable): {best_feat} | Z-Shift: {min_shift:.2f}\n")

    # --- 4. FUNCIÓN PARA GENERAR GRÁFICAS KDE ---
    def generate_kde_plot(feature_name, shift_value, title_desc, filename):
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        past_vals_plot = X_past_scaled[feature_name]
        drift_vals_plot = X_drift_scaled[feature_name]

        sns.kdeplot(past_vals_plot, fill=True, color='#3498db', label='Pasado (Entrenamiento)', alpha=0.5, linewidth=2)
        sns.kdeplot(drift_vals_plot, fill=True, color='#e74c3c', label='Futuro (Concept Drift)', alpha=0.5, linewidth=2)

        plt.axvline(past_vals_plot.mean(), color='#2980b9', linestyle='--', alpha=0.8, linewidth=1.5)
        plt.axvline(drift_vals_plot.mean(), color='#c0392b', linestyle='--', alpha=0.8, linewidth=1.5)

        plt.title(f'{title_desc} (Z-Shift: {shift_value:.2f})\nCaracterística: {feature_name}', 
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Valor Escalado (Z-Score)', fontsize=12, fontweight='bold')
        plt.ylabel('Densidad de Población', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', frameon=True, fontsize=10)
        
        plt.xlim(-4, 4) 
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        output_path = os.path.join(SAVE_DIR, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"🚀 Gráfico guardado en: {output_path}")

    # Generamos ambas gráficas
    generate_kde_plot(worst_feat, max_shift, "Mutación Distributiva por Concept Drift [PEOR CASO]", "G6A_KDE_Peor_Caso.png")
    generate_kde_plot(best_feat, min_shift, "Estabilidad Conservada / ADN Inmutable [MEJOR CASO]", "G6B_KDE_Mejor_Caso.png")

except Exception as e:
    print(f"❌ Error crítico durante la generación: {e}")