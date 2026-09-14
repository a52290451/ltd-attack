"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: Reduccion_Features.py (ACTUALIZADO A DATOS CLEAN + PARIDAD)
🚀 VERSIÓN: 3.0 (Integración de Filtro Maestro de 65 Clases para Estudio de Ablación)

-----------------------------------------------------------------------------------------
DESCRIPCIÓN ACADÉMICA:
    Analizador de "Covariate Shift" mediante el Test de Kolmogorov-Smirnov (KS-Test) con
    Autopsia Visual. Compara empíricamente la distribución estadística de cada Feature 
    en el Pasado (Entrenamiento) contra su distribución en el Futuro (Concept Drift).

    ACTUALIZACIÓN CRÍTICA (PARIDAD ESTRICTA): 
    Se ejecuta sobre los datos purificados (Fase 0), pero ahora ingesta el LabelEncoder 
    de la rama Micro (Vectores) para aislar EXCLUSIVAMENTE las 65 clases de élite que 
    sobrevivieron a los filtros de integridad secuencial. Esto garantiza una comparación
    matemáticamente justa (1:1) en el Estudio de Ablación.

OBJETIVO:
    Identificar el verdadero "ADN inmutable" (KS <= 0.15) libre del ruido estadístico 
    de secuencias rotas, y generar evidencia visual del proceso.
=========================================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os
import sys
import datetime
import joblib
import matplotlib
matplotlib.use('Agg') # Modo headless para evitar errores en servidores sin interfaz gráfica
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.paths import data_path, artifact_path, result_path

# --- 1. RUTAS DE LOS DATOS Y DICCIONARIOS ---
HISTORIC_CSV = data_path('historical', 'CLEAN_final_features_sites.csv')
FUTURE_CSV = data_path('historical', 'CLEAN_final_features_sites_concept_drift.csv')
SAVE_DIR = result_path('macro', 'resultados_analisis')

# [!] RUTA CRÍTICA: Ajusta esta ruta si tu archivo .joblib de vectores está en otra carpeta
VECTOR_LE_PATH = artifact_path('ds3_label_encoder_vec_3000.joblib')

os.makedirs(SAVE_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

print("\n" + "═"*70)
print("🔍 DESCUBRIMIENTO DEL ADN INMUTABLE (KS-TEST CON PARIDAD 65 CLASES)")
print("═"*70)

# --- 2. CARGA DEL FILTRO MAESTRO DE VECTORES (PARIDAD DE ABLACIÓN) ---
print("\n[1/5] Extrayendo filtro de paridad desde la rama de Vectores...")
try:
    le_vectores = joblib.load(VECTOR_LE_PATH)
    sitios_elite = set(le_vectores.classes_.astype(str))
    print(f"   ✅ Filtro cargado exitosamente: {len(sitios_elite)} sitios de élite detectados.")
except Exception as e:
    print(f"   ❌ Error al cargar el LabelEncoder de Vectores.")
    print(f"   Por favor, verifica que la ruta sea correcta: {VECTOR_LE_PATH}")
    print(f"   Detalle del error: {e}")
    sys.exit()

# --- 3. CARGA DE DATOS Y APLICACIÓN DE PARIDAD ESTRICTA ---
print("\n[2/5] Cargando datasets purificados y aplicando filtro de paridad...")
df_past = pd.read_csv(HISTORIC_CSV)
df_fut = pd.read_csv(FUTURE_CSV)

# Convertir a string para comparación segura
df_past['site_label'] = df_past['site_label'].astype(str)
df_fut['site_label'] = df_fut['site_label'].astype(str)

# [!] Aplicar el filtro de las 65 clases de élite
df_past = df_past[df_past['site_label'].isin(sitios_elite)].copy()
df_fut = df_fut[df_fut['site_label'].isin(sitios_elite)].copy()

# Garantizar que evaluamos exactamente las mismas clases (Intersección final defensiva)
common_sites = np.intersect1d(df_past['site_label'].unique(), df_fut['site_label'].unique())
df_past = df_past[df_past['site_label'].isin(common_sites)].copy()
df_fut = df_fut[df_fut['site_label'].isin(common_sites)].copy()

print(f"   ✅ Clases sincronizadas bajo paridad: {len(common_sites)} sitios web evaluados.")
print(f"   -> Muestras Pasado (Élite): {len(df_past)}")
print(f"   -> Muestras Futuro (Élite): {len(df_fut)}")

# --- 4. EXTRACCIÓN INTELIGENTE DE FEATURES (IGNORANDO METADATOS) ---
# Lista de columnas que sabemos que NO son features predictivas estadísticas
metadata_cols = ['pcap_uid', 'site_label', 'pcap_name', 'category', 'site', 
                 'vector_len', 'matches', 'n_errors', 'direction_vector', 
                 'time_vector', 'size_vector', 'hour_bin', 'date_id']

# Tomamos solo columnas numéricas y descartamos los metadatos
numeric_cols = df_past.select_dtypes(include=[np.number]).columns.tolist()
feature_columns = [col for col in numeric_cols if col not in metadata_cols]

# Comprobamos que las mismas columnas existan en el futuro
feature_columns = [col for col in feature_columns if col in df_fut.columns]

print(f"\n[3/5] Ejecutando Kolmogorov-Smirnov Test sobre {len(feature_columns)} features...")
results = []

for feature in feature_columns:
    # Extraemos los valores ignorando valores nulos o infinitos
    past_vals = df_past[feature].replace([np.inf, -np.inf], np.nan).dropna().values
    fut_vals = df_fut[feature].replace([np.inf, -np.inf], np.nan).dropna().values
    
    # [!] FILTRO VITAL DE VARIANZA CERO: 
    # Descartamos variables "muertas" (constantes) porque un modelo no puede aprender de ellas,
    # y evitamos que el KS-Test devuelva un falso 0.00 perfecto.
    if len(past_vals) == 0 or len(fut_vals) == 0 or np.std(past_vals) == 0 or np.std(fut_vals) == 0:
        continue
    
    # El estadístico KS mide la distancia máxima entre las dos distribuciones. 
    # Cerca de 0 = Distribución idéntica (Invariante). Cerca de 1 = Distribuciones totalmente distintas (Volátil).
    stat, p_value = ks_2samp(past_vals, fut_vals)
    
    results.append({
        'Feature': feature,
        'KS_Distance': stat,  
        'P_Value': p_value
    })

# --- 5. CLASIFICACIÓN Y GUARDADO DE INVARIANTES ---
print("\n[4/5] Clasificando Supervivientes...")
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='KS_Distance', ascending=True)

# Umbral científico: Si la distancia es menor o igual a 0.15, es una Invariante Fuerte
umbral_estabilidad = 0.15
df_invariantes = df_results[df_results['KS_Distance'] <= umbral_estabilidad]

print("\n" + "🏆"*20)
print(f"RESUMEN DE SUPERVIVENCIA DE FEATURES (PARIDAD 65 CLASES)")
print("🏆"*20)
print(f"Total Features Válidas Evaluadas: {len(df_results)}")
print(f"Features Invariantes Reales (KS <= {umbral_estabilidad}): {len(df_invariantes)}")
print(f"Features Volátiles (Ruido/Basura temporal): {len(df_results) - len(df_invariantes)}")

print("\n💎 TOP 5 FEATURES MÁS ESTABLES (El verdadero ADN del modelo):")
print(df_results.head(5)[['Feature', 'KS_Distance']].to_string(index=False))

print("\n☠️ TOP 5 FEATURES MÁS VOLÁTILES (Las que destruyen la generalización):")
print(df_results.tail(5)[['Feature', 'KS_Distance']].to_string(index=False))

# --- 6. GENERACIÓN DE EVIDENCIA VISUAL (AUTOPSIA GRÁFICA) ---
print("\n[5/5] Generando Evidencia Visual (Gráficas)...")

# Identificar las mejores y peores para la gráfica KDE
top_stable = df_results.iloc[0]['Feature']
top_volatile = df_results.iloc[-1]['Feature']

# GRÁFICA A: Histograma de Distribución KS global
plt.figure(figsize=(10, 6))
sns.histplot(data=df_results, x='KS_Distance', bins=40, color='gray', edgecolor='black')
plt.axvline(umbral_estabilidad, color='green', linestyle='--', linewidth=2, label=f'Umbral de Invarianza ({umbral_estabilidad})')

# Colorear barras: Verde (Estable) vs Rojo (Volátil)
for p in plt.gca().patches:
    if p.get_x() <= umbral_estabilidad:
        p.set_color('#2ecc71')
    else:
        p.set_color('#e74c3c')

plt.title('Distribución Global del Concept Drift en Features (65 Clases de Élite)', fontsize=14, fontweight='bold')
plt.xlabel('Distancia de Kolmogorov-Smirnov (0 = Invariante, 1 = Destruida)', fontsize=12)
plt.ylabel('Cantidad de Características', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'RF_G1_Distribucion_KS_Paridad.jpg'), dpi=300)
plt.close()

# GRÁFICA B: Contraste KDE (La Más Estable vs La Más Volátil)
plt.figure(figsize=(14, 6))

# Subplot 1: La variable Invariante
plt.subplot(1, 2, 1)
sns.kdeplot(df_past[top_stable].dropna(), fill=True, color='#3498db', label='Pasado (Entrenamiento)')
sns.kdeplot(df_fut[top_stable].dropna(), fill=True, color='#e74c3c', label='Futuro (Concept Drift)')
plt.title(f'Feature INVARIANTE: {top_stable}\n(Distancia KS = {df_results.iloc[0]["KS_Distance"]:.3f})', fontweight='bold', fontsize=12)
plt.legend(loc='upper right')

# Subplot 2: La variable Volátil
plt.subplot(1, 2, 2)
sns.kdeplot(df_past[top_volatile].dropna(), fill=True, color='#3498db', label='Pasado (Entrenamiento)')
sns.kdeplot(df_fut[top_volatile].dropna(), fill=True, color='#e74c3c', label='Futuro (Concept Drift)')
plt.title(f'Feature VOLÁTIL: {top_volatile}\n(Distancia KS = {df_results.iloc[-1]["KS_Distance"]:.3f})', fontweight='bold', fontsize=12)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'RF_G2_Contraste_KDE_Paridad.jpg'), dpi=300)
plt.close()

# --- 7. GUARDADO FINAL DE RESULTADOS ---
out_csv = os.path.join(SAVE_DIR, f"ks_test_results_CLEAN_PARITY_{timestamp}.csv")
df_results.to_csv(out_csv, index=False)

# [CRÍTICO] Sobrescribimos el archivo txt del que dependerán los demás scripts (GEN_02, EXP1...)
out_txt = os.path.join(SAVE_DIR, "new_features_invariantes_seguras.txt")
with open(out_txt, "w") as f:
    for feature in df_invariantes['Feature']:
        f.write(f"{feature}\n")

print(f"\n📄 Resultados completos guardados en: {out_csv}")
print(f"🔒 LISTA OFICIAL ACTUALIZADA GUARDADA EN: {out_txt}")
print(f"📊 Gráficas de paridad guardadas exitosamente en el directorio de resultados.")
print("===========================================================================\n")
