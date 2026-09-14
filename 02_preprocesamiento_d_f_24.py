"""
=====================================================================================
ANÁLISIS MASIVO DE FIRMAS HORARIAS EN TRÁFICO DE RED WEB
=====================================================================================

Descripción general
-------------------
Este script implementa un pipeline integral de análisis masivo de características de tráfico
de red web, orientado a la identificación de **firmas temporales robustas**. Está diseñado
para procesar datos de múltiples sitios web y múltiples días, incluyendo series horarias
completas (24 mediciones por día).

El objetivo principal es generar **features discriminativas y estables** que puedan
ser utilizadas para análisis de comportamiento del tráfico, fingerprinting de sitios web,
y como entrada a modelos de aprendizaje automático basados en embeddings de series temporales
(Hourly Encoder y MetaFeatures).

Fases principales
-----------------
0. **Preprocesamiento y completado de series horarias**  
   - Completa cualquier dato faltante, asegurando que cada día tenga 24 mediciones.  
   - Genera un CSV con todas las features completas.  
   - Prepara los datos en formato listo para el Hourly Encoder (tensor D × F × 24).  

1. **Discretización temporal y análisis intra-día**  
   - Discretiza los datos en intervalos de 30 minutos (`hour bins`).  
   - Evalúa diferencias significativas entre horas mediante la prueba de **Kruskal-Wallis**.  
   - Verifica la estabilidad de la varianza entre intervalos usando **Levene** (centrada en la mediana).  

2. **Verificación de recurrencia temporal**  
   - Determina si el patrón significativo se repite en al menos un número mínimo de días (`MIN_RECURRENCE_DAYS`).  
   - Solo se conservan las features que cumplen los tres criterios: **significancia**, **estabilidad** y **recurrencia**.  
   - Genera un CSV de features robustas listo para pasar a la fase de codificación con Hourly Encoder.  

3. **Visualización y documentación de firmas horarias**  
   - Para cada feature candidata final, genera gráficos multilínea mostrando su evolución a lo largo del día y la recurrencia en distintas fechas.  
   - Facilita la identificación visual de patrones estables y discriminativos.  

Parámetros clave
----------------
- **ALPHA**: Nivel de significancia estadística (0.01)  
- **Resolución temporal**: 30 minutos  
- **MIN_RECURRENCE_DAYS**: Número mínimo de días para considerar un patrón recurrente  
- **Pruebas estadísticas**: Kruskal-Wallis y Levene  

Entradas
--------
- Archivo CSV con las features extraídas de capturas PCAP, incluyendo:
    - Identificador de sitio web (`site_label`)
    - Fecha de captura (`date_id`)
    - Serie horaria de cada feature (intervalos de 30 min)
    - Metadatos (hora, minuto, archivo PCAP, etc.)

Salidas
-------
- **CSV con todas las features completas** (24 valores por día)
- **CSV con features robustas filtradas** tras aplicar los tres criterios
- **Gráficos de evidencia multilínea** para cada feature robusta

Aplicabilidad
-------------
- Web Fingerprinting y análisis de tráfico cifrado
- Identificación de sitios web basada en patrones de tráfico
- Estudios de estabilidad, recurrencia y comportamiento temporal de métricas de red
- Generación de embeddings (Hourly Encoder) y MetaFeatures para aprendizaje automático

Diseño
------
El script prioriza:
- **Robustez estadística**
- **Reproducibilidad experimental**
- **Escalabilidad** a grandes volúmenes de datos y características
- Preparación directa de datos para modelos basados en series temporales (transformers, CNNs, etc.)
=====================================================================================
"""

import pandas as pd 
import numpy as np 
from scipy import stats 
import re 
import os 
import matplotlib # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# =============================================================================
# =============================================================================  
# CONFIGURACIÓN E INICIALIZACIÓN 
# =============================================================================
# =============================================================================  

try: 
    matplotlib.use('Agg')  
except: 
    pass  

FILE_PATH_FEATURES = 'output/final_features_sites.csv'  
OUTPUT_DIR = 'output/graficos_finales_masivos/' # Directorio de gráficos
OUTPUT_DIR_PRE = 'output/preprocessed/'         # Directorio CSVs preprocesados
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_PRE, exist_ok=True)

ALPHA = 0.01 # Umbral de significancia estadística 
MIN_RECURRENCE_DAYS = 15 # Mínimo de días para considerar el patrón recurrente 
HOURS_24 = [f'{h:02d}:00' for h in range(24)]
TIME_BINS_ORDER = HOURS_24

# Columnas a excluir (Metadatos) 
METADATA_COLS = [ 
    'pcap_uid', 'site_label', 'pcap_name', 'category', 'site', 'time_str', 'hour', 'minute', 'date_id',  
    'hour_bin', 'deviation', 'hour_bin_norm','sample_uid'
] 

# =============================================================================
# ============================================================================= 
# 1. FUNCIONES AUXILIARES 
# =============================================================================
# ============================================================================= 

def extract_time_from_pcap(filename):
    """
    ===========================================================================
    Extrae la hora y minuto de un archivo PCAP a partir de su nombre.
    El nombre del archivo debe contener un patrón del tipo "-HHMMxx.pcap", 
    donde HH es la hora y MM los minutos.

    Args:
        filename (str): Nombre del archivo PCAP.
    Returns:
        tuple:
            - str: Hora formateada como "HH:MM".
            - int: Hora como entero (0-23). Si no se encuentra, -1.
            - int: Minuto como entero (0-59). Si no se encuentra, -1.
    Ejemplo:
        >>> extract_time_from_pcap("site1_20240612-083012.pcap")
        ('08:30', 8, 30)
    ===========================================================================
    """
    match = re.search(r'-(\d{2})(\d{2})\d{2}\.pcap', filename) 
    if match: 
        hour = int(match.group(1)) 
        minute = int(match.group(2)) 
        return f"{hour:02d}:{minute:02d}", hour, minute 
    return "00:00", -1, -1  

def extract_date_from_pcap(filename): 
    """
    ===========================================================================
    Extrae la fecha de un archivo PCAP a partir de su nombre.
    El nombre del archivo debe contener un patrón del tipo "_YYYYMMDD-", 
    donde YYYY es el año, MM el mes y DD el día.

    Args:
        filename (str): Nombre del archivo PCAP.
    Returns:
        str: Fecha en formato "YYYYMMDD". Si no se encuentra, devuelve
             "UNKNOWN_DATE".
    
    Ejemplo:
        >>> extract_date_from_pcap("site1_20240612-083012.pcap")
        '2024-06-12'
    ===========================================================================
    """
    match = re.search(r'_(\d{8})-', filename) 
    if match: 
        return match.group(1) 
    return 'UNKNOWN_DATE' 

def bin_time(hour, minute):
    """Normaliza a 24 bins (60 min) ignorando los minutos."""
    return f"{hour:02d}:00"

def create_output_directory(): 
    """
    ===========================================================================
    Crea el directorio de salida si no existe.

    Usa la variable global OUTPUT_DIR como ruta destino.
    Imprime un mensaje indicando que el directorio ha sido creado.
    ===========================================================================
    """
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR) 
        print(f"Directorio creado: {OUTPUT_DIR}") 

# =============================================================================
# =============================================================================  
# 2. FUNCIÓN DE VALIDACIÓN DE FRECUENCIA 
# =============================================================================
# ============================================================================= 
def check_recurrence(df_site, feature, p_value_kw, min_days): 
    """ 
    Verifica si el patrón significativo se debe a una recurrencia en al menos 'min_days' diferentes, 
    contando los días que superan un umbral de desviación significativo en la hora pico. 
    """ 
    if p_value_kw >= ALPHA: 
        return False, None 

    df_local = df_site[['hour_bin', 'date_id', feature]].copy() 
     
    global_median = df_local[feature].median() 
    df_local['deviation'] = abs(df_local[feature] - global_median) 
     
    if df_local['deviation'].sum() == 0: 
        return False, None 
         
    mean_deviation_by_hour = df_local.groupby('hour_bin')['deviation'].mean() 

    if mean_deviation_by_hour.empty: 
        return False, None 

    peak_hour_bin = mean_deviation_by_hour.idxmax() 
     
    # Usamos el cuantil 75% para definir qué es una "desviación extrema" 
    deviation_threshold = df_local['deviation'].quantile(0.75)  

    if deviation_threshold == 0 and df_local['deviation'].max() > 0: 
        deviation_threshold = df_local['deviation'].max() * 0.1 
         
    df_peak = df_local[df_local['hour_bin'] == peak_hour_bin] 
    recurrent_days_count = df_peak[df_peak['deviation'] > deviation_threshold]['date_id'].nunique() 
    is_recurrent = recurrent_days_count >= min_days 
     
    return is_recurrent, peak_hour_bin 

# =============================================================================
# ============================================================================= 
# 3. PROCESAMIENTO INICIAL DE DATOS 
# =============================================================================
# =============================================================================  
print("--- Paso 1: Cargando y Preparando Datos para Análisis Masivo ---") 

# --- BASE 1: FEATURES --- 
try: 
    df_features = pd.read_csv(FILE_PATH_FEATURES) 
except FileNotFoundError: 
    print(f"Error: Archivo no encontrado en {FILE_PATH_FEATURES}") 
    exit() 

# --- CORRECCIÓN CRÍTICA: Extracción de tiempo desde pcap_name ---
if 'hour_bin' not in df_features.columns:
    print("Extrayendo variables temporales desde pcap_name...")
    
    # Aplicamos la función extract_time_from_pcap a la columna pcap_name
    # Tu función actual busca '-HHMMSS', ajustamos la lógica si es necesario
    def apply_extraction(row):
        time_str, hour, minute = extract_time_from_pcap(row['pcap_name'])
        return pd.Series([time_str, hour, minute])

    # Creamos las columnas que faltan
    df_features[['time_str', 'hour', 'minute']] = df_features.apply(apply_extraction, axis=1)
    
    # Ahora generamos el hour_bin (ej. "12:00")
    df_features['hour_bin'] = df_features['hour'].apply(lambda h: f"{h:02d}:00" if h != -1 else "00:00")

# Aseguramos que date_id exista (extrayéndolo de pcap_name si es UNKNOWN)
if 'date_id' not in df_features.columns or (df_features['date_id'] == 'UNKNOWN_DATE').any():
    df_features['date_id'] = df_features['pcap_name'].apply(extract_date_from_pcap)

dias_disponibles = sorted(df_features['date_id'].unique())
if len(dias_disponibles) > 2:
    dia_inicio = dias_disponibles[0]
    dia_final = dias_disponibles[-1]
    print(f"Excluyendo días de borde (posiblemente incompletos): {dia_inicio} y {dia_final}")
    df_features = df_features[(df_features['date_id'] != dia_inicio) & (df_features['date_id'] != dia_final)]
else:
    print("⚠️ Advertencia: No hay suficientes días para excluir inicio y fin.")

print("DATA FRAME 1")
print(df_features[['site_label', 'date_id', 'hour_bin']].head(50))
# Ahora el Categorical tendrá exactamente 24 categorías
df_features['hour_bin'] = pd.Categorical( 
    df_features['hour_bin'], categories=HOURS_24, ordered=True 
) 

SITE_IDS = sorted(df_features['site_label'].unique()) 

# DEFINICIÓN DE FEATURES A PROBAR (TODAS LAS COLUMNAS NO METADATOS) 
FEATURES_TO_TEST = [col for col in df_features.columns if col not in METADATA_COLS] 

print(f"Total de sitios: {len(SITE_IDS)}. Total de Features a probar: {len(FEATURES_TO_TEST)}") 

# =============================================================================
# ============================================================================= 
# 4. COMPLETADO DE DATOS FALTANTES
# =============================================================================
# ============================================================================= 

print("--- Paso 2: Normalizando y Generando Matriz Completa (24 Bins) ---")


# 1. Aseguramos que el sitio sea string para evitar errores de tipo en el UID
df_features['site_label'] = df_features['site_label'].astype(str)

# 2. Promediar valores duplicados dentro de la misma hora
df_features = df_features.groupby(['site_label', 'date_id', 'hour_bin'])[FEATURES_TO_TEST].mean().reset_index()

# 3. CREAR EL ESQUELETO: Esto asegura que CADA SITIO y CADA DÍA tenga las 24 horas representadas
all_sites = df_features['site_label'].unique()
all_dates = df_features['date_id'].unique()

skeleton = pd.MultiIndex.from_product(
    [all_sites, all_dates, HOURS_24],
    names=['site_label', 'date_id', 'hour_bin']
).to_frame(index=False)

# 4. Unimos los datos reales con el esqueleto
df_complete = skeleton.merge(df_features, on=['site_label', 'date_id', 'hour_bin'], how='left')

# 5. Generamos el Identificador Único por Muestra (Sitio + Fecha)
df_complete['sample_uid'] = df_complete['site_label'] + "_" + df_complete['date_id'].astype(str)

# 6. RELLENO DE HUECOS (Interpolación + Relleno de seguridad)
# 6. RELLENO DE HUECOS (Interpolación Lineal + Ceros en los extremos)
print("Aplicando relleno: Configurando todas las horas sin tráfico a 0.0...")
df_complete[FEATURES_TO_TEST] = df_complete[FEATURES_TO_TEST].fillna(0.0)

# Ordenamos estrictamente
df_features = df_complete.sort_values(['sample_uid', 'hour_bin'])

print("\n--- Evaluando densidad de datos por sitio ---")

sites_to_keep = []
total_records_per_site = len(all_dates) * 24  # Total esperado de bins (puntos)

print("\n--- TOTAL --- ",total_records_per_site)

for site in SITE_IDS:
    # Filtramos los datos de este sitio
    df_site_temp = df_features[df_features['site_label'] == str(site)]
    
    # Contamos cuántos registros son diferentes a 0 en una feature de referencia (ej. vector_len)
    # Si prefieres que sea 'alguna' feature, puedes usar .any(axis=1)
    puntos_activos = (df_site_temp['vector_len'] > 0).sum()
    
    porcentaje_actividad = (puntos_activos / total_records_per_site) * 100
    
    # Tu condición: máximo 40% faltantes = mínimo 60% actividad
    # (Ajusta a 50% si prefieres la segunda cifra que mencionaste)
    if porcentaje_actividad >= 60.0:
        sites_to_keep.append(site)
        # print(f"✅ Sitio {site}: {porcentaje_actividad:.1f}% de actividad. (CONSERVADO)")
    else:
        print(f"❌ Sitio {site}: Solo {porcentaje_actividad:.1f}% de actividad. (ELIMINADO por alta escasez)")

# Actualizamos la lista de sitios para el análisis
print(f"\nSitios originales: {len(SITE_IDS)} | Sitios tras filtro de densidad: {len(sites_to_keep)}")
SITE_IDS = sites_to_keep

# Filtramos el DataFrame principal para liberar memoria y no analizar basura
df_features = df_features[df_features['site_label'].isin([str(s) for s in SITE_IDS])]

# Guardar CSV Global (Ahora sí tendrá exactamente 24 filas por día/sitio)
all_features_csv = os.path.join(OUTPUT_DIR_PRE, '01_all_features.csv')
df_features.to_csv(all_features_csv, index=False)
print(f"CSV consolidado (Matriz 24h) guardado en: {all_features_csv}")

print(df_features[['site_label','sample_uid', 'date_id', 'hour_bin','vector_len']].head(50))

# =============================================================================
# ============================================================================= 
# 5. FUNCIÓN DE GRAFICACIÓN CONDICIONAL (SOLO CANDIDATOS FINALES) 
# =============================================================================
# ============================================================================= 

def plot_final_candidate(df_data, site_id, feature, p_value_kw): 
    """ Genera y guarda el gráfico multilínea solo para un candidato triple éxito. """ 
     
    df_plot = df_data[df_data['site_label'] == site_id].copy().dropna(subset=[feature]) 
     
    if df_plot.empty: 
        return 

    plt.figure(figsize=(18, 8)) 
    sns.lineplot(x='hour_bin',  
                 y=feature,  
                 data=df_plot,  
                 marker='o',  
                 errorbar=None,  
                 hue='date_id',  
                 palette='tab10')  

    plt.title(f'CANDIDATO FINAL (Recurrente y Estable): {feature} (Site {site_id}) | P_KW = {p_value_kw:.3e}', fontsize=16) 
    plt.ylabel(f'Valor Promedio de {feature}') 
    plt.xlabel('Hora del Día (Agrupado :00 / :30)') 
    plt.xticks(rotation=45, ha='right', fontsize=10) 
    plt.legend(title='Fecha de Captura', bbox_to_anchor=(1.05, 1), loc=2, fontsize=8)  
    plt.grid(True, axis='both', linestyle='--', alpha=0.6) 
    plt.tight_layout(rect=[0, 0, 0.85, 1])  

    output_filename = os.path.join(OUTPUT_DIR, f'site_{site_id}_{feature}_FINAL.png') 
    plt.savefig(output_filename) 
    plt.close() 
    print(f"   [GUARDADO FINAL] Gráfico de evidencia en: {output_filename}") 


# =============================================================================
# BLOQUE DE DIAGNÓSTICO: Inspección de calidad de datos
# =============================================================================
print("\n🔍 --- DIAGNÓSTICO DE DATOS (Check de Integridad) ---")

# Convertimos a string para asegurar coincidencia con el DataFrame preprocesado
sample_site = str(SITE_IDS[0]) 
sample_feat = FEATURES_TO_TEST[0] 

df_check = df_features[df_features['site_label'] == sample_site]

if df_check.empty:
    print(f"⚠️ ERROR: No se encontraron datos para el sitio {sample_site}. Verifique tipos de datos.")
else:
    dias_detectados = df_check['date_id'].nunique()
    print(f"Inspeccionando Sitio: {sample_site} | Feature: {sample_feat}")
    print(f"Días únicos: {dias_detectados} | Filas: {len(df_check)}")

    # Verificamos si hay variación real
    if len(df_check) > 0:
        primer_dia = df_check['date_id'].iloc[0]
        serie_24h = df_check[df_check['date_id'] == primer_dia][sample_feat].values
        print(f"Serie 24h ({primer_dia}):\n{serie_24h}")
        
        # Validar si el relleno es puro 0 o repetición
        if np.all(serie_24h == serie_24h[0]) and serie_24h[0] != 0:
            print("⚠️ ADVERTENCIA: Relleno detectado como repetición constante.")
        elif np.all(serie_24h == 0):
            print("ℹ️ INFO: El día seleccionado es todo ceros.")

# =============================================================================
# ============================================================================= 
# 6. ANÁLISIS MASIVO Y PRUEBAS ESTADÍSTICAS 
# =============================================================================
# ============================================================================= 

print("\n--- Paso 3: Ejecutando Kruskal-Wallis, Levene y Recurrencia sobre todas las Features ---") 

# DataFrames para almacenar los resultados (solo los P-Values, para no saturar la memoria) 
results_kruskal_df = pd.DataFrame(index=SITE_IDS, columns=FEATURES_TO_TEST) 
results_levene_df = pd.DataFrame(index=SITE_IDS, columns=FEATURES_TO_TEST) 
results_recurrence_df = pd.DataFrame(index=SITE_IDS, columns=FEATURES_TO_TEST) 

# Limpiar filas con datos faltantes en las features a probar
df_features_clean = df_features.dropna(subset=FEATURES_TO_TEST, axis=0).copy() 
total_optimal_and_recurrent = 0 

print("Validaciòn de limpieza")
print(len(df_features_clean))
print(df_features_clean[['site_label','sample_uid', 'date_id', 'hour_bin','vector_len']].head(50))

# Iterar sobre cada sitio y cada feature 
for site_id in SITE_IDS: 
    df_site = df_features_clean[df_features_clean['site_label'].astype(str) == str(site_id)].copy()
    print(len(df_site))

    if df_site.empty:
        continue

    for feature in FEATURES_TO_TEST: 
        try:
            # 1. Validación básica de datos
            if df_site[feature].nunique() < 2: 
                results_kruskal_df.loc[site_id, feature] = 1.0 
                results_levene_df.loc[site_id, feature] = 0.0 
                results_recurrence_df.loc[site_id, feature] = False 
                continue  
            
            # 2. Agrupar por hora
            groups_binned = [ 
                df_site[df_site['hour_bin'] == h][feature].values
                for h in TIME_BINS_ORDER 
            ]
            groups_binned = [g for g in groups_binned if len(g) > 1] 
            #print("GROUP BINNED")
            #print(groups_binned)
            if len(groups_binned) < 2: 
                continue 
                 
            # 3. Test de Kruskal-Wallis (Diferencia horaria)
            stat_kw, p_value_kw = stats.kruskal(*groups_binned) 
            results_kruskal_df.loc[site_id, feature] = p_value_kw 
             
            # 4. Test de Levene (Estabilidad) - RELAJADO
            try: 
                stat_levene, p_value_levene = stats.levene(*groups_binned, center='median') 
            except ValueError: 
                # Si falla la varianza, asumimos 1.0 para que no bloquee la feature
                p_value_levene = 1.0 
            
            results_levene_df.loc[site_id, feature] = p_value_levene 
                 
            # 5. Chequeo de Recurrencia
            is_recurrent = False 
            if p_value_kw < ALPHA: 
                 is_recurrent, _ = check_recurrence(df_site, feature, p_value_kw, MIN_RECURRENCE_DAYS) 
             
            results_recurrence_df.loc[site_id, feature] = is_recurrent 

            # 6. CRITERIO DE ÉXITO (Levene relajado a 0.0001)
            if p_value_kw < ALPHA and p_value_levene > 0.0001 and is_recurrent: 
                total_optimal_and_recurrent += 1 
                print(f"   ✅ ÉXITO: Site {site_id}, Feature {feature} (p_KW: {p_value_kw:.2e}, p_Lev: {p_value_levene:.2e})") 

        except Exception as e:
            # En análisis masivo, si una feature falla, saltamos a la siguiente
            continue
         
    # Progreso por sitio
    print(f"   [PROGRESO] Site {site_id} completado.")


# =============================================================================
# ============================================================================= 
# 7. INTERPRETACIÓN Y RESUMEN FINAL - FEATURES ROBUSTAS
# =============================================================================
# =============================================================================  
print("\n--- Paso 4: Resultados Consolidados y graficación---") 

# Filtramos las columnas que pasaron las 3 pruebas (KW, Levene y Recurrencia)
final_candidates = (results_kruskal_df < ALPHA) & (results_levene_df >= ALPHA) & results_recurrence_df
robust_features = final_candidates.columns[final_candidates.any()].tolist()
print(f"Total features robustas: {len(robust_features)}")

# Utilizamos el sample_uid como identificador único en lugar del pcap_name
id_cols = ['sample_uid','site_label', 'date_id', 'hour_bin']

df_features_robust = df_features[id_cols + robust_features].copy()

print("\n--- Aplicando Filtro de Consenso Global (50% de los sitios) ---")

conteo_por_feature = final_candidates.sum(axis=0)
total_sitios_analizados = len(SITE_IDS)
umbral_sitios = total_sitios_analizados * 0.2  # <--- Aquí puedes cambiar el 0.5 por 0.3 si es muy estricto

features_maestras = conteo_por_feature[conteo_por_feature >= umbral_sitios].index.tolist()

print(f"Sitios analizados: {total_sitios_analizados}")
print(f"Umbral requerido (20%): {umbral_sitios} sitios")
print(f"Features que cumplen el consenso: {len(features_maestras)}")

if len(features_maestras) > 0:
    print("\n💎 FEATURES MAESTRAS SELECCIONADAS (Consenso Global):")
    for f in features_maestras:
        print(f" - {f} (Presente en {conteo_por_feature[f]} sitios)")
    
    # Guardar el CSV solo con las maestras
    df_features_maestras = df_features_robust[id_cols + features_maestras]
    path_maestras = os.path.join(OUTPUT_DIR_PRE, '03_features_maestras.csv')
    df_features_maestras.to_csv(path_maestras, index=False)
    print(f"✅ CSV de features maestras guardado en: {path_maestras}")
else:
    print("⚠️ Ninguna feature alcanzó el consenso del 20%. No se generó el archivo 03_features_maestras.csv")



# --- BLOQUE DE VALIDACIÓN DE 24 HORAS ---
counts = df_features_robust.groupby('sample_uid').size()
if not (counts == 24).all():
    print("\n⚠️ ADVERTENCIA: Hay muestras que no tienen exactamente 24 horas.")
    print(counts[counts != 24])
else:
    print("\n✅ VALIDACIÓN EXITOSA: Todas las muestras tienen exactamente 24 bins temporales.")
# ----------------------------------------

robust_features_csv = os.path.join(OUTPUT_DIR_PRE, '02_features_robust_.csv')
df_features_robust.to_csv(robust_features_csv, index=False)
print(f"CSV de features robustas guardado en: {robust_features_csv}")

print("\n--- FASE 0 COMPLETA: Datos listos para HOURLY ENCODER ---")

results_kruskal_df = results_kruskal_df.astype(float) 
results_levene_df = results_levene_df.astype(float) 

final_candidates_df = (results_kruskal_df < ALPHA) & (results_levene_df >= ALPHA) & results_recurrence_df 

print("\n## 🏆 CANDIDATOS FINALES (Mediana Diferente Y Varianza Estable Y RECURRENTE)") 
print(f"TOTAL DE CANDIDATOS ÓPTIMOS Y RECURRENTES (N>={MIN_RECURRENCE_DAYS} DÍAS): {final_candidates_df.sum().sum()}") 

print("\n### 🥇 Las Mejores Características (Conteo de Sitios donde son Óptimas):") 
print(final_candidates_df.sum().sort_values(ascending=False).head(10).to_string()) 

print("\n### 🥈 Los Mejores Sitios (Conteo de Características Óptimas en cada Sitio):") 
print(final_candidates_df.sum(axis=1).sort_values(ascending=False).head(10).to_string()) 

print("\n---") 
print("El análisis masivo ha concluido. Los gráficos de las firmas horarias más robustas se encuentran en el directorio:") 
print(f"-> {OUTPUT_DIR}")