"""
=========================================================================================
                                  V_revision.py
=========================================================================================

DESCRIPCIÓN:
    Herramienta de auditoría de integridad de datos para el proyecto de 
    Website Fingerprinting. Compara el dataset de vectores bidimensionales 
    (dirección y peso) con el dataset de características (features) estadísticas.

OBJETIVOS PRINCIPALES:
    1. Validar que la cantidad de muestras por clase coincida entre ambos datasets.
    2. Analizar e identificar anomalías específicas de desbalanceo detectadas en el 
       dataset (específicamente las clases 49 y 5).
    3. Realizar un cruce de datos por identificador único ('pcap_uid') para detectar 
       registros "huérfanos" (archivos procesados exitosamente en la fase de extracción 
       de vectores, pero que fallaron en la de features, o viceversa).

ENTRADAS REQUERIDAS:
    - VECTORS_CSV  : data_path('historical', 'final_vectors_sites.csv')
    - FEATURES_CSV : data_path('historical', 'final_features_sites.csv')

SALIDAS:
    - Reporte de validación en consola que incluye:
        * Comparativa Top 10 clases por volumen.
        * Detalle y seguimiento de las clases anómalas.
        * Análisis estadístico de intersección de identificadores.

METADATOS:
    - Autor   : BRAYAN LEONARDO SIERRA FORERO
    - Fecha   : 15/02/2026
    - Versión : 1.0
    
=========================================================================================
"""

import pandas as pd
import numpy as np
from src.utils.paths import data_path

# --- RUTAS ---
VECTORS_CSV = data_path('historical', 'final_vectors_sites.csv')
FEATURES_CSV = data_path('historical', 'final_features_sites.csv')

print("🔍" + "═"*50)
print("AUDITORÍA DE INTEGRIDAD: VECTORES VS FEATURES")
print("═"*50)

# 1. Cargar Datasets
try:
    df_vec = pd.read_csv(VECTORS_CSV)
    df_feat = pd.read_csv(FEATURES_CSV)
    print(f"✅ Archivos cargados correctamente.")
except Exception as e:
    print(f"❌ Error al cargar archivos: {e}")
    exit()

# 2. Comparación de Totales por Clase
count_vec = df_vec['site_label'].value_counts()
count_feat = df_feat['site_label'].value_counts()

comp = pd.DataFrame({
    'Muestras_Vectores': count_vec,
    'Muestras_Features': count_feat
}).fillna(0).astype(int)

comp['Diferencia'] = comp['Muestras_Vectores'] - comp['Muestras_Features']

print("\n📊 COMPARATIVA TOP 10 CLASES (Por volumen en Vectores):")
print(comp.sort_values(by='Muestras_Vectores', ascending=False).head(10))

# 3. Análisis específico de la anomalía (Clases 49 y 5)
anomalas = [49, 5]
print("\n🕵️ DETALLE DE CLASES ANÓMALAS:")
for cls in anomalas:
    if cls in comp.index:
        v = comp.loc[cls, 'Muestras_Vectores']
        f = comp.loc[cls, 'Muestras_Features']
        print(f"   - Clase {cls}: Vectores({v}) vs Features({f}) | Dif: {v-f}")
    else:
        print(f"   - Clase {cls}: No encontrada en el dataset.")

# 4. Verificación de Intersección por pcap_uid
if 'pcap_uid' in df_vec.columns and 'pcap_uid' in df_feat.columns:
    ids_vec = set(df_vec['pcap_uid'])
    ids_feat = set(df_feat['pcap_uid'])
    
    comunes = ids_vec.intersection(ids_feat)
    solo_vec = ids_vec - ids_feat
    solo_feat = ids_feat - ids_vec
    
    print("\n🆔 ANÁLISIS DE IDENTIFICADORES (pcap_uid):")
    print(f"   - UID en ambos archivos: {len(comunes)}")
    print(f"   - UID presentes SOLO en Vectores: {len(solo_vec)}")
    print(f"   - UID presentes SOLO en Features: {len(solo_feat)}")
    
    if len(solo_vec) > 0:
        print(f"\n⚠️  ALERTA: Se encontraron {len(solo_vec)} registros en Vectores que no existen en Features.")
        print(f"   Etiquetas de estos registros huérfanos: {df_vec[df_vec['pcap_uid'].isin(list(solo_vec)[:10])]['site_label'].unique()}")
else:
    print("\n⚠️  No se pudo comparar por pcap_uid porque falta la columna en uno de los archivos.")

# 5. Ayuda para verificar URLs (Si tienes el archivo de mapeo original)
# Aquí intentamos deducir si el sitio label 49 y 5 se repiten
print("\n🔗 SUGERENCIA DE VERIFICACIÓN DE URL:")
print("Si usaste un diccionario o archivo para mapear 'URL -> site_label',")
print("revisa si las posiciones 49 y 5 en tu lista original están duplicadas.")
print("Comúnmente sucede al procesar 'google.com' y 'www.google.com' como etiquetas distintas.")

print("\n" + "═"*50)
