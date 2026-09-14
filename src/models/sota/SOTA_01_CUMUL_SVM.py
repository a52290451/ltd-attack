"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: SOTA_01_CUMUL_SVM.py
🚀 VERSIÓN: 1.0
👤 INVESTIGADOR: bsierra@zeus
📅 FECHA DE CREACIÓN: 20 de Julio, 2026

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este script implementa el baseline CUMUL basado en Support Vector Machine (SVM)
    para Website Fingerprinting, siguiendo la aproximación del Estado del Arte clásico
    (Panchenko et al.).

    La metodología CUMUL extrae características estadísticas de tráfico de red a nivel
    Macro (flujo de paquete por flujo) para evaluar el impacto real del Concept Drift
    en la precisión de clasificación de sitios web.

    [!] ENFOQUE MACRO:
    - Utiliza 94 características estadísticas invariantes del tráfico de red.
    - Aplicar StandardScaler ajustado exclusivamente sobre el dataset del Pasado.
    - Modelo SVM (RBF kernel) entrenado sobre datos del Pasado.
    - Evaluación en Pasado (Train Accuracy) y Futuro (Concept Drift Accuracy).

📐 ARQUITECTURA DEL MODELO:
    Support Vector Machine (SVM) con kernel RBF (Radial Basis Function).
    Parámetros: C=1.0, gamma='scale'.

⚙️ ENTRADAS:
    - ../../output/CLEAN_final_features_sites.csv (Pasado)
    - ../../output/CLEAN_final_features_sites_concept_drift.csv (Futuro)
    - ../vectores/resultados/ds3_label_encoder_vec_3000.joblib
    - ../vectores_features/resultados_analisis/new_features_invariantes_seguras.txt
=======================================================================================
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os
import datetime
import matplotlib
matplotlib.use('Agg')  # Modo headless para ejecución en servidor sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE ENTORNO Y RUTAS
# ---------------------------------------------------------------------------------------

# Rutas a los datasets de features purificadas
HISTORIC_CSV = '../../output/CLEAN_final_features_sites.csv'
DRIFT_CSV = '../../output/CLEAN_final_features_sites_concept_drift.csv'

# Ruta al LabelEncoder de vectores (filtro de paridad de 65 clases)
VECTOR_LE_PATH = '../vectores/resultados/ds3_label_encoder_vec_3000.joblib'

# Ruta a la lista de 94 características invariantes
FEATURES_LIST_TXT = '../vectores_features/resultados_analisis/new_features_invariantes_seguras.txt'

# Directorio de resultados
SAVE_DIR = './resultados'
os.makedirs(SAVE_DIR, exist_ok=True)

# Configuración del Logger dual (consola + archivo)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(SAVE_DIR, f'log_SOTA_01_CUMUL_{timestamp}.txt')

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)

print("\n" + "═" * 70)
print("🛡️  SOTA 01: CUMUL BASELINE (SVM) — Evaluación frente al Concept Drift")
print("═" * 70)
print(f"📄 Log de ejecución: {log_file}")

# ---------------------------------------------------------------------------------------
# 2. CARGA DEL FILTRO DE PARIDAD Y LAS FEATURES DE ÉLITE
# ---------------------------------------------------------------------------------------

# Cargar LabelEncoder de vectores (65 clases de élite)
try:
    le = joblib.load(VECTOR_LE_PATH)
    sitios_elite = set(le.classes_.astype(str))
    print(f"\n📦 Filtro Maestro cargado: {len(sitios_elite)} clases de élite detectadas.")
    print(f"   Clases (primeras 10): {list(sitios_elite)[:10]}")
except Exception as e:
    print(f"\n❌ Error crítico al cargar el LabelEncoder: {e}")
    print(f"   Verifica la ruta: {VECTOR_LE_PATH}")
    sys.exit()

# Cargar lista de 94 características invariantes
try:
    with open(FEATURES_LIST_TXT, 'r') as f:
        features_list = [line.strip() for line in f.readlines() if line.strip()]
    print(f"\n📊 Núcleo de Features cargado: {len(features_list)} características estadísticas.")
    print(f"   Features (primeras 10): {features_list[:10]}")
except Exception as e:
    print(f"\n❌ Error crítico al cargar las features: {e}")
    sys.exit()

# ---------------------------------------------------------------------------------------
# 3. CARGA DE DATOS Y FILTRADO POR PARIDAD
# ---------------------------------------------------------------------------------------

print("\n⏳ Cargando dataset del PASADO...")
df_pasado = pd.read_csv(HISTORIC_CSV)
df_pasado = df_pasado.dropna(subset=['site_label'])
df_pasado['site_label'] = df_pasado['site_label'].astype(str)
initial_pasado = len(df_pasado)
df_pasado = df_pasado[df_pasado['site_label'].isin(sitios_elite)].copy()
print(f"   -> Muestras PASADO: {initial_pasado} -> {len(df_pasado)} (filtrado por paridad)")

print("\n⏳ Cargando dataset del FUTURO (Concept Drift)...")
df_futuro = pd.read_csv(DRIFT_CSV)
df_futuro = df_futuro.dropna(subset=['site_label'])
df_futuro['site_label'] = df_futuro['site_label'].astype(str)
initial_futuro = len(df_futuro)
df_futuro = df_futuro[df_futuro['site_label'].isin(sitios_elite)].copy()
print(f"   -> Muestras FUTURO: {initial_futuro} -> {len(df_futuro)} (filtrado por paridad)")

# ---------------------------------------------------------------------------------------
# 4. EXTRACCIÓN DE FEATURES Y PREPARACIÓN DE X, y
# ---------------------------------------------------------------------------------------

# Verificar que todas las features existen en los datasets
missing_pasado = [f for f in features_list if f not in df_pasado.columns]
missing_futuro = [f for f in features_list if f not in df_futuro.columns]

if missing_pasado:
    print(f"\n⚠️  Advertencia: Features faltantes en PASADO: {missing_pasado}")
if missing_futuro:
    print(f"\n⚠️  Advertencia: Features faltantes en FUTURO: {missing_futuro}")

# Aislar características (X) y etiquetas (y)
X_pasado = df_pasado[features_list].copy()
y_pasado = df_pasado['site_label'].copy()

X_futuro = df_futuro[features_list].copy()
y_futuro = df_futuro['site_label'].copy()

# Eliminar filas con NaNs en las features
nan_count_pasado = X_pasado.isnull().sum().sum()
nan_count_futuro = X_futuro.isnull().sum().sum()
print(f"\n🧹 Filas con NaN en PASADO (features): {nan_count_pasado} valores nulos totales")
print(f"🧹 Filas con NaN en FUTURO (features): {nan_count_futuro} valores nulos totales")

X_pasado = X_pasado.dropna()
y_pasado = y_pasado.loc[X_pasado.index]

X_futuro = X_futuro.dropna()
y_futuro = y_futuro.loc[X_futuro.index]

print(f"   -> Dimensiones PASADO tras limpieza: X={X_pasado.shape}, y={y_pasado.shape[0]}")
print(f"   -> Dimensiones FUTURO tras limpieza: X={X_futuro.shape}, y={y_futuro.shape[0]}")

# Convertir etiquetas a valores numéricos usando LabelEncoder.transform()
y_pasado = le.transform(y_pasado)
y_futuro = le.transform(y_futuro)

num_classes = len(le.classes_)
print(f"\n📋 Número de clases para clasificación: {num_classes}")

# ---------------------------------------------------------------------------------------
# 5. PREPROCESAMIENTO — CERO FUGAS DE DATOS (Zero Data Leakage)
# ---------------------------------------------------------------------------------------

print("\n🔬 Preprocesamiento: StandardScaler (Cero Data Leakage)")
print("   -> Ajuste (.fit()) exclusivamente sobre dataset del PASADO")
print("   -> Transformación (.transform()) sobre PASADO y FUTURO")

scaler = StandardScaler()
scaler.fit(X_pasado)

X_pasado_scaled = scaler.transform(X_pasado)
X_futuro_scaled = scaler.transform(X_futuro)

print(f"   -> PASADO escalado: {X_pasado_scaled.shape}")
print(f"   -> FUTURO escalado: {X_futuro_scaled.shape}")

# ---------------------------------------------------------------------------------------
# 6. ENTRENAMIENTO Y EVALUACIÓN — SVM (CUMUL)
# ---------------------------------------------------------------------------------------

print("\n🏋️  Entrenamiento: Support Vector Machine (SVM, kernel RBF)")
print("   -> Parámetros: C=1.0, gamma='scale'")

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_pasado_scaled, y_pasado)
print("   -> Modelo entrenado exitosamente con datos del PASADO.")

# Predicción sobre dataset del PASADO (Train Accuracy)
y_pasado_pred = svm_model.predict(X_pasado_scaled)
train_accuracy = accuracy_score(y_pasado, y_pasado_pred)
print(f"\n📊 Accuracy de Entrenamiento (PASADO): {train_accuracy * 100:.2f}%")

# Predicción sobre dataset del FUTURO (Concept Drift Accuracy)
y_futuro_pred = svm_model.predict(X_futuro_scaled)
drift_accuracy = accuracy_score(y_futuro, y_futuro_pred)
print(f"📊 Accuracy frente al Concept Drift (FUTURO): {drift_accuracy * 100:.2f}%")

# Diferencia de precisión (métrica de impacto del Concept Drift)
drift_impact = (train_accuracy - drift_accuracy) * 100
print(f"\n📉 Impacto del Concept Drift (diferencia de precisión): {drift_impact:.2f} pp")

if drift_impact > 0:
    print(f"   ⚠️  El modelo DEGRADA su rendimiento en datos futuros ({drift_impact:.2f}%)." if drift_impact > 5 else f"   ℹ️  Degradación leve: {drift_impact:.2f}%.")
elif drift_impact < 0:
    print(f"   ✅ El modelo MEJORA su rendimiento en datos futuros ({abs(drift_impact):.2f}%)." )
else:
    print(f"   🎯 El modelo mantiene su rendimiento sin degradación.")

# Reporte de clasificación sobre PASADO
print("\n📋 Classification Report (PASADO - Train Set):")
print(classification_report(y_pasado, y_pasado_pred, target_names=[str(c) for c in le.classes_], digits=4))

# Reporte de clasificación sobre FUTURO
print("📋 Classification Report (FUTURO - Drift Set):")
print(classification_report(y_futuro, y_futuro_pred, target_names=[str(c) for c in le.classes_], digits=4))

# ---------------------------------------------------------------------------------------
# 7. REPORTE Y EVIDENCIA VISUAL — Matriz de Confusión Normalizada
# ---------------------------------------------------------------------------------------

print("\n🎨 Generando Matriz de Confusión Normalizada (PASADO - Train Set)...")

# Calcular matriz de confusión normalizada por filas
cm = confusion_matrix(y_pasado, y_pasado_pred, normalize='true')

# Configuración de la gráfica
fig, ax = plt.subplots(figsize=(12, 10))
sns.set_theme(style="whitegrid")

# Crear heatmap
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# Etiquetas y título
title = f'CUMUL SVM — Matriz de Confusión Normalizada (PASADO)\nTrain Accuracy: {train_accuracy * 100:.2f}%'
ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Predicción', fontsize=12)
ax.set_ylabel('Real', fontsize=12)

# Ocultar ticks de las 65 clases para evitar sobrecarga visual
ax.set_xticks([])
ax.set_yticks([])
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# Añadir texto con valores de precisión global
ax.text(0.02, 0.98, f'Acc Train: {train_accuracy * 100:.2f}%\nAcc Drift: {drift_accuracy * 100:.2f}%\nImpacto Drift: {drift_impact:.2f} pp',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Guardar gráfica
graph_path = os.path.join(SAVE_DIR, f'G_SOTA_01_CUMUL_Drift_{timestamp}.png')
plt.savefig(graph_path, dpi=300, bbox_inches='tight')
print(f"✅ Gráfica guardada en: {graph_path}")

# ---------------------------------------------------------------------------------------
# 8. RESUMEN FINAL
# ---------------------------------------------------------------------------------------

print("\n" + "═" * 70)
print("🏆 RESUMEN FINAL — CUMUL SVM (SOTA CLÁSICO)")
print("═" * 70)
print(f"   📊 Accuracy de Entrenamiento (PASADO):  {train_accuracy * 100:.2f}%")
print(f"   📊 Accuracy frente al Concept Drift:     {drift_accuracy * 100:.2f}%")
print(f"   📉 Impacto del Concept Drift:            {drift_impact:.2f} pp")
print(f"   📋 Número de clases:                     {num_classes}")
print(f"   📋 Número de features:                   {len(features_list)}")
print(f"   📋 Muestras PASADO:                      {len(X_pasado)}")
print(f"   📋 Muestras FUTURO:                      {len(X_futuro)}")
print(f"═" * 70)
print(f"   📄 Log completo:         {log_file}")
print(f"   📄 Gráfica:              {graph_path}")
print("🏆" * 35 + "\n")

print("✅ SOTA_01_CUMUL_SVM.py finalizado exitosamente.")