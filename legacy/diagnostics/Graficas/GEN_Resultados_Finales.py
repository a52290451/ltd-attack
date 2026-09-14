"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: GEN_Resultados_Finales.py
🚀 VERSIÓN: 1.0 (Consolidación Final del Estudio de Ablación)
👤 INVESTIGADOR: bsierra@zeus
📅 FECHA DE CREACIÓN: 16 de Julio, 2026

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este script genera las gráficas maestras conclusivas para el documento de tesis.
    Sintetiza todos los resultados empíricos obtenidos en la evaluación del Concept Drift,
    contrastando el impacto crítico del "Data Leakage" (Fuga de Datos temporal por mala
    praxis de escalado) y estableciendo el Estado del Arte real de las arquitecturas 
    híbridas purificadas (65 sitios, 94 variables inmutables).

EVIDENCIAS GENERADAS:
    1. G_Final_1_DataLeakage.png: Demuestra cómo la fuga de datos inflaba artificialmente 
       la resiliencia de los modelos, validando el rigor metodológico aplicado en la Fase 2.
    2. G_Final_2_EstadoDelArte.png: Comparativa absoluta Pasado vs Futuro (Sin Leakage).
       Corona a la arquitectura de Cross-Attention (Gating Dinámico) como la solución 
       más robusta frente a la degradación temporal en redes Tor.
=========================================================================================
"""

import matplotlib
matplotlib.use('Agg')  # Configuración para entornos headless (Servidores Linux como Zeus)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# --- 1. CONFIGURACIÓN DEL ENTORNO VISUAL ---
print("\n" + "═"*70)
print("🎨 GENERANDO GRÁFICAS MAESTRAS DE CONCLUSIÓN (ESTUDIO DE ABLACIÓN)")
print("═"*70)

# Estilo académico limpio y profesional
sns.set_theme(style="whitegrid")
SAVE_DIR = "graficas_tesis_finales"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. DATOS EMPÍRICOS RECOPILADOS ---
# Nombres de las arquitecturas evaluadas
modelos = ['Vectores\n(Micro)', 'Features\n(Macro)', 'Híbrido\n(Concatenación)', 'Híbrido\n(DML)', 'Híbrido\n(Cross-Attention)']

# Datos de la Fase 1: Con Fuga de Datos (Ilusión de Resiliencia)
drift_con_leakage = [33.68, 28.05, 41.38, 43.37, 40.65]

# Datos de la Fase 2: Purificados / Rigor Científico (Realidad)
pasado_sin_leakage = [96.45, 85.57, 87.08, 82.76, 94.69]
drift_sin_leakage = [23.21, 15.86, 21.78, 23.24, 29.69]

x = np.arange(len(modelos))
width = 0.35  # Ancho estándar de las barras

# ===========================================================================
# GRÁFICA 1: LA ILUSIÓN VS LA REALIDAD (El impacto del Data Leakage)
# ===========================================================================
print("⏳ [1/2] Dibujando Gráfica de Impacto del Data Leakage...")

fig1, ax1 = plt.subplots(figsize=(12, 6))

# Barras para la comparativa de la caída en el Drift
rects1 = ax1.bar(x - width/2, drift_con_leakage, width, label='Con Data Leakage (Resultados Inflados)', color='#e74c3c', edgecolor='black')
rects2 = ax1.bar(x + width/2, drift_sin_leakage, width, label='Sin Data Leakage (Rigor Matemático)', color='#2ecc71', edgecolor='black')

# Formato de la Gráfica 1
ax1.set_ylabel('Accuracy frente al Concept Drift (%)', fontsize=12, fontweight='bold')
ax1.set_title('El Impacto del "Data Leakage" en la Evaluación de Resiliencia', fontsize=15, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(modelos, fontsize=11, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.set_ylim(0, 50)  # Límite ajustado para enfocar en la caída baja

# Función auxiliar para añadir los porcentajes sobre las barras
def autolabel(rects, ax):
    """Adjunta una etiqueta de texto sobre cada barra mostrando su altura."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1, ax1)
autolabel(rects2, ax1)

plt.tight_layout()
ruta_g1 = os.path.join(SAVE_DIR, 'G_Final_1_DataLeakage.png')
plt.savefig(ruta_g1, dpi=300)
plt.close()

# ===========================================================================
# GRÁFICA 2: ESTADO DEL ARTE - RESILIENCIA REAL (Solo Modelos Purificados)
# ===========================================================================
print("⏳ [2/2] Dibujando Gráfica del Estado del Arte (Estudio de Ablación)...")

fig2, ax2 = plt.subplots(figsize=(14, 7))

# Barras para la comparativa Pasado (Conocimiento) vs Futuro (Drift)
rects_pasado = ax2.bar(x - width/2, pasado_sin_leakage, width, label='Pasado (Entrenamiento / Conocimiento Estático)', color='#3498db', edgecolor='black')
rects_futuro = ax2.bar(x + width/2, drift_sin_leakage, width, label='Futuro (Concept Drift / Traslación Dinámica)', color='#e67e22', edgecolor='black')

# Formato de la Gráfica 2
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Estudio de Ablación: Degradación Estructural en Tor (Datos Purificados)', fontsize=15, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(modelos, fontsize=11, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')
ax2.set_ylim(0, 115) # Margen extra superior para las flechas/cajas de anotación

# Etiquetas estándar para el pasado
autolabel(rects_pasado, ax2)

# Etiquetas rojas para resaltar la caída en el futuro
for rect in rects_futuro:
    height = rect.get_height()
    ax2.annotate(f'{height}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkred')

# Anotación especial destacando la arquitectura vencedora
indice_ganador = 4 # Posición de Cross-Attention en el array (índice 4)
ax2.annotate('Arquitectura SOTA\n(Resiliencia Dinámica)', 
            xy=(indice_ganador + width/2, drift_sin_leakage[indice_ganador] + 2), 
            xytext=(indice_ganador + width/2, 45),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", fc="#f1c40f", ec="black", lw=1.5))

plt.tight_layout()
ruta_g2 = os.path.join(SAVE_DIR, 'G_Final_2_EstadoDelArte.png')
plt.savefig(ruta_g2, dpi=300)
plt.close()

# --- 3. FINALIZACIÓN Y REPORTE ---
print("\n" + "🏆"*25)
print(" ✅ EJECUCIÓN COMPLETADA EXITOSAMENTE")
print(f" 📂 Las gráficas se han guardado en el directorio: ./{SAVE_DIR}/")
print("    1. G_Final_1_DataLeakage.png")
print("    2. G_Final_2_EstadoDelArte.png")
print("🏆"*25 + "\n")