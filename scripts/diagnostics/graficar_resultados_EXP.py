"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: graficar_resultados_EXP.py
🚀 VERSIÓN: 1.0 (Visualización de Nivel Publicación Académica)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO:
    Generador de la gráfica maestra para la defensa de la tesis.
    Compara el rendimiento de las diferentes arquitecturas en un entorno estático
    (Pasado) vs. el impacto destructivo del Concept Drift (Futuro).
    Incluye una curva de degradación neta para evidenciar la necesidad del
    Deep Metric Learning.
=========================================================================================
"""
import matplotlib
matplotlib.use('Agg') # Backend seguro para servidores SSH

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. DIRECTORIO DE SALIDA ---
SAVE_DIR = './resultados_globales'
os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "═"*70)
print("📊 GENERANDO GRÁFICA MAESTRA DE ABLACIÓN Y CONCEPT DRIFT")
print("═"*70)

# --- 2. INGRESO MANUAL DE RESULTADOS ---
# Sustituye los valores 0.0 con los resultados reales de tus evaluaciones.
# El EXP2 y EXP6 ya tienen los valores de tus pruebas.
RESULTADOS_MANUALES = {
    "EXP1: Vectores": {
        "Pasado": 96.02,
        "Futuro": 37.02
    },
    "EXP2: Híbrido\nNeutro": {
        "Pasado": 96.02,
        "Futuro": 37.02
    },
    "EXP3: Inclinado\nFeatures (30%)": {
        "Pasado": 0.0,   # <-- Actualizar cuando termine EXP3
        "Futuro": 0.0    # <-- Actualizar cuando termine EXP3
    },
    "EXP4: Inclinado\nVectores (30%)": {
        "Pasado": 0.0,   # <-- Actualizar cuando termine EXP4
        "Futuro": 0.0    # <-- Actualizar cuando termine EXP4
    },
    "EXP5: Gated\nCross-Attention": {
        "Pasado": 0.0,   # <-- Actualizar cuando termine EXP5
        "Futuro": 0.0    # <-- Actualizar cuando termine EXP5
    },
    "EXP6: Deep Metric\nLearning (SupCon)": {
        "Pasado": 0.0,   # <-- Actualizar con el Accuracy Interno del EXP6 (cuando lo re-entrenes)
        "Futuro": 0.0    # <-- Actualizar con el Accuracy Drift del EXP6 (cuando lo re-evalúes)
    }
}

# --- 3. PROCESAMIENTO DE DATOS ---
nombres = list(RESULTADOS_MANUALES.keys())
acc_pasado = [RESULTADOS_MANUALES[exp]["Pasado"] for exp in nombres]
acc_futuro = [RESULTADOS_MANUALES[exp]["Futuro"] for exp in nombres]

# Calcular degradación neta (Puntos porcentuales perdidos)
caida_neta = [p - f for p, f in zip(acc_pasado, acc_futuro)]

# --- 4. CONFIGURACIÓN ESTÉTICA (NIVEL PUBLICACIÓN) ---
sns.set_theme(style="white", context="paper")
plt.rcParams['font.family'] = 'sans-serif'

fig, ax1 = plt.subplots(figsize=(14, 8))
x = np.arange(len(nombres))
width = 0.35  # Ancho de las barras

# Colores profesionales
color_pasado = '#2c3e50' # Azul oscuro elegante
color_futuro = '#e74c3c' # Rojo alerta
color_linea = '#f39c12'  # Naranja para la degradación

# --- 5. DIBUJO DE BARRAS ---
barras1 = ax1.bar(x - width/2, acc_pasado, width, label='Validación Estática (Mismo Periodo)', color=color_pasado, edgecolor='black', linewidth=1.2)
barras2 = ax1.bar(x + width/2, acc_futuro, width, label='Concept Drift (+2 Meses)', color=color_futuro, edgecolor='black', linewidth=1.2)

# --- 6. ANOTACIONES SOBRE LAS BARRAS ---
def autolabel(rects, ax, is_future=False):
    """Adjunta una etiqueta de texto sobre cada barra mostrando su altura."""
    for rect in rects:
        height = rect.get_height()
        if height > 0: # Solo etiquetar si el valor ya fue actualizado
            # Si es la barra del futuro, mostrar en rojo para enfatizar la caída
            font_color = color_futuro if is_future else 'black'
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 6),  # Desplazamiento vertical en puntos
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color=font_color)

autolabel(barras1, ax1, is_future=False)
autolabel(barras2, ax1, is_future=True)

# --- 7. DIBUJO DE LA LÍNEA DE DEGRADACIÓN (EJE SECUNDARIO) ---
ax2 = ax1.twinx()
linea_caida = ax2.plot(x, caida_neta, color=color_linea, linestyle='-', marker='D', markersize=10, linewidth=3, label='Degradación Neta (Puntos %)', zorder=5)

# Anotaciones para la línea (Muestra cuánto % perdió exactamente)
for i, txt in enumerate(caida_neta):
    if txt > 0:
        ax2.annotate(f'-{txt:.1f}%', (x[i], caida_neta[i]), textcoords="offset points", xytext=(0,-20), ha='center', fontsize=10, fontweight='bold', color=color_linea, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_linea, lw=1, alpha=0.8))

# --- 8. FORMATO Y ESTILIZACIÓN DE EJES ---
ax1.set_ylabel('Accuracy Predictivo (%)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Pérdida Neta por Concept Drift (%)', fontsize=13, fontweight='bold', color='#d35400')

ax1.set_xticks(x)
ax1.set_xticklabels(nombres, fontsize=12, fontweight='bold')
ax1.set_ylim(0, 115) # Espacio extra para las etiquetas
ax2.set_ylim(0, 105)

# Cuadrícula sutil solo en el eje Y principal
ax1.yaxis.grid(True, linestyle='--', alpha=0.6)
ax1.xaxis.grid(False)
ax2.grid(False)

# Título del Gráfico
plt.title('Evaluación de Resiliencia Arquitectónica frente al Concept Drift\nTransición de Clasificación Estática hacia Deep Metric Learning', fontsize=16, fontweight='black', pad=20)

# Unificación de Leyendas
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=11, framealpha=0.95, edgecolor='black')

sns.despine(ax=ax1, right=False, left=False)
plt.tight_layout()

# --- 9. GUARDADO DE LA IMAGEN ---
grafica_path = os.path.join(SAVE_DIR, 'G_Ablacion_Maestra_Publicacion.png')
plt.savefig(grafica_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Evidencia visual maestra generada en: {grafica_path}")
print("   [!] Recuerda editar el script y añadir los valores conforme finalicen tus scripts.")
print("===========================================================================\n")