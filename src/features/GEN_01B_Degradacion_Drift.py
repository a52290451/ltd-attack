"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: GEN_01B_Degradacion_Drift.py
🚀 VERSIÓN: 3.0 (Actualizado a Resultados de Paridad: 94 Invariantes)
👤 INVESTIGADOR: bsierra@zeus
📅 FECHA DE ACTUALIZACIÓN: 12 de Junio, 2026

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este script genera el Gráfico 2 (G2), diseñado para ilustrar la vulnerabilidad 
    intrínseca de las características estadísticas frente al Concept Drift.
    
    [!] ACTUALIZACIÓN V3.0: Los valores han sido calibrados con los resultados 
    definitivos del Estudio de Ablación bajo Paridad Estricta (65 sitios de élite). 
    Se demuestra que, a pesar de aislar las 94 Invariantes Reales (eliminando el ruido 
    volumétrico determinista), el modelo Macro Puro colapsa severamente, justificando
    la transición hacia arquitecturas de Deep Metric Learning.
=========================================================================================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils.paths import result_path

# --- 1. RUTAS Y DIRECTORIOS ---
SAVE_DIR = result_path('features', 'graficas_tesis')
os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "═"*70)
print("🎨 GENERANDO GRÁFICO 2: DEGRADACIÓN POR CONCEPT DRIFT (ADN INMUTABLE)")
print("═"*70)

try:
    # --- 2. DATOS EXTRAÍDOS DEL ESTUDIO DE ABLACIÓN (PARIDAD 65 CLASES) ---
    # Valores obtenidos empíricamente en EXP1_Train_Invariantes y EXP1_Evaluar_Invariantes
    etiquetas = ['Pasado\n(Entrenamiento & Validación)', 'Futuro\n(Concept Drift: +2 Meses)']
    valores_accuracy = [89.06, 28.05]
    colores = ['#3498db', '#e74c3c'] # Azul (Conocimiento Estático) y Rojo (Colapso)

    # --- 3. GENERACIÓN DEL GRÁFICO ---
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    # Dibujar barras con estilo académico
    barras = plt.bar(etiquetas, valores_accuracy, color=colores, width=0.45, edgecolor='black', linewidth=1.2)

    # Añadir los porcentajes exactos sobre las barras
    for barra in barras:
        alto = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., alto + 2,
                 f'{alto:.2f}%',
                 ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Añadir flecha indicadora de la caída dramática
    # Ajustamos las coordenadas de la flecha a los nuevos valores
    plt.annotate('', xy=(1, 35), xytext=(0, 85),
                 arrowprops=dict(arrowstyle="->, head_width=0.5, head_length=0.8", 
                                 color="black", lw=2, ls='--'))
    
    # Etiqueta de la magnitud de la caída
    caida = valores_accuracy[0] - valores_accuracy[1]
    plt.text(0.5, 60, f'Colapso Predictivo\n(-{caida:.2f}%)', ha='center', va='center',
             fontsize=12, fontweight='bold', color='darkred',
             bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.6', alpha=0.9))

    # --- 4. FORMATO Y ESTILO ---
    plt.ylim(0, 110) # Margen superior para la legibilidad de las etiquetas
    plt.ylabel('Precisión / Accuracy (%)', fontsize=13, fontweight='bold')
    
    plt.title('Impacto del Concept Drift en el ADN Inmutable (94 Features Depuradas)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11)

    plt.tight_layout()
    output_path = os.path.join(SAVE_DIR, 'G2_Degradacion_Concept_Drift_Paridad.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Gráfico 2 generado exitosamente en: {output_path}")
    print("===========================================================================\n")

except Exception as e:
    import traceback
    print(f"❌ Error al generar el gráfico:\n{traceback.format_exc()}")
