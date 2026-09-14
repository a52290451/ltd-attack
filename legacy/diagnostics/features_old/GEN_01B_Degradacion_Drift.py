import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. RUTAS ---
SAVE_DIR = './graficas_tesis'
os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "="*60)
print("🎨 GENERANDO GRÁFICO 2: DEGRADACIÓN POR CONCEPT DRIFT")
print("="*60)

try:
    # --- 2. DATOS EXTRAÍDOS DE LA ABLACIÓN ---
    # Solo tomamos los valores del modelo clásico de 230 features
    etiquetas = ['Pasado\n(Entrenamiento & Validación)', 'Futuro\n(Concept Drift: +2 Meses)']
    valores_accuracy = [94.00, 1.85]
    colores = ['#3498db', '#e74c3c'] # Azul (Estabilidad) y Rojo (Colapso)

    # --- 3. GENERACIÓN DEL GRÁFICO ---
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    # Dibujar barras
    barras = plt.bar(etiquetas, valores_accuracy, color=colores, width=0.45, edgecolor='black', linewidth=1.2)

    # Añadir los porcentajes exactos sobre las barras
    for barra in barras:
        alto = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., alto + 2,
                 f'{alto:.2f}%',
                 ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Añadir flecha indicadora de la caída dramática
    plt.annotate('', xy=(1, 5), xytext=(0, 90),
                 arrowprops=dict(arrowstyle="->, head_width=0.5, head_length=0.8", 
                                 color="black", lw=2, ls='--'))
    
    # Etiqueta de la caída
    caida = valores_accuracy[0] - valores_accuracy[1]
    plt.text(0.5, 50, f'Colapso Predictivo\n(-{caida:.2f}%)', ha='center', va='center',
             fontsize=12, fontweight='bold', color='darkred',
             bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.6', alpha=0.9))

    # --- 4. FORMATO Y ESTILO ---
    plt.ylim(0, 110) # Damos espacio arriba para los números
    plt.ylabel('Precisión / Accuracy (%)', fontsize=13, fontweight='bold')
    
    # Título omitido o integrado sutilmente según tu instrucción
    plt.title('Impacto del Concept Drift en Características Estáticas (230 Features)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Ajuste de las etiquetas del eje X
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11)

    plt.tight_layout()
    output_path = os.path.join(SAVE_DIR, 'G2_Degradacion_Concept_Drift.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"🚀 Gráfico 2 generado exitosamente en: {output_path}")

except Exception as e:
    print(f"❌ Error al generar el gráfico: {e}")