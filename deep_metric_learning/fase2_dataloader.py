"""
=========================================================================================
FASE 2: DISEÑO DEL CARGADOR DE DATOS Y MUESTREO ESTRATÉGICO (P-K SAMPLER)
=========================================================================================
Objetivo: Implementar la partición estratificada del dataset (anti-fuga de datos) y 
          construir el mecanismo de alimentación en lotes especializados para DML.

Requisitos de entrada:
- ./resultados/X_hourly_raw.npy : Tensor de características (Días, 24, MAX_SEQ, 2).
- ./resultados/y_hourly_raw.npy : Etiquetas asociadas a cada día.

Reglas aplicadas:
1. Partición estratificada 80/20 (Train/Validation) sobre el dataset del Pasado.
2. Aislamiento absoluto: El dataset del Futuro (Concept Drift) no interviene aquí.
3. Construcción del PKBatchSampler: Cada lote de tamaño B contendrá P instancias 
   de K clases diferentes, garantizando pares "Ancla-Positivo" para Contrastive Loss.
4. Optimización PyTorch: Uso de mmap_mode y Datasets nativos para control de RAM.

Salida:
- Archivos serializados: X_train, y_train, X_val, y_val en ./resultados/
- G2_Validacion_Lote_PK.png : Evidencia gráfica de la composición del lote estratégico.
=========================================================================================
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ==========================================
# CONFIGURACIÓN DE HIPERPARÁMETROS FASE 2
# ==========================================
INPUT_DIR = "./resultados/"
OUTPUT_DIR = "./resultados/"
TEST_SIZE = 0.20        # 20% para validación, 80% entrenamiento
RANDOM_SEED = 42        # Para reproducibilidad en la partición

# Configuración del Sampler PxK
K_CLASSES = 16          # Número de clases distintas (sitios web) por lote
P_INSTANCES = 4         # Número de muestras (días) por cada clase
BATCH_SIZE = K_CLASSES * P_INSTANCES  # Total del lote: 64

# ==========================================
# 1. PARTICIÓN ESTRATIFICADA (TRAIN / VAL)
# ==========================================
def perform_train_val_split(X_path, y_path):
    print("[1/4] Ejecutando partición estratificada 80/20...")
    
    # Usar mmap_mode='r' permite leer los metadatos sin cargar los gigabytes a la RAM
    X_full = np.load(X_path, mmap_mode='r')
    y_full = np.load(y_path)
    
    # StratifiedShuffleSplit garantiza que la proporción de clases se mantenga
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, val_idx = next(sss.split(np.zeros(len(y_full)), y_full))
    
    print(f"      -> Total muestras: {len(y_full)}")
    print(f"      -> Entrenamiento: {len(train_idx)} | Validación: {len(val_idx)}")
    
    # Guardar los subconjuntos físicos en el disco para la Fase 3
    print("      -> Escribiendo tensores particionados a disco (esto puede demorar un momento)...")
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_full[train_idx])
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_full[train_idx])
    np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_full[val_idx])
    np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_full[val_idx])
    
    return train_idx, val_idx, y_full[train_idx]

# ==========================================
# 2. DATASET OPTIMIZADO (PYTORCH)
# ==========================================
class TorHourlyDataset(Dataset):
    """
    Dataset nativo de PyTorch optimizado para tensores grandes.
    Carga desde memoria mapeada (mmap) si es necesario.
    """
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path, mmap_mode='r')
        self.y = np.load(y_path)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        # Convertir a tensor de PyTorch (float32, el estándar para las redes)
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_label = self.y[idx]
        return x_tensor, y_label

# ==========================================
# 3. EL MUESTREADOR ESTRATÉGICO PxK
# ==========================================
class PKBatchSampler(Sampler):
    """
    Sampler DML: Garantiza lotes con K clases diferentes y P instancias por clase.
    Esencial para Contrastive / Triplet Loss.
    """
    def __init__(self, labels, k_classes, p_instances):
        self.labels = labels
        self.k_classes = k_classes
        self.p_instances = p_instances
        
        # Agrupar los índices por clase
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
            
        self.unique_classes = list(self.label_to_indices.keys())
        
    def __iter__(self):
        # Barajar las clases y los índices internos por época
        np.random.shuffle(self.unique_classes)
        for label in self.unique_classes:
            np.random.shuffle(self.label_to_indices[label])
            
        # Generar lotes
        batch = []
        # Iteramos sobre las clases barajadas en bloques de tamaño K
        for i in range(0, len(self.unique_classes), self.k_classes):
            selected_classes = self.unique_classes[i : i + self.k_classes]
            
            # Si no hay suficientes clases para un bloque K completo, lo saltamos (Drop last)
            if len(selected_classes) < self.k_classes:
                break
                
            # Por cada clase seleccionada, tomamos P instancias
            for cls in selected_classes:
                indices = self.label_to_indices[cls]
                # Si una clase tiene más de P instancias, tomamos las P primeras
                # Si tiene menos (raro por el preprocesamiento), usamos reemplazo
                if len(indices) >= self.p_instances:
                    batch.extend(indices[:self.p_instances])
                else:
                    batch.extend(np.random.choice(indices, self.p_instances, replace=True))
                    
            yield batch
            batch = []
            
    def __len__(self):
        return len(self.unique_classes) // self.k_classes

# ==========================================
# 4. EVIDENCIA GRÁFICA (COMPOSICIÓN DEL LOTE)
# ==========================================
def generate_g2_plot(dataloader):
    print("[4/4] Extrayendo un lote aleatorio y generando evidencia gráfica (G2)...")
    
    # Obtener el primer lote del Dataloader
    iterator = iter(dataloader)
    x_batch, y_batch = next(iterator)
    
    # Configurar gráfica
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Convertir labels a string para mejor visualización categórica
    y_batch_str = [str(y.item() if isinstance(y, torch.Tensor) else y) for y in y_batch]
    x_indices = np.arange(len(y_batch_str))
    
    # Crear un scatter plot codificado por color
    unique_labels = list(set(y_batch_str))
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    colors = [color_map[label] for label in y_batch_str]
    
    plt.scatter(x_indices, y_batch_str, c=colors, s=150, edgecolor='black', linewidth=0.5)
    
    # Líneas divisorias cada P instancias para resaltar visualmente los bloques
    for i in range(0, len(y_batch_str), P_INSTANCES):
        plt.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.5)
        
    plt.title(f"G2: Composición del Lote PxK (K={K_CLASSES} clases, P={P_INSTANCES} muestras)", fontsize=14, fontweight='bold')
    plt.xlabel(f"Índice de la muestra dentro del Lote (Total = {BATCH_SIZE})", fontsize=12)
    plt.ylabel("Etiqueta del Sitio Web (Clase)", fontsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, 'G2_Validacion_Lote_PK.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"      -> Gráfica G2_Validacion_Lote_PK.png guardada en: {plot_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    raw_x_path = os.path.join(OUTPUT_DIR, 'X_hourly_raw.npy')
    raw_y_path = os.path.join(OUTPUT_DIR, 'y_hourly_raw.npy')
    
    if not os.path.exists(raw_x_path) or not os.path.exists(raw_y_path):
        print("ERROR: No se encontraron los archivos de la Fase 1 en ./resultados/")
        exit(1)
        
    # Paso 1: Partición
    train_idx, val_idx, y_train_subset = perform_train_val_split(raw_x_path, raw_y_path)
    
    # Paso 2 y 3: Inicializar Dataset y Dataloader para Entrenamiento
    print("[2/4] Instanciando TorHourlyDataset para PyTorch...")
    train_dataset = TorHourlyDataset(
        os.path.join(OUTPUT_DIR, 'X_train.npy'),
        os.path.join(OUTPUT_DIR, 'y_train.npy')
    )
    
    print("[3/4] Acoplando PKBatchSampler al DataLoader...")
    pk_sampler = PKBatchSampler(y_train_subset, k_classes=K_CLASSES, p_instances=P_INSTANCES)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=pk_sampler,
        num_workers=2,        # Ajustable según tu CPU
        pin_memory=True       # Acelera la transferencia a la GPU en Fase 3
    )
    
    # Paso 4: Generar Gráfica
    generate_g2_plot(train_dataloader)
    
    print("=== FASE 2 COMPLETADA ===")
    print("-> El ecosistema de datos está preparado para la Arquitectura Neuronal (Fase 3).")