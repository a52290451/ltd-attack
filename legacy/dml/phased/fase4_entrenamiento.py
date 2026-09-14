"""
=========================================================================================
FASE 4: ENTRENAMIENTO ESTRICTO Y PROYECCIÓN TOPOLÓGICA EN EL ESPACIO LATENTE
=========================================================================================
Objetivo: Ejecutar el bucle de aprendizaje métrico (DML) para optimizar las distancias 
          relativas usando SupCon Loss, monitorear la convergencia y proyectar 
          topológicamente los vectores resultantes.

Requisitos de entrada:
- ./resultados/X_train.npy, y_train.npy, X_val.npy, y_val.npy (Desde Fase 2).
- Módulos de arquitectura y función de pérdida (Desde Fase 3).

Reglas aplicadas:
1. Optimizador AdamW con Weight Decay para prevenir sobreajuste en el Transformer.
2. Monitoreo exclusivo de Contrastive Loss (sin Accuracy categórico en el bucle).
3. Early Stopping basado en la validación para preservar la mejor topología.
4. Extracción final de embeddings (128D) usando el modelo congelado.
5. Inferencia Geométrica Interna: Cálculo de Accuracy Base usando clasificación por 
   Centroides en el conjunto de validación del Pasado para establecer la línea de referencia.
6. Reducción de dimensionalidad no lineal (UMAP) para visualización G3.

Salida:
- modelo_dml_pesos.pth : Red entrenada.
- embeddings_train.npy / embeddings_val.npy : Espacio latente proyectado.
- G3_A_Curvas_Convergencia.png : Historial de pérdida.
- G3_B_Topologia_Latente_UMAP.png : Visualización 2D de clústeres (Validación).
=========================================================================================
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import umap  # pip install umap-learn
from sklearn.metrics import accuracy_score

# Importamos las herramientas construidas en las fases previas
from fase2_dataloader import TorHourlyDataset, PKBatchSampler
from torch.utils.data import DataLoader
from fase3_arquitectura import HourlyEncoderDML, SupervisedContrastiveLoss

# ==========================================
# CONFIGURACIÓN DE HIPERPARÁMETROS FASE 4
# ==========================================
OUTPUT_DIR = "./resultados/"
EPOCHS = 200
PATIENCE = 30             # Early stopping (épocas sin mejora)
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-configuración del Sampler (Debe coincidir con Fase 2)
K_CLASSES = 32
P_INSTANCES = 4
BATCH_SIZE = K_CLASSES * P_INSTANCES
LATENT_DIM = 256         # Aumentado para dar más espacio a las 115 clases
LOSS_TEMPERATURE = 0.05  # Reducido para forzar mayor separación

def train_model():
    print(f"[1/5] Preparando entorno de entrenamiento en dispositivo: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Cargar Datasets
    print("      -> Cargando Datasets mapeados en memoria...")
    train_dataset = TorHourlyDataset(os.path.join(OUTPUT_DIR, 'X_train.npy'), os.path.join(OUTPUT_DIR, 'y_train.npy'))
    val_dataset = TorHourlyDataset(os.path.join(OUTPUT_DIR, 'X_val.npy'), os.path.join(OUTPUT_DIR, 'y_val.npy'))
    
    # 2. Configurar Dataloaders (Train usa el PK Sampler, Val usa batch estándar)
    pk_sampler = PKBatchSampler(train_dataset.y, k_classes=K_CLASSES, p_instances=P_INSTANCES)
    train_loader = DataLoader(train_dataset, batch_sampler=pk_sampler, num_workers=2, pin_memory=True)
    
    # Validación no necesita el P-K Sampler estricto, solo evaluar pérdida global
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Inicializar Arquitectura y Optimizador
    print("      -> Instanciando HourlyEncoderDML y AdamW...")
    model = HourlyEncoderDML(latent_dim=LATENT_DIM).to(DEVICE)
    criterion = SupervisedContrastiveLoss(temperature=LOSS_TEMPERATURE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Variables de rastreo
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(OUTPUT_DIR, 'modelo_dml_pesos.pth')

    print("\n[2/5] Iniciando bucle de optimización (Deep Metric Learning)...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- ENTRENAMIENTO ---
        model.train()
        train_loss_acum = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            embeddings = model(x_batch)
            loss = criterion(embeddings, y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss_acum += loss.item()
            
        avg_train_loss = train_loss_acum / len(train_loader)
        
        # --- VALIDACIÓN ---
        model.eval()
        val_loss_acum = 0.0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                embeddings = model(x_batch)
                loss = criterion(embeddings, y_batch)
                val_loss_acum += loss.item()
                
        avg_val_loss = val_loss_acum / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        print(f"Época [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Tiempo: {epoch_time:.1f}s")
        
        # --- EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print("  --> ¡Mejora detectada! Modelo guardado.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n[!] Early Stopping activado. No hay mejora en {PATIENCE} épocas.")
                break
                
    return model, history, train_loader, val_loader

def generate_g3_a_plot(history):
    print("\n[3/5] Generando G3_A: Curvas de Convergencia (Contrastive Loss)...")
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Entrenamiento (Train Loss)', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Validación (Val Loss)', color='orange', linewidth=2)
    plt.title('G3_A: Convergencia de Deep Metric Learning (SupCon Loss)', fontsize=14)
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (Distancia Geométrica)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'G3_A_Curvas_Convergencia.png'), dpi=300)
    plt.close()

def extract_embeddings(model, dataloader, dataset_name):
    """Pasa los datos por el modelo congelado para extraer el espacio latente"""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(DEVICE)
            embeddings = model(x_batch)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(y_batch.numpy() if hasattr(y_batch, 'numpy') else y_batch)
            
    return np.vstack(all_embeddings), np.array(all_labels)

def calculate_validation_accuracy(emb_train, lab_train, emb_val, lab_val):
    """Prueba Geométrica: Mide el Accuracy Base en el conjunto de Validación del Pasado"""
    print("      -> Calculando Prototipos (Centroides) del set de Entrenamiento...")
    centroids = {}
    unique_classes = np.unique(lab_train)
    
    for cls in unique_classes:
        cls_embeddings = emb_train[lab_train == cls]
        centroid = cls_embeddings.mean(axis=0)
        centroids[cls] = centroid / (np.linalg.norm(centroid) + 1e-9)
        
    centroid_labels = list(centroids.keys())
    centroid_matrix = np.array([centroids[lbl] for lbl in centroid_labels])
    
    print("      -> Clasificando el set de Validación mediante proximidad Coseno...")
    y_pred = []
    for emb in emb_val:
        similarities = np.dot(centroid_matrix, emb)
        best_match_idx = np.argmax(similarities)
        y_pred.append(centroid_labels[best_match_idx])
        
    acc_base = accuracy_score(lab_val, y_pred)
    return acc_base

def generate_g3_b_umap(embeddings, labels):
    print("[5/5] Generando G3_B: Proyección Topológica UMAP del Espacio Latente...")
    
    # UMAP reduce los 128D a 2D conservando la estructura global
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=embedding_2d[:, 0], 
        y=embedding_2d[:, 1], 
        hue=labels, 
        palette="tab20", 
        legend=False, 
        alpha=0.7,
        edgecolor=None
    )
    plt.title('G3_B: Espacio Latente 2D (Datos de Validación Pasado)\nCada "galaxia" es la identidad topológica de un sitio web', fontsize=14)
    plt.xlabel('Dimensión UMAP 1')
    plt.ylabel('Dimensión UMAP 2')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'G3_B_Topologia_Latente_UMAP.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    # 1 y 2. Entrenamiento y Curvas
    trained_model, loss_history, t_loader, v_loader = train_model()
    generate_g3_a_plot(loss_history)
    
    # 3. Extraer Embeddings definitivos (Usando los mejores pesos guardados)
    print("\n[4/5] Extrayendo Embeddings Latentes (128D) del Pasado...")
    trained_model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'modelo_dml_pesos.pth')))
    
    emb_train, lab_train = extract_embeddings(trained_model, t_loader, "Train")
    emb_val, lab_val = extract_embeddings(trained_model, v_loader, "Val")
    
    np.save(os.path.join(OUTPUT_DIR, 'embeddings_train.npy'), emb_train)
    np.save(os.path.join(OUTPUT_DIR, 'labels_train.npy'), lab_train)
    np.save(os.path.join(OUTPUT_DIR, 'embeddings_val.npy'), emb_val)
    np.save(os.path.join(OUTPUT_DIR, 'labels_val.npy'), lab_val)
    
    # --- PRUEBA GEOMÉTRICA DE ACCURACY BASE ---
    acc_base_val = calculate_validation_accuracy(emb_train, lab_train, emb_val, lab_val)
    print(f"\n=======================================================")
    print(f" ACCURACY BASE (MISMO PERIODO TEMPORAL): {acc_base_val * 100:.2f}%")
    print(f"=======================================================\n")
    
    # 4. Generar gráfica UMAP
    generate_g3_b_umap(emb_val, lab_val)
    
    print("\n=== FASE 4 COMPLETADA ===")
    print("-> El modelo ha forjado el espacio latente y calculado su precisión de referencia.")
