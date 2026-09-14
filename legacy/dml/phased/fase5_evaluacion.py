"""
=========================================================================================
FASE 5: CLASIFICACIÓN POR PROTOTIPOS (NEAREST CENTROID) Y EVALUACIÓN FINAL
=========================================================================================
Objetivo: Evaluar la robustez de la arquitectura DML frente al Concept Drift utilizando
          un clasificador dinámico topográfico basado en proximidad angular (Centroides).

Requisitos de entrada:
- ../../output/final_features_sites_concept_drift.csv : Características del Futuro.
- ../../output/final_vectors_sites_concept_drift.csv  : Vectores del Futuro.
- ./resultados/embeddings_train.npy, labels_train.npy : Espacio latente del Pasado.
- ./resultados/modelo_dml_pesos.pth                    : Pesos entrenados del modelo.

Reglas aplicadas (Alineación estricta con el Pasado):
1. Procesamiento idéntico: Mismos filtros de outliers, tamaños y límites de horas (12h).
2. Normalización homóloga: División de tamaños entre el mismo factor constante (3000.0).
3. ADN Inmutable: Generación de Prototipos L2 (Centroides) del Pasado.
4. Clasificación Angular: Proyección del Futuro y emparejamiento mediante Similitud Coseno.

Salida:
- G4_A_Matriz_Confusion_Final.png : Matriz de confusión sobre el Concept Drift.
- G4_B_Contraste_Rendimiento.png   : Gráfico comparativo frente al límite clásico (17.90%).
- reporte_evaluacion_final.csv    : Métricas detalladas por clase en el futuro.
=========================================================================================
"""

import os
import ast
import re
import gc
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Importamos la arquitectura desde la Fase 3
from fase3_arquitectura import HourlyEncoderDML

# ==========================================
# CONFIGURACIÓN DE HIPERPARÁMETROS FASE 5
# ==========================================
MIN_PACKETS = 50              
MIN_CAPTURES_PER_SITE = 500   
MIN_HOURS_PER_DAY = 12        
MAX_SEQ_LEN = 3000            
NORMALIZATION_DIVISOR = 3000.0 
BASELINE_ACCURACY = 17.90       # El techo del enfoque tabular clásico bajo Concept Drift
OUTPUT_DIR = "./resultados/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. PIPELINE DE PREPROCESAMIENTO HOMÓLOGO
# ==========================================
def parse_pcap_name(pcap_name):
    match = re.search(r'(\d{8}-\d{6})', str(pcap_name))
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d-%H%M%S')
        except ValueError:
            return None
    return None

def process_concept_drift_dataset(features_file, vectors_file, valid_classes):
    print("[1/6] Ingestando y sincronizando el dataset de Concept Drift (Futuro)...")
    df_feat = pd.read_csv(features_file, usecols=['pcap_uid', 'site_label', 'pcap_name'])
    df_vect = pd.read_csv(vectors_file, usecols=['pcap_uid', 'direction_vector', 'size_vector'])
    
    df = pd.merge(df_feat, df_vect, on='pcap_uid')
    
    df['direction_vector'] = df['direction_vector'].apply(ast.literal_eval)
    df['size_vector'] = df['size_vector'].apply(ast.literal_eval)
    
    df['datetime'] = df['pcap_name'].apply(parse_pcap_name)
    df = df.dropna(subset=['datetime'])
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    
    print("[2/6] Aplicando filtros estrictos de alineación temporal (Outliers y Duplicados)...")
    # Eliminar duplicados en la misma hora del CD
    df = df.drop_duplicates(subset=['site_label', 'date', 'hour'], keep='first')
    
    # Filtro secuencias cortas
    df['seq_len'] = df['direction_vector'].apply(len)
    df = df[df['seq_len'] >= MIN_PACKETS]
    
    # Filtro sitios minoritarios (manteniendo la misma condición de consistencia)
    df = df[df['site_label'].isin(valid_classes)]
    
    print("[3/6] Estructurando perfiles horarios de 24h para el dataset del Futuro...")
    profiles = defaultdict(lambda: defaultdict(dict))
    for _, row in df.iterrows():
        site, date, hour = row['site_label'], row['date'], row['hour']
        if hour not in profiles[site][date]:
            profiles[site][date][hour] = (row['direction_vector'], row['size_vector'])
            
    X_tensor, y_labels = [], []
    empty_hour = np.zeros((MAX_SEQ_LEN, 2), dtype=np.float16)
    
    for site, dates in profiles.items():
        for date, hours_data in dates.items():
            if len(hours_data) < 1: # Relajamos el límite para que acepte días con pocas capturas en el futuro
                continue
                
            profile_24h = []
            for h in range(24):
                if h in hours_data:
                    dirs_raw, sizes_raw = hours_data[h]
                    dirs = np.array(dirs_raw[:MAX_SEQ_LEN], dtype=np.float16)
                    sizes = np.array(sizes_raw[:MAX_SEQ_LEN], dtype=np.float16) / NORMALIZATION_DIVISOR
                    
                    if len(dirs) < MAX_SEQ_LEN:
                        pad_len = MAX_SEQ_LEN - len(dirs)
                        dirs = np.pad(dirs, (0, pad_len), 'constant', constant_values=0)
                        sizes = np.pad(sizes, (0, pad_len), 'constant', constant_values=0)
                        
                    hour_tensor = np.stack([dirs, sizes], axis=-1)
                    profile_24h.append(hour_tensor)
                else:
                    profile_24h.append(empty_hour)
                    
            X_tensor.append(np.array(profile_24h, dtype=np.float16))
            y_labels.append(site)
            
    return np.array(X_tensor, dtype=np.float16), np.array(y_labels)

# ==========================================
# 2. GENERACIÓN DE PROTOTIPOS (CENTROIDES)
# ==========================================
def calculate_past_prototypes(embeddings_path, labels_path):
    print("[4/6] Calculando Prototipos de Clase (Centroides) basados en el Pasado...")
    emb_train = np.load(embeddings_path)
    lab_train = np.load(labels_path)
    
    centroids = {}
    unique_classes = np.unique(lab_train)
    
    for cls in unique_classes:
        # Extraer los embeddings de la clase e identificar su centro de gravedad
        cls_embeddings = emb_train[lab_train == cls]
        centroid = cls_embeddings.mean(axis=0)
        
        # Re-normalizar el centroide L2 para mantener consistencia esférica angular
        centroids[cls] = centroid / (np.linalg.norm(centroid) + 1e-9)
        
    print(f"      -> {len(centroids)} Prototipos de Identidad forjados exitosamente.")
    return centroids

# ==========================================
# 3. PROYECCIÓN E INFERENCIA ANGULAR
# ==========================================
def evaluate_concept_drift(X_future, y_future, centroids, model_path):
    print("[5/6] Cargando HourlyEncoderDML congelado y proyectando el Futuro...")
    
    # IMPORTANTE: Alinear los hiperparámetros con los cambios de la Fase 3
    # cnn_out_dim=64, windows=50, latent_dim=256
    model = HourlyEncoderDML(cnn_out_dim=64, windows=50, latent_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    future_embeddings = []
    # Procesar en pequeños batches nativos para evitar desbordamiento de memoria
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(X_future), batch_size):
            x_b = torch.tensor(X_future[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            emb_b = model(x_b)
            future_embeddings.append(emb_b.cpu().numpy())
            
    future_embeddings = np.vstack(future_embeddings)
    
    # Clasificación por proximidad angular (Similitud Coseno)
    print("      -> Ejecutando Clasificación por Proximidad Angular (Nearest Centroid)...")
    y_pred = []
    
    centroid_labels = list(centroids.keys())
    centroid_matrix = np.array([centroids[lbl] for lbl in centroid_labels]) # (N_Clases, 128)
    
    for emb in future_embeddings:
        # Como ambos están normalizados L2, el producto punto es la Similitud Coseno exacta
        similarities = np.dot(centroid_matrix, emb)
        best_match_idx = np.argmax(similarities)
        y_pred.append(centroid_labels[best_match_idx])
        
    return np.array(y_pred), future_embeddings

# ==========================================
# 4. MÉTRICAS Y GENERACIÓN GRÁFICA (G4)
# ==========================================
def generate_final_reports_and_plots(y_true, y_pred, final_acc):
    print("[6/6] Generando reportes estadísticos y evidencias gráficas finales (G4)...")
    
    # 1. Guardar reporte tabular por clase
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    report_save_path = os.path.join(OUTPUT_DIR, 'reporte_evaluacion_final.csv')
    df_report.to_csv(report_save_path)
    print(f"      -> Reporte detallado guardado en: {report_save_path}")
    
    # 2. Gráfica G4_A: Matriz de Confusión Final
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    # Graficar versión limpia para evitar saturación si hay demasiadas clases
    sns.heatmap(cm, cmap="Blues", cbar=True, xticklabels=False, yticklabels=False)
    plt.title('G4_A: Matriz de Confusión Final frente al Concept Drift(Espacio Latente Métrico con Nearest Centroid)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicciones (Sitios Web Futuro)')
    plt.ylabel('Etiquetas Reales (Sitios Web Pasado)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'G4_A_Matriz_Confusion_Final.png'), dpi=300)
    plt.close()
    
    # 3. Gráfica G4_B: Gráfico de Barras Comparativo de Rendimiento
    plt.figure(figsize=(8, 6))
    categories = ['Enfoque Tabular Clásico(Fronteras Estáticas - XGBoost)', 'Nuevo Enfoque Geométrico(Hourly Encoder + SupCon + Centroides)']
    accuracies = [BASELINE_ACCURACY, final_acc * 100]
    
    colors = ['#ff4d4d', '#33cc33']
    bars = plt.bar(categories, accuracies, color=colors, edgecolor='black', width=0.5)
    
    # Añadir las etiquetas de los valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2, f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.ylim(0, 105)
    plt.ylabel('Accuracy Predictivo (%)', fontsize=12)
    plt.title('G4_B: Contraste de Rendimiento ante el Concept Drift (+2 Meses)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'G4_B_Contraste_Rendimiento.png'), dpi=300)
    plt.close()
    print("      -> Gráficas obligatorias G4_A y G4_B generadas en ./resultados/")

if __name__ == "__main__":
    # Definición de rutas físicas de los archivos del futuro (Concept Drift)
    FILE_FEATURES_CD = "../../output/final_features_sites_concept_drift.csv"
    FILE_VECTORS_CD = "../../output/final_vectors_sites_concept_drift.csv"
    
    MODEL_PATH = os.path.join(OUTPUT_DIR, 'modelo_dml_pesos.pth')
    EMB_TRAIN_PATH = os.path.join(OUTPUT_DIR, 'embeddings_train.npy')
    LAB_TRAIN_PATH = os.path.join(OUTPUT_DIR, 'labels_train.npy')
    
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Asegúrate de haber completado la Fase 4 y tener 'modelo_dml_pesos.pth' listo.")
        exit(1)
        
   # Step 1: Calcular el mapa de centroides del Pasado PRIMERO
    past_centroids = calculate_past_prototypes(EMB_TRAIN_PATH, LAB_TRAIN_PATH)
    valid_past_classes = list(past_centroids.keys())
    
    # Step 2: Preprocesamiento del Futuro filtrando solo clases conocidas
    X_fut, y_fut = process_concept_drift_dataset(FILE_FEATURES_CD, FILE_VECTORS_CD, valid_past_classes)
    
    # Step 3: Inferencia métrica en el futuro
    preds_fut, emb_fut = evaluate_concept_drift(X_fut, y_fut, past_centroids, MODEL_PATH)
    
    # Step 4: Evaluación y Métricas Finales
    acc_final = accuracy_score(y_fut, preds_fut)
    print(f"=======================================================")
    print(f" ACCURACY FINAL OBTENIDO ANTE CONCEPT DRIFT: {acc_final * 100:.2f}%")
    print(f"=======================================================")
    
    # ==========================================
    # AUTOPSIA DE DATOS DEL FUTURO
    # ==========================================
    print(f"\n=== DIAGNÓSTICO DE INTEGRIDAD DE DATOS DEL FUTURO ===")
    print(f"-> Total de muestras reales a evaluar: {len(y_fut)}")
    
    unique_fut_classes, counts_fut = np.unique(y_fut, return_counts=True)
    print(f"-> Clases del pasado con presencia real en el futuro: {len(unique_fut_classes)} de {len(past_centroids)}")
    
    # Evaluar si la red predijo lo mismo todo el tiempo (Efecto Captcha/Bot)
    unique_preds, counts_preds = np.unique(preds_fut, return_counts=True)
    sorted_preds = sorted(zip(unique_preds, counts_preds), key=lambda x: x[1], reverse=True)
    print("-> Top 5 predicciones (¿A dónde se fue el tráfico?):")
    for cls, count in sorted_preds[:5]:
        porcentaje = (count / len(preds_fut)) * 100
        print(f"   - Centroide {cls}: {count} muestras ({porcentaje:.1f}%)")
    print("=======================================================\n")
    
    generate_final_reports_and_plots(y_fut, preds_fut, acc_final)
    print("=== FASE 5 COMPLETADA CON ÉXITO ===")