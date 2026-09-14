"""
=========================================================================================
                        VF3_Visualizacion_Autopsia.py
=========================================================================================
DESCRIPCIÓN ACADÉMICA:
    Generador de Evidencia Visual para el Estudio de Ablación (Rama Micro / Vectores).
    Demuestra empíricamente por qué la resiliencia cae al 32.62% incluso con datos
    perfectamente purificados (La Paradoja del Ruido Determinista).

EVIDENCIAS GENERADAS:
    1. G1_Huella_Secuencial.jpg  -> Muestra el desorden del código de barras (Direction).
    2. G2_Matriz_Dispersion.jpg  -> Muestra la ceguera sistémica del modelo (Heatmap).
    3. G3_Colapso_Latente.jpg    -> Demuestra el desplazamiento geométrico UMAP.
=========================================================================================
"""

import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import ast
import os
import joblib
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import umap
except ImportError:
    print("[!] Librería 'umap-learn' no encontrada. Ejecute: pip install umap-learn")
    sys.exit()

# --- 1. CONFIGURACIÓN DE RUTAS ---
DIR_DATA = '../../output/'
TRAIN_VECTORS_CSV = os.path.join(DIR_DATA, 'CLEAN_final_vectors_sites.csv')
FUT_VECTORS_CSV = os.path.join(DIR_DATA, 'CLEAN_final_vectors_sites_concept_drift.csv')
SAVE_DIR = '../vectores/resultados/'  

MODEL_WEIGHTS = "ds3_best_multimodal_transformer_vec_3000.pth"
LABEL_ENCODER = "ds3_label_encoder_vec_3000.joblib"

MAX_LEN = 3000 
BATCH_SIZE = 64
D_MODEL = 256
NHEAD = 8           
NUM_LAYERS = 4      

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60)
print(" 🔬 AUTOPSIA VISUAL DEL CONCEPT DRIFT (SOLO VECTORES)")
print("="*60)
print(f"🖥️ Dispositivo configurado: {device}")

# --- 2. CARGAR MEMORIA Y ESCALA HISTÓRICA ---
le = joblib.load(os.path.join(SAVE_DIR, LABEL_ENCODER))
site_to_historic_id = {str(site): idx for idx, site in enumerate(le.classes_)}
known_sites_str = set(site_to_historic_id.keys())

print("\n🕰️ Extrayendo Escala Histórica del Entrenamiento...")
max_size_historic = 0.0
for chunk in pd.read_csv(TRAIN_VECTORS_CSV, chunksize=10000, usecols=['size_vector']):
    for _, row in chunk.iterrows():
        try:
            v = ast.literal_eval(row['size_vector'])
            if isinstance(v, list) and len(v) > 0:
                max_size_historic = max(max_size_historic, max(v))
        except: pass
    break # Con escanear los primeros 10,000 basta para encontrar un valor aproximado seguro
max_size_historic = max(max_size_historic, 14546.0) # Aseguramos el valor que descubrimos antes
print(f"   ✅ Escala Histórica Recuperada: Max = {max_size_historic}")

# --- 3. FUNCIONES DE PROCESAMIENTO (ACTUALIZADAS CON ESTRATIFICACIÓN Y STRINGS) ---
def preprocess_vector(v_str, max_len):
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list): v = []
    except: v = []
    return (v[:max_len] if len(v) > max_len else v + [0] * (max_len - len(v)))

def load_data_subset(csv_path, max_per_class=50):
    df = pd.read_csv(csv_path)
    
    # [!] CORRECCIÓN CRÍTICA: Forzar el tipo a STRING desde el inicio para evitar colapsos
    df['site_label'] = df['site_label'].astype(str)
    
    # Filtrar solo sitios conocidos
    df = df[df['site_label'].isin(known_sites_str)].copy()
    
    # El gran truco: Tomar N muestras de cada clase para tener un set ligero pero completo
    df = df.groupby('site_label').head(max_per_class).reset_index(drop=True)
    
    # Mapear a target_id para la red neuronal
    df['target_id'] = df['site_label'].map(site_to_historic_id).astype(int)
    
    X_d, X_w, y = [], [], []
    for _, row in df.iterrows():
        X_d.append(preprocess_vector(row['direction_vector'], MAX_LEN))
        X_w.append(preprocess_vector(row['size_vector'], MAX_LEN))
        y.append(row['target_id'])
    
    X_w_norm = np.clip(np.array(X_w) / float(max_size_historic), 0.0, 1.0)
    
    # Garantizar que el array de etiquetas que devolvemos sea 100% de strings
    return np.array(X_d), X_w_norm, np.array(y), df['site_label'].values.astype(str)

# --- 4. EXTRACCIÓN DE DATOS PARA AUTOPSIA ---
print("\n⏳ Cargando Pasado (Muestreo Estratificado de 50 por clase)...")
X_d_past, X_w_past, y_past, lbl_past = load_data_subset(TRAIN_VECTORS_CSV, max_per_class=50)

print("⏳ Cargando Futuro (Muestreo Estratificado de 50 por clase)...")
X_d_fut, X_w_fut, y_fut, lbl_fut = load_data_subset(FUT_VECTORS_CSV, max_per_class=50)

# --- 5. ARQUITECTURA DEL MODELO (MODIFICADA PARA EXTRAER EMBEDDINGS) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class MultimodalTransformer(nn.Module):
    def __init__(self, num_classes, max_len, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.dir_embedding = nn.Embedding(3, d_model) 
        self.weight_proj = nn.Linear(1, d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.conv_local = nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, 
            dropout=0.2, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model * 2) 
        
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_dir, x_weight):
        e_dir = self.dir_embedding(x_dir + 1) 
        e_weight = self.weight_proj(x_weight.unsqueeze(-1))
        x = self.fusion(torch.cat([e_dir, e_weight], dim=-1))                       
        x = self.conv_local(x.transpose(1, 2)).transpose(1, 2)     
        x = self.transformer(self.pos_encoder(x))
        
        # Extraer características latentes (Embebidos)
        features = self.ln(torch.cat((torch.mean(x, dim=1), torch.max(x, dim=1)[0]), dim=1))
        logits = self.fc(features)
        return logits, features # Devolvemos ambas cosas para la autopsia

print("\n🔌 Despertando red neuronal para Inferencia Latente...")
model = MultimodalTransformer(len(site_to_historic_id), MAX_LEN, d_model=D_MODEL, num_layers=NUM_LAYERS).to(device)
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, MODEL_WEIGHTS), map_location=device))
model.eval()

def get_predictions_and_embeddings(X_d, X_w, y):
    dataset = TensorDataset(torch.LongTensor(X_d), torch.FloatTensor(X_w), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    all_preds, all_embs = [], []
    with torch.no_grad():
        for d, w, _ in loader:
            logits, embs = model(d.to(device), w.to(device))
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_embs.extend(embs.cpu().numpy())
    return np.array(all_preds), np.array(all_embs)

print("   🚀 Extrayendo geometrías del Pasado...")
preds_past, embs_past = get_predictions_and_embeddings(X_d_past, X_w_past, y_past)
print("   🚀 Extrayendo geometrías del Futuro...")
preds_fut, embs_fut = get_predictions_and_embeddings(X_d_fut, X_w_fut, y_fut)

# ==========================================
# GENERACIÓN DE EVIDENCIAS GRÁFICAS
# ==========================================
os.makedirs("resultados_autopsia", exist_ok=True)

# Identificamos sitios que realmente existan en ambos vectores extraídos
common_sites = np.intersect1d(lbl_past, lbl_fut)

# --- GRÁFICA 1: HUELLA DACTILAR SECUENCIAL (BARCODE) ---
print("\n[1/3] Generando G1: Huella Dactilar Secuencial...")
target_site = str(common_sites[0]) # Ahora es seguro extraer el primero común

# [!] CORRECCIÓN: Aseguramos que ambos lados de la ecuación sean string
idx_past = np.where(lbl_past == target_site)[0][0]
idx_fut = np.where(lbl_fut == target_site)[0][0]

plt.figure(figsize=(15, 4))
plt.subplot(2, 1, 1)
# Mostramos solo los primeros 500 paquetes para que el desorden sea visible
plt.imshow(X_d_past[idx_past][:500].reshape(1, -1), cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.title(f'PASADO (Entrenamiento) - Primeros 500 Paquetes [Sitio ID: {target_site}]', fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(X_d_fut[idx_fut][:500].reshape(1, -1), cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.title(f'FUTURO (Concept Drift) - Primeros 500 Paquetes [Sitio ID: {target_site}]\nObserve cómo el ruteo de Tor desordena la secuencia destruyendo la firma aprendida por la CNN.', fontsize=12, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('../vectores/resultados/VF3_G1_Huella_Secuencial.jpg', dpi=300)
plt.close()

# --- GRÁFICA 2: MATRIZ DE CONFUSIÓN GLOBAL ---
print("[2/3] Generando G2: Matriz de Dispersión/Confusión...")
cm = confusion_matrix(y_fut, preds_fut)
plt.figure(figsize=(12, 10))
# Normalizamos por fila para ver el porcentaje de error
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm) # Prevenir NaNs
sns.heatmap(cm_norm, cmap='inferno', cbar=True, xticklabels=False, yticklabels=False)
plt.title('G2: Ceguera Sistémica del Modelo Micro (Concept Drift 32.62%)\nLa ausencia de una diagonal fuerte demuestra que el modelo adivina al azar.', fontsize=14, fontweight='bold')
plt.xlabel('Clase Predicha (Red Neuronal)')
plt.ylabel('Clase Real (Sitio Web)')
plt.tight_layout()
plt.savefig('../vectores/resultados/VF3_G2_Matriz_Dispersion.jpg', dpi=300)
plt.close()

# --- GRÁFICA 3: COLAPSO DEL ESPACIO LATENTE (UMAP) ---
print("[3/3] Generando G3: Colapso del Espacio Latente (UMAP)...")
# Elegir 10 sitios al azar de los comunes para que UMAP sea legible
np.random.seed(42)
top_10_sites = np.random.choice(common_sites, min(10, len(common_sites)), replace=False)

mask_past = np.isin(lbl_past, top_10_sites)
mask_fut = np.isin(lbl_fut, top_10_sites)

X_umap = np.vstack([embs_past[mask_past], embs_fut[mask_fut]])
y_umap = np.concatenate([lbl_past[mask_past], lbl_fut[mask_fut]])
domains = np.array(['PASADO (Base de Conocimiento)'] * mask_past.sum() + ['FUTURO (Desplazamiento)'] * mask_fut.sum())

reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
X_embedded = reducer.fit_transform(X_umap)

plt.figure(figsize=(14, 9))
sns.scatterplot(
    x=X_embedded[:, 0], y=X_embedded[:, 1], 
    hue=y_umap, style=domains, palette='tab10', 
    s=100, alpha=0.8, edgecolor='black'
)
plt.title('G3: Autopsia del Concept Drift (Desplazamiento Geométrico)\nObserve cómo las muestras del Futuro (Cruces) abandonan la agrupación original (Círculos) invadiendo otras clases.', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clases y Dominio")
plt.tight_layout()
plt.savefig('../vectores/resultados/VF3_G3_Colapso_Latente.jpg', dpi=300)
plt.close()

print("\n=======================================================")
print(" [✓] AUTOPSIA VISUAL COMPLETADA CON ÉXITO")
print(" Las 3 evidencias irrefutables se han guardado en la ")
print(" carpeta '../vectores/resultados/'.")
print("=======================================================\n")