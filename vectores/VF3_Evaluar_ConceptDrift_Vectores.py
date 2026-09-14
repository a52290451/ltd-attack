"""
=========================================================================================
                        VF3_Evaluar_ConceptDrift_Vectores.py
=========================================================================================
DESCRIPCIÓN ACADÉMICA (ESTUDIO DE ABLACIÓN):
    Script de evaluación para aislar y medir el rendimiento de la rama "Micro" 
    (Vectores de red: Dirección + Tamaño) frente al fenómeno de Concept Drift.
    
PROPÓSITO EMPÍRICO:
    Este script forma parte del 'Ablation Study' del artículo. Evalúa un modelo 
    entrenado EXCLUSIVAMENTE con la secuencia de paquetes (sin la rama de features
    macro-temporales de 24h). 
    
    Al evaluar este modelo "Solo-Micro" frente a tráfico capturado 2 meses después,
    esperamos demostrar matemáticamente que las firmas basadas puramente en secuencias 
    de paquetes son altamente volátiles y sufren la degradación más severa por Concept Drift.
    Este resultado justificará la necesidad de la arquitectura híbrida (LTD-Attack).

ENTRADAS:
    - Datos del futuro (Concept Drift) - Solo vectores.
    - Pesos del modelo "Solo-Micro" entrenado previamente (3000 paquetes).
=========================================================================================
"""

import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
import ast
import os
import joblib
import time
import sys
import datetime

# --- CLASE PARA LOGGER ---
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

# --- 1. CONFIGURACIÓN DE RUTAS ---
VECTORS_CSV = '../../output/CLEAN_final_vectors_sites_concept_drift.csv'
# Directorio donde se guardó el modelo original de solo vectores
SAVE_DIR = '../vectores/resultados/'  

# Nombre esperado del modelo y LabelEncoder de 3000 paquetes
MODEL_WEIGHTS = "ds3_best_multimodal_transformer_vec_3000.pth"
LABEL_ENCODER = "ds3_label_encoder_vec_3000.joblib"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(SAVE_DIR, f"ds3_evaluacion_ABLACION_SoloVectores_{timestamp}.txt")
sys.stdout = Logger(log_filename)

print("\n" + "═"*60)
print("🛡️ ESTUDIO DE ABLACIÓN: SOLO VECTORES (MICRO) vs CONCEPT DRIFT")
print("═"*60)

MAX_LEN = 3000 
BATCH_SIZE = 64
D_MODEL = 256
NHEAD = 8           
NUM_LAYERS = 4      

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Dispositivo configurado: {device}")

# --- 2. EL TRADUCTOR HISTÓRICO Y MEMORIA DEL MODELO ---
print("\n📖 Cargando memoria del modelo Solo-Micro...")
try:
    le = joblib.load(os.path.join(SAVE_DIR, LABEL_ENCODER))
    memorized_ids = set(range(len(le.classes_))) 
    
    # Creamos un mapeo directo de las clases memorizadas
    site_to_historic_id = {site: idx for idx, site in enumerate(le.classes_)}
    known_sites_names = set(le.classes_)
    
    print(f"   ✅ Memoria recuperada: {len(known_sites_names)} sitios estrictamente entrenados.")
except Exception as e:
    print(f"   ❌ Error al cargar memoria. Verifique que los archivos de 3000 paquetes existan: {e}")
    sys.exit()

# --- 3. CONSTRUCCIÓN DEL TEST SET DEL FUTURO ---
print("\n🕰️ Extrayendo escalas matemáticas del entrenamiento original (CLEAN PASADO)...")
start_scale = time.time()

TRAIN_VECTORS_CSV = '../../output/CLEAN_final_vectors_sites.csv'
max_size_historic = 0.0

# Escanear el pasado para encontrar la escala de normalización real
for chunk in pd.read_csv(TRAIN_VECTORS_CSV, chunksize=5000, usecols=['size_vector']):
    for _, row in chunk.iterrows():
        try:
            v = ast.literal_eval(row['size_vector'])
            if isinstance(v, list) and len(v) > 0:
                current_max = max(v)
                if current_max > max_size_historic:
                    max_size_historic = current_max
        except:
            pass

print(f"   ✅ Escala Histórica Recuperada: Max = {max_size_historic} (Tomó {time.time() - start_scale:.1f}s)")

print("\n⏳ Estructurando los datos del futuro (Solo Vectores)...")
start_prep = time.time()

df_vec = pd.read_csv(VECTORS_CSV)
initial_count = len(df_vec)

# 1. Filtramos los sitios que el modelo no conoce
known_sites_str = set(str(c) for c in known_sites_names)
df_vec = df_vec[df_vec['site_label'].astype(str).isin(known_sites_str)].copy()
print(f"   - Vectores descartados (sitios no entrenados): {initial_count - len(df_vec)}")

# 2. Mapeamos el Label de forma segura
site_to_historic_id_str = {str(k): v for k, v in site_to_historic_id.items()}
df_vec['target_id'] = df_vec['site_label'].astype(str).map(site_to_historic_id_str).astype(int)

def preprocess_vector(v_str, max_len):
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list): v = []
    except: v = []
    if len(v) > max_len: return v[:max_len]
    return v + [0] * (max_len - len(v))

print("   📦 Empaquetando tensores y aplicando Calibración Histórica...")
X_dir_list, X_w_list, y_list = [], [], []
for _, row in df_vec.iterrows():
    X_dir_list.append(preprocess_vector(row['direction_vector'], MAX_LEN))
    X_w_list.append(preprocess_vector(row['size_vector'], MAX_LEN))
    y_list.append(row['target_id'])

X_d_ts = np.array(X_dir_list)

# CORRECCIÓN CRÍTICA: Normalizar con la escala histórica en lugar de 1500
X_w_ts = np.clip(np.array(X_w_list) / float(max_size_historic), 0.0, 1.0)

y_ts = np.array(y_list)

test_loader = DataLoader(
    TensorDataset(torch.LongTensor(X_d_ts), torch.FloatTensor(X_w_ts), torch.LongTensor(y_ts)), 
    batch_size=BATCH_SIZE
)
print(f"   ✅ Test Set definitivo listo: {len(y_ts)} capturas de vectores. (Tomó {time.time() - start_prep:.1f}s)")

if len(y_ts) == 0:
    print("\n   ❌ CRÍTICO: Test Set vacío.")
    sys.exit()

# --- 4. ARQUITECTURA EXACTA DEL MODELO SOLO-MICRO ---
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
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :] 
        return self.dropout(x)

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
        x = torch.cat([e_dir, e_weight], dim=-1) 
        x = self.fusion(x)                       
        x = x.transpose(1, 2)
        x = self.conv_local(x)    
        x = x.transpose(1, 2)     
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        avg_pool = torch.mean(x, dim=1) 
        max_pool, _ = torch.max(x, dim=1) 
        x = torch.cat((avg_pool, max_pool), dim=1) 
        x = self.ln(x)
        return self.fc(x)

# --- 5. CARGAR PESOS E INFERENCIA ---
print(f"\n🔌 Despertando a la red neuronal Solo-Micro...")
num_classes = len(memorized_ids)
model = MultimodalTransformer(num_classes, MAX_LEN, d_model=D_MODEL, num_layers=NUM_LAYERS).to(device)

try:
    pth_path = os.path.join(SAVE_DIR, MODEL_WEIGHTS)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    print("   ✅ Cerebro Ablación (.pth) restaurado exitosamente.")
except Exception as e:
    print(f"   ❌ Error al cargar los pesos. Detalle: {e}")
    sys.exit()

model.eval()
y_true, y_pred = [], []
print("   🚀 Pasando los datos de Concept Drift por el modelo (Tomará unos minutos)...")

with torch.no_grad():
    for d, w, y_batch in test_loader:
        preds_idx = torch.argmax(model(d.to(device), w.to(device)), dim=1).cpu().numpy()
        preds = le.inverse_transform(preds_idx)
        
        # Guardamos los textos
        y_true_labels = le.inverse_transform(y_batch.numpy())
        y_true.extend(y_true_labels)
        y_pred.extend(preds)

# --- 6. RESULTADOS FINALES ---
acc = accuracy_score(y_true, y_pred)
print("\n" + "🏆"*20)
print(f"   ACCURACY FINAL ABLACIÓN SOLO-MICRO (DRIFT): {acc*100:.2f}%")
print("🏆"*20)

etiquetas_presentes = np.unique(y_true)
report = classification_report(y_true, y_pred, labels=etiquetas_presentes, zero_division=0)

with open(os.path.join(SAVE_DIR, "ds3_ablacion_resultado_solo_vectores.txt"), "w") as f:
    f.write(f"ACCURACY GLOBAL ABLACIÓN SOLO-MICRO (DRIFT 2 MESES): {acc*100:.2f}%\n\n{report}")
print("\n📄 Se ha guardado un reporte detallado en: ds3_ablacion_resultado_solo_vectores.txt")