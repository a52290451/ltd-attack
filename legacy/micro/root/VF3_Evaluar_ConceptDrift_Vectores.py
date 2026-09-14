"""
=========================================================================================
                        VF3_Evaluar_ConceptDrift_Vectores.py
=========================================================================================
DESCRIPCIÓN ACADÉMICA (ESTUDIO DE ABLACIÓN):
    Script de evaluación para aislar y medir el rendimiento de la rama "Micro" 
    (Vectores de red: Dirección + Tamaño) frente al fenómeno de Concept Drift.
    
PROPÓSITO EMPÍRICO:
    Este script forma parte del 'Ablation Study' del artículo. 
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
HISTORIC_VECTORS_CSV = '../../output/final_vectors_sites.csv' # <--- PARA CALCULAR LA ESCALA
VECTORS_CSV = '../../output/final_vectors_sites_concept_drift.csv'
RAW_FEATURES_CSV = '../../output/final_features_sites_concept_drift.csv'
HISTORIC_RAW_CSV = '../../output/final_features_sites.csv' 
SAVE_DIR = '../vectores/resultados'  

MODEL_WEIGHTS = "ds2_best_multimodal_transformer_vec_3000.pth"
LABEL_ENCODER = "ds2_label_encoder_vec_3000.joblib"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(SAVE_DIR, f"evaluacion_ABLACION_SoloVectores_{timestamp}.txt")
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
print("\n📖 Construyendo el diccionario histórico y cargando memoria...")
try:
    df_historic = pd.read_csv(HISTORIC_RAW_CSV, usecols=['site', 'site_label']).drop_duplicates()
    site_to_historic_id = dict(zip(df_historic['site'], df_historic['site_label']))
    
    le = joblib.load(os.path.join(SAVE_DIR, LABEL_ENCODER))
    memorized_ids = set(le.classes_) 
    
    id_to_site = {v: k for k, v in site_to_historic_id.items()}
    known_sites_names = {id_to_site[val] for val in memorized_ids if val in id_to_site}
    
    print(f"   ✅ Memoria recuperada: {len(known_sites_names)} sitios estrictamente entrenados.")
except Exception as e:
    print(f"   ❌ Error al cargar memoria. Detalle: {e}")
    sys.exit()

def preprocess_vector(v_str, max_len):
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list): v = []
    except: v = []
    if len(v) > max_len: return v[:max_len]
    return v + [0] * (max_len - len(v))

# --- 2.5 RECONSTRUCCIÓN DE LA ESCALA HISTÓRICA (MIN-MAX SCALER) ---
print("\n🕰️ Extrayendo escalas matemáticas del entrenamiento original...")
start_scale = time.time()
df_hist_vec = pd.read_csv(HISTORIC_VECTORS_CSV, usecols=['site_label', 'size_vector'])

# Aplicar los mismos filtros que VP1
df_hist_vec = df_hist_vec[~df_hist_vec['site_label'].isin([49, 5])].copy()
counts = df_hist_vec['site_label'].value_counts()
valid_sites = counts[counts >= counts.max() * 0.5].index
df_hist_vec = df_hist_vec[df_hist_vec['site_label'].isin(valid_sites)].copy()

# Encontrar el máximo y mínimo absoluto de los pesos
print("   - Calculando valores... (Esto tomará un minuto)")
X_weight_hist = np.array([preprocess_vector(row, MAX_LEN) for row in df_hist_vec['size_vector']])
HISTORIC_MIN = np.min(X_weight_hist)
HISTORIC_MAX = np.max(X_weight_hist)

# Liberar memoria
del df_hist_vec, X_weight_hist

print(f"   ✅ Escala Histórica Recuperada: Min = {HISTORIC_MIN}, Max = {HISTORIC_MAX} (Tomó {time.time() - start_scale:.1f}s)")

# --- 3. CONSTRUCCIÓN DEL TEST SET DEL FUTURO ---
print("\n⏳ Estructurando los datos del futuro (Solo Vectores)...")
start_prep = time.time()

df_vec = pd.read_csv(VECTORS_CSV)

print("   🔗 Cruzando metadatos para recuperar nombres reales de sitios...")
df_bridge = pd.read_csv(RAW_FEATURES_CSV, usecols=['pcap_uid', 'site']).drop_duplicates()
df_vec = pd.merge(df_vec, df_bridge, on='pcap_uid', how='inner')

initial_count = len(df_vec)
df_vec = df_vec[df_vec['site'].astype(str).isin(known_sites_names)].copy()
print(f"   - Vectores descartados (sitios no entrenados): {initial_count - len(df_vec)}")

df_vec['target_id'] = df_vec['site'].map(site_to_historic_id).astype(int)

print("   📦 Empaquetando tensores y aplicando Calibración Histórica...")
X_dir_list, X_w_list, y_list = [], [], []
for _, row in df_vec.iterrows():
    X_dir_list.append(preprocess_vector(row['direction_vector'], MAX_LEN))
    X_w_list.append(preprocess_vector(row['size_vector'], MAX_LEN))
    y_list.append(row['target_id'])

X_d_ts = np.array(X_dir_list)

# ⚠️ LA MAGIA OCURRE AQUÍ: Escalamos el futuro usando la regla del pasado.
# $X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$
X_weight_raw = np.array(X_w_list)
X_w_ts = (X_weight_raw - HISTORIC_MIN) / (HISTORIC_MAX - HISTORIC_MIN + 1e-8)

# Hacemos un clip por si hay un paquete anómalo gigante en el futuro, no desborde la red
X_w_ts = np.clip(X_w_ts, 0.0, 1.0) 

y_ts = np.array(y_list)

test_loader = DataLoader(
    TensorDataset(torch.LongTensor(X_d_ts), torch.FloatTensor(X_w_ts), torch.LongTensor(y_ts)), 
    batch_size=BATCH_SIZE
)
print(f"   ✅ Test Set definitivo listo: {len(y_ts)} capturas de vectores. (Tomó {time.time() - start_prep:.1f}s)")

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
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.2, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model * 2) 
        self.fc = nn.Sequential(nn.Linear(d_model * 2, 512), nn.GELU(), nn.Dropout(0.4), nn.Linear(512, num_classes))

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
num_classes_model = len(memorized_ids)
model = MultimodalTransformer(num_classes_model, MAX_LEN, d_model=D_MODEL, num_layers=NUM_LAYERS).to(device)

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
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds)

# --- 6. RESULTADOS FINALES ---
acc = accuracy_score(y_true, y_pred)
print("\n" + "🏆"*20)
print(f"   ACCURACY FINAL ABLACIÓN SOLO-MICRO (DRIFT): {acc*100:.2f}%")
print("🏆"*20)

etiquetas_presentes = np.unique(y_true)
nombres_etiquetas = [id_to_site[val] for val in etiquetas_presentes if val in id_to_site]

report = classification_report(y_true, y_pred, labels=etiquetas_presentes, target_names=nombres_etiquetas, zero_division=0)

with open(os.path.join(SAVE_DIR, "ablacion_resultado_solo_vectores.txt"), "w") as f:
    f.write(f"ACCURACY GLOBAL ABLACIÓN SOLO-MICRO (DRIFT 2 MESES): {acc*100:.2f}%\n\n{report}")
print("\n📄 Se ha guardado un reporte detallado en: ablacion_resultado_solo_vectores.txt")