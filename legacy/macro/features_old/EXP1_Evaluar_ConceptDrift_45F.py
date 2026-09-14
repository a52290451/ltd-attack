"""
=========================================================================================
                    EXP1_Evaluar_ConceptDrift_45F.py
=========================================================================================
OBJETIVO:
    Evaluar la resistencia frente al Concept Drift del modelo entrenado exclusivamente 
    con las 45 Invariantes.
=========================================================================================
"""
import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import os
import joblib
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
DRIFT_FEATURES_CSV = '../../output/preprocessed/02_features_robust_drift.csv'
HISTORIC_RAW_CSV = '../../output/final_features_sites.csv' 
SAVE_DIR = './resultados'  

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Logger(os.path.join(SAVE_DIR, f"log_eval_EXP1_{timestamp}.txt"))

print("\n" + "═"*60)
print("🛡️ EXP 1: EVALUACIÓN CONCEPT DRIFT (45 INVARIANTES)")
print("═"*60)

BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Dispositivo configurado: {device}")

# --- 2. CARGAR MEMORIA DEL MODELO ---
print("\n📖 Cargando memoria del modelo...")
try:
    df_historic_raw = pd.read_csv(HISTORIC_RAW_CSV, usecols=['site', 'site_label']).drop_duplicates()
    site_to_historic_id = dict(zip(df_historic_raw['site'], df_historic_raw['site_label']))
    
    le = joblib.load(os.path.join(SAVE_DIR, "le_45.joblib"))
    scaler = joblib.load(os.path.join(SAVE_DIR, "scaler_45.joblib"))
    features_45 = joblib.load(os.path.join(SAVE_DIR, "features_45_list.joblib"))
    
    input_dim_features = len(features_45)
    known_historic_ids = set(le.classes_) 
    print(f"   ✅ Memoria recuperada: {len(known_historic_ids)} sitios en el LabelEncoder.")
except Exception as e:
    print(f"❌ Error al cargar los metadatos: {e}")
    sys.exit()

# --- 3. ARQUITECTURA ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=24):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe

class TemporalEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_norm = nn.LayerNorm(d_model)
    def forward(self, x): return self.output_norm(self.transformer_encoder(self.pos_encoder(x)).mean(dim=1))

class SetAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        x = self.norm(x + attn_out)
        return self.norm2(x + self.ffn(x))

class GlobalSiteEncoder(nn.Module):
    def __init__(self, num_features, num_classes, d_model=64):
        super().__init__()
        self.feature_projection = nn.Linear(1, d_model)
        self.temporal_engine = TemporalEncoder(d_model)
        self.aggregation_engine = nn.Sequential(SetAttentionBlock(d_model), SetAttentionBlock(d_model))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Linear(d_model, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, num_classes))
    def forward(self, x):
        b, f, t = x.shape
        x = self.feature_projection(x.view(b * f, t, 1))
        hourly_embs = self.temporal_engine(x) 
        x = self.aggregation_engine(hourly_embs.view(b, f, -1)) 
        daily_emb = self.global_pool(x.transpose(1, 2)).squeeze(-1) 
        return self.classifier(daily_emb)

# --- 4. PREPARACIÓN DE DATOS DRIFT ---
print("\n⏳ Estructurando datos del futuro (Drift)...")
df_drift = pd.read_csv(DRIFT_FEATURES_CSV)

if 'site_label' not in df_drift.columns and 'site' in df_drift.columns:
    df_drift['site_label'] = df_drift['site'].map(site_to_historic_id)

initial_len = len(df_drift)
df_drift = df_drift.dropna(subset=['site_label'])
df_drift = df_drift[df_drift['site_label'].isin(known_historic_ids)].copy()
print(f"   -> Registros filtrados (no entrenados o sin mapeo): {initial_len - len(df_drift)}")

df_drift[features_45] = scaler.transform(df_drift[features_45]) 

X_list_drift, y_list_drift = [], []
for (site_id, date), group in df_drift.groupby(['site_label', 'date_id']):
    tensor = np.zeros((input_dim_features, 24))
    for _, row in group.iterrows():
        hora_str = str(row['hour_bin'])
        hora = int(hora_str.split(':')[0])
        if 0 <= hora < 24:
            tensor[:, hora] = row[features_45].values
    X_list_drift.append(tensor)
    
    # IMPORTANTE: Convertir el ID histórico numérico al índice interno (0-N) de la red para calcular accuracy
    # Usamos le.transform para obtener la etiqueta mapeada.
    target_idx = le.transform([int(site_id)])[0]
    y_list_drift.append(target_idx)

X_drift = np.array(X_list_drift)
y_drift = np.array(y_list_drift)

if len(y_drift) == 0:
    print("   ❌ Set de Drift vacío.")
    sys.exit()

drift_loader = DataLoader(TensorDataset(torch.FloatTensor(X_drift), torch.LongTensor(y_drift)), batch_size=BATCH_SIZE)
print(f"   -> Generados {len(y_drift)} tensores para inferencia.")

# --- 5. INFERENCIA ---
print("\n🔌 Despertando red neuronal...")
model = GlobalSiteEncoder(input_dim_features, len(known_historic_ids), d_model=64).to(device)
try:
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model_45F.pth"), map_location=device))
    print("   ✅ Pesos restaurados correctamente.")
except Exception as e:
    print(f"   ❌ Error al cargar .pth: {e}")
    sys.exit()

model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for f, y_b in drift_loader:
        logits = model(f.to(device))
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y_b.numpy())
        y_pred.extend(preds.cpu().numpy())

acc_drift = accuracy_score(y_true, y_pred)
print("\n" + "🏆"*20)
print(f"   ACCURACY FINAL (DRIFT CON 45 INVARIANTES): {acc_drift*100:.2f}%")
print("🏆"*20)