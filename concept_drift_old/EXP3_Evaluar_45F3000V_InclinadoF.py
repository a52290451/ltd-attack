"""
=========================================================================================
                    EXP3_Evaluar_45F3000V_InclinadoF.py
=========================================================================================
"""
import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import ast
import os
import joblib
import sys
import datetime
import re

# --- 1. CONFIGURACIÓN DE RUTAS ---
VECTORS_DRIFT_CSV = '../../output/final_vectors_sites_concept_drift.csv'
FEATURES_DRIFT_CSV = '../../output/preprocessed/02_features_robust_drift.csv'
RAW_FEATURES_DRIFT_CSV = '../../output/final_features_sites_concept_drift.csv'
HISTORIC_RAW_CSV = '../../output/final_features_sites.csv' 
SAVE_DIR = '../vectores_features/resultados_40F3000V_IncF'

print("\n" + "═"*60)
print("🛡️ EXP 3: EVALUACIÓN CONCEPT DRIFT (HÍBRIDO INCLINADO + 45F)")
print("═"*60)

MAX_LEN = 3000
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. CARGAR MEMORIA DEL MODELO ---
try:
    df_historic_raw = pd.read_csv(HISTORIC_RAW_CSV, usecols=['site', 'site_label']).drop_duplicates()
    site_to_historic_id = dict(zip(df_historic_raw['site'], df_historic_raw['site_label']))
    
    le = joblib.load(os.path.join(SAVE_DIR, "le_45_inclinado.joblib"))
    scaler = joblib.load(os.path.join(SAVE_DIR, "scaler_45_inclinado.joblib"))
    
    FEATURES_LIST_TXT = '../vectores_features/resultados_analisis/features_invariantes_seguras.txt'
    with open(FEATURES_LIST_TXT, 'r') as f:
        features_45 = [line.strip() for line in f.readlines() if line.strip()]
    input_dim_features = len(features_45)
    
    memorized_ids = set(le.classes_)
    id_to_site = {v: k for k, v in site_to_historic_id.items()}
    known_sites_names = {id_to_site[val] for val in memorized_ids if val in id_to_site}
except Exception as e:
    print(f"❌ Error al cargar memoria: {e}")
    sys.exit()

# --- 3. PREPARACIÓN DATOS DRIFT ---
df_feat = pd.read_csv(FEATURES_DRIFT_CSV)
df_feat = df_feat[df_feat['site'].astype(str).isin(known_sites_names)].copy()
df_feat['target_id'] = df_feat['site'].map(site_to_historic_id)
df_feat[features_45] = scaler.transform(df_feat[features_45]) 

site_date_to_feature_tensor = {}
for uid, group in df_feat.groupby('sample_uid'):
    site_id = int(group['target_id'].iloc[0]) 
    date = str(group['date_id'].iloc[0])
    tensor = np.zeros((input_dim_features, 24))
    for _, row in group.iterrows():
        hora_str = str(row['hour_bin'])
        hora = int(hora_str.split(':')[0])
        if 0 <= hora < 24:
            tensor[:, hora] = row[features_45].values
    site_date_to_feature_tensor[(site_id, date)] = tensor

df_vec = pd.read_csv(VECTORS_DRIFT_CSV)
df_bridge = pd.read_csv(RAW_FEATURES_DRIFT_CSV, usecols=['pcap_uid', 'site', 'pcap_name']).drop_duplicates(subset=['pcap_uid'])

def extract_date(name_str):
    match = re.search(r'_(\d{8})-', str(name_str))
    return str(match.group(1)) if match else "UNKNOWN"

df_bridge['date_id'] = df_bridge['pcap_name'].apply(extract_date)
df_vec = pd.merge(df_vec, df_bridge[['pcap_uid', 'site', 'date_id']], on='pcap_uid', how='inner')
df_vec = df_vec[df_vec['site'].astype(str).isin(known_sites_names)].copy()
df_vec['target_id'] = df_vec['site'].map(site_to_historic_id).astype(int)

def preprocess_vector(v_str, max_len):
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list): v = []
    except: v = []
    if len(v) > max_len: return v[:max_len]
    return v + [0] * (max_len - len(v))

X_dir_list, X_w_list, X_feat_list, y_list = [], [], [], []
for _, row in tqdm(df_vec.iterrows(), total=len(df_vec), desc="Cruzando tensores finales..."):
    site_id = row['target_id']
    date = row['date_id']
    if (site_id, date) in site_date_to_feature_tensor:
        X_dir_list.append(preprocess_vector(row['direction_vector'], MAX_LEN))
        X_w_list.append(preprocess_vector(row['size_vector'], MAX_LEN))
        X_feat_list.append(site_date_to_feature_tensor[(site_id, date)])
        y_list.append(site_id)

X_d_ts, X_w_ts, X_f_ts, y_ts = np.array(X_dir_list), np.array(X_w_list), np.array(X_feat_list), np.array(y_list)
X_w_ts = np.clip(X_w_ts / 1500.0, 0.0, 1.0)
drift_loader = DataLoader(TensorDataset(torch.LongTensor(X_d_ts), torch.FloatTensor(X_w_ts), torch.FloatTensor(X_f_ts), torch.LongTensor(y_ts)), batch_size=BATCH_SIZE)

# --- 4. ARQUITECTURA E INFERENCIA ---
# (Se asume la misma arquitectura MegaHybridModel)
class VecPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class TemporalEncoder(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model, 4, 128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, 2)
    def forward(self, x): return self.transformer(x).mean(dim=1)

class SetAttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        attn, _ = self.mha(x, x, x)
        return self.norm(x + attn)

class MegaHybridModel(nn.Module):
    def __init__(self, num_classes, input_dim_feat, d_vec=256, d_feat=64):
        super().__init__()
        self.dir_emb = nn.Embedding(3, d_vec) 
        self.weight_proj = nn.Linear(1, d_vec)
        self.fusion_vec = nn.Linear(d_vec * 2, d_vec)
        self.conv = nn.Conv1d(d_vec, d_vec, 5, 2, 2)
        self.pos_vec = VecPositionalEncoding(d_vec)
        self.transformer_vec = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_vec, 8, 1024, 0.2, batch_first=True), 4)
        self.ln_vec = nn.LayerNorm(d_vec * 2) 

        self.feat_proj = nn.Linear(1, d_feat)
        self.temp_engine = TemporalEncoder(d_feat)
        self.agg_engine = nn.Sequential(SetAttentionBlock(d_feat), SetAttentionBlock(d_feat))
        self.pool_feat = nn.AdaptiveAvgPool1d(1) 

        self.classifier = nn.Sequential(nn.Linear((d_vec * 2) + d_feat, 512), nn.GELU(), nn.Dropout(0.4), nn.Linear(512, num_classes))

    def forward(self, x_d, x_w, x_f):
        v = torch.cat([self.dir_emb(x_d + 1), self.weight_proj(x_w.unsqueeze(-1))], dim=-1)
        v = self.fusion_vec(v).transpose(1, 2)
        v = self.conv(v).transpose(1, 2)
        v = self.transformer_vec(self.pos_vec(v))
        vec_out = self.ln_vec(torch.cat([v.mean(1), v.max(1)[0]], 1)) 
        
        b, f, t = x_f.shape
        ft = x_f.view(b * f, t, 1)
        ft = self.temp_engine(self.feat_proj(ft))
        ft = self.agg_engine(ft.view(b, f, -1))
        feat_out = self.pool_feat(ft.transpose(1, 2)).squeeze(-1) 
        
        return self.classifier(torch.cat([vec_out, feat_out], dim=1))

print("\n🔌 Despertando red híbrida inclinada...")
model = MegaHybridModel(len(memorized_ids), input_dim_features).to(device)
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model_hibrido_inclinado_45F.pth"), map_location=device))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for d, w, f, y_batch in drift_loader:
        logits = model(d.to(device), w.to(device), f.to(device))
        preds_idx = torch.argmax(logits, dim=1).cpu().numpy()
        preds = le.inverse_transform(preds_idx)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds)

acc = accuracy_score(y_true, y_pred)
print("\n" + "🏆"*20)
print(f"   ACCURACY FINAL (HÍBRIDO INCLINADO 45F vs DRIFT): {acc*100:.2f}%")
print("🏆"*20)