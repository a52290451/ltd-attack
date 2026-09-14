"""
=========================================================================================
                                VF1_Trans_dir_size_feat.py
=========================================================================================

🔍 DESCRIPCIÓN:
    El Modelo Definitivo (End-to-End). Une la arquitectura de análisis temporal-secuencial 
    (Vectores de Red) con el análisis de series de tiempo jerárquico (Features).

🎯 SOLUCIONES ARQUITECTÓNICAS APLICADAS:
    1. Stratified Time-Series Split: División 80/20 cronológica POR CLASE.
    2. Modality Dropout: Desactiva la rama tabular el 40% del tiempo en Train.
    3. Normalización sin Leakage: Normaliza los tamaños de paquete usando el MTU (1500).
    4. Extracción de Fecha y Caché: Caché permanente para acelerar el merge.

💾 LOGGING: Guarda toda la salida de consola en un archivo .txt automáticamente.
=========================================================================================
"""

import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_cosine_schedule_with_warmup
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import ast
import time
import datetime
import os
import sys
import joblib
import gc
import re

# --- CLASE PARA GUARDAR LOGS EN TXT ---
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

# --- 1. CONFIGURACIÓN ---
VECTORS_CSV = '../../output/final_vectors_sites.csv'
FEATURES_CSV = '../../output/preprocessed/02_features_robust_.csv' 
RAW_FEATURES_CSV = '../../output/final_features_sites.csv' 
CACHED_VECTORS_CSV = '../../output/cached_vectors_with_dates.csv' 
SAVE_DIR = './resultados'
os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(SAVE_DIR, f"mega_entrenamiento_{timestamp}.txt")
sys.stdout = Logger(log_filename)

print("\n" + "═"*60)
print("🧠 INICIANDO ENTRENAMIENTO: MEGA-MODELO HÍBRIDO")
print(f"📄 Guardando copia del log en: {log_filename}")
print("═"*60)

MAX_LEN = 3000 
BATCH_SIZE = 64
EPOCHS = 100
D_MODEL = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Dispositivo: {device}")

# --- 2. CARGA Y SINCRONIZACIÓN DE DATOS ---
start_data_load = time.time()
print(f"\n📦 [Fase 1/4] Cargando datasets de Features...")
df_feat = pd.read_csv(FEATURES_CSV)

excluded = ['sample_uid', 'site_label', 'date_id', 'hour_bin', 'direction_vector', 'size_vector', 'pcap_uid']
features_names = [col for col in df_feat.columns if col not in excluded]
input_dim_features = len(features_names)
print(f"   - Features estadísticos extraídos: {input_dim_features}")

# (A) Preprocesamiento de Vectores y CRUCE DE FECHAS (CON CACHÉ)
start_vec = time.time()
print("\n⏳ [Fase 2/4] Preprocesando Vectores de Red y alineando fechas...")

if os.path.exists(CACHED_VECTORS_CSV):
    print(f"   🔄 Caché encontrado. Cargando {CACHED_VECTORS_CSV} (Omitiendo merge y regex)...")
    df_vec = pd.read_csv(CACHED_VECTORS_CSV)
else:
    print(f"   ⚠️ Caché no encontrado. Extrayendo fechas y cruzando datos (Esto tomará un momento)...")
    df_vec = pd.read_csv(VECTORS_CSV)
    
    df_bridge = pd.read_csv(RAW_FEATURES_CSV, usecols=['pcap_uid', 'pcap_name']).drop_duplicates(subset=['pcap_uid'])
    
    def extract_date(name_str):
        match = re.search(r'_(\d{8})-', str(name_str))
        return int(match.group(1)) if match else 0
        
    df_bridge['date_id'] = df_bridge['pcap_name'].apply(extract_date)
    df_bridge = df_bridge.drop(columns=['pcap_name']) 
    
    df_vec = pd.merge(df_vec, df_bridge, on='pcap_uid', how='inner')
    df_vec.to_csv(CACHED_VECTORS_CSV, index=False)
    print(f"   💾 Caché guardado en {CACHED_VECTORS_CSV} para acelerar ejecuciones futuras.")

def preprocess_vector(v_str, max_len):
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list): v = []
    except: v = []
    if len(v) > max_len: return v[:max_len]
    return v + [0] * (max_len - len(v))

df_vec = df_vec[~df_vec['site_label'].isin([49, 5])].copy()
counts = df_vec['site_label'].value_counts()
valid_sites = counts[counts >= counts.max() * 0.5].index
df_vec = df_vec[df_vec['site_label'].isin(valid_sites)].copy()

print(f"   ✅ Vectores listos y alineados cronológicamente. Tiempo: {time.time() - start_vec:.2f}s")

# (B) Preprocesamiento de Features (Perfil POR DÍA)
start_feat = time.time()
print("\n⏳ [Fase 3/4] Estructurando tensores horarios de Features POR DÍA...")
le = LabelEncoder()
df_feat['target'] = le.fit_transform(df_feat['site_label'])
scaler = StandardScaler()
df_feat[features_names] = scaler.fit_transform(df_feat[features_names])

joblib.dump(le, os.path.join(SAVE_DIR, "mega_label_encoder_robust_3000.joblib"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "mega_scaler_robust_3000.joblib"))
joblib.dump(features_names, os.path.join(SAVE_DIR, "mega_features_list_robust_3000.joblib"))

site_date_to_feature_tensor = {}
for (site, date), group in df_feat.groupby(['site_label', 'date_id']):
    if site not in valid_sites: continue
    tensor = np.zeros((input_dim_features, 24))
    for _, row in group.iterrows():
        # ⚠️ CORRECCIÓN AQUÍ: Separamos la hora del string "00:00"
        hora_str = str(row['hour_bin'])
        hora = int(hora_str.split(':')[0])
        
        if 0 <= hora < 24:
            tensor[:, hora] = row[features_names].values
    site_date_to_feature_tensor[(site, date)] = tensor

print(f"   ✅ Perfiles diarios creados. Tiempo: {time.time() - start_feat:.2f}s")

# (C) Construcción de la Matriz Final Alineada
start_merge = time.time()
print(f"\n🔗 [Fase 4/4] Construyendo Matrices Finales y Split Cronológico...")
X_dir_list, X_w_list, X_feat_list, y_list, dates_list = [], [], [], [], []

for _, row in df_vec.iterrows():
    site = row['site_label']
    date = row['date_id']
    if (site, date) in site_date_to_feature_tensor:
        X_dir_list.append(preprocess_vector(row['direction_vector'], MAX_LEN))
        X_w_list.append(preprocess_vector(row['size_vector'], MAX_LEN))
        X_feat_list.append(site_date_to_feature_tensor[(site, date)])
        y_list.append(le.transform([site])[0])
        dates_list.append(date)

X_dir = np.array(X_dir_list)
X_weight = np.array(X_w_list)
X_feat = np.array(X_feat_list)
y = np.array(y_list)

# ⚠️ CORRECCIÓN EXPERTA: Normalización basada en dominio de red (MTU)
X_weight = np.clip(X_weight / 1500.0, 0.0, 1.0)

# ⚠️ STRATIFIED TIME-SERIES SPLIT
meta_df = pd.DataFrame({'site_label': y, 'date_id': dates_list, 'idx': range(len(y))})
meta_df = meta_df.sort_values(by=['site_label', 'date_id'])

train_indices = []
test_indices = []

for site, group in meta_df.groupby('site_label'):
    n_samples = len(group)
    split_point = int(n_samples * 0.8)
    train_indices.extend(group['idx'].iloc[:split_point].tolist())
    test_indices.extend(group['idx'].iloc[split_point:].tolist())

X_d_tr, X_d_ts = X_dir[train_indices], X_dir[test_indices]
X_w_tr, X_w_ts = X_weight[train_indices], X_weight[test_indices]
X_f_tr, X_f_ts = X_feat[train_indices], X_feat[test_indices]
y_tr, y_ts = y[train_indices], y[test_indices]

print(f"   ✅ Matrices Finales: Vectores {X_dir.shape}, Features {X_feat.shape}")
print(f"   - Muestras de Entrenamiento (Pasado): {len(y_tr)}")
print(f"   - Muestras de Validación    (Futuro): {len(y_ts)}")

train_loader = DataLoader(TensorDataset(torch.LongTensor(X_d_tr), torch.FloatTensor(X_w_tr), torch.FloatTensor(X_f_tr), torch.LongTensor(y_tr)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.LongTensor(X_d_ts), torch.FloatTensor(X_w_ts), torch.FloatTensor(X_f_ts), torch.LongTensor(y_ts)), batch_size=BATCH_SIZE)

num_classes = len(le.classes_)

del X_dir_list, X_w_list, X_feat_list, df_vec, df_feat, meta_df
gc.collect()

# --- 3. ARQUITECTURAS ---

class VecPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class FeatPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=24):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe

class TemporalEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.pos_encoder = FeatPositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.norm(x.mean(dim=1))

class SetAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*2), nn.ReLU(), nn.Linear(d_model*2, d_model))
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        attn, _ = self.mha(x, x, x)
        x = self.norm1(x + attn)
        return self.norm2(x + self.ffn(x))

class MegaHybridModel(nn.Module):
    def __init__(self, num_classes, input_dim_feat, max_len=3000, d_vec=256, d_feat=64):
        super().__init__()
        
        self.dir_emb = nn.Embedding(3, d_vec) 
        self.weight_proj = nn.Linear(1, d_vec)
        self.fusion_vec = nn.Linear(d_vec * 2, d_vec)
        self.conv = nn.Conv1d(d_vec, d_vec, 5, 2, 2)
        self.pos_vec = VecPositionalEncoding(d_vec, max_len=max_len)
        self.transformer_vec = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_vec, 8, 1024, 0.2, batch_first=True, activation='gelu'), 4
        )
        self.ln_vec = nn.LayerNorm(d_vec * 2) 

        self.feat_proj = nn.Linear(1, d_feat)
        self.temp_engine = TemporalEncoder(d_feat)
        self.agg_engine = nn.Sequential(SetAttentionBlock(d_feat), SetAttentionBlock(d_feat))
        self.pool_feat = nn.AdaptiveAvgPool1d(1) 

        fusion_dim = (d_vec * 2) + d_feat 
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_d, x_w, x_f):
        v = torch.cat([self.dir_emb(x_d + 1), self.weight_proj(x_w.unsqueeze(-1))], dim=-1)
        v = self.fusion_vec(v).transpose(1, 2)
        v = self.conv(v).transpose(1, 2)
        v = self.transformer_vec(self.pos_vec(v))
        vec_out = self.ln_vec(torch.cat([torch.mean(v, 1), torch.max(v, 1)[0]], 1)) 
        
        if self.training and torch.rand(1).item() < 0.40:
            feat_out = torch.zeros(x_f.size(0), 64, device=x_f.device)
        else:
            b, f, t = x_f.shape
            ft = x_f.view(b * f, t, 1)
            ft = self.temp_engine(self.feat_proj(ft))
            ft = self.agg_engine(ft.view(b, f, -1))
            feat_out = self.pool_feat(ft.transpose(1, 2)).squeeze(-1) 
        
        return self.classifier(torch.cat([vec_out, feat_out], dim=1))

# --- 4. ENTRENAMIENTO ---
print(f"\n⚙️ Instanciando Mega-Modelo en la GPU...")
model = MegaHybridModel(num_classes, input_dim_features, MAX_LEN, D_MODEL).to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

steps = len(train_loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(steps*0.1), num_training_steps=steps)

best_acc = 0.0
trigger = 0

print(f"\n🏋️ INICIANDO BUCLE DE ENTRENAMIENTO (Épocas: {EPOCHS})")
print("═"*60)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    start = time.time()
    for d, w, f, y_batch in train_loader:
        d, w, f, y_batch = d.to(device), w.to(device), f.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(d, w, f), y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for d, w, f, y_batch in test_loader:
            preds = torch.argmax(model(d.to(device), w.to(device), f.to(device)), dim=1)
            correct += (preds == y_batch.to(device)).sum().item()
            
    acc = correct / len(y_ts)
    print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | {time.time()-start:.1f}s")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"{SAVE_DIR}/mega_best_model_robust_3000.pth")
        print(f" ⭐ Nuevo récord: {best_acc:.4f}")
        trigger = 0
    else:
        trigger += 1
        if trigger >= 15: 
            print(f"🛑 Early stopping activado en la época {epoch+1}. Sin mejoras recientes.")
            break

print("\n" + "═"*60)
print(f"✅ ENTRENAMIENTO FINALIZADO.")
print(f"🏆 Mejor Accuracy en Validación (Sobre días no vistos): {best_acc*100:.2f}%")
print("═"*60)

sys.stdout = sys.__stdout__