"""
=========================================================================================
                    EXP3_Train_45F3000V_InclinadoF.py
=========================================================================================
OBJETIVO:
    Entrenar el modelo Híbrido aplicando Modality Dropout (40%) a los Vectores. 
    Esto fuerza a la red a "inclinarse" hacia las 45 Features Invariantes, extrayendo
    su máximo potencial de desanonimato sin perder el conocimiento estructural de los paquetes.
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
from tqdm import tqdm
import ast
import os
import joblib
import sys
import datetime
import re

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
FEATURES_LIST_TXT = '../vectores_features/resultados_analisis/features_invariantes_seguras.txt'
HISTORIC_FEATURES_CSV = '../../output/preprocessed/02_features_robust_.csv'
CACHED_VECTORS_CSV = '../../output/cached_vectors_with_dates.csv' 

SAVE_DIR = './resultados_40F3000V_IncF'
os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Logger(os.path.join(SAVE_DIR, f"log_train_EXP3_{timestamp}.txt"))

print("\n" + "═"*60)
print("🛡️ EXP 3: ENTRENAMIENTO HÍBRIDO CON MODALITY DROPOUT (50% VECTORES)")
print("═"*60)

MAX_LEN = 3000
BATCH_SIZE = 64
EPOCHS = 60
D_MODEL = 256
D_FEAT = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. CARGA DE LAS 45 FEATURES ---
with open(FEATURES_LIST_TXT, 'r') as f:
    features_45 = [line.strip() for line in f.readlines() if line.strip()]
input_dim_features = len(features_45)

# --- 3. PREPARACIÓN DE DATOS ---
print("\n⏳ [1/3] Procesando Features Macro...")
df_feat = pd.read_csv(HISTORIC_FEATURES_CSV)

print("⏳ [2/3] Cargando Vectores Micro...")
df_vec = pd.read_csv(CACHED_VECTORS_CSV)

def preprocess_vector(v_str, max_len):
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list): v = []
    except: v = []
    if len(v) > max_len: return v[:max_len]
    return v + [0] * (max_len - len(v))

print("⏳ [3/3] Filtrando, escalando y empaquetando...")
df_vec = df_vec[~df_vec['site_label'].isin([49, 5])].copy()
counts = df_vec['site_label'].value_counts()
valid_sites = counts[counts >= counts.max() * 0.5].index
df_vec = df_vec[df_vec['site_label'].isin(valid_sites)].copy()

le = LabelEncoder()
df_feat['target'] = le.fit_transform(df_feat['site_label'])
scaler = StandardScaler()
df_feat[features_45] = scaler.fit_transform(df_feat[features_45])

joblib.dump(le, os.path.join(SAVE_DIR, "le_45_inclinado.joblib"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_45_inclinado.joblib"))

site_date_to_feature_tensor = {}
for (site, date), group in df_feat.groupby(['site_label', 'date_id']):
    if site not in valid_sites: continue
    tensor = np.zeros((input_dim_features, 24))
    for _, row in group.iterrows():
        hora_str = str(row['hour_bin'])
        hora = int(hora_str.split(':')[0])
        if 0 <= hora < 24:
            tensor[:, hora] = row[features_45].values
    site_date_to_feature_tensor[(site, date)] = tensor

X_dir_list, X_w_list, X_feat_list, y_list, dates_list = [], [], [], [], []
for _, row in tqdm(df_vec.iterrows(), total=len(df_vec)):
    site = row['site_label']
    date = row['date_id']
    if (site, date) in site_date_to_feature_tensor:
        X_dir_list.append(preprocess_vector(row['direction_vector'], MAX_LEN))
        X_w_list.append(preprocess_vector(row['size_vector'], MAX_LEN))
        X_feat_list.append(site_date_to_feature_tensor[(site, date)])
        y_list.append(le.transform([site])[0])
        dates_list.append(date)

X_dir, X_weight, X_feat, y = np.array(X_dir_list), np.array(X_w_list), np.array(X_feat_list), np.array(y_list)
X_weight = np.clip(X_weight / 1500.0, 0.0, 1.0)

meta_df = pd.DataFrame({'site_label': y, 'date_id': dates_list, 'idx': range(len(y))})
meta_df = meta_df.sort_values(by=['site_label', 'date_id'])
train_indices, test_indices = [], []
for site, group in meta_df.groupby('site_label'):
    split = int(len(group) * 0.8)
    train_indices.extend(group['idx'].iloc[:split].tolist())
    test_indices.extend(group['idx'].iloc[split:].tolist())

train_loader = DataLoader(TensorDataset(torch.LongTensor(X_dir[train_indices]), torch.FloatTensor(X_weight[train_indices]), torch.FloatTensor(X_feat[train_indices]), torch.LongTensor(y[train_indices])), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.LongTensor(X_dir[test_indices]), torch.FloatTensor(X_weight[test_indices]), torch.FloatTensor(X_feat[test_indices]), torch.LongTensor(y[test_indices])), batch_size=BATCH_SIZE)
num_classes = len(le.classes_)

# --- 4. ARQUITECTURA ---
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

# --- 5. ENTRENAMIENTO CON MODALITY DROPOUT ---
print("\n⚙️ Instanciando Modelo Híbrido en la GPU...")
model = MegaHybridModel(num_classes, input_dim_features).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
steps = len(train_loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps*0.1), steps)

best_acc = 0.0
trigger = 0

print(f"\n🏋️ INICIANDO BUCLE DE ENTRENAMIENTO (MODALITY DROPOUT = 50% en Vectores)")
for epoch in range(EPOCHS):
    model.train(); total_loss = 0
    for d, w, f, y_batch in train_loader:
        optimizer.zero_grad()
        
        # ⚠️ MODALITY DROPOUT: Apagamos los vectores el 50% de las veces
        if np.random.rand() < 0.5:
            d_input = torch.zeros_like(d).to(device)
            w_input = torch.zeros_like(w).to(device)
        else:
            d_input = d.to(device)
            w_input = w.to(device)
            
        f_input = f.to(device)
        
        loss = criterion(model(d_input, w_input, f_input), y_batch.to(device))
        loss.backward(); optimizer.step(); scheduler.step()
        total_loss += loss.item()
    
    model.eval(); correct = 0
    with torch.no_grad():
        for d, w, f, y_batch in test_loader:
            preds = torch.argmax(model(d.to(device), w.to(device), f.to(device)), dim=1)
            correct += (preds == y_batch.to(device)).sum().item()
            
    acc = correct / len(test_indices)
    print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_model_hibrido_inclinado_45F.pth")
        trigger = 0
    else:
        trigger += 1
        if trigger >= 15: 
            print(f"🛑 Early stopping activado.")
            break

print("\n" + "═"*60)
print(f"🏆 ENTRENAMIENTO COMPLETADO. MEJOR ACCURACY: {best_acc*100:.2f}%")
print("═"*60)