"""
=========================================================================================
                    EXP2_Train_45F3000V_Neutro.py
=========================================================================================
OBJETIVO:
    Entrenar la arquitectura Híbrida (Vectores + Features) de forma NEUTRA (0% Dropout),
    pero utilizando EXCLUSIVAMENTE las 45 features "Invariantes".
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
VECTORS_CSV = '../../output/final_vectors_sites.csv'
RAW_FEATURES_CSV = '../../output/final_features_sites.csv' 
CACHED_VECTORS_CSV = '../../output/cached_vectors_with_dates.csv' 

SAVE_DIR = './resultados_40F3000V'
os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Logger(os.path.join(SAVE_DIR, f"log_train_EXP2_{timestamp}.txt"))

print("\n" + "═"*60)
print("🛡️ EXP 2: ENTRENAMIENTO HÍBRIDO NEUTRO (VECTORES + 45 INVARIANTES)")
print("═"*60)

MAX_LEN = 3000
BATCH_SIZE = 64
EPOCHS = 60
D_MODEL = 256
D_FEAT = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. CARGA DE LAS 45 FEATURES ---
try:
    with open(FEATURES_LIST_TXT, 'r') as f:
        features_45 = [line.strip() for line in f.readlines() if line.strip()]
    input_dim_features = len(features_45)
    print(f"✅ Cargadas {input_dim_features} features de élite.")
except Exception as e:
    print(f"❌ Error al cargar la lista de features: {e}")
    sys.exit()

# --- 3. PREPARACIÓN DE DATOS (MACRO) ---
print("\n⏳ [1/4] Procesando Features Macro...")
df_feat = pd.read_csv(HISTORIC_FEATURES_CSV)

print("⏳ [2/4] Cargando y alineando Vectores Micro...")
if os.path.exists(CACHED_VECTORS_CSV):
    df_vec = pd.read_csv(CACHED_VECTORS_CSV)
else:
    df_vec = pd.read_csv(VECTORS_CSV)
    df_bridge = pd.read_csv(RAW_FEATURES_CSV, usecols=['pcap_uid', 'pcap_name']).drop_duplicates()
    def extract_date(name_str):
        match = re.search(r'_(\d{8})-', str(name_str))
        return int(match.group(1)) if match else 0
    df_bridge['date_id'] = df_bridge['pcap_name'].apply(extract_date)
    df_vec = pd.merge(df_vec, df_bridge.drop(columns=['pcap_name']), on='pcap_uid', how='inner')
    df_vec.to_csv(CACHED_VECTORS_CSV, index=False)

def preprocess_vector(v_str, max_len):
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list): v = []
    except: v = []
    if len(v) > max_len: return v[:max_len]
    return v + [0] * (max_len - len(v))

print("⏳ [3/4] Filtrando y escalando...")
df_vec = df_vec[~df_vec['site_label'].isin([49, 5])].copy()
counts = df_vec['site_label'].value_counts()
valid_sites = counts[counts >= counts.max() * 0.5].index
df_vec = df_vec[df_vec['site_label'].isin(valid_sites)].copy()

le = LabelEncoder()
df_feat['target'] = le.fit_transform(df_feat['site_label'])
scaler = StandardScaler()
df_feat[features_45] = scaler.fit_transform(df_feat[features_45])

joblib.dump(le, os.path.join(SAVE_DIR, "le_hibrido_neutro.joblib"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_hibrido_neutro.joblib"))

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

print("⏳ [4/4] Empaquetando tensores Híbridos (Vectores + Features)...")
X_dir_list, X_w_list, X_feat_list, y_list, dates_list = [], [], [], [], []

for _, row in tqdm(df_vec.iterrows(), total=len(df_vec), desc="Cruzando datos"):
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

# --- 4. ARQUITECTURA HÍBRIDA ---
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

# --- 5. ENTRENAMIENTO ---
print("\n⚙️ Instanciando Modelo Híbrido en la GPU...")
model = MegaHybridModel(num_classes, input_dim_features).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
steps = len(train_loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps*0.1), steps)

best_acc = 0.0
trigger = 0

print(f"\n🏋️ INICIANDO BUCLE DE ENTRENAMIENTO")
for epoch in range(EPOCHS):
    model.train(); total_loss = 0
    for d, w, f, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(d.to(device), w.to(device), f.to(device)), y_batch.to(device))
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
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_model_hibrido_neutro.pth")
        trigger = 0
    else:
        trigger += 1
        if trigger >= 15: 
            print(f"🛑 Early stopping activado.")
            break

print("\n" + "═"*60)
print(f"🏆 ENTRENAMIENTO COMPLETADO. MEJOR ACCURACY: {best_acc*100:.2f}%")
print("═"*60)