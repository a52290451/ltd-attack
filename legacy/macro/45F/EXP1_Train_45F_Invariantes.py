"""
=========================================================================================
                    EXP1_Train_45F_Invariantes.py
=========================================================================================
OBJETIVO:
    Entrenar un modelo de Solo-Features (Macro Puro) utilizando EXCLUSIVAMENTE 
    las 45 features "Invariantes".
=========================================================================================
"""
import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
FEATURES_LIST_TXT = '../vectores_features/resultados_analisis/features_invariantes_seguras.txt'
HISTORIC_FEATURES_CSV = '../../output/preprocessed/02_features_robust_.csv'

SAVE_DIR = './resultados'
os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Logger(os.path.join(SAVE_DIR, f"log_train_EXP1_{timestamp}.txt"))

print("\n" + "═"*60)
print("🛡️ EXP 1: ENTRENAMIENTO MACRO PURO CON 45 FEATURES INVARIANTES")
print("═"*60)

BATCH_SIZE = 64
EPOCHS = 60
D_MODEL = 64
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

# --- 3. PREPARACIÓN DE DATOS ---
print("\n⏳ Estructurando datos de entrenamiento...")
df_hist = pd.read_csv(HISTORIC_FEATURES_CSV)
df_hist = df_hist.dropna(subset=['site_label'])

le = LabelEncoder()
df_hist['target'] = le.fit_transform(df_hist['site_label'])
scaler = StandardScaler()
df_hist[features_45] = scaler.fit_transform(df_hist[features_45])

joblib.dump(le, os.path.join(SAVE_DIR, "le_45.joblib"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_45.joblib"))
joblib.dump(features_45, os.path.join(SAVE_DIR, "features_45_list.joblib"))

X_list_hist, y_list_hist = [], []
for (target_idx, date), group in df_hist.groupby(['target', 'date_id']):
    tensor = np.zeros((input_dim_features, 24))
    for _, row in group.iterrows():
        hora_str = str(row['hour_bin'])
        hora = int(hora_str.split(':')[0])
        if 0 <= hora < 24:
            tensor[:, hora] = row[features_45].values
    X_list_hist.append(tensor)
    y_list_hist.append(target_idx)

X_hist = np.array(X_list_hist)
y_hist = np.array(y_list_hist)

X_train, X_val, y_train, y_val = train_test_split(X_hist, y_hist, test_size=0.15, stratify=y_hist, random_state=42)
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=BATCH_SIZE)

num_classes = len(le.classes_)
print(f"   -> Set de Entrenamiento: {len(X_train)} tensores.")
print(f"   -> Set de Validación: {len(X_val)} tensores.")

# --- 4. ARQUITECTURA ---
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

# --- 5. ENTRENAMIENTO ---
print("\n🏋️ Iniciando Entrenamiento...")
model = GlobalSiteEncoder(input_dim_features, num_classes, d_model=D_MODEL).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_val_acc = 0
patience, trigger = 10, 0

for epoch in range(EPOCHS):
    model.train()
    for f, y_b in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(f.to(device)), y_b.to(device))
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for f, y_b in val_loader:
            preds = torch.argmax(model(f.to(device)), dim=1)
            correct += (preds == y_b.to(device)).sum().item()
    
    val_acc = correct / len(y_val)
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:02d} | Val Acc: {val_acc*100:.2f}%")
        
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model_45F.pth"))
        trigger = 0
    else:
        trigger += 1
        if trigger >= patience:
            print(f"  🛑 Early stopping en época {epoch+1}.")
            break

print(f"\n🏆 ENTRENAMIENTO COMPLETADO. MEJOR ACCURACY (VALIDACIÓN): {best_val_acc*100:.2f}%")