"""
=========================================================================================
                        EXP7_02_Prototipos_Vectoriales.py
=========================================================================================
OBJETIVO:
    Validación de "Perfiles Temporales Vectoriales" (Deep Metric Learning).
    Toma secuencias CRUDAS (Dirección + Peso), las agrupa por DÍA (24h) y entrena 
    una CNN 1D para generar un Embedding Diario estable frente al Concept Drift.
=========================================================================================
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import ast
import os
import re
from tqdm import tqdm

# --- 1. CONFIGURACIÓN ---
print("\n" + "═"*60)
print("🧠 EXP 7.2: APRENDIZAJE MÉTRICO SOBRE PERFILES VECTORIALES DE 24H")
print("═"*60)

SAVE_DIR = './resultados_metric'
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_LEN = 3000
EMBEDDING_DIM = 128
BATCH_SIZE = 16 
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. CARGA Y PREPROCESAMIENTO ---
def preprocess_vector(v_str, max_len):
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list): v = []
    except: v = []
    if len(v) > max_len: return v[:max_len]
    return v + [0] * (max_len - len(v))

def load_and_group_by_day(vec_csv, feat_csv, scale_min=None, scale_max=None):
    print(f"\n⏳ Cargando datos desde {os.path.basename(vec_csv)}...")
    
    df_vec = pd.read_csv(vec_csv)
    df_feat = pd.read_csv(feat_csv, usecols=['pcap_uid', 'site_label', 'pcap_name'])
    df = pd.merge(df_vec, df_feat, on=['pcap_uid', 'site_label'], how='inner')
    
    # Filtro de anomalías
    df = df[~df['site_label'].isin([49, 5])].copy()
    
    # Extracción de la fecha del pcap_name
    def extract_date(name):
        match = re.search(r'_(\d{8})-', name)
        return match.group(1) if match else "unknown"
    df['date_id'] = df['pcap_name'].apply(extract_date)
    
    # CÁLCULO DE ESCALA GLOBAL (Solo si es Train)
    if scale_min is None or scale_max is None:
        print("📏 Calculando escala de pesos base (Train)...")
        all_weights = [preprocess_vector(row, MAX_LEN) for row in df['size_vector']]
        w_min, w_max = np.min(all_weights), np.max(all_weights)
    else:
        w_min, w_max = scale_min, scale_max
    
    print("📦 Empaquetando tensores por día (Esto tomará unos minutos)...")
    grouped_data = []
    
    # Agrupar
    for (site, date), group in tqdm(df.groupby(['site_label', 'date_id'])):
        if len(group) < 5: continue # Rechazar días con datos insuficientes
        
        day_dirs = []
        day_weights = []
        for _, row in group.iterrows():
            d_vec = preprocess_vector(row['direction_vector'], MAX_LEN)
            w_vec = preprocess_vector(row['size_vector'], MAX_LEN)
            
            # Normalización con valores históricos para evitar Leakage
            w_norm = (np.array(w_vec) - w_min) / (w_max - w_min + 1e-8)
            # Clip por si en el futuro hay un paquete más grande que el histórico
            w_norm = np.clip(w_norm, 0.0, 1.0) 
            
            day_dirs.append(d_vec)
            day_weights.append(w_norm)
            
        grouped_data.append({
            'site_label': site,
            'date_id': date,
            'dir_tensor': np.array(day_dirs),
            'weight_tensor': np.array(day_weights)
        })
        
    return pd.DataFrame(grouped_data), w_min, w_max

# Carga rigurosa con paso de escalas
train_df, global_min, global_max = load_and_group_by_day('../../../output/final_vectors_sites.csv', '../../../output/final_features_sites.csv')
test_df, _, _ = load_and_group_by_day('../../../output/final_vectors_sites_concept_drift.csv', '../../../output/final_features_sites_concept_drift.csv', scale_min=global_min, scale_max=global_max)

# Sincronización de clases
common_sites = list(set(train_df['site_label']).intersection(set(test_df['site_label'])))
train_df = train_df[train_df['site_label'].isin(common_sites)].reset_index(drop=True)
test_df = test_df[test_df['site_label'].isin(common_sites)].reset_index(drop=True)

le = LabelEncoder()
train_df['label_encoded'] = le.fit_transform(train_df['site_label'])
test_df['label_encoded'] = le.transform(test_df['site_label'])
NUM_CLASSES = len(le.classes_)

print(f"\n✅ Perfiles de 24h generados. Train: {len(train_df)} días | Test: {len(test_df)} días")
print(f"✅ Clases comunes: {NUM_CLASSES}")

# --- 3. DATALOADER DINÁMICO ---
class DailyProfileDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        d_tensor = torch.LongTensor(row['dir_tensor'])      
        w_tensor = torch.FloatTensor(row['weight_tensor'])  
        label = torch.tensor(row['label_encoded'], dtype=torch.long)
        return d_tensor, w_tensor, label

def collate_fn(batch):
    return batch 

train_loader = DataLoader(DailyProfileDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(DailyProfileDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- 4. ARQUITECTURA: CNN 1D EXTRACTORA ---
class DailyProfileEncoder(nn.Module):
    def __init__(self, max_len=3000, embed_dim=128):
        super().__init__()
        self.dir_emb = nn.Embedding(3, 16) 
        
        self.conv1 = nn.Conv1d(17, 64, kernel_size=7, stride=3, padding=3)
        self.conv2 = nn.Conv1d(64, embed_dim, kernel_size=5, stride=2, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1) 
        
        self.classifier = nn.Linear(embed_dim, NUM_CLASSES)
        
    def forward(self, d_tensor, w_tensor):
        d_emb = self.dir_emb(d_tensor + 1) 
        w_emb = w_tensor.unsqueeze(-1)     
        x = torch.cat([d_emb, w_emb], dim=-1) 
        
        x = x.transpose(1, 2)              
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))          
        
        x = self.pool(x).squeeze(-1)       
        daily_embedding = torch.mean(x, dim=0) 
        
        logits = self.classifier(daily_embedding.unsqueeze(0))
        return daily_embedding, logits

model = DailyProfileEncoder(embed_dim=EMBEDDING_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# --- 5. ENTRENAMIENTO ROBUSTO (Evitando OOM) ---
print(f"\n🏋️ Iniciando entrenamiento de Extractor (Epocas: {EPOCHS})...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch_loss_val = 0
        
        # Procesamos día por día y acumulamos gradientes
        for d_t, w_t, label in batch:
            _, logits = model(d_t.to(DEVICE), w_t.to(DEVICE))
            loss = criterion(logits, label.unsqueeze(0).to(DEVICE))
            
            # Promediamos la pérdida por el tamaño del batch
            loss = loss / len(batch)
            loss.backward() # Computamos gradiente sin acumular los grafos en VRAM
            
            batch_loss_val += loss.item() * len(batch)
            
        optimizer.step()
        total_loss += batch_loss_val
        
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] - Loss: {total_loss/len(train_df):.4f}")

# --- 6. EVALUACIÓN (BÚSQUEDA DE SIMILITUD) ---
print("\n🔍 Evaluando robustez frente a Concept Drift (Búsqueda por Similitud Coseno)...")
model.eval()

# 1. Generar la base de datos de "Centroides" del Pasado
print("   - Generando Prototipos Maestros del Pasado...")
past_embeddings = {}
with torch.no_grad():
    for site in common_sites:
        site_data = train_df[train_df['site_label'] == site]
        site_embs = []
        for _, row in site_data.iterrows():
            d_t = torch.LongTensor(row['dir_tensor']).to(DEVICE)
            w_t = torch.FloatTensor(row['weight_tensor']).to(DEVICE)
            emb, _ = model(d_t, w_t)
            site_embs.append(emb.cpu().numpy())
        if site_embs:
            past_embeddings[site] = np.mean(site_embs, axis=0)

# 2. Evaluar días del Futuro
print("   - Procesando Perfiles del Futuro y emparejando por Coseno...")
y_true = []
y_pred = []

with torch.no_grad():
    for batch in test_loader:
        for d_t, w_t, label in batch:
            emb_futuro, _ = model(d_t.to(DEVICE), w_t.to(DEVICE))
            emb_futuro = emb_futuro.cpu().numpy().reshape(1, -1)
            
            best_site = -1
            best_sim = -2.0
            
            for site, proto_emb in past_embeddings.items():
                sim = cosine_similarity(emb_futuro, proto_emb.reshape(1, -1))[0,0]
                if sim > best_sim:
                    best_sim = sim
                    best_site = site
                    
            y_true.append(le.inverse_transform([label.item()])[0])
            y_pred.append(best_site)

acc = accuracy_score(y_true, y_pred)
print("\n" + "🏆"*20)
print(f"   ACCURACY MEDIANTE PROTOTIPOS DE 24H (DRIFT): {acc*100:.2f}%")
print("🏆"*20)
print("\n✅ Proceso Finalizado.")