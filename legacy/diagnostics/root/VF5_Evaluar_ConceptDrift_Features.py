"""
=========================================================================================
                    VF5_Evaluar_ConceptDrift_SoloFeatures.py
=========================================================================================
ESTUDIO DE ABLACIÓN: PRUEBA DE FUEGO (PASADO vs FUTURO)
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

print("\n" + "═"*60)
print("🛡️ ESTUDIO DE ABLACIÓN: PRUEBA DE FUEGO (PASADO vs FUTURO)")
print("═"*60)

# --- 1. CONFIGURACIÓN DE RUTAS ---
HISTORIC_FEATURES_CSV = '../../output/preprocessed/02_features_robust_.csv'
DRIFT_FEATURES_CSV = '../../output/preprocessed/02_features_robust_drift.csv'
HISTORIC_RAW_CSV = '../../output/final_features_sites.csv'   # <--- AÑADE ESTA LÍNEA AQUÍ
SAVE_DIR = '../output/transformers_final'

BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Dispositivo configurado: {device}")

# --- 2. CARGANDO MEMORIA DEL MODELO ---
try:
    le = joblib.load(os.path.join(SAVE_DIR, "x_label_encoder_robust.joblib"))
    scaler = joblib.load(os.path.join(SAVE_DIR, "x_scaler_robust.joblib"))
    features_names = joblib.load(os.path.join(SAVE_DIR, "x_features_list.joblib"))
    
    input_dim_features = len(features_names)
    known_historic_ids = set(le.classes_) 
    print(f"📖 Memoria recuperada: {len(known_historic_ids)} sitios en el LabelEncoder.")
except Exception as e:
    print(f"❌ Error al cargar los metadatos: {e}")
    exit()

# --- 3. ARQUITECTURA EXACTA DEL MODELO ---
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

# --- 4. CARGAR PESOS DE LA RED ---
print("\n🔌 Despertando red neuronal Solo-Features...")
model = GlobalSiteEncoder(input_dim_features, len(known_historic_ids), d_model=64).to(device)
try:
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_global_encoder_robust_v2.pth"), map_location=device))
    print("   ✅ Pesos restaurados correctamente.")
except Exception as e:
    print(f"   ❌ Error al cargar .pth: {e}")
    exit()

model.eval()

# --- 5. MOTOR DE EVALUACIÓN ---
# Recuperamos el diccionario histórico para traducir los nombres a IDs si es necesario
df_historic_raw = pd.read_csv(HISTORIC_RAW_CSV, usecols=['site', 'site_label']).drop_duplicates()
site_to_historic_id = dict(zip(df_historic_raw['site'], df_historic_raw['site_label']))

def evaluate_dataset(csv_path, dataset_name):
    print(f"\n" + "-"*50)
    print(f"⏳ Evaluando: {dataset_name}")
    print("-"*50)
    
    df = pd.read_csv(csv_path)
    
    # FIX DINÁMICO: Si el CSV tiene 'site' pero no 'site_label', lo traducimos.
    if 'site_label' not in df.columns and 'site' in df.columns:
        df['site_label'] = df['site'].map(site_to_historic_id)
    
    # Filtramos sitios no conocidos y evitamos NaNs
    initial_len = len(df)
    df = df.dropna(subset=['site_label'])
    df = df[df['site_label'].isin(known_historic_ids)].copy()
    print(f"   -> Registros filtrados (sitios no entrenados o sin mapeo): {initial_len - len(df)}")
    
    # Escalamos las variables usando la foto del pasado
    df[features_names] = scaler.transform(df[features_names]) 

    site_date_to_feature_tensor = {}
    y_list = []
    for (site_id, date), group in df.groupby(['site_label', 'date_id']):
        tensor = np.zeros((input_dim_features, 24))
        for _, row in group.iterrows():
            hora_str = str(row['hour_bin'])
            hora = int(hora_str.split(':')[0])
            if 0 <= hora < 24:
                tensor[:, hora] = row[features_names].values
        site_date_to_feature_tensor[(site_id, date)] = tensor
        y_list.append(int(site_id))

    X_f = np.array(list(site_date_to_feature_tensor.values()))
    y = np.array(y_list)
    
    if len(y) == 0:
        print("   ❌ Set de datos vacío.")
        return
        
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_f), torch.LongTensor(y)), batch_size=BATCH_SIZE)
    print(f"   -> Generados {len(y)} tensores de 24h.")
    print(f"   🚀 Ejecutando inferencia...")
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for f, y_batch in loader:
            logits = model(f.to(device))
            preds_idx = torch.argmax(logits, dim=1).cpu().numpy()
            
            preds_labels = le.inverse_transform(preds_idx)
            
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds_labels)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n🏆 ACCURACY EN {dataset_name.upper()}: {acc*100:.2f}%")

# --- 6. EJECUCIÓN ---
evaluate_dataset(HISTORIC_FEATURES_CSV, "Datos del Pasado (Entrenamiento/Validación)")
evaluate_dataset(DRIFT_FEATURES_CSV, "Datos del Futuro (Concept Drift)")