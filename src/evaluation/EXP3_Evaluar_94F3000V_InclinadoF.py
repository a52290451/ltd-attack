"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: EXP3_Evaluar_94F3000V_InclinadoF.py
🚀 VERSIÓN: 3.1.2 (Autopsia del Drift sobre Híbrido Inclinado a Features)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este script somete al modelo Híbrido entrenado con Modality Dropout (50% en Vectores)
    a la prueba de Concept Drift. 
    
    El objetivo es evaluar si "obligar" a la red a enfocarse en las 94 Invariantes
    durante el entrenamiento (apagando la rama Micro la mitad del tiempo) le otorgó
    mayor resiliencia para sobrevivir al envejecimiento de las firmas en el futuro,
    en comparación con el modelo neutro.

    [!] ACTUALIZACIÓN V3.1.2 (RIGOR DE ABLACIÓN):
    - Fix KeyError en Pandas: Prevención de colisiones de sufijos durante el merge.
    - Auditoría de Alineación: Sanity check de tensores integrado.
    - Evidencia Visual: Generación de la Matriz de Confusión Post-Drift (Gráfica 18).
=========================================================================================
"""
import matplotlib
matplotlib.use('Agg') # Backend seguro para servidores

import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
FEATURES_LIST_TXT = '../vectores_features/resultados_analisis/new_features_invariantes_seguras.txt'

VECTORS_DRIFT_CSV = '../../output/CLEAN_final_vectors_sites_concept_drift.csv'
FEATURES_DRIFT_CSV = '../../output/CLEAN_final_features_sites_concept_drift.csv'

SAVE_DIR = '../vectores_features_v2/resultados_94F3000V_IncF'

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Logger(os.path.join(SAVE_DIR, f"log_eval_EXP3_InclinadoF_{timestamp}.txt"))

print("\n" + "═"*70)
print("🛡️ EXP 3: EVALUACIÓN CONCEPT DRIFT (HÍBRIDO INCLINADO A FEATURES)")
print("═"*70)

MAX_LEN = 3000
BATCH_SIZE = 64
D_MODEL = 256
D_FEAT = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Dispositivo de cómputo detectado: {device}")

# --- 2. CARGAR MEMORIA DEL MODELO ---
print("\n📖 Cargando memoria y artefactos matemáticos del pasado...")
try:
    le_entrenamiento = joblib.load(os.path.join(SAVE_DIR, "le_hibrido_incF.joblib"))
    scaler = joblib.load(os.path.join(SAVE_DIR, "scaler_hibrido_incF.joblib"))
    
    sitios_elite_entrenados = set(le_entrenamiento.classes_.astype(str))
    
    with open(FEATURES_LIST_TXT, 'r') as f:
        features_inv = [line.strip() for line in f.readlines() if line.strip()]
    input_dim_features = len(features_inv)
    
    print(f"   ✅ Memoria recuperada: {input_dim_features} características invariantes.")
    print(f"   ✅ Clases entrenadas en el LabelEncoder: {len(sitios_elite_entrenados)} sitios de élite.")
except Exception as e:
    print(f"❌ Error crítico al cargar memoria: {e}")
    sys.exit()

# --- 3. PREPARACIÓN DATOS DEL FUTURO (DRIFT) ---
print("\n⏳ [1/3] Estructurando Features Macro del futuro (Extrayendo Ciclos 24h)...")
df_feat = pd.read_csv(FEATURES_DRIFT_CSV).dropna(subset=['site_label'])
df_feat['site_label'] = df_feat['site_label'].astype(str)

df_feat = df_feat[df_feat['site_label'].isin(sitios_elite_entrenados)].copy()

def parse_time_from_pcap(name_str):
    match = re.search(r'_(\d{8})-(\d{2})\d{4}', str(name_str))
    if match: return int(match.group(1)), int(match.group(2))
    return 0, 0

parsed = df_feat['pcap_name'].apply(parse_time_from_pcap)
df_feat['date_id'] = [p[0] for p in parsed]
df_feat['hour_bin'] = [p[1] for p in parsed]

# Transformación Matemática Segura
df_feat[features_inv] = scaler.transform(df_feat[features_inv]) 

site_date_to_feature_tensor = {}
for (site, date), group in df_feat.groupby(['site_label', 'date_id']):
    tensor = np.zeros((input_dim_features, 24))
    for _, row in group.iterrows():
        hora = int(row['hour_bin'])
        if 0 <= hora < 24:
            tensor[:, hora] = row[features_inv].values
    site_date_to_feature_tensor[(site, date)] = tensor

print("⏳ [2/3] Alineando Vectores Micro del futuro...")
df_vec = pd.read_csv(VECTORS_DRIFT_CSV)

# Prevención de colisiones en el merge
cols_to_drop = [c for c in ['site_label', 'site'] if c in df_vec.columns]
if cols_to_drop:
    df_vec = df_vec.drop(columns=cols_to_drop)

df_bridge = pd.read_csv(FEATURES_DRIFT_CSV, usecols=['pcap_uid', 'site_label', 'pcap_name']).drop_duplicates(subset=['pcap_uid'])

parsed_vec = df_bridge['pcap_name'].apply(parse_time_from_pcap)
df_bridge['date_id'] = [p[0] for p in parsed_vec]

df_vec = pd.merge(df_vec, df_bridge[['pcap_uid', 'site_label', 'date_id']], on='pcap_uid', how='inner')
df_vec['site_label'] = df_vec['site_label'].astype(str)
df_vec = df_vec[df_vec['site_label'].isin(sitios_elite_entrenados)].copy()

def preprocess_vector(v_str, max_len):
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list): v = []
    except: v = []
    if len(v) > max_len: return v[:max_len]
    return v + [0] * (max_len - len(v))

print("⏳ [3/3] Cruzando tensores finales de inferencia...")
X_dir_list, X_w_list, X_feat_list, y_list = [], [], [], []
for _, row in tqdm(df_vec.iterrows(), total=len(df_vec), desc="Alineando modalidades"):
    site = str(row['site_label'])
    date = int(row['date_id'])
    if (site, date) in site_date_to_feature_tensor:
        X_dir_list.append(preprocess_vector(row['direction_vector'], MAX_LEN))
        X_w_list.append(preprocess_vector(row['size_vector'], MAX_LEN))
        X_feat_list.append(site_date_to_feature_tensor[(site, date)])
        y_list.append(le_entrenamiento.transform([site])[0])

X_d_ts, X_w_ts, X_f_ts, y_ts = np.array(X_dir_list), np.array(X_w_list), np.array(X_feat_list), np.array(y_list)
X_w_ts = np.clip(X_w_ts / 1500.0, 0.0, 1.0)

# --- AUDITORÍA DE ALINEACIÓN (SANITY CHECK) ---
print("\n" + "🔎"*20)
print("   AUDITORÍA DE ALINEACIÓN DE TENSORES")
print(f"   -> Total de muestras cruzadas (Match): {len(y_ts)}")
print(f"   -> Tensor Direcciones (Micro): {X_d_ts.shape}")
print(f"   -> Tensor Tamaños (Micro):     {X_w_ts.shape}")
print(f"   -> Tensor Features (Macro):    {X_f_ts.shape}")
print(f"   -> Vector Etiquetas (Target):  {y_ts.shape}")

if len(y_ts) > 0:
    sample_idx = 0
    site_decoded = le_entrenamiento.inverse_transform([y_ts[sample_idx]])[0]
    print(f"\n   [Radiografía de la Muestra 0]")
    print(f"   - Clase original (Site): {site_decoded} | ID codificado: {y_ts[sample_idx]}")
    print(f"   - Longitud Vector Dirección: {len(X_d_ts[sample_idx])} | Longitud Vector Tamaño: {len(X_w_ts[sample_idx])}")
    print(f"   - Dimensiones de Matriz Macro (Features x 24h): {X_f_ts[sample_idx].shape}")
print("🔎"*20 + "\n")

drift_loader = DataLoader(TensorDataset(torch.LongTensor(X_d_ts), torch.FloatTensor(X_w_ts), torch.FloatTensor(X_f_ts), torch.LongTensor(y_ts)), batch_size=BATCH_SIZE, pin_memory=True)

# --- 4. ARQUITECTURA (IDÉNTICA AL ENTRENAMIENTO) ---
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
        self.conv = nn.Conv1d(d_vec, d_vec, kernel_size=7, stride=2, padding=3)
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

# --- 5. INFERENCIA DEL FUTURO ---
print("🔌 Despertando red neuronal híbrida (Inclinada a Features)...")
model = MegaHybridModel(len(sitios_elite_entrenados), input_dim_features).to(device)
try:
    raw_state = torch.load(os.path.join(SAVE_DIR, "best_model_hibrido_inclinadoF.pth"), map_location=device, weights_only=True)
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
    model.load_state_dict(clean_state)
    print("   ✅ Pesos restaurados correctamente.")
except Exception as e:
    print(f"   ❌ Error al cargar .pth: {e}")
    sys.exit()

model.eval()
y_true, y_pred = [], []

print("   ⏳ Analizando tráfico del futuro...")
with torch.inference_mode():
    for d, w, f, y_batch in drift_loader:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(d.to(device), w.to(device), f.to(device))
        preds_idx = torch.argmax(logits, dim=1).cpu().numpy()
        
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds_idx)

if len(y_true) == 0:
    print("❌ Error: No se cruzó ninguna muestra. Revisa las fechas/etiquetas en los CSV de drift.")
    sys.exit()

acc = accuracy_score(y_true, y_pred)
print("\n" + "🏆"*20)
print(f"   ACCURACY FINAL (HÍBRIDO INCLINADO A FEATURES vs DRIFT): {acc*100:.2f}%")
print("🏆"*20 + "\n")

# --- 6. GENERACIÓN DE EVIDENCIA VISUAL (GRÁFICA 18) ---
print("🎨 Generando Gráfica 18: Matriz de Confusión del Concept Drift...")

cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized) 

plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, cmap="Purples", cbar_kws={'label': 'Proporción de Predicciones'})

plt.title('Colapso de Fronteras: Matriz de Confusión Post-Drift\n(Modelo Inclinado a Features vs Futuro)', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Predicción del Modelo (Sitio Inferido)', fontsize=12, fontweight='bold')
plt.ylabel('Verdad Terreno (Sitio Real)', fontsize=12, fontweight='bold')

plt.tight_layout()
grafica_path = os.path.join(SAVE_DIR, 'G18_Matriz_Drift_Hibrido_InclinadoF.png')
plt.savefig(grafica_path, dpi=300)
plt.close()

print(f"✅ Evidencia visual guardada exitosamente en: {grafica_path}")
print("===========================================================================\n")