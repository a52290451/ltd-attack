"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: EXP6_Evaluar_DML_InclinadoV.py
🚀 VERSIÓN: 4.0.1 (Fix Dimensionalidad: Alineación exacta de pesos con el Train)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Evaluación final del paradigma Deep Metric Learning (DML) frente al Concept Drift. 
    Se cargan los embeddings del pasado para consolidar Prototipos de Identidad de Clase 
    (Centroides normalizados en la hiper-esfera L2). 
    
    Posteriormente, las secuencias complejas del futuro (+2 meses) son proyectadas por 
    el HourlyEncoderDML híbrido y clasificadas dinámicamente mediante proximidad angular 
    (Similitud Coseno / Nearest Centroid), rompiendo la rigidez de las fronteras absolutas.
=========================================================================================
"""
import matplotlib
matplotlib.use('Agg') # Backend seguro para ejecución desatendida en servidores SSH

import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ast
import os
import joblib
import sys
import datetime
import re
from src.utils.paths import data_path, artifact_path, result_path

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

# --- 1. CONFIGURACIÓN DE RUTAS Y ARTEFACTOS ---
FEATURES_LIST_TXT = result_path('macro', 'resultados_analisis', 'new_features_invariantes_seguras.txt')
VECTOR_LE_PATH = artifact_path('ds3_label_encoder_vec_3000.joblib')

VECTORS_DRIFT_CSV = data_path('historical', 'CLEAN_final_vectors_sites_concept_drift.csv')
FEATURES_DRIFT_CSV = data_path('historical', 'CLEAN_final_features_sites_concept_drift.csv')

SAVE_DIR = result_path('dml', 'resultados_DML_IncV')

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Logger(os.path.join(SAVE_DIR, f"log_eval_EXP6_DML_{timestamp}.txt"))

print("\n" + "═"*70)
print("🛡️ EXP 6: EVALUACIÓN CONCEPT DRIFT (PARADIGMA DEEP METRIC LEARNING)")
print("═"*70)

MAX_LEN = 3000
BATCH_SIZE = 64
LATENT_DIM = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Entorno de ejecución configurado en hardware: {device}")

# --- 2. RECUPERAR MEMORIA HISTÓRICA Y CONSOLIDAR PROTOTIPOS L2 ---
print("\n[1/3] Cargando huella latente histórica y forjando Prototipos...")
try:
    le_entrenamiento = joblib.load(os.path.join(SAVE_DIR, "le_dml_incV.joblib"))
    scaler = joblib.load(os.path.join(SAVE_DIR, "scaler_dml_incV.joblib"))
    sitios_elite_entrenados = set(le_entrenamiento.classes_.astype(str))
    
    with open(FEATURES_LIST_TXT, 'r') as f:
        features_inv = [line.strip() for line in f.readlines() if line.strip()]
    input_dim_features = len(features_inv)
    
    # Ingesta del espacio geométrico del pasado (Fase 4 Homóloga)
    emb_train = np.load(os.path.join(SAVE_DIR, 'embeddings_train.npy'))
    lab_train = np.load(os.path.join(SAVE_DIR, 'labels_train.npy'))
    
    # Cálculo analítico de Centroides Prototípicos con re-normalización esférica
    centroids = {}
    for cls in np.unique(lab_train):
        cls_embeddings = emb_train[lab_train == cls]
        centroid = cls_embeddings.mean(axis=0)
        # Normalización L2 para asegurar coherencia en el cálculo de distancias cosenoidales
        centroids[cls] = centroid / (np.linalg.norm(centroid) + 1e-9)
        
    print(f"   ✅ Memoria del pasado restaurada: {input_dim_features} variables robustas.")
    print(f"   ✅ Forjados exitosamente {len(centroids)} Prototipos de Identidad de Clase.")
except Exception as e:
    print(f"❌ Error crítico en la restauración de artefactos del pasado: {e}")
    sys.exit()

# --- 3. PREPARACIÓN Y ALINEACIÓN DE TENSORES DEL FUTURO (DRIFT) ---
print("\n[2/3] Ingestando y sincronizando el dataset del Futuro (Concept Drift)...")
df_feat = pd.read_csv(FEATURES_DRIFT_CSV).dropna(subset=['site_label'])
df_feat['site_label'] = df_feat['site_label'].astype(str)

# Filtrado estricto de paridad (Ablación cerrada sobre los 65 sitios conocidos)
df_feat = df_feat[df_feat['site_label'].isin(sitios_elite_entrenados)].copy()

def parse_time_from_pcap(name_str):
    match = re.search(r'_(\d{8})-(\d{2})\d{4}', str(name_str))
    if match: return int(match.group(1)), int(match.group(2))
    return 0, 0

parsed = df_feat['pcap_name'].apply(parse_time_from_pcap)
df_feat['date_id'] = [p[0] for p in parsed]
df_feat['hour_bin'] = [p[1] for p in parsed]

# Transformación Z-Score usando parámetros del pasado (Cero fuga de información)
df_feat[features_inv] = scaler.transform(df_feat[features_inv]) 

site_date_to_feature_tensor = {}
for (site, date), group in df_feat.groupby(['site_label', 'date_id']):
    tensor = np.zeros((input_dim_features, 24))
    for _, row in group.iterrows():
        hora = int(row['hour_bin'])
        if 0 <= hora < 24:
            tensor[:, hora] = row[features_inv].values
    site_date_to_feature_tensor[(site, date)] = tensor

print(" Decodificando y empaquetando trazas Micro del futuro...")
df_vec = pd.read_csv(VECTORS_DRIFT_CSV)

# [!] PREVENCIÓN DE ERROR PANDAS MERGE: Limpieza de colisiones de sufijos (_x, _y)
cols_to_drop = [c for c in ['site_label', 'site'] if c in df_vec.columns]
if cols_to_drop: 
    df_vec = df_vec.drop(columns=cols_to_drop)

df_bridge = pd.read_csv(FEATURES_DRIFT_CSV, usecols=['pcap_uid', 'site_label', 'pcap_name']).drop_duplicates(subset=['pcap_uid'])
parsed_vec = df_bridge['pcap_name'].apply(parse_time_from_pcap)
df_bridge['date_id'] = [p[0] for p in parsed_vec]

# Fusión temporal Micro-Macro libre de fuga distributiva
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

X_dir_list, X_w_list, X_feat_list, y_list = [], [], [], []
for _, row in tqdm(df_vec.iterrows(), total=len(df_vec), desc="Sincronizando modalidades"):
    site = str(row['site_label'])
    date = int(row['date_id'])
    if (site, date) in site_date_to_feature_tensor:
        X_dir_list.append(preprocess_vector(row['direction_vector'], MAX_LEN))
        X_w_list.append(preprocess_vector(row['size_vector'], MAX_LEN))
        X_feat_list.append(site_date_to_feature_tensor[(site, date)])
        y_list.append(le_entrenamiento.transform([site])[0])

X_d_ts, X_w_ts, X_f_ts, y_ts = np.array(X_dir_list), np.array(X_w_list), np.array(X_feat_list), np.array(y_list)
X_w_ts = np.clip(X_w_ts / 1500.0, 0.0, 1.0)

# --- AUDITORÍA DE ALINEACIÓN METICULOSA (SANITY CHECK) ---
print("\n" + "🔎"*25)
print(" INFORME DE INTEGRIDAD Y ALINEACIÓN DE TENSORES (DRIFT)")
print(f"   -> Total de instancias multimodales consolidadas: {len(y_ts)}")
print(f"   -> Rama Micro - Matriz Direcciones: {X_d_ts.shape}")
print(f"   -> Rama Micro - Matriz Tamaños:     {X_w_ts.shape}")
print(f"   -> Rama Macro - Tensor Circadiano:  {X_f_ts.shape}")
print(f"   -> Vector Target - Etiquetas:       {y_ts.shape}")

if len(y_ts) > 0:
    site_decoded = le_entrenamiento.inverse_transform([y_ts[0]])[0]
    print(f"\n   [Rayos X de la Muestra 0 - Inferencia]")
    print(f"   - Identidad Terreno: {site_decoded} | ID Interno: {y_ts[0]}")
    print(f"   - Estructura Secuencial: {X_d_ts[0].shape} posiciones normalizadas.")
    print(f"   - Estructura Temporal Agregada: {X_f_ts[0].shape} (94 Features x 24 Horas)")
print("🔎"*20 + "\n")

drift_loader = DataLoader(TensorDataset(torch.LongTensor(X_d_ts), torch.FloatTensor(X_w_ts), torch.FloatTensor(X_f_ts), torch.LongTensor(y_ts)), batch_size=BATCH_SIZE)

# --- 4. ARQUITECTURA GEOMÉTRICA (CORREGIDA) ---
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

class MegaHybridModelDML(nn.Module):
    def __init__(self, input_dim_feat, d_vec=256, d_feat=64, latent_dim=256):
        super().__init__()
        self.dir_emb = nn.Embedding(3, d_vec) 
        self.weight_proj = nn.Linear(1, d_vec)
        self.fusion_vec = nn.Linear(d_vec * 2, d_vec)
        self.conv = nn.Conv1d(d_vec, d_vec, 5, 2, 2)
        self.pos_vec = VecPositionalEncoding(d_vec)
        self.transformer_vec = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_vec, 8, 1024, 0.2, batch_first=True), 4)
        self.ln_vec = nn.LayerNorm(d_vec * 2) 

        # [FIX] CORREGIDA LA DIMENSIONALIDAD PARA COINCIDIR EXACTAMENTE CON EL ENTRENAMIENTO
        self.feat_proj = nn.Linear(1, d_feat)
        self.temp_engine = TemporalEncoder(d_feat)
        self.agg_engine = nn.Sequential(SetAttentionBlock(d_feat), SetAttentionBlock(d_feat))
        self.pool_feat = nn.AdaptiveAvgPool1d(1) 

        self.projection_head = nn.Sequential(
            nn.Linear((d_vec * 2) + d_feat, 512), 
            nn.BatchNorm1d(512),
            nn.GELU(), 
            nn.Dropout(0.4), 
            nn.Linear(512, latent_dim)
        )

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
        
        latent_vector = self.projection_head(torch.cat([vec_out, feat_out], dim=1))
        return F.normalize(latent_vector, p=2, dim=1)

# --- 5. EJECUCIÓN INFERENCIA ANGULAR (NEAREST CENTROID) ---
print("[3/3] Congelando pesos y mapeando el futuro en la hiper-esfera esférica...")
model = MegaHybridModelDML(input_dim_features, latent_dim=LATENT_DIM).to(device)
try:
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model_dml_incV.pth"), map_location=device))
    print("   ✅ Parámetros de red cargados exitosamente.")
except Exception as e:
    print(f"   ❌ Error fatal al mapear pesos .pth: {e}")
    sys.exit()

model.eval()
future_embeddings, y_true = [], []

with torch.no_grad():
    for d, w, f, y_batch in drift_loader:
        emb = model(d.to(device), w.to(device), f.to(device))
        future_embeddings.append(emb.cpu().numpy())
        y_true.extend(y_batch.numpy())

future_embeddings = np.vstack(future_embeddings)

# Construcción de la matriz compacta de centroides históricos
centroid_labels = list(centroids.keys())
centroid_matrix = np.array([centroids[lbl] for lbl in centroid_labels]) # Dimensión: (65, 256)

print("   ⏳ Ejecutando matching angular vía producto punto...")
y_pred = []
for emb in future_embeddings:
    # Dado que ambos vectores están normalizados L2, el producto punto es la Similitud Coseno exacta
    similarities = np.dot(centroid_matrix, emb)
    best_match_idx = np.argmax(similarities)
    y_pred.append(centroid_labels[best_match_idx])

acc = accuracy_score(y_true, y_pred)
print("\n" + "🏆"*25)
print(f"   🚀 ACCURACY FINAL DML (HÍBRIDO INCLINADO VECTORES vs DRIFT): {acc*100:.2f}%")
print("🏆"*25 + "\n")

# Guardar métrica en crudo para reportes analíticos de la memoria
with open(os.path.join(SAVE_DIR, "resultado_drift_DML_IncV.txt"), "w") as f:
    f.write(f"ACCURACY DRIFT EXP6 DEEP METRIC LEARNING: {acc*100:.2f}%\n")

# --- 6. GENERACIÓN DE EVIDENCIA VISUAL EN EL FUTURO (GRÁFICA 22) ---
print("🎨 Generando Gráfica 22: Matriz de Confusión Geométrica (Nearest Centroid)...")

cm = confusion_matrix(y_true, y_pred)
cm_normalized = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

plt.figure(figsize=(13, 11))
sns.heatmap(cm_normalized, cmap="Greens", cbar_kws={'label': 'Proporción de Similitud Coseno Estructural'})

plt.title('Resiliencia Geométrica: Matriz de Confusión Post-Drift (+2 Meses)\n(Arquitectura Híbrida + Supervised Contrastive Learning)', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Sitio Web Inferido (Prototipo / Centroide L2 Más Cercano)', fontsize=12, fontweight='bold')
plt.ylabel('Sitio Web Real (Verdad Terreno Histórica)', fontsize=12, fontweight='bold')

plt.tight_layout()
grafica_path = os.path.join(SAVE_DIR, 'G22_Matriz_Drift_DML_IncV.png')
plt.savefig(grafica_path, dpi=300)
plt.close()

print(f"✅ Evidencia visual exportada exitosamente en: {grafica_path}")
print("===========================================================================\n")
