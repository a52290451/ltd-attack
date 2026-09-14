"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: EXP7_Evaluar_Transfer_DML.py
🚀 VERSIÓN: 1.0.0 (Evaluación Concept Drift — Transfer Learning DML Two-Stage)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este script evalúa el modelo Two-Stage DML (EXP7) frente al Concept Drift.
    
    Stage 1: Encoder pre-entrenado del EXP4 (congelado) — CNN 1D + Transformer + 
              TemporalEncoder + SetAttention.
    Stage 2: Proyección DML entrenada con SupervisedContrastiveLoss (entrenable solo 
              en fine-tuning, ya congelada en evaluación).
    
    La clasificación se realiza por Nearest Centroid con similitud coseno 
    (producto punto sobre embeddings L2-normalizados).

    [!] NOVEDADES:
    - Carga de centroides desde embeddings_train.npy con normalización L2.
    - Inferencia con producto punto (coseno) en lugar de distancia L2.
    - Matriz de Confusión Post-Drift (Gráfica 22).
=========================================================================================
"""
import matplotlib
matplotlib.use('Agg')  # Backend seguro para servidores

import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
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
from src.utils.paths import data_path, result_path


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
FEATURES_LIST_TXT = result_path('macro', 'resultados_analisis', 'new_features_invariantes_seguras.txt')

VECTORS_DRIFT_CSV = data_path('historical', 'CLEAN_final_vectors_sites_concept_drift.csv')
FEATURES_DRIFT_CSV = data_path('historical', 'CLEAN_final_features_sites_concept_drift.csv')

TRAIN_RESULTS_DIR = result_path('dml', 'resultados_EXP7_Transfer_DML')
SAVE_DIR = TRAIN_RESULTS_DIR  # Guardamos en la misma carpeta de resultados del EXP7

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Logger(os.path.join(SAVE_DIR, f"log_eval_EXP7_Transfer_DML_{timestamp}.txt"))

print("\n" + "═"*70)
print("🛡️ EXP 7: EVALUACIÓN CONCEPT DRIFT (TRANSFER LEARNING DML — TWO-STAGE)")
print("═"*70)

MAX_LEN = 3000
BATCH_SIZE = 64
D_VEC = 256
D_FEAT = 64
LATENT_DIM = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Dispositivo de cómputo detectado: {device}")


# --- 2. CARGAR MEMORIA DEL PASADO ---
print("\n📖 [1/4] Cargando memoria y artefactos matemáticos del pasado...")
try:
    le_entrenamiento = joblib.load(os.path.join(TRAIN_RESULTS_DIR, "le_dml_transfer.joblib"))
    scaler = joblib.load(os.path.join(TRAIN_RESULTS_DIR, "scaler_dml_transfer.joblib"))

    sitios_elite_entrenados = set(le_entrenamiento.classes_.astype(str))

    with open(FEATURES_LIST_TXT, 'r') as f:
        features_inv = [line.strip() for line in f.readlines() if line.strip()]
    input_dim_features = len(features_inv)

    print(f"   ✅ Memoria recuperada: {input_dim_features} características invariantes.")
    print(f"   ✅ Clases entrenadas en el LabelEncoder: {len(sitios_elite_entrenados)} sitios de élite.")
except Exception as e:
    print(f"❌ Error crítico al cargar memoria: {e}")
    sys.exit()

# --- 2.1 CARGAR EMBEDDINGS Y CONSTRUIR CENTROIDES NORMALIZADOS ---
print("\n📐 Construyendo Centroides de Referencia (Normalizados L2)...")
try:
    embeddings_train = np.load(os.path.join(TRAIN_RESULTS_DIR, 'embeddings_train.npy'))
    labels_train = np.load(os.path.join(TRAIN_RESULTS_DIR, 'labels_train.npy'))

    centroids = {}
    for label in np.unique(labels_train):
        mask = labels_train == label
        centroid = embeddings_train[mask].mean(axis=0)
        # Normalización L2 del centroide
        centroids[label] = centroid / (np.linalg.norm(centroid) + 1e-9)

    # Matriz de centroides ordenada por clase para producto punto eficiente
    sorted_labels = sorted(centroids.keys())
    centroids_matrix = np.stack([centroids[lbl] for lbl in sorted_labels])
    centroid_labels = np.array(sorted_labels)

    print(f"   ✅ {len(centroids)} centroides construidos y normalizados (dimensión: {centroids_matrix.shape[1]}).")
    print(f"   🔍 Rango de clases: {sorted_labels[0]} → {sorted_labels[-1]}")
except Exception as e:
    print(f"❌ Error crítico al cargar embeddings/centroides: {e}")
    sys.exit()


# --- 3. PREPARACIÓN DATOS DEL FUTURO (DRIFT) ---
print("\n⏳ [2/4] Estructurando Features Macro del futuro (Extrayendo Ciclos 24h)...")
df_feat = pd.read_csv(FEATURES_DRIFT_CSV).dropna(subset=['site_label'])
df_feat['site_label'] = df_feat['site_label'].astype(str)

df_feat = df_feat[df_feat['site_label'].isin(sitios_elite_entrenados)].copy()


def parse_time_from_pcap(name_str):
    match = re.search(r'_(\d{8})-(\d{2})\d{4}', str(name_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


parsed = df_feat['pcap_name'].apply(parse_time_from_pcap)
df_feat['date_id'] = [p[0] for p in parsed]
df_feat['hour_bin'] = [p[1] for p in parsed]

# Transformación Matemática Segura (idéntica al entrenamiento)
df_feat[features_inv] = scaler.transform(df_feat[features_inv])

site_date_to_feature_tensor = {}
for (site, date), group in df_feat.groupby(['site_label', 'date_id']):
    tensor = np.zeros((input_dim_features, 24))
    for _, row in group.iterrows():
        hora = int(row['hour_bin'])
        if 0 <= hora < 24:
            tensor[:, hora] = row[features_inv].values
    site_date_to_feature_tensor[(site, date)] = tensor

print("⏳ [3/4] Alineando Vectores Micro del futuro...")
df_vec = pd.read_csv(VECTORS_DRIFT_CSV)

# Prevención de colisiones en el merge
cols_to_drop = [c for c in ['site_label', 'site'] if c in df_vec.columns]
if cols_to_drop:
    df_vec = df_vec.drop(columns=cols_to_drop)

df_bridge = pd.read_csv(
    FEATURES_DRIFT_CSV,
    usecols=['pcap_uid', 'site_label', 'pcap_name']
).drop_duplicates(subset=['pcap_uid'])

parsed_vec = df_bridge['pcap_name'].apply(parse_time_from_pcap)
df_bridge['date_id'] = [p[0] for p in parsed_vec]

df_vec = pd.merge(df_vec, df_bridge[['pcap_uid', 'site_label', 'date_id']], on='pcap_uid', how='inner')
df_vec['site_label'] = df_vec['site_label'].astype(str)
df_vec = df_vec[df_vec['site_label'].isin(sitios_elite_entrenados)].copy()


def preprocess_vector(v_str, max_len):
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list):
            v = []
    except Exception:
        v = []
    if len(v) > max_len:
        return v[:max_len]
    return v + [0] * (max_len - len(v))


print("⏳ [4/4] Cruzando tensores finales de inferencia...")
X_dir_list, X_w_list, X_feat_list, y_list = [], [], [], []
for _, row in tqdm(df_vec.iterrows(), total=len(df_vec), desc="Alineando modalidades"):
    site = str(row['site_label'])
    date = int(row['date_id'])
    if (site, date) in site_date_to_feature_tensor:
        X_dir_list.append(preprocess_vector(row['direction_vector'], MAX_LEN))
        X_w_list.append(preprocess_vector(row['size_vector'], MAX_LEN))
        X_feat_list.append(site_date_to_feature_tensor[(site, date)])
        y_list.append(le_entrenamiento.transform([site])[0])

X_d_ts, X_w_ts, X_f_ts, y_ts = (
    np.array(X_dir_list),
    np.array(X_w_list),
    np.array(X_feat_list),
    np.array(y_list),
)
X_w_ts = np.clip(X_w_ts / 1500.0, 0.0, 1.0)

# --- AUDITORÍA DE ALINEACIÓN (SANITY CHECK) ---
print("\n" + "🔎" * 20)
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
print("🔎" * 20 + "\n")

drift_loader = DataLoader(
    TensorDataset(
        torch.LongTensor(X_d_ts),
        torch.FloatTensor(X_w_ts),
        torch.FloatTensor(X_f_ts),
        torch.LongTensor(y_ts),
    ),
    batch_size=BATCH_SIZE,
    pin_memory=True,
)


# --- 4. ARQUITECTURA (IDÉNTICA AL ENTRENAMIENTO EXP4 + EXP7) ---
class VecPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TemporalEncoder(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model, 4, 128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, 2)

    def forward(self, x):
        return self.transformer(x).mean(dim=1)


class SetAttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn, _ = self.mha(x, x, x)
        return self.norm(x + attn)


class MegaHybridModel(nn.Module):
    """
    Réplica EXACTA del encoder usado en EXP4.
    Sin cabeza clasificadora — solo extracción de features.
    """
    def __init__(self, num_classes, input_dim_feat, d_vec=256, d_feat=64):
        super().__init__()
        self.dir_emb = nn.Embedding(3, d_vec)
        self.weight_proj = nn.Linear(1, d_vec)
        self.fusion_vec = nn.Linear(d_vec * 2, d_vec)
        self.conv = nn.Conv1d(d_vec, d_vec, kernel_size=7, stride=2, padding=3)
        self.pos_vec = VecPositionalEncoding(d_vec)
        self.transformer_vec = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_vec, 8, 1024, 0.2, batch_first=True), 4
        )
        self.ln_vec = nn.LayerNorm(d_vec * 2)

        self.feat_proj = nn.Linear(1, d_feat)
        self.temp_engine = TemporalEncoder(d_feat)
        self.agg_engine = nn.Sequential(
            SetAttentionBlock(d_feat), SetAttentionBlock(d_feat)
        )
        self.pool_feat = nn.AdaptiveAvgPool1d(1)

        # Clasificador original (no se usa en DML, pero necesario para cargar pesos)
        self.classifier = nn.Sequential(
            nn.Linear((d_vec * 2) + d_feat, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_d, x_w, x_f):
        # --- Rama Vectores ---
        v = torch.cat(
            [self.dir_emb(x_d + 1), self.weight_proj(x_w.unsqueeze(-1))], dim=-1
        )
        v = self.fusion_vec(v).transpose(1, 2)
        v = self.conv(v).transpose(1, 2)
        v = self.transformer_vec(self.pos_vec(v))
        vec_out = self.ln_vec(torch.cat([v.mean(1), v.max(1)[0]], 1))

        # --- Rama Features 24h ---
        b, f, t = x_f.shape
        ft = x_f.view(b * f, t, 1)
        ft = self.temp_engine(self.feat_proj(ft))
        ft = self.agg_engine(ft.view(b, f, -1))
        feat_out = self.pool_feat(ft.transpose(1, 2)).squeeze(-1)

        return self.classifier(torch.cat([vec_out, feat_out], dim=1))


class MegaHybridModelDML_Transfer(nn.Module):
    """
    Wrapper DML para Transfer Learning.
    Encoder congelado del EXP4 + projection_head entrenada con SupCon.
    forward() retorna embeddings L2-normalizados.
    """
    def __init__(self, pretrained_model, latent_dim=256):
        super().__init__()
        # Copiar componentes del encoder pre-entrenado
        self.dir_emb = pretrained_model.dir_emb
        self.weight_proj = pretrained_model.weight_proj
        self.fusion_vec = pretrained_model.fusion_vec
        self.conv = pretrained_model.conv
        self.pos_vec = pretrained_model.pos_vec
        self.transformer_vec = pretrained_model.transformer_vec
        self.ln_vec = pretrained_model.ln_vec

        self.feat_proj = pretrained_model.feat_proj
        self.temp_engine = pretrained_model.temp_engine
        self.agg_engine = pretrained_model.agg_engine
        self.pool_feat = pretrained_model.pool_feat

        # Proyección DML entrenada (congelada en evaluación, ya tiene los pesos cargados)
        self.projection_head = nn.Sequential(
            nn.Linear(
                (pretrained_model.dir_emb.embedding_dim * 2)
                + pretrained_model.feat_proj.out_features,
                512,
            ),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x_d, x_w, x_f):
        # --- Rama Vectores ---
        v = torch.cat(
            [self.dir_emb(x_d + 1), self.weight_proj(x_w.unsqueeze(-1))], dim=-1
        )
        v = self.fusion_vec(v).transpose(1, 2)
        v = self.conv(v).transpose(1, 2)
        v = self.transformer_vec(self.pos_vec(v))
        vec_out = self.ln_vec(torch.cat([v.mean(1), v.max(1)[0]], 1))

        # --- Rama Features 24h ---
        b, f, t = x_f.shape
        ft = x_f.view(b * f, t, 1)
        ft = self.temp_engine(self.feat_proj(ft))
        ft = self.agg_engine(ft.view(b, f, -1))
        feat_out = self.pool_feat(ft.transpose(1, 2)).squeeze(-1)

        # --- Proyección DML ---
        latent_vector = self.projection_head(torch.cat([vec_out, feat_out], dim=1))

        # Normalización L2 obligatoria
        embeddings = F.normalize(latent_vector, p=2, dim=1)

        return embeddings


# --- 5. CARGA DEL MODELO ENTRENADO ---
print("🔌 Construyendo arquitectura Two-Stage DML y cargando pesos...")

# Instanciar encoder base con las dimensiones correctas
num_classes = len(sitios_elite_entrenados)
base_model = MegaHybridModel(num_classes, input_dim_features, d_vec=D_VEC, d_feat=D_FEAT)

# Envolver en el wrapper DML (los pesos se sobreescribirán con el .pth)
model = MegaHybridModelDML_Transfer(base_model, latent_dim=LATENT_DIM).to(device)

try:
    checkpoint_path = os.path.join(TRAIN_RESULTS_DIR, "best_model_dml_transfer.pth")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state)
    print("   ✅ Pesos Two-Stage DML restaurados correctamente.")
except Exception as e:
    print(f"   ❌ Error al cargar best_model_dml_transfer.pth: {e}")
    sys.exit()

model.eval()

# --- 6. INFERENCIA: EXTRACCIÓN DE EMBEDDINGS + NEAREST CENTROID (COSENO) ---
print("\n🔮 Proyectando tráfico del futuro al espacio latente DML...")
all_embeddings = []
y_true = []

with torch.inference_mode():
    for d, w, f, y_batch in tqdm(drift_loader, desc="Extrayendo embeddings del futuro"):
        d_input = d.to(device, non_blocking=True)
        w_input = w.to(device, non_blocking=True)
        f_input = f.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embeddings = model(d_input, w_input, f_input)

        all_embeddings.append(embeddings.cpu().numpy())
        y_true.extend(y_batch.numpy())

if len(y_true) == 0:
    print("❌ Error: No se cruzó ninguna muestra. Revisa las fechas/etiquetas en los CSV de drift.")
    sys.exit()

all_embeddings = np.vstack(all_embeddings)
y_true = np.array(y_true)

print(f"   ✅ Embeddings extraídos: {all_embeddings.shape[0]} muestras × {all_embeddings.shape[1]} dimensiones.")

# --- Clasificación Nearest Centroid con Similitud Coseno (producto punto) ---
# Los embeddings ya están normalizados L2 por el modelo.
# Los centroides también están normalizados L2.
# Producto punto = similitud coseno.
similarities = np.dot(all_embeddings, centroids_matrix.T)  # (N_muestras, N_clases)
y_pred = centroid_labels[np.argmax(similarities, axis=1)]

accuracy = accuracy_score(y_true, y_pred)
print("\n" + "🏆" * 20)
print(f"   ACCURACY FINAL (TRANSFER DML vs DRIFT): {accuracy * 100:.2f}%")
print("🏆" * 20 + "\n")


# --- 7. GUARDAR ACCURACY EN ARCHIVO .TXT ---
accuracy_path = os.path.join(SAVE_DIR, "accuracy_EXP7_Transfer_DML.txt")
with open(accuracy_path, "w") as f:
    f.write(f"EXP7_Transfer_DML_Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Total muestras evaluadas: {len(y_true)}\n")
    f.write(f"Timestamp: {timestamp}\n")
print(f"📄 Accuracy guardado en: {accuracy_path}")


# --- 8. MATRIZ DE CONFUSIÓN NORMALIZADA (GRÁFICA 22) ---
print("🎨 Generando Gráfica 22: Matriz de Confusión del Concept Drift (Transfer DML)...")

cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm_normalized,
    cmap="Blues",
    cbar_kws={'label': 'Proporción de Predicciones'},
    xticklabels=False,
    yticklabels=False,
)

plt.title(
    'G22: Matriz de Confusión Post-Drift\n(Transfer Learning DML Two-Stage vs Futuro)',
    fontsize=15,
    fontweight='bold',
    pad=15,
)
plt.xlabel('Predicción del Modelo (Sitio Inferido)', fontsize=12, fontweight='bold')
plt.ylabel('Verdad Terreno (Sitio Real)', fontsize=12, fontweight='bold')

plt.tight_layout()
grafica_path = os.path.join(SAVE_DIR, 'G22_Matriz_Drift_Transfer_DML.png')
plt.savefig(grafica_path, dpi=300)
plt.close()

print(f"✅ Evidencia visual guardada exitosamente en: {grafica_path}")
print("=" * 70)
