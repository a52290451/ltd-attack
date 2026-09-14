"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: EXP7_Train_Transfer_DML.py
🚀 VERSIÓN: 1.0 (Two-Stage DML: Transfer Learning + Fine-Tuning Geométrico)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Two-Stage Deep Metric Learning. Se carga el extractor híbrido pre-entrenado del EXP4
    (96% accuracy en 65 clases vía Cross-Entropy), se reemplaza la cabeza clasificadora
    por una projection_head para SupCon Loss, y se realiza Fine-Tuning geométrico
    exclusivamente sobre el espacio latente.

    Pipeline:
      Stage 1 (EXP4): Clasificación supervisada → extractor de features robusto.
      Stage 2 (EXP7): Congelar encoder → entrenar proyector DML con SupCon Loss.

    [!] CONFIGURACIÓN:
    - Pesos pre-entrenados: result_path('hybrid', 'resultados_94F3000V_IncV', ...)
    - Épocas: 30 (solo se entrena la cabeza de proyección)
    - Sin Modality Dropout (la red ya sabe extraer features)
    - AMP bfloat16 + GradScaler + inference_mode + pin_memory
=========================================================================================
"""
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import Sampler
from transformers import get_cosine_schedule_with_warmup
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
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


# --- 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
FEATURES_LIST_TXT = result_path('macro', 'resultados_analisis', 'new_features_invariantes_seguras.txt')
HISTORIC_FEATURES_CSV = data_path('historical', 'CLEAN_final_features_sites.csv')
CACHED_VECTORS_CSV = data_path('historical', 'cached_vectors_with_dates.csv')
VECTOR_LE_PATH = artifact_path('ds3_label_encoder_vec_3000.joblib')

# Ruta exacta de los pesos pre-entrenados del EXP4
PRETRAINED_PATH = result_path('hybrid', 'resultados_94F3000V_IncV', 'best_model_hibrido_inclinado_vec.pth')

SAVE_DIR = result_path('dml', 'resultados_EXP7_Transfer_DML')
os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Logger(os.path.join(SAVE_DIR, f"log_train_EXP7_{timestamp}.txt"))

print("\n" + "═"*70)
print("🛡️ EXP 7: TWO-STAGE DML (TRANSFER LEARNING + FINE-TUNING GEOMÉTRICO)")
print("═"*70)

MAX_LEN = 3000
K_CLASSES = 16
P_INSTANCES = 4
BATCH_SIZE = K_CLASSES * P_INSTANCES
EPOCHS = 30
LATENT_DIM = 256
LOSS_TEMPERATURE = 0.1
LEARNING_RATE = 1e-3                 # LR más alto: solo se entrena la cabeza de proyección
D_VEC = 256
D_FEAT = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️ Dispositivo: {device}")
print(f"📐 Batch Size: {BATCH_SIZE} ({K_CLASSES} clases × {P_INSTANCES} instancias)")
print(f"🔧 AMP: bfloat16 | Épocas: {EPOCHS} | LR: {LEARNING_RATE} | Latent Dim: {LATENT_DIM}")
print(f"📦 Pesos pre-entrenados: {PRETRAINED_PATH}")


# --- 2. HERRAMIENTAS DML: SAMPLER Y LOSS ---
class PKBatchSampler(Sampler):
    """Muestreador P-K: selecciona K clases y P instancias por clase en cada batch.
       OBLIGATORIO para que SupCon Loss funcione correctamente."""
    def __init__(self, labels, k_classes, p_instances):
        self.labels = labels
        self.k_classes = k_classes
        self.p_instances = p_instances
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        self.unique_classes = list(self.label_to_indices.keys())

    def __iter__(self):
        np.random.shuffle(self.unique_classes)
        for label in self.unique_classes:
            np.random.shuffle(self.label_to_indices[label])
        batch = []
        for i in range(0, len(self.unique_classes), self.k_classes):
            selected_classes = self.unique_classes[i : i + self.k_classes]
            if len(selected_classes) < self.k_classes:
                break
            for cls in selected_classes:
                indices = self.label_to_indices[cls]
                if len(indices) >= self.p_instances:
                    batch.extend(indices[:self.p_instances])
                else:
                    batch.extend(np.random.choice(indices, self.p_instances, replace=True))
            yield batch
            batch = []

    def __len__(self):
        return len(self.unique_classes) // self.k_classes


class SupervisedContrastiveLoss(nn.Module):
    """
    Fórmula canónica de Khosla et al. (NeurIPS 2020) con Log-Sum-Exp Trick.
    Loss = -1/|P(i)| Σ_{p∈P(i)} log[ exp(z_i·z_p / τ) / Σ_{a∈A(i)} exp(z_i·z_a / τ) ]
    donde P(i) son los positivos (misma clase, excluyendo i) y A(i) todos los demás.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Matriz de similitud coseno escalada por temperatura
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Eliminar auto-contraste (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # Log-Sum-Exp: log_prob = log[ exp(sim) / Σ exp(sim) ]
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Promedio sobre todos los positivos de cada ancla
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        loss = -mean_log_prob_pos
        return loss.mean()


# --- 3. CARGA DE DATOS ---
print("\n[1/7] Extrayendo datos y ADN Inmutable...")
try:
    le_vectores_maestro = joblib.load(VECTOR_LE_PATH)
    sitios_elite = set(le_vectores_maestro.classes_.astype(str))

    with open(FEATURES_LIST_TXT, 'r') as f:
        features_inv = [line.strip() for line in f.readlines() if line.strip()]
    input_dim_features = len(features_inv)
    print(f"   ✅ Cargadas {input_dim_features} features de élite.")
except Exception as e:
    print(f"❌ Error al inicializar: {e}")
    sys.exit()

df_feat = pd.read_csv(HISTORIC_FEATURES_CSV).dropna(subset=['site_label'])
df_feat['site_label'] = df_feat['site_label'].astype(str)
df_feat = df_feat[df_feat['site_label'].isin(sitios_elite)].copy()


def parse_time_from_pcap(name_str):
    match = re.search(r'_(\d{8})-(\d{2})\d{4}', str(name_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


parsed = df_feat['pcap_name'].apply(parse_time_from_pcap)
df_feat['date_id'] = [p[0] for p in parsed]
df_feat['hour_bin'] = [p[1] for p in parsed]

df_vec = pd.read_csv(CACHED_VECTORS_CSV)
df_vec['site_label'] = df_vec['site_label'].astype(str)
df_vec = df_vec[df_vec['site_label'].isin(sitios_elite)].copy()


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


le = LabelEncoder()
df_feat['target'] = le.fit_transform(df_feat['site_label'])
num_classes = len(le.classes_)
print(f"   ✅ Clases únicas (sitios): {num_classes}")

scaler = StandardScaler()
df_feat[features_inv] = scaler.fit_transform(df_feat[features_inv])

joblib.dump(le, os.path.join(SAVE_DIR, "le_dml_transfer.joblib"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_dml_transfer.joblib"))

site_date_to_feature_tensor = {}
for (site, date), group in df_feat.groupby(['site_label', 'date_id']):
    tensor = np.zeros((input_dim_features, 24))
    for _, row in group.iterrows():
        hora = int(row['hour_bin'])
        if 0 <= hora < 24:
            tensor[:, hora] = row[features_inv].values
    site_date_to_feature_tensor[(site, date)] = tensor

X_dir_list, X_w_list, X_feat_list, y_list, dates_list = [], [], [], [], []
for _, row in tqdm(df_vec.iterrows(), total=len(df_vec), desc="Cruzando tensores"):
    site = str(row['site_label'])
    date = int(row['date_id'])
    if (site, date) in site_date_to_feature_tensor:
        X_dir_list.append(preprocess_vector(row['direction_vector'], MAX_LEN))
        X_w_list.append(preprocess_vector(row['size_vector'], MAX_LEN))
        X_feat_list.append(site_date_to_feature_tensor[(site, date)])
        y_list.append(le.transform([site])[0])
        dates_list.append(date)

X_dir, X_weight, X_feat, y = (
    np.array(X_dir_list), np.array(X_w_list),
    np.array(X_feat_list), np.array(y_list)
)
X_weight = np.clip(X_weight / 1500.0, 0.0, 1.0)

# División train/test cronológica (80/20 por sitio, sin fuga temporal)
meta_df = pd.DataFrame({'site_label': y, 'date_id': dates_list, 'idx': range(len(y))})
meta_df = meta_df.sort_values(by=['site_label', 'date_id'])
train_indices, test_indices = [], []

for site, group in meta_df.groupby('site_label'):
    split = int(len(group) * 0.8)
    train_indices.extend(group['idx'].iloc[:split].tolist())
    test_indices.extend(group['idx'].iloc[split:].tolist())

print(f"   ✅ Train: {len(train_indices)} muestras | Test: {len(test_indices)} muestras")

print("\n[2/7] Instanciando Samplers DML...")
pk_sampler_train = PKBatchSampler(y[train_indices], K_CLASSES, P_INSTANCES)

train_loader = DataLoader(
    TensorDataset(
        torch.LongTensor(X_dir[train_indices]),
        torch.FloatTensor(X_weight[train_indices]),
        torch.FloatTensor(X_feat[train_indices]),
        torch.LongTensor(y[train_indices])
    ),
    batch_sampler=pk_sampler_train,
    pin_memory=True,
)

test_loader = DataLoader(
    TensorDataset(
        torch.LongTensor(X_dir[test_indices]),
        torch.FloatTensor(X_weight[test_indices]),
        torch.FloatTensor(X_feat[test_indices]),
        torch.LongTensor(y[test_indices])
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
)


# --- 4. ARQUITECTURAS ---
# 4.1. Componentes del extractor (idénticos a EXP4)
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


# 4.2. Réplica exacta de MegaHybridModel (EXP4) — solo para cargar pesos
class MegaHybridModel(nn.Module):
    """Extractor híbrido entrenado en EXP4 con 96% accuracy en 65 clases."""
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
        self.agg_engine = nn.Sequential(SetAttentionBlock(d_feat), SetAttentionBlock(d_feat))
        self.pool_feat = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear((d_vec * 2) + d_feat, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
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

        return self.classifier(torch.cat([vec_out, feat_out], dim=1))


# 4.3. Modelo DML para Transfer Learning: mismo encoder + cabeza de proyección
class MegaHybridModelDML_Transfer(nn.Module):
    """
    Wrapper DML con encoder congelado del EXP4 + projection_head entrenable.
    forward() retorna el vector latente normalizado L2.
    """
    def __init__(self, pretrained_model, latent_dim=256):
        super().__init__()
        # Copiar todos los componentes del encoder pre-entrenado
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

        # Congelar el encoder completo (CNN + Transformer + agregación de features)
        for param in self.parameters():
            param.requires_grad = False

        # Nueva cabeza de proyección DML (entrenable)
        self.projection_head = nn.Sequential(
            nn.Linear((pretrained_model.dir_emb.embedding_dim * 2) + pretrained_model.feat_proj.out_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x_d, x_w, x_f):
        # --- Rama Vectores (congelada) ---
        v = torch.cat([self.dir_emb(x_d + 1), self.weight_proj(x_w.unsqueeze(-1))], dim=-1)
        v = self.fusion_vec(v).transpose(1, 2)
        v = self.conv(v).transpose(1, 2)
        v = self.transformer_vec(self.pos_vec(v))
        vec_out = self.ln_vec(torch.cat([v.mean(1), v.max(1)[0]], 1))

        # --- Rama Features 24h (congelada) ---
        b, f, t = x_f.shape
        ft = x_f.view(b * f, t, 1)
        ft = self.temp_engine(self.feat_proj(ft))
        ft = self.agg_engine(ft.view(b, f, -1))
        feat_out = self.pool_feat(ft.transpose(1, 2)).squeeze(-1)

        # --- Proyección DML (entrenable) ---
        latent_vector = self.projection_head(torch.cat([vec_out, feat_out], dim=1))

        # Normalización L2 obligatoria para SupCon Loss
        embeddings = F.normalize(latent_vector, p=2, dim=1)

        return embeddings


# --- 5. CARGA DE PESOS PRE-ENTRENADOS Y CONSTRUCCIÓN DEL MODELO DML ---
print("\n[3/7] 🔄 Instanciando arquitectura híbrida (réplica EXP4)...")
base_model = MegaHybridModel(num_classes, input_dim_features, d_vec=D_VEC, d_feat=D_FEAT)

print(f"[4/7] 📥 Cargando pesos pre-entrenados desde: {PRETRAINED_PATH}")
if not os.path.exists(PRETRAINED_PATH):
    print(f"❌ ERROR: No se encuentra el archivo de pesos pre-entrenados en {PRETRAINED_PATH}")
    print("   Asegúrate de haber ejecutado EXP4_Train_94F3000V_InclinadoV.py primero.")
    sys.exit()

checkpoint = torch.load(PRETRAINED_PATH, map_location=device, weights_only=True)
clean_state = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
base_model.load_state_dict(clean_state)
print("   ✅ Pesos cargados exitosamente (Cross-Entropy head incluida pero será ignorada).")

print("[5/7] 🧬 Mutando a DML: reemplazando cabeza clasificadora por projection_head...")
model = MegaHybridModelDML_Transfer(base_model, latent_dim=LATENT_DIM).to(device)

# Verificar congelación
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   ✅ Encoder congelado: {frozen_params:,} parámetros")
print(f"   🔥 Proyección entrenable: {trainable_params:,} parámetros")

# torch.compile para acelerar inferencia del encoder congelado
try:
    model = torch.compile(model, mode="reduce-overhead")
    print("   ✅ torch.compile activado (modo reduce-overhead).")
except Exception as e:
    print(f"   ⚠️ torch.compile no disponible ({e}), continuando en modo eager.")


# --- 6. ENTRENAMIENTO DML (SOLO PROYECCIÓN) ---
print(f"\n[6/7] ⚙️ Fine-Tuning Geométrico (30 épocas, solo projection_head)...")
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
criterion_supcon = SupervisedContrastiveLoss(temperature=LOSS_TEMPERATURE).to(device)
scaler = GradScaler("cuda")

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * 0.1)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

best_loss = float('inf')
history_train_loss, history_val_loss = [], []

for epoch in range(EPOCHS):
    # ==================== ENTRENAMIENTO ====================
    model.train()
    total_loss = 0.0

    for d, w, f, y_batch in train_loader:
        optimizer.zero_grad(set_to_none=True)

        d_input = d.to(device, non_blocking=True)
        w_input = w.to(device, non_blocking=True)
        f_input = f.to(device, non_blocking=True)
        y_target = y_batch.to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.bfloat16):
            embeddings = model(d_input, w_input, f_input)
            loss = criterion_supcon(embeddings, y_target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    history_train_loss.append(avg_train_loss)

    # ==================== VALIDACIÓN ====================
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        for d, w, f, y_batch in test_loader:
            d_input = d.to(device, non_blocking=True)
            w_input = w.to(device, non_blocking=True)
            f_input = f.to(device, non_blocking=True)
            y_target = y_batch.to(device, non_blocking=True)

            with autocast("cuda", dtype=torch.bfloat16):
                embeddings = model(d_input, w_input, f_input)
                loss = criterion_supcon(embeddings, y_target)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(test_loader)
    history_val_loss.append(avg_val_loss)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train SupCon: {avg_train_loss:.4f} | Val SupCon: {avg_val_loss:.4f}")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_model_dml_transfer.pth")
        print(f"      ✅ Mejor modelo guardado (Val SupCon: {best_loss:.4f})")

print(f"\n🏁 Fine-Tuning completado. Mejor Val SupCon: {best_loss:.4f}")


# --- 7. EXTRACCIÓN DE EMBEDDINGS Y EVALUACIÓN ---
print("\n[7/7] 💾 Extrayendo Embeddings y calculando Accuracy L2...")
model.load_state_dict(
    torch.load(
        f"{SAVE_DIR}/best_model_dml_transfer.pth", map_location=device, weights_only=True
    )
)
model.eval()

# Extraer embeddings de entrenamiento
ext_train_loader = DataLoader(
    TensorDataset(
        torch.LongTensor(X_dir[train_indices]),
        torch.FloatTensor(X_weight[train_indices]),
        torch.FloatTensor(X_feat[train_indices]),
        torch.LongTensor(y[train_indices]),
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
)

train_embeddings = []
with torch.inference_mode():
    for d, w, f, _ in tqdm(ext_train_loader, desc="Proyectando Pasado (Train)"):
        emb = model(d.to(device, non_blocking=True),
                     w.to(device, non_blocking=True),
                     f.to(device, non_blocking=True))
        train_embeddings.append(emb.cpu().numpy())
train_embeddings = np.vstack(train_embeddings)

np.save(os.path.join(SAVE_DIR, 'embeddings_train.npy'), train_embeddings)
np.save(os.path.join(SAVE_DIR, 'labels_train.npy'), y[train_indices])

# Construir Centroides con el conjunto de entrenamiento
# Construir Centroides con el conjunto de entrenamiento
centroids = {}
for label in np.unique(y[train_indices]):
    mask = y[train_indices] == label
    centroid_crudo = train_embeddings[mask].mean(axis=0)
    # Re-normalizar el centroide hacia la superficie de la hiper-esfera L2
    centroids[label] = centroid_crudo / (np.linalg.norm(centroid_crudo) + 1e-9)

# Extraer embeddings del conjunto de test
ext_test_loader = DataLoader(
    TensorDataset(
        torch.LongTensor(X_dir[test_indices]),
        torch.FloatTensor(X_weight[test_indices]),
        torch.FloatTensor(X_feat[test_indices]),
        torch.LongTensor(y[test_indices]),
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
)

test_embeddings = []
with torch.inference_mode():
    for d, w, f, _ in tqdm(ext_test_loader, desc="Proyectando Futuro (Test)"):
        emb = model(d.to(device, non_blocking=True),
                     w.to(device, non_blocking=True),
                     f.to(device, non_blocking=True))
        test_embeddings.append(emb.cpu().numpy())
test_embeddings = np.vstack(test_embeddings)

np.save(os.path.join(SAVE_DIR, 'embeddings_test.npy'), test_embeddings)
np.save(os.path.join(SAVE_DIR, 'labels_test.npy'), y[test_indices])

# Clasificación Nearest Centroid (L2)
centroids_matrix = np.stack([centroids[lbl] for lbl in sorted(centroids.keys())])
centroid_labels = np.array(sorted(centroids.keys()))
test_labels = y[test_indices]

dists = np.linalg.norm(
    test_embeddings[:, np.newaxis, :] - centroids_matrix[np.newaxis, :, :], axis=2
)
pred_labels = centroid_labels[np.argmin(dists, axis=1)]
accuracy_l2 = accuracy_score(test_labels, pred_labels)
print(f"   🎯 Accuracy L2 (Nearest Centroid): {accuracy_l2 * 100:.2f}%")


# --- 8. GRÁFICA 21: CURVAS DE CONVERGENCIA ---
print("\n📈 Generando Gráfica 21: Convergencia Fine-Tuning DML (Two-Stage)...")
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

epochs_range = range(1, EPOCHS + 1)
plt.plot(epochs_range, history_train_loss, 'b-', linewidth=2.5, marker='o', markersize=5, label='Train SupCon Loss')
plt.plot(epochs_range, history_val_loss, 'r-', linewidth=2.5, marker='s', markersize=5, label='Val SupCon Loss')
plt.axhline(best_loss, color='gray', linestyle='--', alpha=0.5, label=f'Best Val Loss: {best_loss:.4f}')

plt.xlabel('Época', fontsize=12)
plt.ylabel('Supervised Contrastive Loss', fontsize=12)
plt.title('EXP7: Fine-Tuning DML — Transfer Learning desde EXP4 (96% Acc)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'G21_Curvas_Convergencia_Transfer_DML.png'), dpi=150)
plt.close()
print(f"   ✅ Gráfica 21 guardada en {SAVE_DIR}/G21_Curvas_Convergencia_Transfer_DML.png")

print("\n" + "═"*70)
print(f"🏆 EXP7 FINALIZADO CON ÉXITO | Accuracy L2: {accuracy_l2*100:.2f}% | Best Val SupCon: {best_loss:.4f}")
print(f"📁 Resultados guardados en: {SAVE_DIR}/")
print("═"*70)
