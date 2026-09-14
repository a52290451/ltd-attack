"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: EXP6_Train_DML_InclinadoV.py
🚀 VERSIÓN: 5.0 (Multi-Task Joint Loss + AMP bfloat16 + torch.compile)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Transición al paradigma de Aprendizaje Métrico Profundo (DML) con estabilización
    multi-tarea. Se optimiza el espacio latente mediante Supervised Contrastive Loss
    (SupCon) + Cross-Entropy como señal de anclaje para prevenir colapso dimensional.

    [!] ACTUALIZACIÓN V5.0 (AUDITORÍA CLINE — ESTABILIZACIÓN MULTI-TAREA):
    - Multi-Task Joint Loss: CrossEntropy(logits, y) + λ · SupCon(embeddings, y) con λ=1.0
    - AMP nativo: torch.amp.autocast("cuda", dtype=bfloat16) + GradScaler("cuda")
    - Optimización: torch.inference_mode() en validación, pin_memory=True en DataLoaders
    - Arquitectura: Cabeza clasificadora auxiliar sobre espacio latente normalizado (L2)
    - Seguridad: weights_only=True en carga de checkpoints, SetAttentionBlock con need_weights=False
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

SAVE_DIR = result_path('dml', 'resultados_DML_IncV')
os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Logger(os.path.join(SAVE_DIR, f"log_train_EXP6_{timestamp}.txt"))

print("\n" + "═"*70)
print("🛡️ EXP 6: ENTRENAMIENTO DML MULTI-TAREA (HÍBRIDO INCLINADO A VECTORES)")
print("═"*70)

MAX_LEN = 3000
K_CLASSES = 16
P_INSTANCES = 4
BATCH_SIZE = K_CLASSES * P_INSTANCES
EPOCHS = 60
LATENT_DIM = 256
LOSS_TEMPERATURE = 0.1
LEARNING_RATE = 5e-4
LAMBDA_SUPCON = 1.0          # Peso de la pérdida SupCon en el Joint Loss
MODALITY_DROPOUT_RATE = 0.3   # Probabilidad de anular features (inercia a vectores)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️ Dispositivo: {device}")
print(f"📐 Batch Size: {BATCH_SIZE} ({K_CLASSES} clases × {P_INSTANCES} instancias)")
print(f"🔧 AMP: bfloat16 | λ_SupCon: {LAMBDA_SUPCON} | Modality Dropout: {MODALITY_DROPOUT_RATE*100:.0f}%")


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
print("\n[1/5] Extrayendo datos y ADN Inmutable...")
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

joblib.dump(le, os.path.join(SAVE_DIR, "le_dml_incV.joblib"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_dml_incV.joblib"))

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

print("\n[2/5] Instanciando Samplers DML...")
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


# --- 4. ARQUITECTURA DML MULTI-TAREA ---
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
        attn, _ = self.mha(x, x, x, need_weights=False)
        return self.norm(x + attn)


class MegaHybridModelDML(nn.Module):
    """
    Extractor híbrido multimodal (CNN 1D + Transformer) para Deep Metric Learning.
    V5.0: Salida dual — (embeddings L2-normalizados, logits de clasificación).
    """
    def __init__(self, num_classes, input_dim_feat, d_vec=256, d_feat=64, latent_dim=256):
        super().__init__()
        # --- Rama de Vectores (Micro) ---
        self.dir_emb = nn.Embedding(3, d_vec)
        self.weight_proj = nn.Linear(1, d_vec)
        self.fusion_vec = nn.Linear(d_vec * 2, d_vec)
        self.conv = nn.Conv1d(d_vec, d_vec, 5, 2, 2)
        self.pos_vec = VecPositionalEncoding(d_vec)
        self.transformer_vec = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_vec, 8, 1024, 0.2, batch_first=True), 4
        )
        self.ln_vec = nn.LayerNorm(d_vec * 2)

        # --- Rama de Features (Macro 24h) ---
        self.feat_proj = nn.Linear(1, d_feat)
        self.temp_engine = TemporalEncoder(d_feat)
        self.agg_engine = nn.Sequential(
            SetAttentionBlock(d_feat), SetAttentionBlock(d_feat)
        )
        self.pool_feat = nn.AdaptiveAvgPool1d(1)

        # --- Proyector DML (Sin Dropout para estabilidad geométrica) ---
        self.projection_head = nn.Sequential(
            nn.Linear((d_vec * 2) + d_feat, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )

        # --- Cabeza de clasificación auxiliar (opera sobre el embedding normalizado) ---
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x_d, x_w, x_f):
        # --- Rama Vectores ---
        v = torch.cat(
            [self.dir_emb(x_d + 1), self.weight_proj(x_w.unsqueeze(-1))], dim=-1
        )
        v = self.fusion_vec(v).transpose(1, 2)
        v = self.conv(v).transpose(1, 2)
        v = self.transformer_vec(self.pos_vec(v))
        vec_out = self.ln_vec(torch.cat([v.mean(1), v.max(1)[0]], 1))

        # --- Rama Features ---
        b, f, t = x_f.shape
        ft = x_f.view(b * f, t, 1)
        ft = self.temp_engine(self.feat_proj(ft))
        ft = self.agg_engine(ft.view(b, f, -1))
        feat_out = self.pool_feat(ft.transpose(1, 2)).squeeze(-1)

        # --- Fusión y proyección ---
        latent_vector = self.projection_head(torch.cat([vec_out, feat_out], dim=1))

        # Normalización L2 obligatoria para SupCon Loss
        embeddings = F.normalize(latent_vector, p=2, dim=1)

        # Cabeza clasificadora sobre el embedding normalizado (prototipos en hiperesfera)
        logits = self.classifier(embeddings)

        return embeddings, logits


# --- 5. ENTRENAMIENTO MULTI-TAREA CON AMP ---
print("\n[3/5] ⚙️ Iniciando Optimización Multi-Tarea (SupCon + CrossEntropy) con AMP bfloat16...")
model = MegaHybridModelDML(num_classes, input_dim_features, latent_dim=LATENT_DIM).to(device)

# torch.compile para acelerar el grafo computacional (PyTorch 2.0+)
try:
    model = torch.compile(model, mode="reduce-overhead")
    print("   ✅ torch.compile activado (modo reduce-overhead).")
except Exception as e:
    print(f"   ⚠️ torch.compile no disponible ({e}), continuando en modo eager.")

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion_supcon = SupervisedContrastiveLoss(temperature=LOSS_TEMPERATURE).to(device)
criterion_ce = nn.CrossEntropyLoss().to(device)
scaler = GradScaler("cuda")

# Cosine schedule con warmup
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * 0.1)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

best_loss = float('inf')
history_train_loss, history_val_loss = [], []
history_train_ce, history_train_supcon = [], []

for epoch in range(EPOCHS):
    # ==================== ENTRENAMIENTO ====================
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_supcon = 0.0

    for d, w, f, y_batch in train_loader:
        optimizer.zero_grad(set_to_none=True)

        d_input, w_input = d.to(device, non_blocking=True), w.to(device, non_blocking=True)
        # Modality Dropout: 30% de probabilidad de anular features (inductancia a vectores)
        if np.random.rand() < MODALITY_DROPOUT_RATE:
            f_input = torch.zeros_like(f, device=device)
        else:
            f_input = f.to(device, non_blocking=True)
        y_target = y_batch.to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.bfloat16):
            embeddings, logits = model(d_input, w_input, f_input)
            loss_ce = criterion_ce(logits, y_target)
            loss_supcon = criterion_supcon(embeddings, y_target)
            loss = loss_ce + LAMBDA_SUPCON * loss_supcon

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_supcon += loss_supcon.item()

    n_batches = len(train_loader)
    avg_train_loss = total_loss / n_batches
    avg_train_ce = total_ce / n_batches
    avg_train_supcon = total_supcon / n_batches
    history_train_loss.append(avg_train_loss)
    history_train_ce.append(avg_train_ce)
    history_train_supcon.append(avg_train_supcon)

    # ==================== VALIDACIÓN ====================
    model.eval()
    val_loss = 0.0
    val_ce = 0.0
    val_supcon = 0.0
    with torch.inference_mode():
        for d, w, f, y_batch in test_loader:
            d_input, w_input = d.to(device, non_blocking=True), w.to(device, non_blocking=True)
            f_input = f.to(device, non_blocking=True)
            y_target = y_batch.to(device, non_blocking=True)

            with autocast("cuda", dtype=torch.bfloat16):
                embeddings, logits = model(d_input, w_input, f_input)
                loss_ce = criterion_ce(logits, y_target)
                loss_supcon = criterion_supcon(embeddings, y_target)
                loss = loss_ce + LAMBDA_SUPCON * loss_supcon

            val_loss += loss.item()
            val_ce += loss_ce.item()
            val_supcon += loss_supcon.item()

    n_val_batches = len(test_loader)
    avg_val_loss = val_loss / n_val_batches
    avg_val_ce = val_ce / n_val_batches
    avg_val_supcon = val_supcon / n_val_batches
    history_val_loss.append(avg_val_loss)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train Joint: {avg_train_loss:.4f} (CE: {avg_train_ce:.4f}, SupCon: {avg_train_supcon:.4f}) | "
        f"Val Joint: {avg_val_loss:.4f} (CE: {avg_val_ce:.4f}, SupCon: {avg_val_supcon:.4f})"
    )

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_model_dml_incV.pth")
        print(f"      ✅ Mejor modelo guardado (Val Loss: {best_loss:.4f})")

print(f"\n🏁 Entrenamiento completado. Mejor Val Loss: {best_loss:.4f}")


# --- 6. EXTRACCIÓN Y PRUEBA DE VIDA (ACCURACY L2) ---
print("\n[4/5] 💾 Extrayendo Embeddings de Entrenamiento y Validación...")
model.load_state_dict(
    torch.load(
        f"{SAVE_DIR}/best_model_dml_incV.pth", map_location=device, weights_only=True
    )
)
model.eval()

ext_loader = DataLoader(
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
    for d, w, f, _ in tqdm(ext_loader, desc="Proyectando Pasado (Train)"):
        emb, _ = model(d.to(device, non_blocking=True),
                       w.to(device, non_blocking=True),
                       f.to(device, non_blocking=True))
        train_embeddings.append(emb.cpu().numpy())
train_embeddings = np.vstack(train_embeddings)

np.save(os.path.join(SAVE_DIR, 'embeddings_train.npy'), train_embeddings)
np.save(os.path.join(SAVE_DIR, 'labels_train.npy'), y[train_indices])

print("\n[5/5] 📊 Calculando Accuracy de Referencia L2 (Nearest Centroid)...")
# Construir Centroides con el conjunto de entrenamiento
centroids = {}
for label in np.unique(y[train_indices]):
    mask = y[train_indices] == label
    centroids[label] = train_embeddings[mask].mean(axis=0)

# Extraer embeddings del conjunto de test
test_loader2 = DataLoader(
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
    for d, w, f, _ in tqdm(test_loader2, desc="Proyectando Futuro (Test)"):
        emb, _ = model(d.to(device, non_blocking=True),
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


# --- 7. GRÁFICA 21: CURVAS DE CONVERGENCIA ---
print("\n📈 Generando Gráfica 21: Curvas de Convergencia Multi-Tarea...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

epochs_range = range(1, EPOCHS + 1)

# Subplot 1: Train/Val Joint Loss
ax1.plot(epochs_range, history_train_loss, 'b-', linewidth=2, label='Train Joint Loss')
ax1.plot(epochs_range, history_val_loss, 'r-', linewidth=2, label='Val Joint Loss')
ax1.set_xlabel('Época', fontsize=12)
ax1.set_ylabel('Pérdida Conjunta', fontsize=12)
ax1.set_title('EXP6: Convergencia Multi-Tarea (Joint Loss)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Subplot 2: Componentes CE y SupCon (Train)
ax2.plot(epochs_range, history_train_ce, 'g--', linewidth=2, label='Train CrossEntropy')
ax2.plot(epochs_range, history_train_supcon, 'm--', linewidth=2, label='Train SupCon Loss')
ax2.set_xlabel('Época', fontsize=12)
ax2.set_ylabel('Componentes de Pérdida', fontsize=12)
ax2.set_title('EXP6: Desglose de Señales (CE + λ·SupCon)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'G21_Curvas_Convergencia_MultiTarea.png'), dpi=150)
plt.close()
print(f"   ✅ Gráfica 21 guardada en {SAVE_DIR}/G21_Curvas_Convergencia_MultiTarea.png")

print("\n" + "═"*70)
print(f"🏆 EXP6 FINALIZADO CON ÉXITO | Accuracy L2: {accuracy_l2*100:.2f}% | Best Val Loss: {best_loss:.4f}")
print(f"📁 Resultados guardados en: {SAVE_DIR}/")
print("═"*70)
