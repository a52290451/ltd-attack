from pathlib import Path
"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: EXP5_Train_94F3000V_CrossAttention.py
🚀 VERSIÓN: 3.2 (Híbrido Avanzado Gated Cross-Attention + AMP + torch.compile)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este experimento representa el cénit arquitectónico bajo el paradigma de 
    clasificación estática (Cross-Entropy). En lugar de concatenar vectores y features, 
    se implementa un mecanismo de GATED CROSS-ATTENTION.
    
    LÓGICA: La firma dura del paquete (Vector) actúa como "Query", consultando 
    dinámicamente el perfil de 24 horas (Features) como "Keys/Values" para 
    desempatar la incertidumbre generada por el Concept Drift.

    [!] ACTUALIZACIÓN V3.2 (OPTIMIZACIÓN DE RENDIMIENTO):
    - StandardScaler fitteado exclusivamente sobre train_indices (sin data leakage).
    - Conv1d con kernel_size=7 para mejor captura de patrones locales de ráfaga.
    - torch.compile("reduce-overhead") para reducir overhead del framework.
    - AMP bfloat16 + GradScaler para ~1.5× speedup y ~40% menos VRAM.
    - CosineAnnealingWarmRestarts nativo (sin dependencia de HuggingFace transformers).
    - Gradient Clipping (max_norm=1.0) para estabilidad.
    - torch.inference_mode() en evaluación.
    - pin_memory=True en DataLoaders.
=========================================================================================
"""
import matplotlib
matplotlib.use('Agg')  # Backend seguro para servidores sin interfaz gráfica

import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder
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


# --- 1. CONFIGURACIÓN DE RUTAS ---
FEATURES_LIST_TXT = str(Path.home() / 'ltd-attack/configs/features/MACRO-94.txt')
HISTORIC_FEATURES_CSV = str(Path.home() / 'ltd-storage/datasets/historical/CLEAN_final_features_sites.csv')
CACHED_VECTORS_CSV = str(Path.home() / 'ltd-storage/datasets/historical/cached_vectors_with_dates.csv')
VECTOR_LE_PATH = str(Path.home() / 'ltd-storage/artifacts/micro/MICRO-002-R/ds3_label_encoder_vec_3000.joblib')

SAVE_DIR = str(Path.home() / 'ltd-storage/results/canonical/HYB-006-R')
os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Logger(os.path.join(SAVE_DIR, f"log_train_EXP5_{timestamp}.txt"))

print("\n" + "═"*70)
print("🛡️ EXP 5 v3.2: ENTRENAMIENTO HÍBRIDO CON GATED CROSS-ATTENTION (24H PERFIL) + AMP + torch.compile")
print("═"*70)

MAX_LEN = 3000
BATCH_SIZE = 64
EPOCHS = 60
D_MODEL = 256
D_FEAT = 128  # Dimensionalidad latente ampliada para empatar mejor con el vector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. CARGA DE LAS 94 FEATURES Y PARIDAD ---
try:
    print("\n[1/5] Extrayendo filtro de paridad y ADN Inmutable...")
    le_vectores_maestro = joblib.load(VECTOR_LE_PATH)
    sitios_elite = set(le_vectores_maestro.classes_.astype(str))

    with open(FEATURES_LIST_TXT, 'r') as f:
        features_inv = [line.strip() for line in f.readlines() if line.strip()]
    input_dim_features = len(features_inv)
    print(f"   ✅ Cargadas {input_dim_features} features de élite.")
except Exception as e:
    print(f"❌ Error al inicializar: {e}")
    sys.exit()

# --- 3. PREPARACIÓN DE DATOS (SIN SCALER AÚN — SE FITEARÁ SOLO CON TRAIN) ---
print("\n⏳ [2/5] Procesando Features Macro purificadas...")
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

print("⏳ [3/5] Cargando y filtrando Vectores Micro...")
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


# LabelEncoder sobre df_feat completo (no causa leakage: solo mapea strings → ints)
le = LabelEncoder()
df_feat['target'] = le.fit_transform(df_feat['site_label'])

# --- Construir tensores de features SIN escalar (valores crudos) ---
site_date_to_feature_tensor = {}
for (site, date_val), group in df_feat.groupby(['site_label', 'date_id']):
    tensor = np.zeros((input_dim_features, 24))
    for _, row in group.iterrows():
        hora = int(row['hour_bin'])
        if 0 <= hora < 24:
            tensor[:, hora] = row[features_inv].values
    site_date_to_feature_tensor[(site, date_val)] = tensor

print("⏳ [4/5] Empaquetando tensores Híbridos (Vectores + Features sin escalar)...")
X_dir_list, X_w_list, X_feat_list, y_list, dates_list, site_str_list = [], [], [], [], [], []

for _, row in tqdm(df_vec.iterrows(), total=len(df_vec), desc="Cruzando datos"):
    site = str(row['site_label'])
    date_val = int(row['date_id'])
    if (site, date_val) in site_date_to_feature_tensor:
        X_dir_list.append(preprocess_vector(row['direction_vector'], MAX_LEN))
        X_w_list.append(preprocess_vector(row['size_vector'], MAX_LEN))
        X_feat_list.append(site_date_to_feature_tensor[(site, date_val)])
        y_list.append(le.transform([site])[0])
        dates_list.append(date_val)
        site_str_list.append(site)

X_dir = np.array(X_dir_list)
X_weight = np.array(X_w_list)
X_feat = np.array(X_feat_list)
y = np.array(y_list)

X_weight = np.clip(X_weight / 1500.0, 0.0, 1.0)

# --- División temporal cronológica (80/20 por sitio) ---
meta_df = pd.DataFrame({'site_label': y, 'date_id': dates_list, 'idx': range(len(y))})
meta_df = meta_df.sort_values(by=['site_label', 'date_id'])
train_indices, test_indices = [], []

for site_enc, group in meta_df.groupby('site_label'):
    split = int(len(group) * 0.8)
    train_indices.extend(group['idx'].iloc[:split].tolist())
    test_indices.extend(group['idx'].iloc[split:].tolist())

# --- ⚠️ FIX DATA LEAKAGE: StandardScaler fitteado SOLO con train ---
print("   🔒 Fitteando StandardScaler exclusivamente sobre train_indices...")
train_site_dates = set()
for idx in train_indices:
    train_site_dates.add((site_str_list[idx], dates_list[idx]))

train_mask = df_feat.apply(
    lambda r: (str(r['site_label']), int(r['date_id'])) in train_site_dates, axis=1
)

scaler = StandardScaler()
scaler.fit(df_feat.loc[train_mask, features_inv])

# Transformar X_feat in-place usando los parámetros del scaler
mean = scaler.mean_
std = scaler.scale_
for i in range(input_dim_features):
    X_feat[:, i, :] = (X_feat[:, i, :] - mean[i]) / (std[i] + 1e-8)

print(f"   ✅ Scaler fitteado sobre {train_mask.sum()} muestras de train ({len(train_site_dates)} sitios-fecha únicos).")

joblib.dump(le, os.path.join(SAVE_DIR, "le_crossattn.joblib"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_crossattn.joblib"))

# --- DataLoaders con pin_memory ---
train_loader = DataLoader(
    TensorDataset(
        torch.LongTensor(X_dir[train_indices]),
        torch.FloatTensor(X_weight[train_indices]),
        torch.FloatTensor(X_feat[train_indices]),
        torch.LongTensor(y[train_indices])
    ),
    batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)
test_loader = DataLoader(
    TensorDataset(
        torch.LongTensor(X_dir[test_indices]),
        torch.FloatTensor(X_weight[test_indices]),
        torch.FloatTensor(X_feat[test_indices]),
        torch.LongTensor(y[test_indices])
    ),
    batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
)
num_classes = len(le.classes_)


# --- 4. NUEVA ARQUITECTURA: GATED CROSS-ATTENTION (Conv1d kernel=7, padding=3) ---
class VecPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class CrossAttentionHybridModel(nn.Module):
    def __init__(self, num_classes, input_dim_feat, d_vec=256, d_feat=128):
        super().__init__()
        # RAMA VECTORES (Se mantiene fuerte)
        self.dir_emb = nn.Embedding(3, d_vec)
        self.weight_proj = nn.Linear(1, d_vec)
        self.fusion_vec = nn.Linear(d_vec * 2, d_vec)
        # ← kernel_size 5→7, padding 2→3 para mantener dimensionalidad de salida
        self.conv = nn.Conv1d(d_vec, d_vec, kernel_size=7, stride=2, padding=3)
        self.pos_vec = VecPositionalEncoding(d_vec)
        self.transformer_vec = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_vec, 8, 1024, 0.2, batch_first=True), 4
        )
        self.ln_vec = nn.LayerNorm(d_vec)

        # RAMA FEATURES (Procesa las 24 horas pero NO las promedia)
        self.feat_proj = nn.Linear(input_dim_feat, d_feat)
        self.transformer_feat = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_feat, 4, 256, 0.2, batch_first=True), 2
        )

        # BLOQUE CROSS-ATTENTION Y GATING
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_vec, kdim=d_feat, vdim=d_feat, num_heads=4, batch_first=True
        )
        self.gate_layer = nn.Sequential(
            nn.Linear(d_vec * 2, d_vec),
            nn.Sigmoid()
        )
        self.ln_fusion = nn.LayerNorm(d_vec)

        # CLASIFICADOR FINAL
        self.classifier = nn.Sequential(
            nn.Linear(d_vec, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_d, x_w, x_f):
        # 1. Extraer representación del Vector
        v = torch.cat([self.dir_emb(x_d + 1), self.weight_proj(x_w.unsqueeze(-1))], dim=-1)
        v = self.fusion_vec(v).transpose(1, 2)
        v = self.conv(v).transpose(1, 2)
        v = self.transformer_vec(self.pos_vec(v))
        vec_query = self.ln_vec(v.mean(dim=1))  # (Batch, d_vec) -> El "Query" global del paquete

        # 2. Extraer secuencia de 24 horas de Features
        f = x_f.transpose(1, 2)  # (Batch, 24, num_features)
        feat_seq = self.feat_proj(f)
        feat_seq = self.transformer_feat(feat_seq)  # (Batch, 24, d_feat) -> Los "Keys/Values"

        # 3. Cross-Attention: El vector consulta las 24 horas
        q = vec_query.unsqueeze(1)  # (Batch, 1, d_vec)
        attn_out, _ = self.cross_attn(query=q, key=feat_seq, value=feat_seq)
        attn_out = attn_out.squeeze(1)  # (Batch, d_vec) -> Contexto extraído

        # 4. Mecanismo de Compuerta (Gating)
        gate_input = torch.cat([vec_query, attn_out], dim=-1)
        g = self.gate_layer(gate_input)  # (Batch, d_vec), valores entre 0 y 1

        # 5. Fusión Final Desempatada
        fusion = vec_query + (g * attn_out)
        fusion = self.ln_fusion(fusion)

        return self.classifier(fusion)


# --- 5. ENTRENAMIENTO OPTIMIZADO (AMP bfloat16 + torch.compile + Gradient Clipping) ---
print("\n[5/5] ⚙️ Instanciando Modelo Cross-Attention y compilando con torch.compile...")
model = CrossAttentionHybridModel(num_classes, input_dim_features).to(device)
model = torch.compile(model, mode="reduce-overhead")

optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler_amp = GradScaler("cuda")

# CosineAnnealingWarmRestarts nativo (sin dependencia de HuggingFace)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2, eta_min=1e-6
)

best_acc = 0.0
best_epoch = 0
trigger = 0

# --- TELEMETRÍA ---
history_loss = []
history_acc = []

print(f"\n🏋️ INICIANDO BUCLE DE ENTRENAMIENTO (GATED CROSS-ATTENTION + AMP bfloat16)")
for epoch in range(EPOCHS):
    # ─── FASE DE ENTRENAMIENTO ───
    model.train()
    total_loss = 0.0

    for d, w, f, y_batch in train_loader:
        optimizer.zero_grad(set_to_none=True)

        # Transferencia unificada a GPU con non_blocking
        d = d.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        f = f.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.bfloat16):
            loss = criterion(model(d, w, f), y_batch)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    history_loss.append(avg_loss)

    scheduler.step()

    # ─── FASE DE EVALUACIÓN ───
    model.eval()
    correct = 0
    with torch.inference_mode():
        for d, w, f, y_batch in test_loader:
            d = d.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            f = f.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            preds = torch.argmax(model(d, w, f), dim=1)
            correct += (preds == y_batch).sum().item()

    acc = correct / len(test_indices)
    history_acc.append(acc * 100)

    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Val Acc: {acc*100:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e}")

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_model_crossattn.pth")
        trigger = 0
    else:
        trigger += 1
        if trigger >= 15:
            print(f"🛑 Early stopping activado en la época {epoch+1}.")
            break

print("\n" + "═"*70)
print(f"🏆 ENTRENAMIENTO COMPLETADO. MEJOR ACCURACY (VALIDACIÓN PASADO): {best_acc*100:.2f}% (Época {best_epoch})")
print("═"*70)

# --- 6. GENERACIÓN DE EVIDENCIA VISUAL (GRÁFICA 16) ---
print("\n🎨 Generando Gráfica 16: Dinámica de Convergencia (Cross-Attention)...")

plt.figure(figsize=(14, 6))
sns.set_theme(style="whitegrid")

# Subplot 1: Curva de Pérdida
plt.subplot(1, 2, 1)
plt.plot(range(1, len(history_loss) + 1), history_loss, color='#c0392b', linewidth=2.5, marker='*', markersize=6)
plt.axvline(best_epoch, color='gray', linestyle='--', alpha=0.7, label=f'Mejor Época ({best_epoch})')
plt.title('Minimización del Error (Cross-Attention)', fontsize=14, fontweight='bold', pad=10)
plt.xlabel('Época de Entrenamiento', fontsize=12)
plt.ylabel('Pérdida (Loss)', fontsize=12)
plt.legend()

# Subplot 2: Curva de Precisión
plt.subplot(1, 2, 2)
plt.plot(range(1, len(history_acc) + 1), history_acc, color='#f39c12', linewidth=2.5, marker='*', markersize=6)
plt.axvline(best_epoch, color='gray', linestyle='--', alpha=0.7, label=f'Mejor Época ({best_epoch})')
plt.title('Estabilización de Precisión (Fusión Dinámica)', fontsize=14, fontweight='bold', pad=10)
plt.xlabel('Época de Entrenamiento', fontsize=12)
plt.ylabel('Precisión (%)', fontsize=12)
plt.legend()

plt.suptitle('Dinámica de Convergencia: Híbrido Gated Cross-Attention (AMP v3.2)', fontsize=16, fontweight='black', y=1.05)
plt.tight_layout()

# Guardar la imagen
grafica_path = os.path.join(SAVE_DIR, 'G16_Convergencia_CrossAttention.png')
plt.savefig(grafica_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Evidencia visual guardada exitosamente en: {grafica_path}")
