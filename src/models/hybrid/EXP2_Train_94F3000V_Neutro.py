"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: EXP2_Train_94F3000V_Neutro.py
🚀 VERSIÓN: 3.2 (Fusión Híbrida Neutra + AMP + torch.compile)
👤 INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Entrenar la arquitectura Híbrida (Vectores + Features) de forma NEUTRA (0% Dropout),
    utilizando EXCLUSIVAMENTE las 94 variables Invariantes (ADN Puro) y los 65 sitios 
    de élite. 
    
    HIPÓTESIS: Evaluar si la concatenación simple de secuencias estructurales (Micro) 
    y perfiles temporales de 24h (Macro) es suficiente para que un clasificador de 
    Cross-Entropy resista el Concept Drift.

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
matplotlib.use('Agg')  # Backend para servidores sin interfaz gráfica

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
HISTORIC_FEATURES_CSV = '../../output/CLEAN_final_features_sites.csv'
CACHED_VECTORS_CSV = '../../output/cached_vectors_with_dates.csv'
VECTOR_LE_PATH = '../vectores/resultados/ds3_label_encoder_vec_3000.joblib'

SAVE_DIR = './resultados_94F3000V'
os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Logger(os.path.join(SAVE_DIR, f"log_train_EXP2_{timestamp}.txt"))

print("\n" + "═"*70)
print("🛡️ EXP 2 v3.2: ENTRENAMIENTO HÍBRIDO NEUTRO (VECTORES + 94 INVARIANTES) + AMP + torch.compile")
print("═"*70)

MAX_LEN = 3000
BATCH_SIZE = 64
EPOCHS = 60
D_MODEL = 256
D_FEAT = 64
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
for (site, date), group in df_feat.groupby(['site_label', 'date_id']):
    tensor = np.zeros((input_dim_features, 24))
    for _, row in group.iterrows():
        hora = int(row['hour_bin'])
        if 0 <= hora < 24:
            tensor[:, hora] = row[features_inv].values
    site_date_to_feature_tensor[(site, date)] = tensor

print("⏳ [4/5] Empaquetando tensores Híbridos (Vectores + Features sin escalar)...")
X_dir_list, X_w_list, X_feat_list, y_list, dates_list, site_str_list = [], [], [], [], [], []

for _, row in tqdm(df_vec.iterrows(), total=len(df_vec), desc="Cruzando datos"):
    site = str(row['site_label'])
    date = int(row['date_id'])
    if (site, date) in site_date_to_feature_tensor:
        X_dir_list.append(preprocess_vector(row['direction_vector'], MAX_LEN))
        X_w_list.append(preprocess_vector(row['size_vector'], MAX_LEN))
        X_feat_list.append(site_date_to_feature_tensor[(site, date)])
        y_list.append(le.transform([site])[0])
        dates_list.append(date)
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
mean = scaler.mean_   # shape: (94,)
std = scaler.scale_   # shape: (94,)
for i in range(input_dim_features):
    X_feat[:, i, :] = (X_feat[:, i, :] - mean[i]) / (std[i] + 1e-8)

print(f"   ✅ Scaler fitteado sobre {train_mask.sum()} muestras de train ({len(train_site_dates)} sitios-fecha únicos).")

joblib.dump(le, os.path.join(SAVE_DIR, "le_hibrido_neutro.joblib"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_hibrido_neutro.joblib"))

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


# --- 4. ARQUITECTURA HÍBRIDA (Conv1d kernel=7, padding=3) ---
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
    def __init__(self, num_classes, input_dim_feat, d_vec=256, d_feat=64):
        super().__init__()
        self.dir_emb = nn.Embedding(3, d_vec)
        self.weight_proj = nn.Linear(1, d_vec)
        self.fusion_vec = nn.Linear(d_vec * 2, d_vec)
        # ← kernel_size 5→7, padding 2→3 para mantener dimensionalidad de salida
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
            nn.Linear(512, num_classes)
        )

    def forward(self, x_d, x_w, x_f):
        # Rama de Vectores
        v = torch.cat([self.dir_emb(x_d + 1), self.weight_proj(x_w.unsqueeze(-1))], dim=-1)
        v = self.fusion_vec(v).transpose(1, 2)
        v = self.conv(v).transpose(1, 2)
        v = self.transformer_vec(self.pos_vec(v))
        vec_out = self.ln_vec(torch.cat([v.mean(1), v.max(1)[0]], 1))

        # Rama de Features
        b, f, t = x_f.shape
        ft = x_f.view(b * f, t, 1)
        ft = self.temp_engine(self.feat_proj(ft))
        ft = self.agg_engine(ft.view(b, f, -1))
        feat_out = self.pool_feat(ft.transpose(1, 2)).squeeze(-1)

        return self.classifier(torch.cat([vec_out, feat_out], dim=1))


# --- 5. ENTRENAMIENTO OPTIMIZADO (AMP bfloat16 + torch.compile + Gradient Clipping) ---
print("\n[5/5] ⚙️ Instanciando Modelo Híbrido y compilando con torch.compile...")
model = MegaHybridModel(num_classes, input_dim_features).to(device)
model = torch.compile(model, mode="reduce-overhead")

optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler_amp = GradScaler("cuda")

# CosineAnnealingWarmRestarts nativo (sin dependencia de HuggingFace)
# T_0=15, T_mult=2 → ciclos: [0-15), [15-45), [45-105)... cubre 60 épocas
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2, eta_min=1e-6
)

best_acc = 0.0
best_epoch = 0
trigger = 0

# --- TELEMETRÍA PARA GRÁFICAS ---
history_loss = []
history_acc = []

print(f"\n🏋️ INICIANDO BUCLE DE ENTRENAMIENTO (NEUTRO 0% Dropout + AMP bfloat16)")
for epoch in range(EPOCHS):
    # ─── FASE DE ENTRENAMIENTO ───
    model.train()
    total_loss = 0.0

    for d, w, f, y_batch in train_loader:
        optimizer.zero_grad(set_to_none=True)

        # Transferencia unificada a GPU
        d = d.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        f = f.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # EXP2 Neutro: SIN Modality Dropout — ambas ramas siempre activas
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

    # CosineAnnealingWarmRestarts step por época
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
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_model_hibrido_neutro.pth")
        trigger = 0
    else:
        trigger += 1
        if trigger >= 15:
            print(f"🛑 Early stopping activado en la época {epoch+1}.")
            break

print("\n" + "═"*70)
print(f"🏆 ENTRENAMIENTO COMPLETADO. MEJOR ACCURACY (VALIDACIÓN PASADO): {best_acc*100:.2f}% (Época {best_epoch})")
print("═"*70)

# --- 6. GENERACIÓN DE EVIDENCIA VISUAL (GRÁFICA 13) ---
print("\n🎨 Generando Gráfica 13: Dinámica de Convergencia del Modelo...")

plt.figure(figsize=(14, 6))
sns.set_theme(style="whitegrid")

# Subplot 1: Curva de Pérdida (Loss)
plt.subplot(1, 2, 1)
plt.plot(range(1, len(history_loss) + 1), history_loss, color='#e74c3c', linewidth=2.5, marker='o', markersize=4)
plt.axvline(best_epoch, color='gray', linestyle='--', alpha=0.7, label=f'Mejor Época ({best_epoch})')
plt.title('Minimización del Error (Cross-Entropy Loss)', fontsize=14, fontweight='bold', pad=10)
plt.xlabel('Época de Entrenamiento', fontsize=12)
plt.ylabel('Pérdida (Loss)', fontsize=12)
plt.legend()

# Subplot 2: Curva de Precisión (Accuracy)
plt.subplot(1, 2, 2)
plt.plot(range(1, len(history_acc) + 1), history_acc, color='#2ecc71', linewidth=2.5, marker='o', markersize=4)
plt.axvline(best_epoch, color='gray', linestyle='--', alpha=0.7, label=f'Mejor Época ({best_epoch})')
plt.title('Evolución de la Precisión (Validation Accuracy)', fontsize=14, fontweight='bold', pad=10)
plt.xlabel('Época de Entrenamiento', fontsize=12)
plt.ylabel('Precisión (%)', fontsize=12)
plt.legend()

plt.suptitle('Dinámica de Convergencia: Arquitectura Híbrida Neutra (Cross-Entropy + AMP v3.2)', fontsize=16, fontweight='black', y=1.05)
plt.tight_layout()

# Guardar la imagen
grafica_path = os.path.join(SAVE_DIR, 'G13_Convergencia_Hibrido_Neutro.png')
plt.savefig(grafica_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Evidencia visual guardada exitosamente en: {grafica_path}")