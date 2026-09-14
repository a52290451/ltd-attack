"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: EXP1_Train_Invariantes.py
🚀 VERSIÓN: 4.0 (PyTorch 2.0 Best Practices — Rigor de Producción)
👤 INVESTIGADOR: bsierra@zeus
📅 FECHA DE ACTUALIZACIÓN: 08 de Julio, 2026

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este script implementa el entrenamiento de la rama "Macro-Temporal" (Macro Puro).

    [!] ACTUALIZACIÓN V4.0 (PYTORCH 2.0 BEST PRACTICES):
    - 🔴 Data Leakage Fix: StandardScaler ajustado solo sobre X_train, no sobre el dataset
      completo. El escalado se realiza tensor por tensor tras el split temporal.
    - ⚡ torch.compile(mode="reduce-overhead") para optimización JIT de grafos.
    - 🎯 Mixed Precision nativa: torch.amp.autocast("cuda", bfloat16) + GradScaler.
    - 🛡️ Gradient Clipping: clip_grad_norm_(max_norm=1.0) para estabilidad numérica.
    - 📉 CosineAnnealingWarmRestarts(T_0=15, T_mult=2) en lugar de LR fijo.
    - 💾 optimizer.zero_grad(set_to_none=True) para reducir presión sobre el allocator.
    - 🚀 pin_memory=True en DataLoaders.

📐 ARQUITECTURA DE LA RED NEURONAL (HOURLY ENCODER ORIGINAL - MULTILEVEL ATTENTION):
    1. Feature Projection Layers: Proyecta los vectores individuales a un espacio d_model (64).
    2. Temporal Transformer Encoder: Modela las Dependencias Longitudinales mediante
       codificación posicional sinusoidal sobre el ciclo temporal de 24 horas.
    3. Set Attention Blocks (SAB): Bloques de Auto-Atención Multifactorial diseñados
       para mapear las correlaciones latentes cruzadas *entre* las diferentes features.
    4. Global Pooling & Classifier: Capa de agregación y MLP con regularización por
       Dropout para la predicción de la clase.

⚙️ ENTRADAS:
    - result_path('macro', 'resultados_analisis', 'new_features_invariantes_seguras.txt')
    - data_path('historical', 'CLEAN_final_features_sites.csv')
    - [!] artifact_path('ds3_label_encoder_vec_3000.joblib')
========================================================================================
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
import re
import matplotlib
matplotlib.use('Agg')  # Modo headless para evitar problemas gráficos en servidores
import matplotlib.pyplot as plt
import seaborn as sns
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


# --- 1. CONFIGURACIÓN DE ENTORNO Y RUTAS ---
FEATURES_LIST_TXT = result_path('macro', 'resultados_analisis', 'new_features_invariantes_seguras.txt')
HISTORIC_FEATURES_CSV = data_path('historical', 'CLEAN_final_features_sites.csv')
SAVE_DIR = result_path('macro')

# [!] RUTA DEL DICCIONARIO DE VECTORES PARA EL FILTRO DE PARIDAD
VECTOR_LE_PATH = artifact_path('ds3_label_encoder_vec_3000.joblib')

os.makedirs(SAVE_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Logger(os.path.join(SAVE_DIR, f"log_train_EXP1_{timestamp}.txt"))

print("\n" + "═" * 70)
print("🛡️  EXP 1: HOURLY ENCODER ORIGINAL (PARIDAD ESTRICTA 65 CLASES)")
print("═" * 70)

BATCH_SIZE = 64
EPOCHS = 100
D_MODEL = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Dispositivo de cómputo configurado: {device}")

# --- 2. CARGA DEL FILTRO DE PARIDAD Y LAS FEATURES DE ÉLITE ---
try:
    le_vectores = joblib.load(VECTOR_LE_PATH)
    sitios_elite = set(le_vectores.classes_.astype(str))
    print(f"   ✅ Filtro Maestro cargado: {len(sitios_elite)} sitios de élite detectados en rama Micro.")
except Exception as e:
    print(f"   ❌ Error crítico al cargar el LabelEncoder de Vectores: {e}")
    print(f"   Verifica que la ruta sea correcta: {VECTOR_LE_PATH}")
    sys.exit()

try:
    with open(FEATURES_LIST_TXT, 'r') as f:
        features_inv = [line.strip() for line in f.readlines() if line.strip()]
    input_dim_features = len(features_inv)
    print(f"   ✅ Núcleo Invariante inyectado con éxito: {input_dim_features} características estadísticas.")
except Exception as e:
    print(f"   ❌ Error crítico al cargar las variables desde el archivo de texto: {e}")
    sys.exit()

# --- 3. PIPELINE DE PREPROCESAMIENTO TEMPORAL Y PARIDAD ---
print("\n⏳ Estructurando datos históricos (Cuantización Horaria y Segmentación de Ciclos)...")
df_hist = pd.read_csv(HISTORIC_FEATURES_CSV)
df_hist = df_hist.dropna(subset=['site_label'])
df_hist['site_label'] = df_hist['site_label'].astype(str)

# [!] APLICACIÓN DEL FILTRO DE PARIDAD (DESCARTANDO LOS SITIOS ROTOS)
initial_len = len(df_hist)
df_hist = df_hist[df_hist['site_label'].isin(sitios_elite)].copy()
print(f"   -> Muestras descartadas por ruido (sitios anómalos sin vectores): {initial_len - len(df_hist)}")


def parse_time_from_pcap(name_str):
    match = re.search(r'_(\d{8})-(\d{2})\d{4}', str(name_str))
    if match:
        return match.group(1), int(match.group(2))
    return None, None


parsed_dimensions = df_hist['pcap_name'].apply(parse_time_from_pcap)
df_hist['date_id'] = [p[0] for p in parsed_dimensions]
df_hist['hour_bin'] = [p[1] for p in parsed_dimensions]
df_hist = df_hist.dropna(subset=['date_id', 'hour_bin'])

le = LabelEncoder()
df_hist['target'] = le.fit_transform(df_hist['site_label'])

# 🔴 CRÍTICO — Data Leakage Fix V4.0:
# Primero construimos los tensores SIN escalar (valores crudos),
# luego hacemos el split temporal, y SOLO ENTONCES ajustamos el scaler sobre X_train.
# Finalmente transformamos X_train y X_val por separado.

# Construcción de tensores con features crudas (sin escala)
X_list_hist_raw, y_list_hist = [], []
for (target_idx, date), group in df_hist.groupby(['target', 'date_id']):
    tensor = np.zeros((input_dim_features, 24))
    for _, row in group.iterrows():
        hora = int(row['hour_bin'])
        if 0 <= hora < 24:
            tensor[:, hora] = row[features_inv].values.astype(np.float64)
    X_list_hist_raw.append(tensor)
    y_list_hist.append(target_idx)

X_hist_raw = np.array(X_list_hist_raw)  # (N_samples, features, 24)
y_hist = np.array(y_list_hist)

# Split temporal estratificado
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_hist_raw, y_hist, test_size=0.15, stratify=y_hist, random_state=42
)

# 🔴 AHORA sí: ajustar scaler SOLO sobre X_train
# Aplanamos (N_train * features * 24) → (N_train * 24, features) para fit bidimensional
N_train, F, T = X_train_raw.shape
X_train_flat = X_train_raw.transpose(0, 2, 1).reshape(N_train * T, F)

scaler = StandardScaler()
scaler.fit(X_train_flat)

# Transformar X_train
X_train_scaled = np.zeros_like(X_train_raw)
for i in range(N_train):
    sample = X_train_raw[i]  # (features, 24)
    sample_flat = sample.T   # (24, features)
    sample_scaled_flat = scaler.transform(sample_flat)
    X_train_scaled[i] = sample_scaled_flat.T  # (features, 24)

# Transformar X_val
N_val = X_val_raw.shape[0]
X_val_scaled = np.zeros_like(X_val_raw)
for i in range(N_val):
    sample = X_val_raw[i]
    sample_flat = sample.T
    sample_scaled_flat = scaler.transform(sample_flat)
    X_val_scaled[i] = sample_scaled_flat.T

# Guardar artefactos sin leakage
joblib.dump(le, os.path.join(SAVE_DIR, "le_invariantes.joblib"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_invariantes.joblib"))
joblib.dump(features_inv, os.path.join(SAVE_DIR, "features_list.joblib"))

# DataLoaders con pin_memory=True
train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)
val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val)),
    batch_size=BATCH_SIZE,
    pin_memory=True
)

num_classes = len(le.classes_)
print(f"   -> Universo de clases activas detectado: {num_classes}")
print(f"   -> Scaler ajustado exclusivamente sobre X_train ({N_train} muestras) ✅ Sin leakage temporal.")
print(f"   -> Tamaño del Tensor de Entrada (Train Set): {X_train_scaled.shape}")
print(f"   -> Tamaño del Tensor de Entrada (Val Set): {X_val_scaled.shape}")

# --- 4. ARQUITECTURA DE LA RED NEURONAL ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=24):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe


class TemporalEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_norm = nn.LayerNorm(d_model)
    def forward(self, x):
        return self.output_norm(self.transformer_encoder(self.pos_encoder(x)).mean(dim=1))


class SetAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
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
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        b, f, t = x.shape
        x = self.feature_projection(x.view(b * f, t, 1))
        hourly_embs = self.temporal_engine(x)
        x = self.aggregation_engine(hourly_embs.view(b, f, -1))
        daily_emb = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        return self.classifier(daily_emb)


# --- 5. MOTOR DE OPTIMIZACIÓN Y ENTRENAMIENTO (PyTorch 2.0 Optimizado) ---
print("\n🏋️  Iniciando proceso de optimización del Hourly Encoder (PyTorch 2.0 Stack)...")
model = GlobalSiteEncoder(input_dim_features, num_classes, d_model=D_MODEL).to(device)

# ⚡ torch.compile con modo reduce-overhead para minimizar overhead de kernel launches
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode="reduce-overhead")
    print("   ✅ Modelo compilado con torch.compile(mode='reduce-overhead')")

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 📉 CosineAnnealingWarmRestarts: reinicios cálidos cada T_0=15 épocas, duplicando el período
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

# 🎯 Mixed Precision: GradScaler para FP16/BF16 seguro
scaler_amp = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

best_val_acc = 0.0
patience, trigger = 10, 0

hist_train_loss, hist_val_loss = [], []
hist_train_acc, hist_val_acc = [], []

for epoch in range(EPOCHS):
    # ── ENTRENAMIENTO ──
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0

    for f, y_b in train_loader:
        optimizer.zero_grad(set_to_none=True)  # 💾 Libera memoria del grafo anterior

        if scaler_amp is not None:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(f.to(device, non_blocking=True))
                loss = criterion(outputs, y_b.to(device, non_blocking=True))
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 🛡️ Gradient Clipping
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            outputs = model(f.to(device))
            loss = criterion(outputs, y_b.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * f.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct_train += (preds == y_b.to(device)).sum().item()
        total_train += f.size(0)

    epoch_train_loss = running_loss / total_train
    epoch_train_acc = correct_train / total_train

    # ── VALIDACIÓN ──
    model.eval()
    running_val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.inference_mode():  # ⚡ Más rápido que torch.no_grad()
        for f, y_b in val_loader:
            if scaler_amp is not None:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(f.to(device, non_blocking=True))
                    loss = criterion(outputs, y_b.to(device, non_blocking=True))
            else:
                outputs = model(f.to(device))
                loss = criterion(outputs, y_b.to(device))

            running_val_loss += loss.item() * f.size(0)  # ✅ .item() libera el tensor
            preds = torch.argmax(outputs, dim=1)
            correct_val += (preds == y_b.to(device)).sum().item()
            total_val += f.size(0)

    epoch_val_loss = running_val_loss / total_val
    epoch_val_acc = correct_val / total_val

    # 📉 Step del scheduler (CosineAnnealingWarmRestarts se actualiza por época)
    scheduler.step()

    # Guardar métricas en el historial
    hist_train_loss.append(epoch_train_loss)
    hist_val_loss.append(epoch_val_loss)
    hist_train_acc.append(epoch_train_acc)
    hist_val_acc.append(epoch_val_acc)

    current_lr = optimizer.param_groups[0]['lr']
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"   Epoch {epoch + 1:02d}/{EPOCHS} | LR: {current_lr:.2e} | "
              f"Train Acc: {epoch_train_acc * 100:.2f}% | "
              f"Val Acc: {epoch_val_acc * 100:.2f}% | Val Loss: {epoch_val_loss:.4f}")

    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model_invariantes.pth"))
        trigger = 0
    else:
        trigger += 1
        if trigger >= patience:
            print(f"\n🛑 Early stopping ejecutado de forma preventiva en la época {epoch + 1}.")
            break

print("\n" + "🏆" * 25)
print(f"   ENTRENAMIENTO (PARIDAD) COMPLETADO CON ÉXITO")
print(f"   🎯 MEJOR ACCURACY ALCANZADO (VALIDACIÓN): {best_val_acc * 100:.2f}%")
print(f"   💾 Pesos consolidados guardados en: {SAVE_DIR}/best_model_invariantes.pth")
print("🏆" * 25 + "\n")

# --- 6. GENERACIÓN DE EVIDENCIA VISUAL (CURVAS DE APRENDIZAJE) ---
print("🎨 Generando Gráficas de Convergencia...")
plt.figure(figsize=(14, 6))
sns.set_theme(style="whitegrid")

plt.subplot(1, 2, 1)
plt.plot(hist_train_loss, label='Train Loss', color='#3498db', linewidth=2)
plt.plot(hist_val_loss, label='Validation Loss', color='#e74c3c', linewidth=2, linestyle='--')
plt.title('Curva de Pérdida (Cross-Entropy Loss)', fontsize=14, fontweight='bold')
plt.xlabel('Época (Epoch)', fontsize=12)
plt.ylabel('Pérdida', fontsize=12)
plt.legend(fontsize=11)

plt.subplot(1, 2, 2)
plt.plot(np.array(hist_train_acc) * 100, label='Train Accuracy', color='#2ecc71', linewidth=2)
plt.plot(np.array(hist_val_acc) * 100, label='Validation Accuracy', color='#f39c12', linewidth=2, linestyle='--')
plt.title('Curva de Precisión (Accuracy)', fontsize=14, fontweight='bold')
plt.xlabel('Época (Epoch)', fontsize=12)
plt.ylabel('Precisión (%)', fontsize=12)
plt.legend(fontsize=11)

plt.tight_layout()
graph_path = os.path.join(SAVE_DIR, f'EXP1_Curvas_Aprendizaje_Paridad_{timestamp}.jpg')
plt.savefig(graph_path, dpi=300)
print(f"✅ Evidencia visual guardada exitosamente en: {graph_path}\n")
