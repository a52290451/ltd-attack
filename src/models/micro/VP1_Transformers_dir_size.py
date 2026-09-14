"""
=========================================================================================
                             VP1_Transformers_dir_size.py
=========================================================================================

DESCRIPCIÓN:
    Modelo de Deep Learning Multimodal basado en arquitectura Transformer para 
    la clasificación de tráfico de red (Website Fingerprinting). Este script 
    implementa una "Fusión Temprana" combinando la secuencia discreta de direcciones 
    (Entrante/Saliente) con la secuencia continua del peso de los paquetes (Tamaño 
    en Bytes) para generar una huella digital altamente precisa del sitio web.

OBJETIVOS PRINCIPALES:
    1. Procesar dos inputs paralelos: Direcciones mediante `nn.Embedding` y Pesos 
       (normalizados por Min-Max Scaling) mediante proyección Lineal (`nn.Linear`).
    2. Fusionar ambas representaciones vectoriales antes de alimentar el bloque 
       del Transformer.
    3. Entrenar secuencias extendidas (hasta 3000 paquetes) mitigando la carga en 
       VRAM mediante un "Downsampling" convolucional (Stride=2, Kernel=7).
    4. Guardar metadatos extendidos (Reporte final, Tiempos, Accuracy Global) en 
       archivos de texto para análisis comparativo posterior.

ENTRADAS REQUERIDAS:
    - INPUT_CSV : data_path('historical', 'CLEAN_final_vectors_sites.csv')
      (Requiere las columnas 'site_label', 'direction_vector' y 'size_vector').

SALIDAS (Guardadas bajo result_path('micro')):
    - Pesos del modelo   : ds3_best_multimodal_transformer_vec_3000.pth
    - Codificador        : ds3_label_encoder_vec_3000.joblib
    - Mapeo de clases    : _ds3_class_mapping_vec_3000.joblib
    - Min-Max scaler     : ds3_minmax_scaler_vec_3000.joblib
    - Reporte de métricas: ds3_resultado_final_3000.txt

METADATOS:
    - Autor   : BRAYAN LEONARDO SIERRA FORERO
    - Fecha   : 08/07/2026 (Refactorización v2.0)
    - Versión : 2.0 — Optimizado con AMP bfloat16, GradScaler, torch.compile,
                CosineAnnealingWarmRestarts, clip_grad_norm, kernel_size=7,
                Min-Max solo sobre train (sin Data Leakage).

=========================================================================================
"""

import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import ast
import time
import os
import joblib
from src.utils.paths import data_path, result_path

# --- 1. CONFIGURACIÓN ---
print("\n" + "═"*60)
print("-- INICIANDO EXPERIMENTO: TRANSFORMER DE VECTORES DE DIRECCIÓN (v2.0)")
print("═"*60)

INPUT_CSV = data_path('historical', 'CLEAN_final_vectors_sites.csv')
SAVE_DIR = result_path('micro')
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_LEN = 3000 
BATCH_SIZE = 32
EPOCHS = 100
D_MODEL = 256       # Dimensión del modelo (ajustada para una sola rama)
NHEAD = 8           # Cabezas de atención - patrones distintos que busca el transformer.
NUM_LAYERS = 4      # Capas del encoder - no necesitamos tantas para este problema.

# Gestión de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--  Dispositivo detectado: {device}")

# --- 2. CARGA Y FILTRADO DETALLADO ---
print(f"-- Cargando datos desde: {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
print(f"-- Carga completa. Registros totales: {len(df)}")

# ELIMINACIÓN DE SITIOS ANÓMALOS (49 y 5)
print(f"-- Eliminando sitios anómalos: 49 y 5...")
df = df[~df['site_label'].isin([49, 5])].copy()
print(f"-- Registros tras eliminar anómalos: {len(df)}")

# Estadísticas de clases
counts = df['site_label'].value_counts()
max_samples = counts.max()
threshold = max_samples * 0.5
print(f"-- Clase mayoritaria: '{counts.index[0]}' con {max_samples} muestras.")
print(f"--  Umbral de filtrado (50%): {threshold:.1f} muestras.")

valid_sites = counts[counts >= threshold].index
df_filtered = df[df['site_label'].isin(valid_sites)].copy()

print(f"-- Sitios que pasan el filtro: {len(valid_sites)} de {len(counts)}")
print(f"-- Registros restantes después del filtro: {len(df_filtered)}")

# --- 3. PROCESAMIENTO DE VECTORES ---
def preprocess_vector(v_str, max_len):
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list): v = []
    except: v = []
        
    if len(v) > max_len: return v[:max_len]
    else: return v + [0] * (max_len - len(v))

print(f"--  Procesando vectores de DIRECCIÓN (Max Len: {MAX_LEN})...")
X_dir = np.array([preprocess_vector(row, MAX_LEN) for row in df_filtered['direction_vector']])

print(f"--  Procesando vectores de PESO (Max Len: {MAX_LEN})...")
X_weight_raw = np.array([preprocess_vector(row, MAX_LEN) for row in df_filtered['size_vector']])

# Codificación de etiquetas
le = LabelEncoder()
y = le.fit_transform(df_filtered['site_label'])
num_classes = len(le.classes_)

# Split Estratificado ANTES de cualquier normalización
X_dir_train, X_dir_test, X_weight_raw_train, X_weight_raw_test, y_train, y_test = train_test_split(
    X_dir, X_weight_raw, y, test_size=0.2, stratify=y, random_state=42
)
print(f"-- Split: Train={len(X_dir_train)} | Test={len(X_dir_test)}")

# --- DATA LEAKAGE FIX: Min-Max Scaling calculado SOLO sobre train ---
print(f"--  Aplicando Min-Max Scaling a los pesos (calculado solo sobre TRAIN)...")
min_weight = np.min(X_weight_raw_train)
max_weight = np.max(X_weight_raw_train)

print(f"    - Peso mínimo (train): {min_weight}")
print(f"    - Peso máximo (train): {max_weight}")

# Guardar min/max para evaluación futura
minmax_scaler = {"min": float(min_weight), "max": float(max_weight)}
joblib.dump(minmax_scaler, f"{SAVE_DIR}/ds3_minmax_scaler_vec_3000.joblib")
print(f"    - Min-Max scaler guardado en: {SAVE_DIR}/ds3_minmax_scaler_vec_3000.joblib")

# Aplicar normalización a train y test con los mismos parámetros
X_weight_train = (X_weight_raw_train - min_weight) / (max_weight - min_weight + 1e-8)
X_weight_test = (X_weight_raw_test - min_weight) / (max_weight - min_weight + 1e-8)

# ========================================================
# 🔍 BLOQUE DE INSPECCIÓN
# ========================================================
print("\n" + "═"*60)
print("-- INSPECCIÓN DEL DATASET (PRE-ENTRENAMIENTO)")
print("═"*60)

# 1. Verificar Balanceo de Clases final
unique, counts = np.unique(y_train, return_counts=True)
print(f"-- Clases totales tras filtrado: {len(unique)}")
print(f"-- Muestras en la clase más poblada (Train): {counts.max()}")
print(f"-- Muestras en la clase menos poblada (Train): {counts.min()}")

# 2. Inspección de un Vector Real (Muestra 0)
sample_idx = 0
sample_dir = X_dir[sample_idx]
sample_weight = X_weight_train[sample_idx]
sample_label = le.inverse_transform([y_train[sample_idx]])[0]

print(f"\n-- Ejemplo de la muestra #{sample_idx}:")
print(f"   - Sitio Web (Label Real): {sample_label}")
print(f"   - Forma del vector Dirección: {sample_dir.shape}")
print(f"   - Primeros 20 paquetes (Dirección): {sample_dir[:20]}")
print(f"   - Primeros 20 paquetes (Pesos Normalizados): {sample_weight[:20]}")

# 3. Verificación de Valores Únicos (Solo para direcciones)
valores_unicos_dir = np.unique(X_dir)
print(f"\n-- Valores únicos detectados en X_dir: {valores_unicos_dir}")
if not set(valores_unicos_dir).issubset({-1, 0, 1}):
    print("⚠️ ADVERTENCIA: Se detectaron valores fuera de [-1, 0, 1] en direcciones.")
else:
    print("-- Consistencia de valores de dirección: OK")

# 4. Estadísticas de Relleno (Padding)
ceros_promedio = (X_dir == 0).sum() / X_dir.size * 100
print(f"--  Porcentaje de Padding (ceros) en el dataset: {ceros_promedio:.2f}%")
print("═"*60 + "\n")


# Creación de DataLoaders (con pin_memory=True para eficiencia GPU)
train_loader = DataLoader(
    TensorDataset(torch.LongTensor(X_dir_train), torch.FloatTensor(X_weight_train), torch.LongTensor(y_train)), 
    batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)
test_loader = DataLoader(
    TensorDataset(torch.LongTensor(X_dir_test), torch.FloatTensor(X_weight_test), torch.LongTensor(y_test)), 
    batch_size=BATCH_SIZE, pin_memory=True
)


# --- 4. ARQUITECTURA TRANSFORMER (Conv1d kernel_size=7) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

    
class MultimodalTransformer(nn.Module):
    def __init__(self, num_classes, max_len, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        
        # 1. Rama de Dirección (Discreta)
        self.dir_embedding = nn.Embedding(3, d_model) 
        
        # 2. Rama de Peso (Continua)
        self.weight_proj = nn.Linear(1, d_model)
        
        # 3. Fusión de ambas ramas
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        # Conv1d con kernel_size=7 (simétrico: padding=3 para stride=2)
        self.conv_local = nn.Conv1d(d_model, d_model, kernel_size=7, stride=2, padding=3)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, 
            dropout=0.2, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.ln = nn.LayerNorm(d_model * 2) 
        
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_dir, x_weight):
        # A. Procesar Direcciones: [Batch, Seq] -> [Batch, Seq, d_model]
        e_dir = self.dir_embedding(x_dir + 1) 
        
        # B. Procesar Pesos: [Batch, Seq] -> añadir dimensión -> [Batch, Seq, 1] -> [Batch, Seq, d_model]
        e_weight = self.weight_proj(x_weight.unsqueeze(-1))
        
        # C. Fusionar: Concatenamos y reducimos de nuevo a d_model
        x = torch.cat([e_dir, e_weight], dim=-1)  # [Batch, Seq, d_model * 2]
        x = self.fusion(x)                         # [Batch, Seq, d_model]
        
        # D. Flujo Transformer estándar
        x = x.transpose(1, 2)
        x = self.conv_local(x)     # [Batch, d_model, Seq/2]
        x = x.transpose(1, 2)      # [Batch, Seq/2, d_model]
        
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        avg_pool = torch.mean(x, dim=1) 
        max_pool, _ = torch.max(x, dim=1) 
        x = torch.cat((avg_pool, max_pool), dim=1) 
        
        x = self.ln(x)
        return self.fc(x)


model = MultimodalTransformer(num_classes, MAX_LEN, d_model=D_MODEL, num_layers=NUM_LAYERS).to(device)

# torch.compile para acelerar entrenamiento
try:
    model = torch.compile(model, mode="reduce-overhead")
    print("   ✅ torch.compile activado (modo reduce-overhead).")
except Exception as e:
    print(f"   ⚠️ torch.compile no disponible ({e}), continuando en modo eager.")

optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler("cuda")

# CosineAnnealingWarmRestarts: reinicios periódicos para explorar el landscape
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

# --- 5. ENTRENAMIENTO OPTIMIZADO ---
print(f"🏋️  Entrenando en {device} con {num_classes} clases...")
best_acc = 0.0
start_train = time.time()
patience = 10
trigger_times = 0

for epoch in range(EPOCHS):
    # ==================== ENTRENAMIENTO ====================
    model.train()
    total_loss = 0.0
    epoch_start = time.time()
    
    for b_dir, b_weight, b_y in train_loader:
        optimizer.zero_grad(set_to_none=True)
        
        b_dir = b_dir.to(device, non_blocking=True)
        b_weight = b_weight.to(device, non_blocking=True)
        b_y = b_y.to(device, non_blocking=True)
        
        with autocast("cuda", dtype=torch.bfloat16):
            loss = criterion(model(b_dir, b_weight), b_y)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping para estabilidad numérica
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
    
    # ==================== VALIDACIÓN ====================
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for b_dir, b_weight, b_y in test_loader:
            b_dir = b_dir.to(device, non_blocking=True)
            b_weight = b_weight.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)
            
            with autocast("cuda", dtype=torch.bfloat16):
                preds = torch.argmax(model(b_dir, b_weight), dim=1)
            
            correct += (preds == b_y).sum().item()
            total += b_y.size(0)
    
    acc = correct / total
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] - LR: {current_lr:.6f} - Loss: {total_loss/len(train_loader):.4f} - Acc: {acc:.4f} - {time.time()-epoch_start:.1f}s")

    if acc > best_acc:
        best_acc = acc
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(raw_model.state_dict(), f"{SAVE_DIR}/ds3_best_multimodal_transformer_vec_3000.pth")
        print(f"***** Nuevo mejor accuracy: {best_acc:.4f} (Modelo guardado)")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping en época {epoch+1}")
            break

# Capturamos los datos finales en variables
tiempo_total = (time.time() - start_train) / 60
resumen_final = f"✅ Entrenamiento finalizado en {tiempo_total:.2f} minutos.\n"
resumen_final += f"🏆 Mejor Accuracy en Test: {best_acc:.4f}\n"

# Imprimimos en consola
print("\n" + resumen_final)

# --- 6. GUARDADO DE METADATOS ---
mapping = dict(zip(range(len(le.classes_)), le.classes_))
joblib.dump(le, f"{SAVE_DIR}/ds3_label_encoder_vec_3000.joblib")
joblib.dump(mapping, f"{SAVE_DIR}/_ds3_class_mapping_vec_3000.joblib") 
print(f"💾 Metadatos guardados en {SAVE_DIR}")

print("\n🧪 GENERANDO REPORTE FINAL...")
raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
raw_model.load_state_dict(torch.load(f"{SAVE_DIR}/ds3_best_multimodal_transformer_vec_3000.pth", map_location=device, weights_only=True))
model.eval()

y_true, y_pred = [], []
with torch.inference_mode():
    for b_dir, b_weight, b_y in test_loader:
        b_dir = b_dir.to(device, non_blocking=True)
        b_weight = b_weight.to(device, non_blocking=True)
        b_y = b_y.to(device, non_blocking=True)
        
        with autocast("cuda", dtype=torch.bfloat16):
            preds = torch.argmax(model(b_dir, b_weight), dim=1)
        
        y_true.extend(b_y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

report = classification_report(y_true, y_pred, target_names=le.classes_.astype(str), zero_division=0)
print(report)

# 💾 GUARDAR TODO EN EL ARCHIVO DE TEXTO
texto_a_guardar = "="*60 + "\n"
texto_a_guardar += "RESUMEN DEL ENTRENAMIENTO\n"
texto_a_guardar += "="*60 + "\n"
texto_a_guardar += resumen_final + "\n"
texto_a_guardar += "="*60 + "\n"
texto_a_guardar += "REPORTE DE CLASIFICACIÓN\n"
texto_a_guardar += "="*60 + "\n\n"
texto_a_guardar += report

with open(f"{SAVE_DIR}/ds3_resultado_final_3000.txt", "w", encoding="utf-8") as f:
    f.write(texto_a_guardar)

print(f"\n📄 Archivo de resultados guardado exitosamente en: {SAVE_DIR}/ds3_resultado_final_3000.txt")
