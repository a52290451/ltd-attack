"""
=========================================================================================
⚡ UNIVERSIDAD / INSTITUTO DE INVESTIGACIÓN Y DESARROLLO DE IA
🔬 LABORATORIO DE CIBERSEGURIDAD Y ANÁLISIS DE TRÁFICO AVANZADO (WFP)
=========================================================================================
📂 ARCHIVO: EXP1_Evaluar_Invariantes.py
🚀 VERSIÓN: 4.0 (Evaluación de Resiliencia con Paridad Estricta de 65 Clases)
👤 INVESTIGADOR: bsierra@zeus
📅 FECHA DE ACTUALIZACIÓN: 08 de Julio, 2026

-----------------------------------------------------------------------------------------
📝 RESUMEN ACADÉMICO / METODOLOGÍA:
    Este script constituye la Prueba de Fuego (Inferencia Post-Drift) para el modelo
    Macro Puro (Hourly Encoder) bajo las condiciones del Estudio de Ablación.

    [!] ACTUALIZACIÓN V4.0 (PYTORCH 2.0 BEST PRACTICES):
    - 🧹 Limpieza de state_dict: elimina prefijo "_orig_mod." de torch.compile.
    - ⚡ torch.inference_mode() en lugar de torch.no_grad() (~10% más rápido).
    - 🎯 autocast("cuda", bfloat16) en inferencia para acelerar pases forward.
    - 🚀 pin_memory=True en DataLoader del drift.
    - 🛡️ torch.load con weights_only=True para seguridad.

🎯 OBJETIVO EMPÍRICO:
    Comparar de forma 100% equitativa (1:1) la supervivencia de la rama Macro frente a
    la rama Micro (que obtuvo 32.62%). Además, se genera evidencia gráfica (Matriz de
    Confusión) para auditar la dispersión de las predicciones en el espacio latente.

⚙️ ENTRADAS:
    - data_path('historical', 'CLEAN_final_features_sites_concept_drift.csv') (Futuro Purificado)
    - Diccionarios, scaler y pesos (.pth) ubicados bajo result_path('macro')
========================================================================================
"""

import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import joblib
import sys
import datetime
import re
import matplotlib
matplotlib.use('Agg')  # Entorno Headless
import matplotlib.pyplot as plt
import seaborn as sns
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
DRIFT_FEATURES_CSV = data_path('historical', 'CLEAN_final_features_sites_concept_drift.csv')
SAVE_DIR = result_path('macro')

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = Logger(os.path.join(SAVE_DIR, f"log_eval_EXP1_{timestamp}.txt"))

print("\n" + "═" * 70)
print("🛡️  EXP 1: EVALUACIÓN DE RESILIENCIA (PARIDAD 65 CLASES vs CONCEPT DRIFT)")
print("═" * 70)

BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Dispositivo de cómputo configurado: {device}")

# Configurar precisión mixta para inferencia
use_amp = device.type == "cuda"

# --- 2. CARGAR MEMORIA DEL MODELO ---
print("\n📖 Cargando memoria y artefactos matemáticos del pasado...")
try:
    le = joblib.load(os.path.join(SAVE_DIR, "le_invariantes.joblib"))
    scaler = joblib.load(os.path.join(SAVE_DIR, "scaler_invariantes.joblib"))
    features_inv = joblib.load(os.path.join(SAVE_DIR, "features_list.joblib"))

    input_dim_features = len(features_inv)
    known_historic_ids = set(le.classes_.astype(str))
    print(f"   ✅ Memoria recuperada: {input_dim_features} características invariantes.")
    print(f"   ✅ Clases entrenadas en el LabelEncoder: {len(known_historic_ids)} sitios de élite.")
except Exception as e:
    print(f"❌ Error crítico al cargar los metadatos: {e}")
    sys.exit()

# --- 3. EXTRACCIÓN TEMPORAL Y ESTRUCTURACIÓN DEL FUTURO (DRIFT) ---
print("\n⏳ Estructurando datos del futuro (Extrayendo Ciclos 24h)...")
df_drift = pd.read_csv(DRIFT_FEATURES_CSV)
df_drift = df_drift.dropna(subset=['site_label'])
df_drift['site_label'] = df_drift['site_label'].astype(str)

initial_len = len(df_drift)

# [!] FILTRO ESTRICTO: Solo evaluamos las 65 clases conocidas
df_drift = df_drift[df_drift['site_label'].isin(known_historic_ids)].copy()
print(f"   -> Muestras filtradas (Sitios desconocidos/anómalos descartados): {initial_len - len(df_drift)}")


# Motor Temporal por Expresiones Regulares
def parse_time_from_pcap(name_str):
    match = re.search(r'_(\d{8})-(\d{2})\d{4}', str(name_str))
    if match:
        return match.group(1), int(match.group(2))
    return None, None


parsed_dimensions = df_drift['pcap_name'].apply(parse_time_from_pcap)
df_drift['date_id'] = [p[0] for p in parsed_dimensions]
df_drift['hour_bin'] = [p[1] for p in parsed_dimensions]
df_drift = df_drift.dropna(subset=['date_id', 'hour_bin'])

# Transformación basada en la memoria del pasado
df_drift['target'] = le.transform(df_drift['site_label'])

# Escalar usando el scaler entrenado sobre X_train (sin leakage)
df_drift_scaled = df_drift.copy()
df_drift_scaled[features_inv] = scaler.transform(df_drift[features_inv])


# Agrupación y relleno de tensores (94 features x 24 horas) — con valores escalados
X_list_drift, y_list_drift = [], []
for (target_idx, date), group in df_drift_scaled.groupby(['target', 'date_id']):
    tensor = np.zeros((input_dim_features, 24))
    for _, row in group.iterrows():
        hora = int(row['hour_bin'])
        if 0 <= hora < 24:
            tensor[:, hora] = row[features_inv].values
    X_list_drift.append(tensor)
    y_list_drift.append(target_idx)

X_drift = np.array(X_list_drift)
y_drift = np.array(y_list_drift)

if len(y_drift) == 0:
    print("   ❌ Set de Drift vacío tras aplicar los filtros de paridad.")
    sys.exit()

drift_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_drift), torch.LongTensor(y_drift)),
    batch_size=BATCH_SIZE,
    pin_memory=True  # 🚀 Acelera transferencia CPU→GPU
)
print(f"   -> Tensores dimensionales de inferencia generados: {len(y_drift)}")


# --- 4. ARQUITECTURA DE LA RED NEURONAL (IDÉNTICA AL ENTRENAMIENTO) ---
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
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        b, f, t = x.shape
        x = self.feature_projection(x.view(b * f, t, 1))
        hourly_embs = self.temporal_engine(x)
        x = self.aggregation_engine(hourly_embs.view(b, f, -1))
        daily_emb = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        return self.classifier(daily_emb)


# --- 5. CARGA LIMPIA DE PESOS Y REGISTRO DE INFERENCIA ACELERADA ---
print("\n🔌 Despertando red neuronal (Cargando pesos de 65 clases)...")
model = GlobalSiteEncoder(input_dim_features, len(known_historic_ids), d_model=64).to(device)
model.eval()

try:
    # 🧹 Cargar con weights_only=True y limpiar prefijo _orig_mod de torch.compile
    state_dict = torch.load(
        os.path.join(SAVE_DIR, "best_model_invariantes.pth"),
        map_location=device,
        weights_only=True
    )
    # Elimina el prefijo "_orig_mod." si los pesos se guardaron desde un modelo compilado
    cleaned_state_dict = {
        (k[10:] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict)
    print("   ✅ Pesos restaurados correctamente (state_dict limpiado de _orig_mod).")
except Exception as e:
    print(f"   ❌ Error al cargar .pth: {e}")
    sys.exit()

# --- 6. INFERENCIA SOBRE EL FUTURO (Modo Acelerado) ---
y_true, y_pred = [], []
with torch.inference_mode():  # ⚡ Más rápido que torch.no_grad()
    for f, y_b in drift_loader:
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(f.to(device, non_blocking=True))
        else:
            logits = model(f.to(device))
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y_b.numpy())
        y_pred.extend(preds.cpu().numpy())

acc_drift = accuracy_score(y_true, y_pred)
print("\n" + "🏆" * 25)
print(f"   ACCURACY FINAL MACRO (DRIFT CON {input_dim_features} INVARIANTES): {acc_drift * 100:.2f}%")
print("🏆" * 25)

# --- 7. GENERACIÓN DE EVIDENCIA VISUAL ---
print("\n🎨 Generando Autopsia Visual (Matriz de Confusión)...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
# Normalizamos por fila para ver el porcentaje de acierto de cada clase
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm)

sns.heatmap(cm_norm, cmap='inferno', cbar=True, xticklabels=False, yticklabels=False)
plt.title(f'Evaluación Concept Drift: Modelo Macro-Temporal ({input_dim_features} Features)\n'
          f'Precisión Retenida: {acc_drift * 100:.2f}%',
          fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Clase Predicha (Red Neuronal)', fontsize=12, fontweight='bold')
plt.ylabel('Clase Real (Sitio Web)', fontsize=12, fontweight='bold')
plt.tight_layout()

graph_path = os.path.join(SAVE_DIR, f'EXP1_G1_Matriz_Drift_Paridad_{timestamp}.jpg')
plt.savefig(graph_path, dpi=300)
print(f"✅ Evidencia visual guardada exitosamente en: {graph_path}")
print("===========================================================================\n")
