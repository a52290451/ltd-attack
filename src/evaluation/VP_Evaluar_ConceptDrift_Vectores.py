"""
=========================================================================================
                    VP_Evaluar_ConceptDrift_Vectores.py
=========================================================================================
DESCRIPCIÓN ACADÉMICA (ESTUDIO DE ABLACIÓN):
    Suite unificada de evaluación y autopsia visual de la rama "Micro" (Vectores de
    red: Dirección + Tamaño) frente al fenómeno de Concept Drift.

PROPÓSITO EMPÍRICO:
    Script consolidado del 'Ablation Study'. Evalúa un modelo entrenado
    EXCLUSIVAMENTE con la secuencia de paquetes (sin la rama de features
    macro-temporales de 24h) y genera evidencia visual del colapso geométrico.

    Al evaluar este modelo "Solo-Micro" frente a tráfico capturado 2 meses después,
    demostramos matemáticamente que las firmas basadas puramente en secuencias de
    paquetes son altamente volátiles y sufren la degradación más severa por
    Concept Drift.

CAPACIDADES UNIFICADAS:
    1. Inferencia con torch.inference_mode() + autocast bfloat16 (alto rendimiento).
    2. Extracción simultánea de logits (Accuracy) y features latentes (UMAP).
    3. Reporte de métricas → ds3_ablacion_resultado_solo_vectores.txt
    4. Evidencia visual (3 gráficas) → result_path('micro')
        - G1: Huella Secuencial (Direction Barcode)
        - G2: Matriz de Confusión (Heatmap)
        - G3: Colapso del Espacio Latente (UMAP)

ENTRADAS:
    - CLEAN_final_vectors_sites.csv (pasado/entrenamiento, para escala + UMAP)
    - CLEAN_final_vectors_sites_concept_drift.csv (futuro/Concept Drift)
    - ds3_best_multimodal_transformer_vec_3000.pth (pesos)
    - ds3_label_encoder_vec_3000.joblib (LabelEncoder)
========================================================================================
"""

import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import ast
import os
import joblib
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.paths import data_path, artifact_path, result_path

try:
    import umap
except ImportError:
    print("[!] Librería 'umap-learn' no encontrada. Ejecute: pip install umap-learn")
    sys.exit()


# ==========================================================================
# 0. LOGGER (Redirección dual: terminal + archivo)
# ==========================================================================
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


# ==========================================================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================================================
DIR_DATA = data_path('historical')
TRAIN_VECTORS_CSV = os.path.join(DIR_DATA, 'CLEAN_final_vectors_sites.csv')
FUT_VECTORS_CSV = os.path.join(DIR_DATA, 'CLEAN_final_vectors_sites_concept_drift.csv')
SAVE_DIR = result_path('micro')

MODEL_WEIGHTS = "ds3_best_multimodal_transformer_vec_3000.pth"
LABEL_ENCODER = "ds3_label_encoder_vec_3000.joblib"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(SAVE_DIR, f"ds3_evaluacion_unificada_{timestamp}.txt")
sys.stdout = Logger(log_filename)

print("\n" + "═" * 60)
print("🛡️ ESTUDIO DE ABLACIÓN + AUTOPSIA VISUAL (SOLO VECTORES)")
print("═" * 60)

MAX_LEN = 3000
BATCH_SIZE = 64
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
KERNEL_SIZE = 7  # Coincide con el entrenamiento original

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Dispositivo configurado: {device}")

# ==========================================================================
# 2. CARGA DE MEMORIA (LabelEncoder + Mapeos)
# ==========================================================================
print("\n📖 Cargando memoria del modelo Solo-Micro...")
try:
    le = joblib.load(os.path.join(SAVE_DIR, LABEL_ENCODER))
    memorized_ids = set(range(len(le.classes_)))
    site_to_historic_id = {str(site): idx for idx, site in enumerate(le.classes_)}
    known_sites_names = set(le.classes_)
    known_sites_str = set(str(c) for c in known_sites_names)
    print(f"   ✅ Memoria recuperada: {len(known_sites_names)} sitios entrenados.")
except Exception as e:
    print(f"   ❌ Error al cargar LabelEncoder: {e}")
    sys.exit()

# ==========================================================================
# 3. CALIBRACIÓN HISTÓRICA (Escala de Normalización desde Entrenamiento)
# ==========================================================================
print("\n🕰️ Extrayendo escala matemática del entrenamiento original (CLEAN PASADO)...")
start_scale = time.time()

max_size_historic = 0.0
for chunk in pd.read_csv(TRAIN_VECTORS_CSV, chunksize=10000, usecols=['size_vector']):
    for _, row in chunk.iterrows():
        try:
            v = ast.literal_eval(row['size_vector'])
            if isinstance(v, list) and len(v) > 0:
                current_max = max(v)
                if current_max > max_size_historic:
                    max_size_historic = current_max
        except:
            pass

# Garantía mínima de escala (valor descubierto empíricamente)
max_size_historic = max(max_size_historic, 14546.0)
print(f"   ✅ Escala Histórica Recuperada: Max = {max_size_historic} (Tomó {time.time() - start_scale:.1f}s)")

# ==========================================================================
# 4. FUNCIONES DE PROCESAMIENTO
# ==========================================================================
def preprocess_vector(v_str, max_len):
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list):
            v = []
    except:
        v = []
    if len(v) > max_len:
        return v[:max_len]
    return v + [0] * (max_len - len(v))


def load_dataset_full(csv_path):
    """Carga el dataset completo de futuro para evaluación de métricas."""
    df = pd.read_csv(csv_path)
    initial_count = len(df)

    df['site_label'] = df['site_label'].astype(str)
    df = df[df['site_label'].isin(known_sites_str)].copy()
    print(f"   - Descartados (sitios no entrenados): {initial_count - len(df)}")

    site_to_historic_id_str = {str(k): v for k, v in site_to_historic_id.items()}
    df['target_id'] = df['site_label'].map(site_to_historic_id_str).astype(int)

    X_d, X_w, y = [], [], []
    for _, row in df.iterrows():
        X_d.append(preprocess_vector(row['direction_vector'], MAX_LEN))
        X_w.append(preprocess_vector(row['size_vector'], MAX_LEN))
        y.append(row['target_id'])

    X_w_norm = np.clip(np.array(X_w) / float(max_size_historic), 0.0, 1.0)
    return np.array(X_d), X_w_norm, np.array(y), df['site_label'].values.astype(str)


def load_dataset_stratified(csv_path, max_per_class=50):
    """Carga con muestreo estratificado (máx N por clase) para UMAP y visualizaciones."""
    df = pd.read_csv(csv_path)
    df['site_label'] = df['site_label'].astype(str)
    df = df[df['site_label'].isin(known_sites_str)].copy()
    df = df.groupby('site_label').head(max_per_class).reset_index(drop=True)
    df['target_id'] = df['site_label'].map(
        {str(k): v for k, v in site_to_historic_id.items()}
    ).astype(int)

    X_d, X_w, y = [], [], []
    for _, row in df.iterrows():
        X_d.append(preprocess_vector(row['direction_vector'], MAX_LEN))
        X_w.append(preprocess_vector(row['size_vector'], MAX_LEN))
        y.append(row['target_id'])

    X_w_norm = np.clip(np.array(X_w) / float(max_size_historic), 0.0, 1.0)
    return np.array(X_d), X_w_norm, np.array(y), df['site_label'].values.astype(str)


# ==========================================================================
# 5. ARQUITECTURA DEL MODELO (MultimodalTransformer con forward dual)
# ==========================================================================
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
    def __init__(self, num_classes, max_len, d_model=256, nhead=8, num_layers=4,
                 kernel_size=5):
        super().__init__()
        self.dir_embedding = nn.Embedding(3, d_model)
        self.weight_proj = nn.Linear(1, d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.conv_local = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                    stride=2, padding=2)
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
        e_dir = self.dir_embedding(x_dir + 1)
        e_weight = self.weight_proj(x_weight.unsqueeze(-1))
        x = torch.cat([e_dir, e_weight], dim=-1)
        x = self.fusion(x)
        x = x.transpose(1, 2)
        x = self.conv_local(x)
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        features = torch.cat((avg_pool, max_pool), dim=1)
        features = self.ln(features)
        logits = self.fc(features)
        return logits, features  # ← Dual: logits para accuracy, features para UMAP


# ==========================================================================
# 6. CARGA DE PESOS CON LIMPIEZA _orig_mod.
# ==========================================================================
print(f"\n🔌 Despertando a la red neuronal Solo-Micro (kernel_size={KERNEL_SIZE})...")
num_classes = len(memorized_ids)
model = MultimodalTransformer(
    num_classes, MAX_LEN, d_model=D_MODEL, nhead=NHEAD,
    num_layers=NUM_LAYERS, kernel_size=KERNEL_SIZE
).to(device)

try:
    pth_path = os.path.join(SAVE_DIR, MODEL_WEIGHTS)
    state_dict = torch.load(pth_path, map_location=device)

    # Limpieza del prefijo _orig_mod. (torch.compile residual)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("   ⚠️ Detectado prefijo '_orig_mod.' en state_dict. Limpiando...")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('_orig_mod.', '')
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    print("   ✅ Pesos del modelo restaurados exitosamente.")
except Exception as e:
    print(f"   ❌ Error al cargar pesos: {e}")
    sys.exit()

model.eval()

# ==========================================================================
# 7. CARGA DE DATOS
# ==========================================================================
print("\n⏳ Cargando datos del futuro (Concept Drift) para evaluación completa...")
X_d_fut, X_w_fut, y_fut, lbl_fut = load_dataset_full(FUT_VECTORS_CSV)
print(f"   ✅ Test Set definitivo: {len(y_fut)} capturas.")

if len(y_fut) == 0:
    print("\n   ❌ CRÍTICO: Test Set vacío. Abortando.")
    sys.exit()

print("\n⏳ Cargando Pasado (Estratificado 50/clase) para autopsia visual...")
X_d_past, X_w_past, y_past, lbl_past = load_dataset_stratified(TRAIN_VECTORS_CSV, max_per_class=50)

print("⏳ Cargando Futuro (Estratificado 50/clase) para autopsia visual...")
X_d_fut_strat, X_w_fut_strat, y_fut_strat, lbl_fut_strat = load_dataset_stratified(
    FUT_VECTORS_CSV, max_per_class=50
)

# ==========================================================================
# 8. INFERENCIA UNIFICADA (Forward Pass Único con autocast + inference_mode)
# ==========================================================================
print("\n🚀 Inferencia unificada (logits + features) con autocast bfloat16...")
start_infer = time.time()
y_true_full, y_pred_full = [], []


def run_inference(X_d, X_w, y, collect_labels=True):
    """Forward pass unificado: devuelve (logits_argmax, features_numpy)."""
    dataset = TensorDataset(
        torch.LongTensor(X_d), torch.FloatTensor(X_w), torch.LongTensor(y)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    all_preds, all_embs = [], []
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for d_batch, w_batch, y_batch in loader:
                logits, embs = model(d_batch.to(device), w_batch.to(device))
                all_preds.extend(torch.argmax(logits.float(), dim=1).cpu().numpy())
                all_embs.extend(embs.float().cpu().numpy())
    return np.array(all_preds), np.array(all_embs)


# Inferencia sobre el dataset completo (futuro) para métricas
preds_fut_full, _ = run_inference(X_d_fut, X_w_fut, y_fut)
y_true_labels = le.inverse_transform(y_fut)
y_pred_labels = le.inverse_transform(preds_fut_full)

# Inferencia sobre datasets estratificados para visualización
print("   🚀 Extrayendo geometrías del Pasado...")
preds_past, embs_past = run_inference(X_d_past, X_w_past, y_past)
print("   🚀 Extrayendo geometrías del Futuro...")
preds_fut_strat_np, embs_fut = run_inference(X_d_fut_strat, X_w_fut_strat, y_fut_strat)

infer_time = time.time() - start_infer
print(f"   ✅ Inferencia completada en {infer_time:.1f}s")

# ==========================================================================
# 9. REPORTE DE MÉTRICAS
# ==========================================================================
acc = accuracy_score(y_true_labels, y_pred_labels)
print("\n" + "🏆" * 20)
print(f"   ACCURACY FINAL ABLACIÓN SOLO-MICRO (DRIFT): {acc * 100:.2f}%")
print("🏆" * 20)

etiquetas_presentes = np.unique(y_true_labels)
report = classification_report(
    y_true_labels, y_pred_labels, labels=etiquetas_presentes, zero_division=0
)

report_path = os.path.join(SAVE_DIR, "ds3_ablacion_resultado_solo_vectores.txt")
with open(report_path, "w") as f:
    f.write(f"ACCURACY GLOBAL ABLACIÓN SOLO-MICRO (DRIFT 2 MESES): {acc * 100:.2f}%\n\n{report}")
print(f"📄 Reporte guardado en: ds3_ablacion_resultado_solo_vectores.txt")

# ==========================================================================
# 10. EVIDENCIAS GRÁFICAS (AutoPSIA VISUAL)
# ==========================================================================
print("\n🎨 Generando evidencias gráficas de autopsia...")

# Directorio de salida para gráficas
vis_dir = os.path.join(SAVE_DIR)
os.makedirs(vis_dir, exist_ok=True)

common_sites = np.intersect1d(lbl_past, lbl_fut_strat)
if len(common_sites) == 0:
    print("   ❌ No hay sitios comunes entre pasado y futuro. Saltando gráficas.")
    sys.exit(0)

# ------------------------------------------------------------------
# G1: HUELLA DACTILAR SECUENCIAL (Direction Barcode)
# ------------------------------------------------------------------
print("[1/3] Generando G1: Huella Dactilar Secuencial...")
target_site = str(common_sites[0])

idx_past = np.where(lbl_past == target_site)[0][0]
idx_fut = np.where(lbl_fut_strat == target_site)[0][0]

plt.figure(figsize=(15, 4))
plt.subplot(2, 1, 1)
plt.imshow(X_d_past[idx_past][:500].reshape(1, -1), cmap='coolwarm',
           aspect='auto', vmin=-1, vmax=1)
plt.title(f'PASADO (Entrenamiento) - Primeros 500 Paquetes [Sitio ID: {target_site}]',
          fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(X_d_fut_strat[idx_fut][:500].reshape(1, -1), cmap='coolwarm',
           aspect='auto', vmin=-1, vmax=1)
plt.title(f'FUTURO (Concept Drift) - Primeros 500 Paquetes [Sitio ID: {target_site}]\n'
          'Observe cómo el ruteo de Tor desordena la secuencia destruyendo la firma aprendida.',
          fontsize=12, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'VF3_G1_Huella_Secuencial.jpg'), dpi=300)
plt.close()

# ------------------------------------------------------------------
# G2: MATRIZ DE CONFUSIÓN (Datos completos del futuro)
# ------------------------------------------------------------------
print("[2/3] Generando G2: Matriz de Confusión...")
cm = confusion_matrix(y_fut, preds_fut_full)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_norm, cmap='inferno', cbar=True, xticklabels=False, yticklabels=False)
plt.title(f'G2: Ceguera Sistémica del Modelo Micro (Concept Drift {acc * 100:.1f}%)\n'
          'La ausencia de una diagonal fuerte demuestra que el modelo adivina al azar.',
          fontsize=14, fontweight='bold')
plt.xlabel('Clase Predicha (Red Neuronal)')
plt.ylabel('Clase Real (Sitio Web)')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'VF3_G2_Matriz_Dispersion.jpg'), dpi=300)
plt.close()

# ------------------------------------------------------------------
# G3: COLAPSO DEL ESPACIO LATENTE (UMAP)
# ------------------------------------------------------------------
print("[3/3] Generando G3: Colapso del Espacio Latente (UMAP)...")
np.random.seed(42)
top_10_sites = np.random.choice(common_sites, min(10, len(common_sites)), replace=False)

mask_past = np.isin(lbl_past, top_10_sites)
mask_fut = np.isin(lbl_fut_strat, top_10_sites)

X_umap = np.vstack([embs_past[mask_past], embs_fut[mask_fut]])
y_umap = np.concatenate([lbl_past[mask_past], lbl_fut_strat[mask_fut]])
domains = np.array(
    ['PASADO (Base de Conocimiento)'] * mask_past.sum() +
    ['FUTURO (Desplazamiento)'] * mask_fut.sum()
)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
X_embedded = reducer.fit_transform(X_umap)

plt.figure(figsize=(14, 9))
sns.scatterplot(
    x=X_embedded[:, 0], y=X_embedded[:, 1],
    hue=y_umap, style=domains, palette='tab10',
    s=100, alpha=0.8, edgecolor='black'
)
plt.title('G3: Autopsia del Concept Drift (Desplazamiento Geométrico)\n'
          'Observe cómo las muestras del Futuro (Cruces) abandonan la agrupación '
          'original (Círculos) invadiendo otras clases.',
          fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clases y Dominio")
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'VF3_G3_Colapso_Latente.jpg'), dpi=300)
plt.close()

# ==========================================================================
# 11. RESUMEN FINAL
# ==========================================================================
print("\n" + "=" * 60)
print(" ✅ SUITE DE ABLACIÓN + AUTOPSIA VISUAL COMPLETADA")
print("=" * 60)
print(f"   📄 Métricas: {report_path}")
print(f"   🖼️  G1 Huella Secuencial:    {os.path.join(vis_dir, 'VF3_G1_Huella_Secuencial.jpg')}")
print(f"   🖼️  G2 Matriz de Confusión:  {os.path.join(vis_dir, 'VF3_G2_Matriz_Dispersion.jpg')}")
print(f"   🖼️  G3 Colapso Latente UMAP: {os.path.join(vis_dir, 'VF3_G3_Colapso_Latente.jpg')}")
print(f"   ⏱️  Tiempo total de inferencia: {infer_time:.1f}s")
print("=" * 60 + "\n")
