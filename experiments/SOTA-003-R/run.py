"""
=========================================================================================
UNIVERSIDAD / INSTITUTO DE INVESTIGACION Y DESARROLLO DE IA
LABORATORIO DE CIBERSEGURIDAD Y ANALISIS DE TRAFICO AVANZADO (WFP)
=========================================================================================
ARCHIVO: SOTA_03_Rimmer_LSTM.py
VERSION: 2.0 (Deep Learning Baseline - Rimmer LSTM Refactored)
INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
RESUMEN ACADÉMICO / METODOLOGIA:
    Tercer baseline del Estado del Arte para evaluar el impacto del Concept Drift en
    Website Fingerprinting (WFP) mediante Deep Learning.

    Implementa la arquitectura LSTM (Rimmer et al., 2018) aplicada directamente a los
    vectores de dirección de paquetes (direction_vector) de tamaño fijo (3000 posiciones).

    La LSTM transforma las secuencias de direcciones de paquetes (+1 saliente, -1 entrante,
    0 padding) en representaciones temporales que capturan patrones secuenciales
    del tráfico de red.

    HIPOTESIS:
    - Una LSTM 1D sobre vectores de dirección puede capturar patrones temporales del
      tráfico que son informativos para la clasificación de sitios web.
    - El modelo sufrirá degradación significativa bajo Concept Drift debido al cambio
      en las rutas de Tor que alteran la firma temporal de los paquetes.

ENTRADAS:
    - data_path('historical', 'CLEAN_final_vectors_sites.csv') (Pasado / Entrenamiento)
    - data_path('historical', 'CLEAN_final_vectors_sites_concept_drift.csv') (Futuro / Concept Drift)
    - artifact_path('ds3_label_encoder_vec_3000.joblib') (LabelEncoder de 65 clases)

SALIDAS:
    - result_path('sota')/log_SOTA_03_LSTM_[timestamp].txt
    - result_path('sota')/best_rimmer_lstm.pth (mejores pesos del modelo)
    - result_path('sota')/G_SOTA_03_LSTM_Curvas_[timestamp].png (curvas de entrenamiento)
=========================================================================================
"""

# =============================================================================
# 0. IMPORTACIONES Y CONFIGURACION GLOBAL
# =============================================================================
import matplotlib
matplotlib.use('Agg')  # Backend headless para servidores

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import ast
import os
import sys
import datetime
import joblib
from src.utils.paths import data_path, artifact_path, result_path

# =============================================================================
# 1. SISTEMA DE LOGGER (Global - copiado de SOTA_02)
# =============================================================================
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


# =============================================================================
# 2. CONFIGURACION DE RUTAS Y DIRECTORIOS
# =============================================================================
DIR_DATA = data_path('historical')
TRAIN_VECTORS_CSV = os.path.join(DIR_DATA, 'CLEAN_final_vectors_sites.csv')
FUT_VECTORS_CSV = r'/home/bsierra/ltd-storage/datasets/future/CLEAN_final_vectors_sites_concept_drift.csv'

LABEL_ENCODER_PATH = r'/home/bsierra/ltd-storage/artifacts/micro/MICRO-002-R/ds3_label_encoder_vec_3000.joblib'

SAVE_DIR = r'/home/bsierra/ltd-storage/results/canonical/SOTA-003-R'
os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filepath = os.path.join(SAVE_DIR, f"log_SOTA_03_LSTM_{timestamp}.txt")
sys.stdout = Logger(log_filepath)

print("\n" + "=" * 70)
print("SOTA_03 - RIMMER LSTM BASELINE TEMPORAL")
print("   Baseline LSTM aplicado a vectores de red (WFP)")
print("=" * 70)

# =============================================================================
# 3. HIPERPARAMETROS
# =============================================================================
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EMBEDDING_DIM = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
MAX_SEQ_LEN = 3000
NUM_CLASSES = None  # Se determina dinámicamente del LabelEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo configurado: {device}")

# Semilla para reproductibilidad
TORCH_SEED = 42
torch.manual_seed(TORCH_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# 4. CARGA DE MEMORIA (LabelEncoder + Filtro de Paridad Estricta)
# =============================================================================
print("\n[Step 1] Cargando LabelEncoder y filtrando clases de elite...")
try:
    le = joblib.load(LABEL_ENCODER_PATH)
    sitios_elite = set(le.classes_.astype(str))
    num_classes = len(le.classes_)
    print(f"   LabelEncoder cargado: {num_classes} clases de elite (65 clases).")
except Exception as e:
    print(f"   ERROR al cargar LabelEncoder: {e}")
    sys.exit()


# =============================================================================
# 5. FUNCION DE PREPROCESAMIENTO (de DIA_01_Aislamiento_Sitios_Estables.py)
# =============================================================================
def preprocess_vector(v_str, max_len):
    """
    Parsea y normaliza un vector de direccion a longitud fija.
    Copia exacta de DIA_01_Aislamiento_Sitios_Estables.py (linea 143).

    Args:
        v_str: String representacion de una lista Python de valores.
        max_len: Longitud objetivo (3000 por defecto).

    Returns:
        Lista de valores paddeada o truncada a max_len posiciones.
        Los vectores de direccion contienen:
            +1 -> paquete entrante (incoming)
            -1 -> paquete saliente (outgoing)
            0  -> padding
    """
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list):
            v = []
    except Exception:
        v = []
    if len(v) > max_len:
        return v[:max_len]
    return v + [0] * (max_len - len(v))


# =============================================================================
# 6. CARGA DE DATOS Y PREPROCESAMIENTO
# =============================================================================
print("\n[Step 2] Cargando datos del PASADO y FUTURO...")

# --- 6.1 PASADO ---
print("\n   Cargando dataset del PASADO...")
df_past = pd.read_csv(TRAIN_VECTORS_CSV)
df_past['site_label'] = df_past['site_label'].astype(str)
initial_past = len(df_past)
df_past = df_past[df_past['site_label'].isin(sitios_elite)].copy()
print(f"   PASADO: {len(df_past)} registros despues de filtrado por elite "
      f"({initial_past - len(df_past)} descartados).")

# --- 6.2 FUTURO ---
print("\n   Cargando dataset del FUTURO (Concept Drift)...")
df_fut = pd.read_csv(FUT_VECTORS_CSV)
df_fut['site_label'] = df_fut['site_label'].astype(str)
initial_fut = len(df_fut)
df_fut = df_fut[df_fut['site_label'].isin(sitios_elite)].copy()
print(f"   FUTURO: {len(df_fut)} registros despues de filtrado por elite "
      f"({initial_fut - len(df_fut)} descartados).")

# --- 6.3 MAPEO DE ETIQUETAS ---
site_to_historic_id = {str(site): idx for idx, site in enumerate(le.classes_)}

# --- 6.4 PROCESAMIENTO DE VECTORES DE DIRECCION ---
print(f"\n[Step 3] Procesando vectores de direccion (max_len={MAX_SEQ_LEN})...")

# Pasado
X_d_past = []
y_past = []
for _, row in df_past.iterrows():
    vec = preprocess_vector(row['direction_vector'], MAX_SEQ_LEN)
    X_d_past.append(vec)
    y_past.append(site_to_historic_id[str(row['site_label'])])

X_d_past = np.array(X_d_past)
y_past = np.array(y_past)
del df_past  # Liberar memoria
print(f"   PASADO procesado: X={X_d_past.shape}, y={y_past.shape}")

# Futuro
X_d_fut = []
y_fut = []
for _, row in df_fut.iterrows():
    vec = preprocess_vector(row['direction_vector'], MAX_SEQ_LEN)
    X_d_fut.append(vec)
    y_fut.append(site_to_historic_id[str(row['site_label'])])

X_d_fut = np.array(X_d_fut)
y_fut = np.array(y_fut)
del df_fut  # Liberar memoria
print(f"   FUTURO procesado: X={X_d_fut.shape}, y={y_fut.shape}")

# --- 6.5 TRAIN/VALIDATION SPLIT ---
print(f"\n[Step 4] Dividiendo PASADO en Train/Validation (80/20)...")
X_train, X_val, y_train, y_val = train_test_split(
    X_d_past, y_past, test_size=0.2, random_state=TORCH_SEED, stratify=y_past
)
print(f"   Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples")
print(f"   Futuro (drift): {X_d_fut.shape[0]} samples")


# =============================================================================
# 7. ARQUITECTURA LSTM (Rimmer et al., 2018)
# =============================================================================
class WebsiteFingerprintingLSTM(nn.Module):
    """
    LSTM para Website Fingerprinting basado en Vera-Rimmer et al. (2018).

    Arquitectura:
    1. nn.Embedding(4, 32) -> codifica direcciones {-1, 0, 1} a {0, 1, 2} 
       en embeddings de 32D. El +1 a la entrada mapea: -1->0, 0->1, 1->2.
       La salida es (batch, seq_len, embed_dim).

    2. LSTM bidireccional (2 capas, hidden=128, dropout=0.2)

    3. Extracción del contexto final (último paso de tiempo)

    4. Clasificador lineal (128 -> NUM_CLASSES)

    El vocab_size es 4 porque:
      - Valores originales: {-1, 0, 1} (3 valores)
      - Tras +1: {0, 1, 2} (3 índices)
      - +1 índice para padding seguro = 4 (vocab_size)
    """

    def __init__(self, num_classes, vocab_size=4, embedding_dim=32, 
                 hidden_size=128, num_layers=2, dropout=0.2):
        super(WebsiteFingerprintingLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) de tipos LongTensor con valores {-1, 0, 1}
        Returns:
            logits: (batch, num_classes)
        """
        # Embedding: suma +1 para mapear {-1,0,1} -> {0,1,2}
        x = x + 1  # (batch, seq_len) con indices {0,1,2}
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_size)

        # Extraer contexto del ultimo paso de tiempo
        last_step = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Clasificador
        out = self.dropout(last_step)
        logits = self.classifier(out)  # (batch, num_classes)
        return logits


# =============================================================================
# 8. ENTRENAMIENTO (PyTorch 2.0 con Precision Mixta)
# =============================================================================
print("\n[Step 5] Instanciando modelo WebsiteFingerprintingLSTM...")
model = WebsiteFingerprintingLSTM(
    num_classes=num_classes,
    vocab_size=4,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

# Conteo de parametros
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Parametros totales: {total_params:,}")
print(f"   Parametros entrenables: {trainable_params:,}")

# --- Loss, Optimizador, GradScaler ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler(device='cuda')

# --- DataLoaders ---
X_train_t = torch.LongTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.LongTensor(X_val)
y_val_t = torch.LongTensor(y_val)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(X_val_t, y_val_t)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Entrenamiento con Early Stopping ---
print(f"\n[Step 6] Entrenamiento (max={NUM_EPOCHS} epochs, early_stop_patience={EARLY_STOP_PATIENCE})...")

best_val_loss = float('inf')
best_model_state = None
patience_counter = 0
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train()
    epoch_train_loss = 0.0
    epoch_train_correct = 0
    epoch_train_total = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # Precision mixta con torch.autocast
        with autocast('cuda' if device.type == 'cuda' else 'cpu'):
            logits = model(batch_X)
            loss = criterion(logits, batch_y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_train_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(logits, 1)
        epoch_train_correct += (predicted == batch_y).sum().item()
        epoch_train_total += batch_X.size(0)

    # --- Validation Phase ---
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_correct = 0
    epoch_val_total = 0

    with torch.inference_mode():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                logits = model(batch_X)
                loss = criterion(logits, batch_y)

            epoch_val_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(logits, 1)
            epoch_val_correct += (predicted == batch_y).sum().item()
            epoch_val_total += batch_X.size(0)

    # Metrics
    avg_train_loss = epoch_train_loss / epoch_train_total
    avg_val_loss = epoch_val_loss / epoch_val_total
    train_accuracy = epoch_train_correct / epoch_train_total
    val_accuracy = epoch_val_correct / epoch_val_total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(train_accuracy)
    val_accs.append(val_accuracy)

    if True:
        print(f"   Epoch [{epoch:03d}/{NUM_EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")

    # --- Early Stopping Check ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n   Early Stopping en epoch {epoch}. Mejor Val Loss: {best_val_loss:.4f}")
            break

# =============================================================================
# 9. GUARDAR MEJORES PESOS
# =============================================================================
best_model_path = os.path.join(SAVE_DIR, 'best_rimmer_lstm.pth')
torch.save(best_model_state, best_model_path)
print(f"\n   Mejores pesos guardados en: {best_model_path}")


# =============================================================================
# 10. CARGAR MEJORES PESOS Y EVALUAR (EN BATCHES PARA EVITAR CUDA OOM)
# =============================================================================
print("\n[Step 7] Evaluando modelo...")
model.load_state_dict(best_model_state)

def get_accuracy_in_batches(X_data, y_data):
    model.eval()
    dataset = TensorDataset(torch.LongTensor(X_data), torch.LongTensor(y_data))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    correct = 0
    total = 0
    all_preds = []
    with torch.inference_mode():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                logits = model(batch_X)
                _, preds = torch.max(logits, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            all_preds.extend(preds.cpu().numpy())
    return correct / total, torch.tensor(all_preds)

# --- 10.1 Train Accuracy ---
train_accuracy_final, _ = get_accuracy_in_batches(X_train, y_train)

# --- 10.2 Validation Accuracy ---
val_accuracy_final, _ = get_accuracy_in_batches(X_val, y_val)

# --- 10.3 Futuro (Concept Drift) Accuracy ---
print("\n   Evaluando sobre dataset del FUTURO (Concept Drift)...")
drift_accuracy, pred_fut = get_accuracy_in_batches(X_d_fut, y_fut)

# =============================================================================
# 11. REPORTE DE METRICAS FINALES
# =============================================================================
print("\n" + "=" * 70)
print("RESULTADOS FINALES - RIMMER LSTM")
print("=" * 70)
print(f"   Train Accuracy (80% Pasado):  {train_accuracy_final * 100:.2f}%")
print(f"   Validation Accuracy (20% Pasado): {val_accuracy_final * 100:.2f}%")
print(f"   Accuracy frente al Concept Drift (FUTURO): {drift_accuracy * 100:.2f}%")
print(f"   Impacto del Concept Drift: {(train_accuracy_final - drift_accuracy) * 100:.2f} pp")
print("=" * 70)

if drift_accuracy < val_accuracy_final:
    print(f"\n   ⚠️  SE DETECTA DEGRADACION por Concept Drift: "
          f"{(val_accuracy_final - drift_accuracy) * 100:.2f} pp de perdida.")
else:
    print(f"\n   ✅ No se detecto degradacion significativa por Concept Drift.")

# Classification Report para Futuro
print("\n   Classification Report (FUTURO - Concept Drift):")
print(classification_report(y_fut, pred_fut.cpu().numpy(), digits=3, zero_division=0))

# =============================================================================
# 12. GRAFICA: Curvas de Train vs Val Loss y Accuracy
# =============================================================================
print("\n[Step 8] Generando grafica de curvas de entrenamiento...")

sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'axes.linewidth': 1.2,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                gridspec_kw={'height_ratios': [1, 1]})
fig.suptitle('SOTA_03 - Rimmer LSTM Baseline Temporal\n'
             'Curvas de Entrenamiento y Validacion',
             fontweight='bold', fontsize=15, y=1.01)

epochs_range = range(len(train_losses))

# --- Panel Superior: Train vs Val Loss ---
ax1.plot(epochs_range, train_losses, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3)
ax1.plot(epochs_range, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (Cross-Entropy)', fontweight='bold')
ax1.set_title('Train vs Validation Loss', fontweight='bold')
ax1.legend(loc='upper right', frameon=True)
ax1.grid(alpha=0.4, linestyle='--')
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

# --- Panel Inferior: Train vs Val Accuracy ---
ax2.plot(epochs_range, train_accs, 'b-', linewidth=2, label='Train Accuracy', marker='o', markersize=3)
ax2.plot(epochs_range, val_accs, 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=3)
ax2.axhline(y=drift_accuracy, color='green', linestyle=':', linewidth=2,
            label=f'Drift Acc: {drift_accuracy:.4f}', alpha=0.7)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy', fontweight='bold')
ax2.set_title('Train vs Validation Accuracy', fontweight='bold')
ax2.legend(loc='lower right', frameon=True)
ax2.grid(alpha=0.4, linestyle='--')
ax2.set_ylim(0, 1.05)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

plt.tight_layout()
curvas_path = os.path.join(SAVE_DIR, f'G_SOTA_03_LSTM_Curvas_{timestamp}.png')
fig.savefig(curvas_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print(f"   Grafica guardada en: {curvas_path}")

# =============================================================================
# 13. CIERRE
# =============================================================================
print(f"\n{'=' * 70}")
print("SOTA_03 - RIMMER LSTM COMPLETADO EXITOSAMENTE")
print(f"{'=' * 70}")
print(f"Log de ejecucion:      {log_filepath}")
print(f"Mejores pesos:         {best_model_path}")
print(f"Grafica de curvas:     {curvas_path}")
print(f"{'=' * 70}")

# Restaurar stdout
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
