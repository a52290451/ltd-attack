import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import ast
import time
import os
import joblib

# --- 1. CONFIGURACIÓN ---
print("\n" + "═"*60)
print("-- INICIANDO EXPERIMENTO: TRANSFORMER DE VECTORES DE DIRECCIÓN")
print("═"*60)

INPUT_CSV = '../../output/final_vectors_sites.csv'
SAVE_DIR = './resultados'
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_LEN = 3000 
BATCH_SIZE = 64
EPOCHS = 100
D_MODEL = 256       # Dimensión del modelo (ajustada para una sola rama)
NHEAD = 8           # Cabezas de atención
NUM_LAYERS = 6      # Capas del encoder

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

print(f"--  Procesando vectores duales (Max Len: {MAX_LEN})...")
X = np.array([preprocess_vector(row, MAX_LEN) for row in df_filtered['direction_vector']])

# Codificación de etiquetas
le = LabelEncoder()
y = le.fit_transform(df_filtered['site_label'])
num_classes = len(le.classes_)

# Split Estratificado para mantener proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"-- Split: Train={len(X_train)} | Test={len(X_test)}")

# ========================================================
# 🔍 AQUÍ VA EL BLOQUE DE INSPECCIÓN
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
sample_vector = X[sample_idx]
sample_label = le.inverse_transform([y_train[sample_idx]])[0]

print(f"\n-- Ejemplo de la muestra #{sample_idx}:")
print(f"   - Sitio Web (Label Real): {sample_label}")
print(f"   - Forma del vector (Shape): {sample_vector.shape}")
print(f"   - Primeros 20 paquetes (Dirección): {sample_vector[:20]}")
print(f"   - Últimos 20 paquetes (Debería haber ceros si hay padding): {sample_vector[-20:]}")

# 3. Verificación de Valores Únicos
valores_unicos = np.unique(X)
print(f"\n-- Valores únicos detectados en X_train: {valores_unicos}")
if not set(valores_unicos).issubset({-1, 0, 1}):
    print("⚠️ ADVERTENCIA: Se detectaron valores fuera de [-1, 0, 1]. Revisa el preprocesamiento.")
else:
    print("-- Consistencia de valores: OK (solo -1, 0 y 1)")

# 4. Estadísticas de Relleno (Padding)
ceros_promedio = (X == 0).sum() / X_train.size * 100
print(f"--  Porcentaje de Padding (ceros) en el dataset: {ceros_promedio:.2f}%")
print("═"*60 + "\n")

# Creación de DataLoaders
train_loader = DataLoader(
    TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train)), 
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test)), 
    batch_size=BATCH_SIZE
)


# --- 4. ARQUITECTURA TRANSFORMER ---
    
class DirectionTransformer(nn.Module):
    def __init__(self, num_classes, max_len, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        # Aumentamos la resolución del embedding
        self.embedding = nn.Embedding(3, d_model) 
        
        # Nueva: Capa Convolucional para detectar "ráfagas" locales antes de la atención
        self.conv_local = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        # Transformer más profundo
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, 
            dropout=0.1, # Menos dropout para que no pierda tanta señal
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.ln = nn.LayerNorm(d_model)
        
        # Clasificador más robusto
        self.fc = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x + 1) # [Batch, Seq, Model]
        
        # 1. Extraer ráfagas locales con Convolución
        # (Cambiamos dimensiones para Conv1d: [Batch, Model, Seq])
        x = x.transpose(1, 2)
        x = self.conv_local(x)
        x = x.transpose(1, 2) # Volvemos a [Batch, Seq, Model]
        
        x = x + self.pos_embedding
        x = self.transformer(x)
        
        # 2. CAMBIO CRÍTICO: Global Max Pooling en lugar de Mean
        # Esto captura la "ráfaga" más distintiva de todo el flujo
        x, _ = torch.max(x, dim=1) 
        
        x = self.ln(x)
        return self.fc(x)


model = DirectionTransformer(num_classes, MAX_LEN, d_model=D_MODEL).to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Sin pesos por ahora para ver si aprende la base
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# --- 5. ENTRENAMIENTO ---
print(f"🏋️  Entrenando en {device} con {num_classes} clases...")
best_acc = 0.0
start_train = time.time()
patience = 10  # Si no mejora en 10 épocas, paramos
trigger_times = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    epoch_start = time.time()
    
    for b_x, b_y in train_loader:
        b_x, b_y = b_x.to(device), b_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(b_x), b_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Evaluación
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b_x, b_y in test_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            preds = torch.argmax(model(b_x), dim=1)
            correct += (preds == b_y).sum().item()
            total += b_y.size(0)
    
    acc = correct / total
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] - Loss: {total_loss/len(train_loader):.4f} - Acc: {acc:.4f} - {time.time()-epoch_start:.1f}s")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"{SAVE_DIR}/d_best_multimodal_transformer_vec_3000.pth")
        print(f"***** Nuevo mejor accuracy: {best_acc:.4f} (Modelo guardado)")
        trigger_times = 0 # Reset
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping en época {epoch+1}")
            break

print(f"\n✅ Entrenamiento finalizado en {(time.time() - start_train)/60:.2f} minutos.")
print(f"🏆 Mejor Accuracy en Test: {best_acc:.4f}")

# --- 6. GUARDADO DE METADATOS ---
mapping = dict(zip(range(len(le.classes_)), le.classes_))
joblib.dump(le, f"{SAVE_DIR}/d_label_encoder_vec_3000.joblib")
joblib.dump(mapping, f"{SAVE_DIR}/d_class_mapping_vec_3000.joblib") # Más fácil de leer luego
print(f"💾 Metadatos guardados en {SAVE_DIR}")

print("\n🧪 GENERANDO REPORTE FINAL...")
model.load_state_dict(torch.load(f"{SAVE_DIR}/d_best_multimodal_transformer_vec_3000.pth"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for b_x, b_y in test_loader:
        b_x, b_y = b_x.to(device), b_y.to(device)
        preds = torch.argmax(model(b_x), dim=1)
        y_true.extend(b_y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

report = classification_report(y_true, y_pred, target_names=le.classes_.astype(str))
with open(f"{SAVE_DIR}/d_resultado_final_3000.txt", "w") as f:
    f.write(report)
print(report)