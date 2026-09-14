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

MAX_LEN = 5000 
BATCH_SIZE = 64
EPOCHS = 100

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
def preprocess_vector(v_str, max_len, is_size=False):
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list): v = []
    except: v = []
    
    # Normalización logarítmica para tamaños de paquete
    if is_size:
        v = [np.log1p(abs(x)) for x in v]
        
    if len(v) > max_len: return v[:max_len]
    else: return v + [0] * (max_len - len(v))

print(f"--  Procesando vectores duales (Max Len: {MAX_LEN})...")
X_dir = np.array([preprocess_vector(row, MAX_LEN) for row in df_filtered['direction_vector']])
X_size = np.array([preprocess_vector(row, MAX_LEN, is_size=True) for row in df_filtered['size_vector']])

print("--  Estandarizando vector de tamaños...")
X_size = (X_size - X_size.mean()) / (X_size.std() + 1e-6)

le = LabelEncoder()
y = le.fit_transform(df_filtered['site_label'])
num_classes = len(le.classes_)

# Split (Incluye ambos vectores)
X_dir_tr, X_dir_ts, X_size_tr, X_size_ts, y_train, y_test = train_test_split(
    X_dir, X_size, y, test_size=0.2, stratify=y, random_state=42
)
print(f"-- Split: Train={len(X_dir_tr)} | Test={len(X_dir_ts)}")

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
sample_vector = X_dir_tr[sample_idx]
sample_label = le.inverse_transform([y_train[sample_idx]])[0]

print(f"\n-- Ejemplo de la muestra #{sample_idx}:")
print(f"   - Sitio Web (Label Real): {sample_label}")
print(f"   - Forma del vector (Shape): {sample_vector.shape}")
print(f"   - Primeros 20 paquetes (Dirección): {sample_vector[:20]}")
print(f"   - Últimos 20 paquetes (Debería haber ceros si hay padding): {sample_vector[-20:]}")

# 3. Verificación de Valores Únicos
valores_unicos = np.unique(X_dir_tr)
print(f"\n-- Valores únicos detectados en X_train: {valores_unicos}")
if not set(valores_unicos).issubset({-1, 0, 1}):
    print("⚠️ ADVERTENCIA: Se detectaron valores fuera de [-1, 0, 1]. Revisa el preprocesamiento.")
else:
    print("-- Consistencia de valores: OK (solo -1, 0 y 1)")

# 4. Estadísticas de Relleno (Padding)
ceros_promedio = (X_dir_tr == 0).sum() / X_dir_tr.size * 100
print(f"--  Porcentaje de Padding (ceros) en el dataset: {ceros_promedio:.2f}%")
print("═"*60 + "\n")

# DataLoaders con 3 elementos por muestra
train_loader = DataLoader(
    TensorDataset(torch.LongTensor(X_dir_tr), torch.FloatTensor(X_size_tr), torch.LongTensor(y_train)), 
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.LongTensor(X_dir_ts), torch.FloatTensor(X_size_ts), torch.LongTensor(y_test)), 
    batch_size=BATCH_SIZE
)


# --- 4. ARQUITECTURA TRANSFORMER ---

class MultiModalTransformer(nn.Module):
    def __init__(self, num_classes, max_len, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.dir_embedding = nn.Embedding(3, 128) 
        self.size_projection = nn.Linear(1, 128)
        
        # Fusión y posición
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, 
            dropout=0.2, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        
        # Capas de salida más profundas
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, d_vec, s_vec):
        # Eliminamos la máscara de padding para depurar el aprendizaje
        d_feat = self.dir_embedding(d_vec + 1)
        s_feat = self.size_projection(s_vec.unsqueeze(-1))
        
        x = torch.cat([d_feat, s_feat], dim=-1) 
        x = x + self.pos_embedding
        
        # Transformer puro sin máscara externa
        x = self.transformer(x)
        
        # Cambiamos a Mean Pooling (promedio) que es más estable que Max al inicio
        x = x.mean(dim=1) 
        
        x = self.ln(x)
        return self.fc(x)


model = MultiModalTransformer(num_classes, MAX_LEN).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
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
    
    for b_dir, b_size, b_y in train_loader:
        b_dir, b_size, b_y = b_dir.to(device), b_size.to(device), b_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(b_dir, b_size), b_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b_dir, b_size, b_y in test_loader:
            b_dir, b_size, b_y = b_dir.to(device), b_size.to(device), b_y.to(device)
            preds = torch.argmax(model(b_dir, b_size), dim=1)
            correct += (preds == b_y).sum().item()
            total += b_y.size(0)
    
    acc = correct / total
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] - Loss: {total_loss/len(train_loader):.4f} - Acc: {acc:.4f} - {time.time()-epoch_start:.1f}s")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"{SAVE_DIR}/ds_best_multimodal_transformer_vec_5000.pth")
        print(f"   ⭐ Nuevo mejor accuracy: {best_acc:.4f} (Modelo guardado)")
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
joblib.dump(le, f"{SAVE_DIR}/ds_label_encoder_vec_5000.joblib")
joblib.dump(mapping, f"{SAVE_DIR}/ds_class_mapping_vec_5000.joblib") # Más fácil de leer luego
print(f"💾 Metadatos guardados en {SAVE_DIR}")

print("\n🧪 GENERANDO REPORTE FINAL...")
model.load_state_dict(torch.load(f"{SAVE_DIR}/ds_best_multimodal_transformer_vec_5000.pth"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for b_dir, b_size, b_y in test_loader:
        b_dir, b_size, b_y = b_dir.to(device), b_size.to(device), b_y.to(device)
        outputs = model(b_dir, b_size)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(b_y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

report = classification_report(y_true, y_pred, target_names=le.classes_.astype(str))
with open(f"{SAVE_DIR}/ds_resultado_final.txt_5000", "w") as f:
    f.write(report)
print(report)