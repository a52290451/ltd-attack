import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import time
import joblib
import os

# --- 1. CONFIGURACIÓN Y CARGA ---
print("Iniciando Experimento: Deep MLP (Registros Independientes)")
INPUT_CSV = '../output/preprocessed/01_all_features.csv'
SAVE_DIR = './output'
os.makedirs(SAVE_DIR, exist_ok=True)

FEATURES_A_ANALIZAR = [
    'ratio_pkts_firstN_bursts_total', 'cumul_out_auc', 'bytes_last10pct',
        'ngrams_4_min_count', 'packets_last25pct', 'bytes_first25pct',
        'iat_size_p90', 'pps_w1_max', 'global_packet_count_75',
        'ngrams_4_top5_count', 'cumul_interp_out_diffs_std', 'bytes_first10pct',
        'cumul_interp_out_kurtosis', 'global_packet_count_50', 'cumul_in_skew',
]

start_time = time.time()

# Cargar datos
print(f"📂 Cargando dataset...")
df = pd.read_csv(INPUT_CSV)

# Definir Features (Puedes filtrar aquí tu conjunto particular si lo deseas)
excluded = ['sample_uid', 'site_label', 'date_id', 'hour_bin']
#X = df.drop(columns=excluded)
X = df[FEATURES_A_ANALIZAR]
y = df['site_label']

# Codificación de etiquetas y Escalado
le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Preparar Tensores y DataLoaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=512, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=512)

print(f"📈 Muestras entrenamiento: {len(X_train)} | Clases: {len(le.classes_)}")
print(f"🖥️  Dispositivo detectado: {device}")

# --- 2. ARQUITECTURA DEL MODELO ---
class DeepClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DeepClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = DeepClassifier(X.shape[1], len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)

# --- 3. BUCLE DE ENTRENAMIENTO ---
epochs = 100
patience = 12
best_loss = float('inf')
counter = 0
print(f"🏋️ Comenzando entrenamiento por {epochs} épocas...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for b_X, b_y in train_loader:
        b_X, b_y = b_X.to(device), b_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(b_X), b_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    
    # Early Stopping Check
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_custom_mlp.pth")
        counter = 0
    else:
        counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    if counter >= patience:
        print(f"🛑 Detenido temprano por falta de mejora en época {epoch+1}")
        break

# --- 4. EVALUACIÓN FINAL ---
model.load_state_dict(torch.load(f"{SAVE_DIR}/best_custom_mlp.pth"))
model.eval()
y_pred = []

with torch.no_grad():
    for b_X, _ in test_loader:
        outputs = model(b_X.to(device))
        preds = torch.argmax(outputs, dim=1)
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_test, y_pred)

# --- 5. GUARDADO Y CIERRE ---
print("\n" + "═"*50)
print(f"       🏆 RESULTADO DEEP MLP (INDEPENDIENTE) 🏆")
print("═"*50)
print(f"🔹 Accuracy Global: {acc:.4f}")
print(f"🔹 Tiempo Ejecución: {time.time() - start_time:.2f}s")
print("═"*50)

# Guardar metadatos para reproducción
joblib.dump(scaler, f"{SAVE_DIR}/custom_scaler.joblib")
joblib.dump(le, f"{SAVE_DIR}/custom_label_encoder.joblib")

# Opcional: Reporte de sitios con peor desempeño para este set
report = classification_report(y_test, y_pred, output_dict=True)
low_f1 = sorted([(k, v['f1-score']) for k, v in report.items() if k.isdigit()], key=lambda x: x[1])[:3]
print(f"⚠️ Sitios más confundidos con estas features: {low_f1}")