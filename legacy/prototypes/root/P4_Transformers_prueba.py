import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import time
import joblib
import os

# --- 1. CONFIGURACIÓN Y RUTAS ---
print("🚀 Iniciando Experimento: Tabular Transformer (Independiente)")
INPUT_CSV = '../output/preprocessed/03_features_maestras.csv'
SAVE_DIR = './output/transformer'
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. GESTIÓN DE DISPOSITIVO ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Dispositivo de entrenamiento: {device}")

# --- 3. CARGA Y PREPROCESAMIENTO ---
df = pd.read_csv(INPUT_CSV)
excluded = ['sample_uid', 'site_label', 'date_id', 'hour_bin']
features_list = [col for col in df.columns if col not in excluded]

X = df[features_list].values
y = df['site_label'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Datasets
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), 
                          batch_size=512, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), 
                         batch_size=512)

## --- 4. ARQUITECTURA MEJORADA (FT-Transformer Lite) ---
class FTTransformerLite(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=64, nhead=8, num_layers=3):
        super().__init__()
        # AJUSTE 1: Feature Tokenizer. Proyectamos cada feature a un vector d_model
        # Usamos un Linear por feature de forma eficiente
        self.tokenizer = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(input_dim)])
        
        # AJUSTE 2: Transformer Encoder con mejores parámetros
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4,
            dropout=0.2,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # AJUSTE 3: Clasificador con Global Average Pooling (GAP)
        self.ln = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [batch, input_dim]
        # Tokenización eficiente
        tokens = [self.tokenizer[i](x[:, i].unsqueeze(-1)) for i in range(len(self.tokenizer))]
        x = torch.stack(tokens, dim=1) # [batch, input_dim, embed_dim]
        
        # Transformer
        x = self.transformer(x) # [batch, input_dim, embed_dim]
        
        # Global Average Pooling (Resumir información de todas las métricas)
        x = x.mean(dim=1) # [batch, embed_dim]
        x = self.ln(x)
        
        return self.classifier(x)

# --- 5. INSTANCIACIÓN Y ENTRENAMIENTO ---
model = FTTransformerLite(input_dim=len(features_list), num_classes=len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

# --- 5. ENTRENAMIENTO ---
epochs = 50
print(f"🏋️ Entrenando por {epochs} épocas en {device}...")

start_time = time.time()
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for b_X, b_y in train_loader:
        b_X, b_y = b_X.to(device), b_y.to(device)
        optimizer.zero_grad()
        outputs = model(b_X)
        loss = criterion(outputs, b_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f} | Time: {time.time()-start_time:.1f}s")

# --- 6. EVALUACIÓN FINAL ---
model.eval()
y_pred = []
with torch.no_grad():
    for b_X, _ in test_loader:
        outputs = model(b_X.to(device))
        y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

acc = accuracy_score(y_test, y_pred)
print("\n" + "═"*50)
print(f"🏆 RESULTADO TRANSFORMER FINAL: {acc:.4f}")
print("═"*50)

# Guardar
torch.save(model.state_dict(), f"{SAVE_DIR}/ft_transformer_best_maestras.pth")
joblib.dump(le, f"{SAVE_DIR}/label_encoder_maestras.joblib")
print(f"💾 Modelo guardado en {SAVE_DIR}")