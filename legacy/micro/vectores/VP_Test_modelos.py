"""
=========================================================================================
                                VP_Test_modelos.py  (v2.0)
=========================================================================================

DESCRIPCIÓN:
    Herramienta de validación masiva (Hold-Out Set Evaluation) para los modelos de 
    Website Fingerprinting. Este script automatiza la carga y prueba de múltiples 
    modelos (Unimodales y Multimodales, con distintas longitudes de secuencia) 
    contra un dataset de capturas de red completamente nuevo, nunca antes visto 
    durante la fase de entrenamiento y pruebas.

OBJETIVOS PRINCIPALES:
    1. Descartar el "Data Leakage" y validar la capacidad de generalización real 
       de los modelos frente a tráfico de red capturado en diferentes periodos de tiempo.
    2. Instanciar dinámicamente las arquitecturas `DirectionTransformer` (Unimodal) 
       y `MultimodalTransformer` (Dirección + Peso) según el modelo evaluado.
    3. Asegurar la integridad de las pruebas filtrando automáticamente el nuevo 
       dataset para evaluar únicamente los sitios web (clases) que cada modelo 
       aprendió a identificar (`LabelEncoder`).
    4. Generar un reporte de "Accuracy" comparativo entre todas las estrategias.

ENTRADAS REQUERIDAS:
    - NUEVO_CSV : '../../output/nuevas_capturas_marzo.csv' (Datos crudos no vistos).
    - Modelos   : '.pth' generados previamente (ej. ds2_best_multimodal...).
    - Encoders  : '.joblib' asociados a cada modelo para decodificar las etiquetas.
    - Scaler    : '.joblib' con min/max del entrenamiento (para modelos multimodales).

SALIDAS:
    - Log de ejecución detallado por cada modelo evaluado.
    - Tabla comparativa final en consola con el rendimiento (Accuracy global) 
      de todos los modelos probados.

METADATOS:
    - Autor   : BRAYAN LEONARDO SIERRA FORERO
    - Fecha   : 08/07/2026 (Refactorización v2.0)
    - Versión : 2.0 — kernel_size=7, MinMax desde joblib, _orig_mod cleanup,
                inference_mode + AMP bfloat16, pin_memory.
=========================================================================================

"""

import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
import ast
import os
import joblib

# ==========================================
# 1. CONFIGURACIÓN DE LA PRUEBA
# ==========================================
print("═"*60)
print("INICIANDO PRUEBA DEFINITIVA DE MODELOS (HOLD-OUT SET) — v2.0")
print("═"*60)

# CAMBIA ESTO POR LA RUTA DE TU NUEVO DATASET
NUEVO_CSV = '../../output/nuevas_capturas_marzo.csv' 
MODEL_DIR = './resultados'

BATCH_SIZE = 64
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"-- Dispositivo: {device}")

# Definimos las pruebas que queremos correr
pruebas_a_ejecutar = [

    {"nombre": "Solo Dirección - 1000", "tipo": "uni", "max_len": 1000, 
     "modelo": "d_best_multimodal_transformer_vec_1000.pth", "encoder": "d_label_encoder_vec_1000.joblib"},
    
    {"nombre": "Solo Dirección - 3000", "tipo": "uni", "max_len": 3000, 
     "modelo": "d_best_multimodal_transformer_vec_3000.pth", "encoder": "d_label_encoder_vec_3000.joblib"},

    {"nombre": "Solo Dirección - 5000", "tipo": "uni", "max_len": 5000, 
     "modelo": "d_best_multimodal_transformer_vec_5000.pth", "encoder": "d_label_encoder_vec_5000.joblib"},
    
    {"nombre": "Multimodal (Dir+Peso) - 1000", "tipo": "multi", "max_len": 1000, 
     "modelo": "ds2_best_multimodal_transformer_vec_1000.pth", "encoder": "ds2_label_encoder_vec_1000.joblib",
     "scaler": "ds2_minmax_scaler_vec_1000.joblib"},
    
    {"nombre": "Multimodal (Dir+Peso) - 3000", "tipo": "multi", "max_len": 3000, 
     "modelo": "ds2_best_multimodal_transformer_vec_3000.pth", "encoder": "ds2_label_encoder_vec_3000.joblib",
     "scaler": "ds2_minmax_scaler_vec_3000.joblib"},

    {"nombre": "Multimodal (Dir+Peso) - 5000", "tipo": "multi", "max_len": 5000, 
     "modelo": "ds2_best_multimodal_transformer_vec_5000.pth", "encoder": "ds2_label_encoder_vec_5000.joblib",
     "scaler": "ds2_minmax_scaler_vec_5000.joblib"}
]

# ==========================================
# 2. CARGA DEL NUEVO DATASET
# ==========================================
print(f"\n-- Cargando nuevo dataset desde: {NUEVO_CSV}...")
df_nuevo = pd.read_csv(NUEVO_CSV)
print(f"-- Carga completa. Registros totales en el nuevo dataset: {len(df_nuevo)}")

def preprocess_vector(v_str, max_len):
    try:
        v = ast.literal_eval(v_str)
        if not isinstance(v, list): v = []
    except: v = []
    if len(v) > max_len: return v[:max_len]
    else: return v + [0] * (max_len - len(v))

# ==========================================
# 3. ARQUITECTURAS (kernel_size=7 en Conv1d)
# ==========================================
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

# Arquitectura 1: Unimodal (Solo Direcciones) — kernel_size=7
class DirectionTransformer(nn.Module):
    def __init__(self, num_classes, max_len, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(3, d_model) 
        self.conv_local = nn.Conv1d(d_model, d_model, kernel_size=7, stride=2, padding=3)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.2, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model * 2) 
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 512), nn.GELU(), nn.Dropout(0.4), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x + 1)
        x = x.transpose(1, 2)
        x = self.conv_local(x)    
        x = x.transpose(1, 2)     
        x = self.pos_encoder(x)
        x = self.transformer(x)
        avg_pool = torch.mean(x, dim=1) 
        max_pool, _ = torch.max(x, dim=1) 
        x = torch.cat((avg_pool, max_pool), dim=1) 
        x = self.ln(x)
        return self.fc(x)

# Arquitectura 2: Multimodal (Direcciones + Pesos) — kernel_size=7
class MultimodalTransformer(nn.Module):
    def __init__(self, num_classes, max_len, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.dir_embedding = nn.Embedding(3, d_model) 
        self.weight_proj = nn.Linear(1, d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.conv_local = nn.Conv1d(d_model, d_model, kernel_size=7, stride=2, padding=3)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.2, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model * 2) 
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 512), nn.GELU(), nn.Dropout(0.4), nn.Linear(512, num_classes)
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
        x = torch.cat((avg_pool, max_pool), dim=1) 
        x = self.ln(x)
        return self.fc(x)


# ==========================================
# 4. BUCLE DE PRUEBAS
# ==========================================
resultados_globales = []

for prueba in pruebas_a_ejecutar:
    print("\n" + "▼"*60)
    print(f"🧪 EVALUANDO MODELO: {prueba['nombre']}")
    print("▼"*60)
    
    # Rutas
    model_path = os.path.join(MODEL_DIR, prueba['modelo'])
    encoder_path = os.path.join(MODEL_DIR, prueba['encoder'])
    scaler_path = os.path.join(MODEL_DIR, prueba.get('scaler', ''))
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print(f"Archivos no encontrados para {prueba['nombre']}. Saltando...")
        continue

    # Cargar Encoder
    le = joblib.load(encoder_path)
    clases_conocidas = le.classes_
    num_classes = len(clases_conocidas)
    
    # Filtrar dataset nuevo: Solo evaluar páginas que el modelo conoce
    df_eval = df_nuevo[df_nuevo['site_label'].isin(clases_conocidas)].copy()
    print(f"   - Registros válidos (clases conocidas): {len(df_eval)} de {len(df_nuevo)}")
    
    if len(df_eval) == 0:
        print("   - No hay datos coincidentes para evaluar.")
        continue

    # Procesar Vectores
    MAX_LEN = prueba['max_len']
    y_test = le.transform(df_eval['site_label'])
    
    X_dir = np.array([preprocess_vector(row, MAX_LEN) for row in df_eval['direction_vector']])
    
    if prueba['tipo'] == "uni":
        # Data Loader Unimodal
        test_dataset = TensorDataset(torch.LongTensor(X_dir), torch.LongTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)
        # Instanciar Modelo Unimodal
        model = DirectionTransformer(num_classes, MAX_LEN, d_model=D_MODEL, num_layers=NUM_LAYERS).to(device)
    
    elif prueba['tipo'] == "multi":
        # Data Loader Multimodal
        X_weight_raw = np.array([preprocess_vector(row, MAX_LEN) for row in df_eval['size_vector']])
        
        # --- DATA LEAKAGE FIX: Cargar min/max del entrenamiento desde joblib ---
        if os.path.exists(scaler_path):
            minmax_scaler = joblib.load(scaler_path)
            min_w = minmax_scaler["min"]
            max_w = minmax_scaler["max"]
            print(f"   - MinMax cargado desde scaler: min={min_w}, max={max_w}")
        else:
            # Fallback (no recomendado, pero evita crash si no existe el archivo)
            print(f"   ⚠️  Scaler no encontrado en {scaler_path}. Calculando sobre test (NO recomendado).")
            min_w = np.min(X_weight_raw)
            max_w = np.max(X_weight_raw)
        
        X_weight = (X_weight_raw - min_w) / (max_w - min_w + 1e-8)
        
        test_dataset = TensorDataset(torch.LongTensor(X_dir), torch.FloatTensor(X_weight), torch.LongTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)
        # Instanciar Modelo Multimodal
        model = MultimodalTransformer(num_classes, MAX_LEN, d_model=D_MODEL, num_layers=NUM_LAYERS).to(device)
    
    # --- Cargar Pesos con limpieza _orig_mod ---
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state)
    model.eval()
    print(f"   ✅ Pesos cargados y limpiados (_orig_mod.).")
    
    # --- Inferencia optimizada: inference_mode + AMP bfloat16 ---
    y_true, y_pred = [], []
    with torch.inference_mode():
        if prueba['tipo'] == "uni":
            for b_dir, b_y in test_loader:
                b_dir = b_dir.to(device, non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    preds = torch.argmax(model(b_dir), dim=1)
                y_true.extend(b_y.numpy())
                y_pred.extend(preds.cpu().numpy())
        else:
            for b_dir, b_weight, b_y in test_loader:
                b_dir = b_dir.to(device, non_blocking=True)
                b_weight = b_weight.to(device, non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    preds = torch.argmax(model(b_dir, b_weight), dim=1)
                y_true.extend(b_y.numpy())
                y_pred.extend(preds.cpu().numpy())

    # Métricas
    acc = accuracy_score(y_true, y_pred)
    print(f"   🎯 Accuracy en Datos Nuevos: {acc:.4f}")
    
    # Guardar resumen
    resultados_globales.append((prueba['nombre'], acc))

# ==========================================
# 5. REPORTE FINAL COMPARATIVO
# ==========================================
print("\n" + "="*60)
print("RESUMEN FINAL COMPARATIVO (NUEVAS CAPTURAS)")
print("="*60)
for nombre, acc in resultados_globales:
    print(f"- {nombre.ljust(35)} : {acc*100:.2f}%")
print("="*60)