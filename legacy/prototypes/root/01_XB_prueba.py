import pandas as pd
import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

print("Iniciando Experimento: Enfoque de Registros Independientes")
start_time = time.time()

# 1. Cargar Datos
INPUT_CSV = '../output/preprocessed/01_all_features.csv'
print(f"📂 Cargando dataset desde: {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
print(f"✅ Dataset cargado. Dimensiones: {df.shape}")


mis_features_seleccionadas = [
    'ratio_pkts_firstN_bursts_total', 'cumul_out_auc', 'bytes_last10pct',
        'ngrams_4_min_count', 'packets_last25pct', 'bytes_first25pct',
        'iat_size_p90', 'pps_w1_max', 'global_packet_count_75',
        'ngrams_4_top5_count', 'cumul_interp_out_diffs_std', 'bytes_first10pct',
        'cumul_interp_out_kurtosis', 'global_packet_count_50', 'cumul_in_skew',
        'burst_histogram_6_10','iat_size_p50','pps_w10_min','cumul_out_skew',
        'ngrams_4_top4_frac','iat_var','iat_mean_out','global_packet_count_25',
        'burst_max','time_to_first_big_burst','last30_out_in_ratio','iat_std_in',
        'time_to_first_response','burst_durations_entropy','burst_durations_max'
    # ... agrega aquí todas las que desees evaluar
]

# 2. Definir Features y Target
excluded = ['sample_uid', 'site_label', 'date_id', 'hour_bin']
#X = df.drop(columns=excluded)
X = df[mis_features_seleccionadas]
y = df['site_label']

# Codificar etiquetas de sitios (0, 1, 2...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(le.classes_)
print(f"Clases detectadas ({n_classes}): {list(le.classes_)}")

# 3. Split de Datos
print("Realizando split de datos (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Entrenamiento: {X_train.shape[0]} muestras | Prueba: {X_test.shape[0]} muestras")

# 4. Configuración y Entrenamiento de XGBoost
# Detectar dispositivo
print(f"Configurando XGBoost para usar GPU (CUDA)...")

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    device='cuda',      # XGBoost gestiona CUDA internamente
    random_state=42
)

print("🏋️ Entrenando modelo (monitoreando eval_set)...")
train_start = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50 
)
train_end = time.time()
print(f"✅ Entrenamiento completado en {train_end - train_start:.2f} segundos.")

# ... (después del entrenamiento)

# 5. Evaluación
print("Realizando predicciones sobre el set de prueba...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "═"*50)
print(f"🏆 RESULTADO ENFOQUE INDEPENDIENTE 🏆")
print("═"*50)
print(f"🔹 Accuracy Global: {acc:.4f}")
print("═"*50)

# 6. Reporte Detallado (Corregido)
print("\nReporte de Clasificación:")
# Convertimos los nombres a string para evitar el error de len()
target_names = [str(c) for c in le.classes_]
print(classification_report(y_test, y_pred, target_names=target_names))

# 7. EXTRACCIÓN DE IMPORTANCIA DE VARIABLES (Muy importante para tu tesis)
print("\nExtrayendo las 10 variables más influyentes...")
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

print(feature_importance_df.head(10))

# Guardar importancia para tu tesis
feature_importance_df.to_csv('feature_importance_independiente.csv', index=False)