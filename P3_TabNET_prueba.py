import torch
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- 1. CONFIGURACIÓN ---
print("🚀 Iniciando Experimento: TabNet (Atención sobre Features)")
INPUT_CSV = '../output/preprocessed/01_all_features.csv'
SAVE_DIR = './output/tabnet'
os.makedirs(SAVE_DIR, exist_ok=True)

# Define aquí tus features si ya tienes la lista, sino usa el filtro estándar
FEATURES_A_ANALIZAR = ['ratio_pkts_firstN_bursts_total', 'cumul_out_auc', 'bytes_last10pct',
        'ngrams_4_min_count', 'packets_last25pct', 'bytes_first25pct',
        'iat_size_p90', 'pps_w1_max', 'global_packet_count_75',
        'ngrams_4_top5_count', 'cumul_interp_out_diffs_std', 'bytes_first10pct',
        'cumul_interp_out_kurtosis', 'global_packet_count_50', 'cumul_in_skew',
        'burst_histogram_6_10','iat_size_p50','pps_w10_min','cumul_out_skew',
        'ngrams_4_top4_frac','iat_var','iat_mean_out','global_packet_count_25',
        'burst_max','time_to_first_big_burst','last30_out_in_ratio','iat_std_in',
        'time_to_first_response','burst_durations_entropy','burst_durations_max'] 

# --- 2. CARGA Y PREPROCESAMIENTO ---
df = pd.read_csv(INPUT_CSV)

excluded = ['sample_uid', 'site_label', 'date_id', 'hour_bin']
#X = df.drop(columns=excluded)
X = df[FEATURES_A_ANALIZAR].values
y = df['site_label'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# TabNet no requiere StandardScaler obligatoriamente (como XGBoost), 
# pero suele ayudar en la convergencia con datos de red.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- 3. DEFINICIÓN DEL MODELO TABNET ---
# Configuramos TabNet para que use la GPU si está disponible (aunque ignore el error 804 de Zeus)
device = "cuda" if torch.cuda.is_available() else "cpu"

clf = TabNetClassifier(
    n_d=64, n_a=64, # Dimensiones de los pasos de predicción y atención
    n_steps=5,      # Número de pasos secuenciales de decisión
    gamma=1.5,
    lambda_sparse=1e-3, # Sparsity para que "elija" features y no use todas a la vez
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax', # 'sparsemax' o 'entmax'
    device_name=device
)

# --- 4. ENTRENAMIENTO ---
print(f"🏋️ Entrenando TabNet en {device}...")
clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_test, y_test)],
    eval_name=['test'],
    eval_metric=['accuracy'],
    max_epochs=100,
    patience=15,
    batch_size=1024, 
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# --- 5. EVALUACIÓN Y GUARDADO ---
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)

print("\n" + "═"*50)
print(f"       🏆 RESULTADO TABNET (INDEPENDIENTE) 🏆")
print("═"*50)
print(f"🔹 Accuracy Global: {acc:.4f}")
print("═"*50)

# Guardar el modelo (formato propio de TabNet que es un zip)
saved_filepath = clf.save_model(f"{SAVE_DIR}/tabnet_model_custom")
joblib.dump(le, f"{SAVE_DIR}/label_encoder.joblib")

# Reporte de métricas por sitio
print("\n📝 Reporte detallado:")
target_names = [str(c) for c in le.classes_]
print(classification_report(y_test, preds, target_names=target_names))