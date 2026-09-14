"""
=========================================================================================
UNIVERSIDAD / INSTITUTO DE INVESTIGACION Y DESARROLLO DE IA
LABORATORIO DE CIBERSEGURIDAD Y ANALISIS DE TRAFICO AVANZADO (WFP)
=========================================================================================
ARCHIVO: DIA_01_Aislamiento_Sitios_Estables.py
VERSION: 1.0 (Auditoria Forense - Separacion de Content Drift vs Network Drift)
INVESTIGADOR: bsierra@zeus

-----------------------------------------------------------------------------------------
RESUMEN ACADEMICO / METODOLOGIA:
    Auditoria forense que disecciona el fenomeno de Concept Drift en dos componentes
    independientes y medibles:

    1. NETWORK DRIFT (Mutacion del Ruteo de Tor):
       - El contenido del sitio web NO cambio.
       - La secuencia de paquetes muto porque Tor re-enruto los circuitos.
       - Se manifiesta en sitios con volumen de trafico estable (Anchor Sites).

    2. CONTENT DRIFT (Mutacion de la Pagina):
       - El contenido HTML/CSS/JS del sitio SI cambio sustancialmente.
       - La estructura del trafico refleja la nueva morfologia de la pagina.
       - Se manifiesta en sitios con cambio drastico de volumen (Volatile Sites).

    METODOLOGIA:
    Fase I  - Calculo del Volumen Total (sum|size_vector|) por sesion, pasado y futuro.
    Fase II - Ranking de estabilidad volumetrica: delta% = |Futuro - Pasado| / Pasado.
    Fase III- Extraccion de los 5 Sitios Ancla (delta% ~ 0) y 5 Sitios Volatiles (delta% >> 0).
    Fase IV - Inferencia del modelo SOTA (MultimodalTransformer sin Data Leakage)
              exclusivamente sobre los registros del Futuro de los 10 sitios.
    Fase V  - Calculo de Accuracy por sitio y generacion de evidencia visual (G23).

    HIPOTESIS FORENSE:
    - Sitios Ancla (Network Drift puro): Accuracy bajo porque el modelo se confunde
      con las nuevas rutas de Tor, a pesar de que el contenido es identico.
    - Sitios Volatiles (Content Drift severo): Accuracy aun mas bajo porque ademas
      del cambio de ruteo, el contenido mutado genera firmas irreconocibles.

ENTRADAS:
    - ../../output/CLEAN_final_vectors_sites.csv (Pasado / Entrenamiento)
    - ../../output/CLEAN_final_vectors_sites_concept_drift.csv (Futuro / Concept Drift)
    - ../vectores/resultados/ds3_label_encoder_vec_3000.joblib (LabelEncoder)
    - ../vectores/resultados/ds3_best_multimodal_transformer_vec_3000.pth (Pesos SOTA)

SALIDAS:
    - ./resultados_auditoria_forense/G23_Auditoria_Network_vs_Content_Drift.png
    - ./resultados_auditoria_forense/reporte_DIA_01_Aislamiento.txt
=========================================================================================
"""
import matplotlib
matplotlib.use('Agg')  # Backend seguro para servidores sin interfaz grafica

import pandas as pd
import numpy as np
import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.lines import Line2D
import ast
import os
import joblib
import sys
import datetime


# ==========================================================================
# 0. SISTEMA DE LOGGER DUAL (Consola + Archivo)
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
# 1. CONFIGURACION DE RUTAS Y DIRECTORIOS
# ==========================================================================
DIR_DATA = '../../output/'
TRAIN_VECTORS_CSV = os.path.join(DIR_DATA, 'CLEAN_final_vectors_sites.csv')
FUT_VECTORS_CSV = os.path.join(DIR_DATA, 'CLEAN_final_vectors_sites_concept_drift.csv')

LABEL_ENCODER_PATH = '../vectores/resultados/ds3_label_encoder_vec_3000.joblib'
MODEL_WEIGHTS_PATH = '../vectores/resultados/ds3_best_multimodal_transformer_vec_3000.pth'

SAVE_DIR = './resultados_auditoria_forense'
os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filepath = os.path.join(SAVE_DIR, f"log_DIA_01_Aislamiento_{timestamp}.txt")
sys.stdout = Logger(log_filepath)

print("\n" + "=" * 70)
print("DIA_01 - AUDITORIA FORENSE: AISLAMIENTO DE SITIOS ESTABLES")
print("   Separacion de Content Drift (Mutacion de Pagina) vs Network Drift (Ruteo Tor)")
print("=" * 70)

# ==========================================================================
# 2. HIPERPARAMETROS DEL MODELO (Deben coincidir con el entrenamiento)
# ==========================================================================
MAX_LEN = 3000
BATCH_SIZE = 64
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
KERNEL_SIZE = 7  # Coincide con el entrenamiento original

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo configurado: {device}")

# ==========================================================================
# 3. CARGA DE MEMORIA (LabelEncoder + Filtro de Paridad Estricta)
# ==========================================================================
print("\n[Fase 0] Cargando memoria del LabelEncoder y aplicando filtro de elite...")
try:
    le = joblib.load(LABEL_ENCODER_PATH)
    sitios_elite = set(le.classes_.astype(str))
    site_to_historic_id = {str(site): idx for idx, site in enumerate(le.classes_)}
    num_classes = len(le.classes_)
    print(f"   LabelEncoder cargado: {num_classes} sitios de elite (paridad estricta).")
except Exception as e:
    print(f"   ERROR al cargar LabelEncoder: {e}")
    sys.exit()

# ==========================================================================
# 4. FUNCIONES AUXILIARES DE PROCESAMIENTO DE VECTORES
# ==========================================================================


def preprocess_vector(v_str, max_len):
    """
    Parsea y normaliza un vector de direccion o tamano a longitud fija.
    Replica exacta del preprocesamiento en VP_Evaluar_ConceptDrift_Vectores.py
    y EXP5_Train_94F3000V_CrossAttention.py.
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


def compute_volume_bytes(size_vector_str):
    """
    Calcula el Volumen Total en Bytes de una sesion:
    sum |size_vector| - suma de los valores absolutos de todos los paquetes.
    """
    try:
        v = ast.literal_eval(size_vector_str)
        if not isinstance(v, list):
            return 0.0
        return float(sum(abs(x) for x in v))
    except Exception:
        return 0.0


# ==========================================================================
# 5. CARGA DE DATOS Y CALCULO DE VOLUMEN POR SITIO
# ==========================================================================
print("\n[Fase I] Cargando datos del PASADO y calculando volumen por sitio...")

# --- 5.1 PASADO (Entrenamiento) ---
df_past = pd.read_csv(TRAIN_VECTORS_CSV)
df_past['site_label'] = df_past['site_label'].astype(str)
initial_past = len(df_past)
df_past = df_past[df_past['site_label'].isin(sitios_elite)].copy()
print(f"   PASADO: {len(df_past)} registros (descartados {initial_past - len(df_past)} "
      f"por no pertenecer a las {num_classes} clases de elite).")

df_past['volume_bytes'] = df_past['size_vector'].apply(compute_volume_bytes)
volumen_pasado = df_past.groupby('site_label')['volume_bytes'].mean()
print(f"   Volumen medio calculado para {len(volumen_pasado)} sitios en el Pasado.")

# --- 5.2 FUTURO (Concept Drift) ---
print("\n[Fase I] Cargando datos del FUTURO (Concept Drift) y calculando volumen por sitio...")
df_fut = pd.read_csv(FUT_VECTORS_CSV)
df_fut['site_label'] = df_fut['site_label'].astype(str)
initial_fut = len(df_fut)
df_fut = df_fut[df_fut['site_label'].isin(sitios_elite)].copy()
print(f"   FUTURO: {len(df_fut)} registros (descartados {initial_fut - len(df_fut)} "
      f"por no pertenecer a las {num_classes} clases de elite).")

df_fut['volume_bytes'] = df_fut['size_vector'].apply(compute_volume_bytes)
volumen_futuro = df_fut.groupby('site_label')['volume_bytes'].mean()
print(f"   Volumen medio calculado para {len(volumen_futuro)} sitios en el Futuro.")

# ==========================================================================
# 6. RANKING DE ESTABILIDAD VOLUMETRICA (El Nucleo Analitico)
# ==========================================================================
print("\n[Fase II] Calculando Diferencia Porcentual Absoluta de Volumen (delta%)...")

# Interseccion de sitios presentes en ambos periodos
sitios_comunes = sorted(set(volumen_pasado.index) & set(volumen_futuro.index))
print(f"   Sitios con datos en ambos periodos: {len(sitios_comunes)}")

if len(sitios_comunes) < 10:
    print(f"   ADVERTENCIA: Solo {len(sitios_comunes)} sitios comunes. "
          f"Se usaran todos los disponibles.")
    n_ancla = min(5, len(sitios_comunes))
    n_volatil = min(5, len(sitios_comunes))
else:
    n_ancla = 5
    n_volatil = 5

# Calculo de delta% para cada sitio
delta_volumen = {}
for sitio in sitios_comunes:
    v_past = volumen_pasado[sitio]
    v_fut = volumen_futuro[sitio]
    if v_past > 0:
        delta_pct = abs(v_fut - v_past) / v_past * 100.0
    else:
        delta_pct = float('inf') if v_fut > 0 else 0.0
    delta_volumen[sitio] = delta_pct

# Ordenar por delta% ascendente
sitios_ordenados = sorted(delta_volumen.items(), key=lambda x: x[1])

# Extraer Top-N Ancla (menor delta%) y Top-N Volatiles (mayor delta%)
sitios_ancla = [(s, d) for s, d in sitios_ordenados[:n_ancla]]
sitios_volatiles = [(s, d) for s, d in sitios_ordenados[-n_volatil:]]

print(f"\n   TOP {n_ancla} SITIOS ANCLA (Network Drift Puro - delta% ~ 0):")
for i, (sitio, delta) in enumerate(sitios_ancla, 1):
    print(f"      {i}. Site {sitio}: delta% = {delta:.4f}%  |  "
          f"Pasado: {volumen_pasado[sitio]:.1f} B  |  "
          f"Futuro: {volumen_futuro[sitio]:.1f} B")

print(f"\n   TOP {n_volatil} SITIOS VOLATILES (Content Drift Severo - delta% >> 0):")
for i, (sitio, delta) in enumerate(sitios_volatiles, 1):
    print(f"      {i}. Site {sitio}: delta% = {delta:.2f}%  |  "
          f"Pasado: {volumen_pasado[sitio]:.1f} B  |  "
          f"Futuro: {volumen_futuro[sitio]:.1f} B")

# Construir conjunto de los 10 sitios seleccionados
sitios_seleccionados = set()
sitios_seleccionados.update(s for s, _ in sitios_ancla)
sitios_seleccionados.update(s for s, _ in sitios_volatiles)
print(f"\n   Total de sitios seleccionados para auditoria forense: {len(sitios_seleccionados)}")

# ==========================================================================
# 7. CALIBRACION DE ESCALA HISTORICA (Normalizacion de size_vector)
# ==========================================================================
print("\n[Fase III] Extrayendo escala matematica del entrenamiento original...")
start_scale = datetime.datetime.now()

max_size_historic = 0.0
for chunk in pd.read_csv(TRAIN_VECTORS_CSV, chunksize=10000, usecols=['size_vector']):
    for _, row in chunk.iterrows():
        try:
            v = ast.literal_eval(row['size_vector'])
            if isinstance(v, list) and len(v) > 0:
                current_max = max(v)
                if current_max > max_size_historic:
                    max_size_historic = current_max
        except Exception:
            pass

max_size_historic = max(max_size_historic, 14546.0)
elapsed_scale = (datetime.datetime.now() - start_scale).total_seconds()
print(f"   Escala Historica Recuperada: Max = {max_size_historic:.1f} "
      f"(Tomo {elapsed_scale:.1f}s)")

# ==========================================================================
# 8. PREPARACION DE DATOS DE INFERENCIA (Solo Futuro, solo 10 sitios)
# ==========================================================================
print("\n[Fase IV] Preparando tensores de inferencia para los 10 sitios seleccionados...")

df_fut_selected = df_fut[df_fut['site_label'].isin(sitios_seleccionados)].copy()
print(f"   Registros del Futuro para los sitios seleccionados: {len(df_fut_selected)}")

site_to_historic_id_str = {str(k): v for k, v in site_to_historic_id.items()}
df_fut_selected['target_id'] = df_fut_selected['site_label'].map(
    site_to_historic_id_str
).astype(int)

X_d, X_w, y, lbls = [], [], [], []
for _, row in df_fut_selected.iterrows():
    X_d.append(preprocess_vector(row['direction_vector'], MAX_LEN))
    X_w.append(preprocess_vector(row['size_vector'], MAX_LEN))
    y.append(row['target_id'])
    lbls.append(str(row['site_label']))

X_d = np.array(X_d)
X_w_norm = np.clip(np.array(X_w) / float(max_size_historic), 0.0, 1.0)
y = np.array(y)
lbls = np.array(lbls)

print(f"   Tensores listos: X_dir={X_d.shape}, X_weight={X_w_norm.shape}, y={y.shape}")

if len(y) == 0:
    print("\n   CRITICO: No hay muestras para inferencia. Abortando.")
    sys.exit()

# ==========================================================================
# 9. ARQUITECTURA DEL MODELO (MultimodalTransformer - Vectores Solamente)
#    Replicada fielmente de VP_Evaluar_ConceptDrift_Vectores.py
#    NOTA: Se emplea el modelo Solo-Micro (vectores) porque los CSVs de entrada
#          no contienen features macro-temporales. Esta es la arquitectura SOTA
#          para vectores sin Data Leakage.
# ==========================================================================
print("\n[Fase V] Instanciando arquitectura MultimodalTransformer (Solo-Vectores)...")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultimodalTransformer(nn.Module):
    """
    Arquitectura Solo-Micro (Vectores de Red: Direccion + Tamano).
    Sin rama de features macro-temporales.
    Replica exacta importada de VP_Evaluar_ConceptDrift_Vectores.py.
    """
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
        return logits


# ==========================================================================
# 10. CARGA DE PESOS CON LIMPIEZA _orig_mod.
# ==========================================================================
print(f"\n[Fase V] Cargando pesos del modelo SOTA (kernel_size={KERNEL_SIZE})...")
model = MultimodalTransformer(
    num_classes, MAX_LEN, d_model=D_MODEL, nhead=NHEAD,
    num_layers=NUM_LAYERS, kernel_size=KERNEL_SIZE
).to(device)

try:
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)

    # Limpieza del prefijo _orig_mod. (residual de torch.compile)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("   Detectado prefijo '_orig_mod.' en state_dict. Limpiando...")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('_orig_mod.', '')
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    print("   Pesos del modelo restaurados exitosamente.")
except Exception as e:
    print(f"   ERROR al cargar pesos: {e}")
    sys.exit()

model.eval()

# ==========================================================================
# 11. INFERENCIA SOBRE LOS 10 SITIOS SELECCIONADOS (SOLO FUTURO)
# ==========================================================================
print("\n[Fase VI] Ejecutando inferencia forense sobre los 10 sitios seleccionados...")
start_infer = datetime.datetime.now()

dataset = TensorDataset(
    torch.LongTensor(X_d), torch.FloatTensor(X_w_norm), torch.LongTensor(y)
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

all_preds = []
with torch.inference_mode():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for d_batch, w_batch, _ in loader:
            logits = model(d_batch.to(device), w_batch.to(device))
            all_preds.extend(torch.argmax(logits.float(), dim=1).cpu().numpy())

y_pred = np.array(all_preds)
y_true = y

infer_time = (datetime.datetime.now() - start_infer).total_seconds()
print(f"   Inferencia completada en {infer_time:.1f}s sobre {len(y_true)} muestras.")

# ==========================================================================
# 12. CALCULO DE ACCURACY POR SITIO
# ==========================================================================
print("\n[Fase VII] Calculando Accuracy individual por sitio...")

# Accuracy global sobre los 10 sitios
acc_global = accuracy_score(y_true, y_pred)
print(f"   Accuracy Global (10 sitios seleccionados, Futuro): {acc_global * 100:.2f}%")

# Accuracy por sitio
accuracy_por_sitio = {}
for sitio in sorted(sitios_seleccionados):
    mask = lbls == sitio
    if mask.sum() == 0:
        accuracy_por_sitio[sitio] = None
        continue
    y_true_site = y_true[mask]
    y_pred_site = y_pred[mask]
    acc_site = accuracy_score(y_true_site, y_pred_site)
    accuracy_por_sitio[sitio] = acc_site

sitios_ancla_dict = dict(sitios_ancla)
sitios_volatiles_dict = dict(sitios_volatiles)

print("\n   +-------------------------------------------------------------------------+")
print("   |  RESULTADOS POR SITIO (Auditoria Forense)                               |")
print("   +-------------------------------------------------------------------------+")
print("   |  Sitio       Tipo           delta% Volumen        Accuracy              |")
print("   +-------------------------------------------------------------------------+")

orden_grafica = []
for sitio in sorted(sitios_seleccionados):
    is_ancla = sitio in sitios_ancla_dict
    tipo = "ANCLA (Net. Drift)" if is_ancla else "VOLATIL (Cont. Drift)"
    delta = sitios_ancla_dict.get(sitio, sitios_volatiles_dict.get(sitio))
    acc = accuracy_por_sitio.get(sitio)
    delta_str = f"{delta:.4f}%" if delta is not None and delta < 100 else f"{delta:.2f}%"
    acc_str = f"{acc * 100:.2f}%" if acc is not None else "N/A"
    icon = "ANCLA" if is_ancla else "VOLATIL"
    print(f"   |  [{icon}] Site {sitio:<6}| {tipo:<18}| {delta_str:<14}| {acc_str:<20}|")
    orden_grafica.append((sitio, is_ancla, delta, acc))
print("   +-------------------------------------------------------------------------+")

print(f"\n   RESUMEN FORENSE:")
acc_ancla_vals = [accuracy_por_sitio[s] for s, _ in sitios_ancla if accuracy_por_sitio.get(s) is not None]
acc_volatil_vals = [accuracy_por_sitio[s] for s, _ in sitios_volatiles if accuracy_por_sitio.get(s) is not None]

if acc_ancla_vals:
    print(f"   [ANCLA] Accuracy promedio Sitios Ancla (Network Drift): {np.mean(acc_ancla_vals) * 100:.2f}%")
if acc_volatil_vals:
    print(f"   [VOLATIL] Accuracy promedio Sitios Volatiles (Content Drift): {np.mean(acc_volatil_vals) * 100:.2f}%")

# ==========================================================================
# 13. GENERACION DE EVIDENCIA VISUAL - G23
# ==========================================================================
print("\n[Fase VIII] Generando grafica forense G23_Auditoria_Network_vs_Content_Drift.png...")

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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                               gridspec_kw={'height_ratios': [1.5, 1]})
fig.suptitle('G23 - Auditoria Forense: Network Drift vs Content Drift\n'
             'Aislamiento del Impacto en el Modelo SOTA (MultimodalTransformer)',
             fontweight='bold', fontsize=15, y=1.01)

# --- Panel Superior: Barras de delta% Volumen + Accuracy ---
n_sites = len(orden_grafica)
x = np.arange(n_sites)
width = 0.35

labels = []
delta_vals = []
acc_vals = []
bar_colors = []
for sitio, is_ancla, delta, acc in orden_grafica:
    labels.append(f"Site {sitio}")
    delta_vals.append(delta if delta is not None else 0)
    acc_vals.append(acc * 100 if acc is not None else 0)
    bar_colors.append('#2ecc71' if is_ancla else '#e74c3c')

bars_delta = ax1.bar(x - width / 2, delta_vals, width, color=bar_colors,
                     edgecolor='black', linewidth=0.8, alpha=0.85,
                     label='delta% Volumen (Pasado -> Futuro)')

bars_acc = ax1.bar(x + width / 2, acc_vals, width,
                   color='white', edgecolor=bar_colors, linewidth=2.5,
                   hatch='///', alpha=0.7,
                   label='Accuracy (%) - Futuro')

# Anotar valores sobre las barras
for i, (bar_d, bar_a) in enumerate(zip(bars_delta, bars_acc)):
    ax1.text(bar_d.get_x() + bar_d.get_width() / 2., bar_d.get_height() + 0.5,
             f'{delta_vals[i]:.1f}%', ha='center', va='bottom', fontsize=8,
             fontweight='bold', color=bar_colors[i])
    ax1.text(bar_a.get_x() + bar_a.get_width() / 2., bar_a.get_height() + 0.5,
             f'{acc_vals[i]:.1f}%', ha='center', va='bottom', fontsize=8,
             fontweight='bold', color='#2c3e50')

ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax1.set_ylabel('Porcentaje (%)', fontweight='bold')
ax1.set_title('Cambio Volumetrico vs Accuracy por Sitio (Futuro / Concept Drift)',
              fontweight='bold', pad=10)
ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
max_y = max(max(delta_vals) * 1.2, max(acc_vals) * 1.2, 105)
ax1.set_ylim(0, max_y)
ax1.grid(axis='y', alpha=0.4, linestyle='--')

# Linea horizontal de referencia para accuracy global
ax1.axhline(y=acc_global * 100, color='#34495e', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.text(n_sites - 0.5, acc_global * 100 + 1.5,
         f'Acc. Global: {acc_global * 100:.1f}%',
         fontsize=9, color='#34495e', fontstyle='italic', ha='right')

# --- Panel Inferior: Scatter plot delta% vs Accuracy ---
ax2.set_facecolor('#fafafa')
for i, (sitio, is_ancla, delta, acc) in enumerate(orden_grafica):
    color = '#2ecc71' if is_ancla else '#e74c3c'
    marker = 's' if is_ancla else '^'
    size = 160
    ax2.scatter(delta, acc * 100, c=color, s=size, marker=marker,
                edgecolors='black', linewidth=1.2, zorder=5, alpha=0.9)
    ax2.annotate(f'Site {sitio}',
                 (delta, acc * 100),
                 textcoords="offset points",
                 xytext=(8, 6),
                 fontsize=8, fontweight='bold',
                 color=color,
                 arrowprops=dict(arrowstyle='->', color=color, lw=0.8, alpha=0.6))

ax2.set_xlabel('delta% Volumen (Pasado -> Futuro) [%]', fontweight='bold')
ax2.set_ylabel('Accuracy en Futuro [%]', fontweight='bold')
ax2.set_title('Correlacion Forense: Estabilidad Volumetrica vs Reconocimiento del Modelo',
              fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

# Leyenda del panel inferior
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71',
           markersize=12, markeredgecolor='black', markeredgewidth=1,
           label='Sitios Ancla (Network Drift)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#e74c3c',
           markersize=12, markeredgecolor='black', markeredgewidth=1,
           label='Sitios Volatiles (Content Drift)')
]
ax2.legend(handles=legend_elements, loc='best', frameon=True, fancybox=True)

plt.tight_layout()
save_path_png = os.path.join(SAVE_DIR, 'G23_Auditoria_Network_vs_Content_Drift.png')
fig.savefig(save_path_png, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print(f"   Grafica guardada: {save_path_png}")

# ==========================================================================
# 14. REPORTE FORENSE DETALLADO
# ==========================================================================
print("\n[Fase IX] Generando reporte detallado reporte_DIA_01_Aislamiento.txt...")

reporte_path = os.path.join(SAVE_DIR, 'reporte_DIA_01_Aislamiento.txt')
with open(reporte_path, 'w', encoding='utf-8') as f:
    f.write("=" * 85 + "\n")
    f.write("REPORTE DE AUDITORIA FORENSE - DIA_01 AISLAMIENTO DE SITIOS ESTABLES\n")
    f.write("=" * 85 + "\n")
    f.write(f"Fecha de ejecucion: {timestamp}\n")
    f.write(f"Dispositivo: {device}\n")
    f.write("Modelo SOTA: MultimodalTransformer (Solo-Vectores, sin Data Leakage)\n")
    f.write(f"Kernel Size: {KERNEL_SIZE}  |  D_MODEL: {D_MODEL}  |  "
            f"NHEAD: {NHEAD}  |  NUM_LAYERS: {NUM_LAYERS}\n")
    f.write(f"Clases de elite: {num_classes}\n")
    f.write(f"Pesos: {MODEL_WEIGHTS_PATH}\n")
    f.write("\n" + "-" * 85 + "\n")
    f.write("HIPOTESIS DE TRABAJO\n")
    f.write("-" * 85 + "\n")
    f.write(
        "El Concept Drift en trafico de red cifrado (Tor) se descompone en dos\n"
        "fenomenos independientes:\n\n"
        "  1. NETWORK DRIFT: Mutacion del ruteo de Tor (circuitos, relays).\n"
        "     El contenido de la pagina NO cambio. Detectable en sitios con\n"
        "     volumen de trafico estable (delta% Volumen ~ 0).\n\n"
        "  2. CONTENT DRIFT: Mutacion del contenido HTML/CSS/JS del sitio.\n"
        "     La pagina cambio su estructura, alterando la firma de trafico.\n"
        "     Detectable en sitios con cambio drastico de volumen (delta% Volumen >> 0).\n"
    )
    f.write("\n" + "-" * 85 + "\n")
    f.write("RESULTADOS CUANTITATIVOS\n")
    f.write("-" * 85 + "\n")
    f.write(f"Accuracy Global (10 sitios, Futuro): {acc_global * 100:.2f}%\n\n")

    f.write(f"--- TOP {n_ancla} SITIOS ANCLA (Network Drift Puro) ---\n")
    for sitio, delta in sitios_ancla:
        acc = accuracy_por_sitio.get(sitio)
        acc_str = f"{acc * 100:.2f}%" if acc is not None else "N/A"
        f.write(f"  Site {sitio}:  Delta% Volumen = {delta:.4f}%  |  "
                f"Accuracy = {acc_str}  |  "
                f"V_Pasado = {volumen_pasado[sitio]:.1f} B  |  "
                f"V_Futuro = {volumen_futuro[sitio]:.1f} B\n")

    if acc_ancla_vals:
        f.write(f"  Accuracy promedio (Sitios Ancla): "
                f"{np.mean(acc_ancla_vals) * 100:.2f}%\n")

    f.write(f"\n--- TOP {n_volatil} SITIOS VOLATILES (Content Drift Severo) ---\n")
    for sitio, delta in sitios_volatiles:
        acc = accuracy_por_sitio.get(sitio)
        acc_str = f"{acc * 100:.2f}%" if acc is not None else "N/A"
        f.write(f"  Site {sitio}:  Delta% Volumen = {delta:.2f}%  |  "
                f"Accuracy = {acc_str}  |  "
                f"V_Pasado = {volumen_pasado[sitio]:.1f} B  |  "
                f"V_Futuro = {volumen_futuro[sitio]:.1f} B\n")

    if acc_volatil_vals:
        f.write(f"  Accuracy promedio (Sitios Volatiles): "
                f"{np.mean(acc_volatil_vals) * 100:.2f}%\n")

    f.write("\n" + "-" * 85 + "\n")
    f.write("INTERPRETACION FORENSE\n")
    f.write("-" * 85 + "\n")
    f.write(
        "Los sitios ancla (Network Drift puro) presentan volumen de trafico\n"
        "practicamente identico entre Pasado y Futuro. Cualquier degradacion\n"
        "en el accuracy del modelo en estos sitios es atribuible exclusivamente\n"
        "a la mutacion del ruteo de Tor (nuevos circuitos, relays distintos),\n"
        "NO a cambios en el contenido de la pagina web.\n\n"
        "Los sitios volatiles (Content Drift severo) presentan una variacion\n"
        "volumetrica drastica, indicando que el contenido HTML/CSS/JS del sitio\n"
        "cambio sustancialmente. La degradacion de accuracy en estos sitios\n"
        "refleja el efecto combinado de Network Drift + Content Drift.\n\n"
        "La diferencia de accuracy entre ambos grupos permite cuantificar\n"
        "el impacto aislado del Content Drift sobre el modelo SOTA.\n"
    )

    f.write("\n" + "-" * 85 + "\n")
    f.write("METADATOS DE EJECUCION\n")
    f.write("-" * 85 + "\n")
    f.write(f"Tiempo de calibracion de escala: {elapsed_scale:.1f}s\n")
    f.write(f"Tiempo de inferencia: {infer_time:.1f}s\n")
    f.write(f"Muestras en Futuro (sitios seleccionados): {len(y_true)}\n")
    f.write(f"Escala historica maxima: {max_size_historic:.1f}\n")
    f.write("\n" + "=" * 85 + "\n")

# ==========================================================================
# 15. CIERRE
# ==========================================================================
print(f"\n{'=' * 70}")
print("DIA_01 - AUDITORIA FORENSE COMPLETADA EXITOSAMENTE")
print(f"{'=' * 70}")
print(f"Reporte detallado: {reporte_path}")
print(f"Grafica forense:   {save_path_png}")
print(f"Log de ejecucion:  {log_filepath}")
print(f"\nTotal de sitios analizados: {len(sitios_seleccionados)}")
print(f"Sitios Ancla (Network Drift):    {len(sitios_ancla)}")
print(f"Sitios Volatiles (Content Drift): {len(sitios_volatiles)}")
print(f"Accuracy Global (Futuro, 10 sitios): {acc_global * 100:.2f}%")
if acc_ancla_vals:
    print(f"Accuracy Promedio Ancla:   {np.mean(acc_ancla_vals) * 100:.2f}%")
if acc_volatil_vals:
    print(f"Accuracy Promedio Volatil: {np.mean(acc_volatil_vals) * 100:.2f}%")
print(f"{'=' * 70}")

# Restaurar stdout para liberar el archivo de log
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
