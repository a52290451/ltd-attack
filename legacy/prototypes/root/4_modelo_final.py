#!/usr/bin/env python3
"""
Módulo de Entrenamiento Predictivo para Identificación de Sitios Web.

Este script realiza la fase final del pipeline:
1. Carga de datos filtrados estadísticamente.
2. Preprocesamiento de etiquetas para compatibilidad con XGBoost (GPU).
3. Entrenamiento comparativo entre modelos lineales y de ensamble.
4. Evaluación mediante métricas de clasificación multiclase y matrices de confusión.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Soporte para aceleración por hardware (NVIDIA CUDA)
try:
    import xgboost as xgb
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    GPU_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as cuRF
    import xgboost as xgb
    GPU_AVAILABLE = False

# Configuración de entorno
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

# =====================================================
# =====================================================
# SECCIÓN 1: PROCESAMIENTO DE DATOS
# =====================================================
# =====================================================

def prepare_dataset(csv_path, top_features, target_col='site_label'):
    """
    Carga, limpia y remapea el dataset para entrenamiento.
    
    Args:
        csv_path (str): Ruta al archivo CSV filtrado.
        top_features (list): Lista de características seleccionadas por ranking.
        target_col (str): Nombre de la columna objetivo.
        
    Returns:
        tuple: (X_scaled, y_mapped, num_classes)
    """
    print(f"[INFO] Leyendo datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    df = filter_by_min_samples(df, 'site_label', min_samples=300)

    # Filtrado preventivo: XGBoost y TrainTestSplit fallan con clases únicas (n=1)
    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts > 1].index
    df = df[df[target_col].isin(valid_classes)].copy()

    # Mapeo de etiquetas: Transforma etiquetas (ej. 102, 505) a rango continuo (0 a N-1)
    # Requisito estricto para el objetivo 'multi:softmax' de XGBoost
    unique_labels = sorted(df[target_col].unique())
    mapping = {old: new for new, old in enumerate(unique_labels)}
    df[target_col] = df[target_col].map(mapping)

    X = df[top_features]
    y = df[target_col]

    # Escalado: Crucial para Logistic Regression y modelos basados en distancias
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, len(unique_labels)

def filter_by_min_samples(df, target_col, min_samples=300):
    """
    Elimina sitios web que no alcanzan el quorum mínimo de muestras.
    
    Args:
        df (pd.DataFrame): Dataset original.
        target_col (str): Columna con los nombres/IDs de los sitios.
        min_samples (int): Mínimo de registros necesarios por sitio.
        
    Returns:
        pd.DataFrame: Dataset saneado.
    """
    counts = df[target_col].value_counts()
    
    # Identificar clases que cumplen el requisito
    kept_classes = counts[counts >= min_samples].index
    removed_classes = counts[counts < min_samples].index
    
    df_filtered = df[df[target_col].isin(kept_classes)].copy()
    
    print(f"[FILTRO] Umbral: {min_samples} muestras.")
    print(f"[FILTRO] Sitios eliminados: {len(removed_classes)}")
    print(f"[FILTRO] Sitios retenidos: {len(kept_classes)}")
    print(f"[FILTRO] Registros totales restantes: {len(df_filtered)}")
    
    return df_filtered

# =====================================================
# =====================================================
# SECCIÓN 2: LÓGICA DE MODELADO
# =====================================================
# =====================================================

def run_predictive_analysis(X, y, num_classes):
    """
    Entrena modelos y genera reportes de desempeño.
    
    Parámetros ajustados para tráfico de red (high-dimensional):
    - XGBoost: Profundidad 8 para capturar interacciones complejas.
    - RandomForest: 400 árboles para reducir varianza en 118 clases.
    """
    # División estratificada: Asegura que cada sitio esté representado en Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Configuración de Modelos
    # Nota: 'tree_method=hist' y 'device=cuda' activan el procesamiento en GPU
    models = {
        "XGBoost_GPU": xgb.XGBClassifier(
            n_estimators=1200,         # Aumentamos los árboles (antes 300)
            learning_rate=0.05,        # Bajamos la tasa para aprender más lento pero mejor
            max_depth=12,              # Árboles más profundos para diferenciar sitios similares
            subsample=0.8,             # Introducimos un poco de aleatoriedad
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda",
            objective="multi:softmax",
            num_class=num_classes,
            n_jobs=-1
        ),
        "RandomForest": cuRF(n_estimators=800) if GPU_AVAILABLE else cuRF(n_estimators=800, n_jobs=-1)
    }

    for name, model in models.items():
        print(f"\n--- Iniciando entrenamiento de {name} ---")
        start = time.time()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        duration = time.time() - start
        
        # Evaluación de métricas clave
        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average='weighted')
        
        print(f"⏱️ Tiempo: {duration:.2f}s")
        print(f"🎯 Accuracy: {acc:.4f}")
        print(f"📊 F1-Score (Weighted): {f1_w:.4f}")

        # Generar Matriz de Confusión para el mejor modelo (XGBoost)
        if "XGB" in name:
            plot_confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(y_true, y_pred):
    """
    Genera un mapa de calor de la matriz de confusión.
    Útil para identificar 'clústeres' de sitios que el modelo confunde.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 11))
    # No mostramos etiquetas de ejes por legibilidad dado el alto número de clases (118)
    sns.heatmap(cm, cmap='magma', cbar=True, xticklabels=False, yticklabels=False)
    plt.title('Matriz de Confusión: Identificación de 118 Sitios')
    plt.xlabel('Predicción del Modelo')
    plt.ylabel('Etiqueta Real (Ground Truth)')
    plt.savefig('confusion_results.png', dpi=300)
    print("[INFO] Matriz de confusión visual guardada como 'confusion_results.png'")

# =====================================================
# =====================================================
# SECCIÓN 3: PUNTO DE ENTRADA
# =====================================================
# =====================================================

if __name__ == "__main__":
    # Estas son las variables que sobrevivieron a tus filtros estadísticos y ranking
    top_features = [
        'ratio_pkts_firstN_bursts_total', 'cumul_out_auc', 'bytes_last10pct',
        'ngrams_4_min_count', 'packets_last25pct', 'bytes_first25pct',
        'iat_size_p90', 'pps_w1_max', 'global_packet_count_75',
        'ngrams_4_top5_count', 'cumul_interp_out_diffs_std', 'bytes_first10pct',
        'cumul_interp_out_kurtosis', 'global_packet_count_50', 'cumul_in_skew'
    ]

    try:
        X, y, n_classes = prepare_dataset("./output/filtered_features.csv", top_features)
        run_predictive_analysis(X, y, n_classes)
    except FileNotFoundError:
        print("[ERROR] No se encontró el archivo 'filtered_features.csv'. Ejecute primero el pipeline de filtrado.")