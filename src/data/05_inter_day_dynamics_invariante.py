"""
=====================================================================================
FASE 5 (VARIANTE) — SET-BASED SITE DYNAMICS (FINGERPRINTING INVARIANTE)
=====================================================================================

    Descripción General
    -------------------
    Este script genera la representación final (embedding) de un sitio web tratando
    su historial de días como un "conjunto" (set) en lugar de una secuencia.

    Motivación Técnica
    ------------------
    En el análisis de tráfico, el orden exacto de los días puede introducir sesgos 
    (ej. si un sitio se midió en vacaciones y otro en días laborales). Al eliminar 
    el orden temporal, obligamos al modelo a aprender la "distribución de estados"
    típica del sitio, creando una firma digital (fingerprint) mucho más robusta.

    Arquitectura: Transformer-based Deep Set
    ----------------------------------------
    1. Input: Tensor [Días x Dimensiones]
    2. Self-Attention: Los días intercambian información para identificar patrones
       comunes y descartar anomalías (outliers), sin importar su fecha.
    3. Global Pooling: Agregación de todos los días en un único vector.
    4. L2 Normalization: Proyección a una esfera unitaria para facilitar la 
       comparación por similitud coseno o clustering.

=====================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os

# =============================================================================
# DEFINICIÓN DEL MODELO
# =============================================================================

class SetDynamicsEncoder(nn.Module):
    """
    Encoder basado en Transformers para conjuntos de datos (Set-Transformer lite).
    Diseñado para ser invariante a las permutaciones de los días.
    """
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(SetDynamicsEncoder, self).__init__()

        # Capa de codificación del Transformer (sin Positional Encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Pooling promedio: Operación simétrica que garantiza invarianza al orden
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x (Tensor): [D, E] -> D días, E dimensiones de embedding diario.
        Returns:
            Tensor: [E] -> Embedding global normalizado del sitio.
        """
        # 1. Preparar batch: [D, E] -> [1, D, E]
        # El Transformer requiere una dimensión de batch inicial
        x = x.unsqueeze(0)

        # 2. Aplicar Self-Attention Global
        # Aquí cada día "atiende" a los demás días para entender el contexto global del sitio
        x = self.transformer(x)  # Output: [1, D, E]

        # 3. Agregación Temporal (D -> 1)
        # Transponemos para que el pooling actúe sobre la dimensión de los días (D)
        x = x.transpose(1, 2)    # [1, E, D]
        x = self.pool(x)         # [1, E, 1]
        
        # 4. Reducción de dimensiones extra
        x = x.squeeze(0).squeeze(-1)  # [E]

        # 5. Normalización L2 (Esencial para Fingerprinting)
        # Hace que la magnitud del vector sea 1.0, facilitando el cálculo de distancias
        return F.normalize(x, p=2, dim=-1)

# =============================================================================
# SCRIPT DE EJECUCIÓN (APLICACIÓN)
# =============================================================================

if __name__ == "__main__":
    # --- Configuración de Entorno ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciando Fase 5 (Variante Set-Based) en: {device}")

    # Rutas de archivos
    #INPUT_CSV = "output/preprocessed/05_01_daily_embeddings_maestras.csv"
    #INPUT_CSV = "output/preprocessed/05_02_daily_embeddings_robust.csv"
    INPUT_CSV = "output/preprocessed/05_03_daily_embeddings_all.csv"
    OUTPUT_DIR = "output/preprocessed/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Carga de Datos ---
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: No se encuentra el archivo {INPUT_CSV}")
        exit()

    df = pd.read_csv(INPUT_CSV)
    
    # Identificamos las columnas del embedding (day_emb_0 ... day_emb_63)
    day_emb_cols = [c for c in df.columns if c.startswith("day_emb_")]
    E = len(day_emb_cols)
    print(f"[INFO] Dimensiones detectadas: {E} | Registros diarios: {len(df)}")

    # --- Inicialización del Modelo ---
    encoder = SetDynamicsEncoder(embed_dim=E).to(device)
    encoder.eval()

    all_site_rows = []

    # --- Procesamiento por Sitio (Invariante al tiempo) ---
    # Agrupamos por sitio; el orden cronológico interno no es relevante
    grouped = df.groupby("site_label")

    print(f"[INFO] Generando firmas para {len(grouped)} sitios...")

    with torch.no_grad():
        for site_label, df_site in grouped:
            # Convertir los embeddings del historial del sitio a tensor
            # X shape: [Días del sitio, E]
            X = torch.tensor(
                df_site[day_emb_cols].values, 
                dtype=torch.float32
            ).to(device)

            # Generar el embedding maestro del sitio
            site_vector = encoder(X)
            
            # Mover a CPU y convertir a numpy para almacenamiento
            site_vector_np = site_vector.cpu().numpy()

            # Estructurar fila de salida
            row = {"site_label": site_label}
            for i, val in enumerate(site_vector_np):
                row[f"site_emb_{i}"] = val
            
            all_site_rows.append(row)

    # --- Guardado de Resultados ---
    df_final = pd.DataFrame(all_site_rows)
    output_path = os.path.join(
        OUTPUT_DIR, "06_03_all_site_embeddings_inv_all.csv"
    )
    df_final.to_csv(output_path, index=False)

    print("-" * 50)
    print(f"✅ PROCESO FINALIZADO CON ÉXITO")
    print(f"📦 Archivo generado: {output_path}")
    print(f"📊 Total de sitios codificados: {len(df_final)}")
    print("-" * 50)