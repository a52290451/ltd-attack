"""
=====================================================================================
FASE 4 — FEATURE AGGREGATION ENCODER (AGREGACIÓN DIARIA POR FEATURES)
=====================================================================================

    Descripción general
    -------------------
    Este script implementa la **fase de agregación por features a nivel diario**
    dentro del pipeline de análisis temporal de tráfico web basado en embeddings.

    Parte de los embeddings generados por el *Hourly Encoder* (Fase 3), donde cada
    feature de cada día ha sido codificada como un vector compacto que resume su
    patrón intra-día (24 horas).

    El objetivo de esta fase es **agregar todas las features de un mismo día** en
    un único vector diario que represente el estado global del tráfico de una
    página web en ese día concreto.

    A diferencia de etapas posteriores, en esta fase:
    - No existe noción temporal entre días
    - No existe orden natural entre features
    - Se modelan relaciones, correlaciones y redundancias entre features

    Por esta razón, la agregación se realiza mediante un **Set Transformer**,
    basado en *self-attention sin orden*, seguido de un pooling global.

    Entrada
    -------
    - CSV generado en la Fase 3 (`03_hourly_embeddings_all_sites.csv`)
    - Cada fila representa:
        - Una página web
        - Un día
        - Una feature
        - Un embedding horario (vector denso)

    Formalmente, la entrada para un día d se modela como:
        H_f,d ∈ ℝ^{E}

    Donde:
        - f : número de features
        - d : número de dìas 
        - E : dimensión del embedding horario

    Salida
    ------
    - Un CSV con embeddings diarios (`04_daily_embeddings_all_sites.csv`)
    - Cada fila representa:
        - Una página web
        - Un día
        - Un embedding diario Z_d ∈ ℝ^{E}

    Este embedding diario:
        - Resume todas las features del día
        - Captura interacciones entre métricas de tráfico
        - Es robusto frente a ruido individual por feature

    Rol en el pipeline
    ------------------
    Esta fase produce la representación diaria necesaria para la siguiente etapa:
    **Inter-Day Dynamics**, donde se modelará la evolución temporal entre días
    mediante transformers temporales.

=====================================================================================
"""

import torch            # type: ignore
import torch.nn as nn   # type: ignore
import pandas as pd
import numpy as np
import os

class SetAttentionBlock(nn.Module):
    """
        Set Attention Block (SAB)

        Implementa un bloque de *self-attention* para conjuntos no ordenados,
        siguiendo la filosofía de los Set Transformers.

        Cada elemento del conjunto (feature) puede atender a cualquier otro,
        permitiendo aprender:
            - Correlaciones entre features
            - Dependencias cruzadas
            - Redundancias informativas

        Entrada:
            h ∈ ℝ^{B × F × E}
                B : batch (días)
                F : número de features
                E : dimensión del embedding

        Salida:
            h ∈ ℝ^{B × F × E}
                Representaciones enriquecidas con contexto inter-feature
    """
    
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()

        # Atención multi-cabeza sin información posicional
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Normalización residual tras la atención
        self.norm1 = nn.LayerNorm(embed_dim)

        # Red feed-forward aplicada de forma independiente a cada feature
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Normalización residual tras la FFN
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward del bloque SAB.

        1. Self-attention entre todas las features
        2. Conexión residual + normalización
        3. Transformación no lineal por feature
        4. Conexión residual + normalización
        """

        # Atención: cada feature atiende a las demás
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward independiente
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x

class FeatureAggregationEncoder(nn.Module):
    """
    Feature Aggregation Encoder

    Agrega los embeddings horarios de todas las features de un día
    en un único vector diario, aprendiendo relaciones inter-feature
    sin asumir ningún orden entre ellas.

    Entrada:
        H ∈ ℝ^{D × F × E}

    Salida:
        Z_day ∈ ℝ^{D × E}

    Interpretación:
        Cada Z_day representa el estado global del tráfico
        de una página web en un día concreto.
    """

    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()

        # Bloques de atención para enriquecer interacciones entre features
        self.sab1 = SetAttentionBlock(embed_dim, num_heads)
        self.sab2 = SetAttentionBlock(embed_dim, num_heads)

        # Pooling global sobre la dimensión de features (F → 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Forward del agregador diario.

        Pasos:
        1. Modelado de interacciones entre features (SAB)
        2. Refinamiento relacional (segundo SAB)
        3. Pooling global para obtener un vector diario único
        """

        # x: [D, F, E]
        x = self.sab1(x)
        x = self.sab2(x)

        # Pooling sobre features (no impone orden)
        x = x.transpose(1, 2)  # [D, E, F]
        x = self.pool(x)       # [D, E, 1]
        x = x.squeeze(-1)      # [D, E]

        return x


# =============================================================================
# =============================================================================
# APLICACIÓN
# =============================================================================
# =============================================================================

if __name__ == "__main__":

    #INPUT_EMB_CSV = "output/preprocessed/04_01_hourly_embeddings_maestras.csv"
    #INPUT_EMB_CSV = "output/preprocessed/04_02_hourly_embeddings_robust.csv"
    INPUT_EMB_CSV = "output/preprocessed/04_03_hourly_embeddings_all.csv"
    OUTPUT_DIR = "output/preprocessed/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Configuración de dispositivo (GPU si está disponible) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Usando dispositivo: {device}")

    # Cargar embeddings horarios generados en la Fase 3
    df = pd.read_csv(INPUT_EMB_CSV)

    # Identificar columnas del embedding horario
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    E = len(emb_cols)

    # Inicializar agregador diario y mover a dispositivo
    aggregator = FeatureAggregationEncoder(embed_dim=E).to(device)
    aggregator.eval()

    all_daily_rows = []

    # Optimización: Agrupar por sitio y fecha para evitar filtrados repetitivos
    grouped = df.groupby(['site_label', 'date_id'])

    # Inferencia sin gradientes para todo el proceso
    with torch.no_grad():
        # Procesar cada grupo (Sitio + Día)
        for (site_id, date), group_data in grouped:
            
            # Tensor de entrada: [1, F, E] movido al dispositivo
            X = torch.tensor(
                group_data[emb_cols].values,
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            # Inferencia: [1, F, E] -> [1, E]
            Z_day = aggregator(X)

            # Mover resultado a CPU para estructurar el CSV
            Z_day_cpu = Z_day.squeeze(0).cpu().numpy()

            # Construir fila de salida
            row = {
                "site_label": site_id,
                "date_id": date
            }

            # Añadir cada dimensión del embedding diario
            for i in range(E):
                row[f"day_emb_{i}"] = Z_day_cpu[i]

            all_daily_rows.append(row)

            # Opcional: imprimir cada vez que cambia de sitio
            # (Aunque groupby mezcla, site_label suele estar ordenado)

        print(f"[INFO] Agregación diaria completada para {len(grouped)} registros.")

    # Guardar embeddings diarios
    df_daily = pd.DataFrame(all_daily_rows)

    output_csv = os.path.join(
        OUTPUT_DIR,
        "05_03_daily_embeddings_all.csv"
    )

    df_daily.to_csv(output_csv, index=False)

    print("\nFeature Aggregation finalizada")
    print("Archivo generado:", output_csv)
    print("Dimensiones finales:", df_daily.shape)
