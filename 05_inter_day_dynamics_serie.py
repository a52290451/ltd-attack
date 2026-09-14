"""
=====================================================================================
FASE 5 — INTER-DAY DYNAMICS (DINÁMICA TEMPORAL ENTRE DÍAS)
=====================================================================================

    Descripción general
    -------------------
    Este script implementa la **fase de modelado temporal entre días** dentro del
    pipeline jerárquico de análisis de tráfico web basado en embeddings.

    Parte de los embeddings diarios generados en la Fase 4 (Feature Aggregation),
    donde cada día de una página web está representado por un vector compacto que
    resume todas sus features horarias.

    El objetivo de esta fase es **aprender patrones de evolución temporal** a nivel
    de página web, tales como:
        - estabilidad vs variabilidad
        - cambios graduales
        - patrones periódicos (semanales, cíclicos)
        - fingerprints temporales característicos de cada sitio

    A diferencia de fases anteriores:
        - Existe orden temporal
        - Los días importan en su secuencia
        - No se analizan horas ni features individualmente

    Para ello, se emplea un **Transformer temporal**, que modela dependencias
    a largo plazo entre días mediante atención auto-regresiva.

    Entrada
    -------
    - CSV generado en la Fase 4 (`daily_embeddings_all_sites.csv`)
    - Para cada sitio web:
        Z ∈ ℝ^{D × E}

    Donde:
        - D : número de días observados
        - E : dimensión del embedding diario

    Salida
    ------
    - Un embedding global por página web:
        G ∈ ℝ^{E}

    Este embedding:
        - Resume la evolución temporal completa del sitio
        - Es robusto frente a ruido diario
        - Puede usarse directamente para clasificación, clustering o fingerprinting

    Rol en el pipeline
    ------------------
    Esta es la última fase de codificación.
    El embedding global resultante constituye la **MetaFeature final** de cada
    página web.

=====================================================================================
"""

import torch            # type: ignore
import torch.nn as nn   # type: ignore
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os

class PositionalEncoding(nn.Module):
    """
        Positional Encoding sinusoidal para secuencias temporales de días.

        Motivación
        ----------
        Los Transformers son modelos invariantes al orden por diseño.
        Esto significa que, sin información adicional, el modelo no puede
        distinguir entre:

            Día 1 → Día 2 → Día 3
        y
            Día 3 → Día 1 → Día 2

        En el contexto de dinámica temporal entre días, el orden ES crítico.
        Por ello, se introduce un encoding posicional que inyecta información
        explícita sobre la posición temporal absoluta de cada día.

        Diseño
        -------
        Se utiliza el positional encoding sinusoidal clásico propuesto en:
            "Attention Is All You Need" (Vaswani et al.)

        Propiedades clave:
        - No introduce parámetros entrenables
        - Permite extrapolar a secuencias más largas
        - Codifica la posición mediante frecuencias senoidales
        - Facilita que el Transformer aprenda relaciones relativas entre días

        Interpretación intuitiva
        ------------------------
        Cada día recibe una "firma temporal" única que indica:
            - su posición absoluta en la secuencia
            - su relación relativa con otros días

        Esto permite aprender patrones como:
            - periodicidad semanal
            - estabilidad prolongada
            - cambios graduales o abruptos

        Parámetros
        ----------
        embed_dim : int
            Dimensión del embedding diario (E)
        max_len : int
            Número máximo de días soportados (por defecto 365)
    """

    def __init__(self, embed_dim, max_len=365):
        super().__init__()

        # Matriz [max_len, embed_dim] que almacenará los encodings
        pe = torch.zeros(max_len, embed_dim)

        # Posiciones temporales: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1)

        # Término de escala para las frecuencias
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim)
        )

        # Componentes senoidales (dimensiones pares)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Componentes cosenoidales (dimensiones impares)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Registrar como buffer (no entrenable, pero movible a GPU)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
            Aplica positional encoding a una secuencia de embeddings diarios.

            Parámetros
            ----------
            x : torch.Tensor
                Tensor de entrada de forma:
                    [D, E]
            Retorna
            -------
            torch.Tensor
                Tensor [D, E] con información temporal añadida
        """
        D = x.size(0)

        # Se suma el encoding posicional correspondiente a cada día
        return x + self.pe[:D]
    
class InterDayDynamicsEncoder(nn.Module):
    """
        Transformer temporal para modelar la dinámica entre días de una página web.

        Objetivo
        --------
        Aprender patrones de evolución temporal a alto nivel a partir de
        embeddings diarios previamente agregados.

        Este bloque NO analiza:
            - horas
            - features individuales

        Su foco exclusivo es:
            - cómo cambian los días entre sí
            - qué tan estable es el comportamiento del sitio
            - qué patrones temporales caracterizan a la página

        Entrada
        -------
        Z ∈ ℝ^{D × E}

        donde:
            D : número de días observados
            E : dimensión del embedding diario (salida de Feature Aggregation)

        Salida
        ------
        g ∈ ℝ^{E}

        Un embedding global que:
            - resume toda la secuencia temporal
            - actúa como fingerprint de la página web
            - es robusto frente a ruido diario

        Arquitectura
        ------------
        1. Positional Encoding (orden temporal explícito)
        2. Transformer Encoder (auto-atención entre días)
        3. Pooling temporal (reducción D → 1)

        Justificación
        -------------
        El Transformer permite capturar:
            - dependencias a largo plazo
            - relaciones no locales entre días
            - patrones periódicos y transiciones suaves

        El pooling final genera una representación fija,
        independiente del número de días observados.
    """

    def __init__(
        self,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        # Inyección explícita de información temporal
        self.positional_encoding = PositionalEncoding(embed_dim)

        # Capa base del Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Encoder completo con múltiples capas
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Pooling temporal para obtener vector global
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
            Propagación hacia adelante del encoder temporal.

            Parámetros
            ----------
            x : torch.Tensor
                Tensor de entrada [D, E]
                Secuencia de embeddings diarios ordenados temporalmente

            Retorna
            -------
            torch.Tensor
                Vector global [E] representando la dinámica completa del sitio
        """

        # 1. Añadir información explícita de posición temporal
        x = self.positional_encoding(x)

        # 2. Añadir dimensión batch requerida por Transformer
        #    [D, E] → [1, D, E]
        x = x.unsqueeze(0)

        # 3. Modelar dependencias temporales entre días
        x = self.transformer(x)  # [1, D, E]

        # 4. Pooling sobre la dimensión temporal (D)
        x = x.transpose(1, 2)    # [1, E, D]
        x = self.pool(x)         # [1, E, 1]
        x = x.squeeze(0).squeeze(-1)          # [E]
        
        return F.normalize(x, p=2, dim=-1)

# =============================================================================
# APLICACIÓN: INTER-DAY DYNAMICS
# =============================================================================
"""
=====================================================================================
INTER-DAY DYNAMICS: GENERACIÓN DE EMBEDDINGS GLOBALES POR SITIO WEB
=====================================================================================

    Entradas
    --------
    - CSV de embeddings diarios (`INPUT_DAILY_CSV`) con columnas:
        - `site_label`: identificador del sitio web
        - `date_id`: fecha del embedding diario
        - `day_emb_0, ..., day_emb_E`: dimensiones del embedding diario

    - Formato esperado del tensor de entrada al encoder:
        - Tensor X ∈ ℝ^{D × E}, donde:
            - D = número de días para el sitio
            - E = dimensión del embedding diario

    Procesamiento
    -------------
    1. Se ordenan los días por `date_id`.
    2. Se construye un tensor [D, E] con los embeddings diarios.
    3. Se aplica `InterDayDynamicsEncoder` con:
        - Positional Encoding sinusoidal
        - Transformer temporal multi-cabeza
        - Pooling promedio temporal
    4. Se obtiene g ∈ ℝ^E: embedding global del sitio

    Salida
    ------
    - CSV `05_site_embeddings_all_sites.csv` con columnas:
        - `site_label`
        - `site_emb_0, ..., site_emb_E`: embedding global del sitio web
    - Cada fila representa un **sitio web completo**, integrando todas las horas y días.

=====================================================================================
"""

if __name__ == "__main__":

    # 1. Configuración de dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Usando dispositivo: {device}")
    
    # --- Configuración de rutas ---
    #INPUT_DAILY_CSV = "output/preprocessed/05_01_daily_embeddings_maestras.csv"
    #INPUT_DAILY_CSV = "output/preprocessed/05_02_daily_embeddings_robust.csv"
    INPUT_DAILY_CSV = "output/preprocessed/05_03_daily_embeddings_all.csv"
    OUTPUT_DIR = "output/preprocessed/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Carga del CSV de embeddings diarios ---
    df = pd.read_csv(INPUT_DAILY_CSV)

    # --- Identificación de columnas de embedding diario ---
    day_emb_cols = [c for c in df.columns if c.startswith("day_emb_")]
    E = len(day_emb_cols)

    # --- Inicialización del encoder de dinámica inter-día ---
    # 2. Modelo a GPU
    encoder = InterDayDynamicsEncoder(embed_dim=E).to(device)
    encoder.eval()

    all_site_rows = []
    
    # 3. Procesamiento optimizado
    # Usamos groupby para asegurar que tratamos cada sitio como una secuencia
    grouped = df.sort_values("date_id").groupby("site_label")

    # --- Iteración por sitio web ---
    with torch.no_grad():
        for site_id, df_site in grouped:
            # Construir tensor [D, E]
            X = torch.tensor(
                df_site[day_emb_cols].values,
                dtype=torch.float32
            ).to(device) # Mover a GPU

            # El Transformer espera [D, E], el forward añade el batch
            g = encoder(X) # Salida: [E]

            # Volver a CPU para guardar
            g_cpu = g.cpu().numpy()

            row = {"site_label": site_id}
            for i in range(E):
                row[f"site_emb_{i}"] = g_cpu[i]

            all_site_rows.append(row)
            
        print(f"[INFO] Procesados {len(all_site_rows)} sitios únicos.")

    # --- Creación del DataFrame final y guardado en CSV ---
    df_sites = pd.DataFrame(all_site_rows)
    output_csv = os.path.join(
        OUTPUT_DIR,
        "06_03_all_site_embeddings_all.csv"
    )
    df_sites.to_csv(output_csv, index=False)

    print("\nInter-Day Dynamics finalizada")
    print("Archivo generado:", output_csv)
    print("Dimensiones finales:", df_sites.shape)

