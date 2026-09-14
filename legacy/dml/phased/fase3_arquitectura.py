"""
=========================================================================================
FASE 3: ARQUITECTURA NEURONAL Y CONFIGURACIÓN DE LA FUNCIÓN DE PÉRDIDA (DEEP METRIC LEARNING)
=========================================================================================
Objetivo: Definir la arquitectura del modelo híbrido (CNN 1D + Transformer) para procesar
          perfiles horarios de tráfico, y configurar la función de pérdida geométrica.

Requisitos de entrada:
- Ninguno en este script directamente (Las matrices X_train, y_train de la Fase 2 serán
  consumidas por el flujo de la Fase 4 que importa este modelo).

Reglas aplicadas (Diseño bajo estrictos estándares SOTA):
1. Extractor Intra-Hora: CNN 1D para capturar ráfagas locales de tráfico (3000 paquetes).
2. Agregador Inter-Hora: Transformer Encoder para modelar dependencias en ventanas de 24h.
3. Cabeza de Proyección: MLP final que proyecta hacia un espacio latente de baja dimensión.
4. Normalización L2: Obligatoria en la salida para forzar los embeddings a una hiper-esfera.
5. Función de Pérdida: Supervised Contrastive Loss (SupCon) optimizada para lotes PxK.

Salida:
- Clases modulares listas para ser instanciadas en el bucle de entrenamiento (Fase 4).
=========================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. FUNCIÓN DE PÉRDIDA GEOMÉTRICA (SUPCON)
# ==========================================
class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning (SupCon).
    Fuerza a las muestras de la misma clase a acercarse (maximizar similitud coseno)
    y a las muestras de clases distintas a repelerse en el espacio latente.
    Diseñado específicamente para ser alimentado por el PKBatchSampler de la Fase 2.
    """
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (Batch, Hidden_Dim) - Embeddings normalizados L2.
        labels: (Batch) - Etiquetas de clase asociadas.
        """
        device = features.device
        batch_size = features.shape[0]

        # Calcular la matriz de similitud de pares (Producto Punto = Similitud Coseno dado que están norm. L2)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Crear una máscara de clases (1 si la fila i y columna j tienen la misma clase)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Enmascarar la diagonal para evitar que una muestra se compare consigo misma
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Estabilidad numérica
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Calcular el promedio de la probabilidad logarítmica sobre los ejemplos positivos
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        # La pérdida total es el negativo del logaritmo esperado
        loss = -mean_log_prob_pos.mean()
        return loss

# ==========================================
# 2. BLOQUE 1: EXTRACTOR LOCAL (CNN 1D)
# ==========================================
class LocalBurstExtractor(nn.Module):
    """
    Procesa la secuencia de paquetes crudos de una sola hora (MAX_SEQ_LEN).
    Comprime la dimensionalidad temporal usando Convoluciones 1D y Max Pooling.
    """
    def __init__(self, in_channels=2, out_channels=128):
        super(LocalBurstExtractor, self).__init__()
        
        # Arquitectura CNN 1D básica
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv1d(128, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        # Extraemos 50 ventanas representativas en lugar de aplastar toda la hora en 1 valor
        self.adaptive_pool = nn.AdaptiveAvgPool1d(50)

    def forward(self, x):
        # x input = (Batch * 24, Length=3000, Channels=2)
        # PyTorch Conv1d espera (Batch, Channels, Length)
        x = x.transpose(1, 2)
        
        # output shape = (Batch * 24, Out_Channels, 1)
        features = self.conv_block(x)
        features = self.adaptive_pool(features)# (Batch*24, Out_Channels, 50)
        # Aplanar las ventanas para darle una representación densa rica de la hora al transformer
        return features.view(features.size(0), -1)

# ==========================================
# 3. BLOQUE 2 Y 3: TRANSFORMER + PROYECCIÓN
# ==========================================
class HourlyEncoderDML(nn.Module):
    """
    Arquitectura principal del modelo.
    Combina el extractor local (CNN) con el agregador temporal de 24h (Transformer),
    y remata con la proyección L2 hacia el espacio latente.
    """
    def __init__(self, cnn_out_dim=64, windows=50, num_heads=8, num_layers=3, latent_dim=256):
        super(HourlyEncoderDML, self).__init__()
        
        # Dimensión total después de aplanar = 64 canales * 50 ventanas
        self.d_model = cnn_out_dim * windows
        
        # 1. Extractor local instanciado
        self.local_extractor = LocalBurstExtractor(in_channels=2, out_channels=cnn_out_dim)
        
        # 2. Embedding Posicional (Para que el Transformer sepa qué hora del día es)
        # 24 horas como máximo
        self.positional_encoding = nn.Parameter(torch.randn(1, 24, self.d_model))
        
        # 3. Transformer Encoder (Agregador Global)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=num_heads, 
            dim_feedforward=self.d_model * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Cabezal de Proyección (Projection Head)
        self.projection_head = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        """
        x: Tensor de entrada con dimensiones (Batch, 24, MAX_SEQ_LEN, 2)
        """
        batch_size, hours, seq_len, channels = x.shape
        
        # --- PASO 1: Procesar cada hora independientemente a través de la CNN ---
        # Aplanamos la dimensión de batch y horas temporalmente
        x_flat = x.view(batch_size * hours, seq_len, channels)
        
        # features_flat: (Batch * 24, cnn_out_dim)
        features_flat = self.local_extractor(x_flat)
        
        # --- PASO 2: Reconstruir la secuencia temporal de 24h ---
        # features_seq: (Batch, 24, cnn_out_dim)
        features_seq = features_flat.view(batch_size, hours, -1)
        
        # Sumar la codificación posicional (¿Es la 1 am o las 10 pm?)
        features_seq = features_seq + self.positional_encoding
        
        # --- PASO 3: Pasar por el Transformer ---
        transformer_out = self.transformer(features_seq)
        
        # Global Average Pooling a lo largo de las 24 horas
        # daily_profile: (Batch, cnn_out_dim)
        daily_profile = transformer_out.mean(dim=1)
        
        # --- PASO 4: Cabezal de Proyección y Normalización L2 ---
        latent_vector = self.projection_head(daily_profile)
        
        # Obligatorio para SupCon / Cosine Similarity: Normalizar sobre la hiper-esfera
        normalized_latent = F.normalize(latent_vector, p=2, dim=1)
        
        return normalized_latent

# Bloque de prueba de dimensiones para validación rápida (Opcional)
if __name__ == "__main__":
    print("Validando arquitectura con un tensor aleatorio ficticio...")
    # Simulando un lote (Batch=2, Horas=24, Longitud=3000, Canales=2)
    dummy_input = torch.randn(2, 24, 3000, 2)
    dummy_labels = torch.tensor([0, 0]) # Simulando que pertenecen a la misma clase
    
    model = HourlyEncoderDML(latent_dim=256)
    loss_fn = SupervisedContrastiveLoss()
    
    out_vector = model(dummy_input)
    loss = loss_fn(out_vector, dummy_labels)
    
    print(f"Formato de Salida Latente: {out_vector.shape}")
    print(f"Norma L2 del primer vector (Debe ser ~1.0): {torch.norm(out_vector[0]).item():.4f}")
    print(f"Valor de Contrastive Loss inicial: {loss.item():.4f}")
    print("La Arquitectura Fase 3 está lista y libre de errores dimensionales.")