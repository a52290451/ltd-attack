# 🛡️ Estado del Arte — Evaluación de Concept Drift en Website Fingerprinting (WFP)

Este directorio contiene las replicaciones del **Estado del Arte (SOTA)** para evaluar el impacto real del **Concept Drift** en **Website Fingerprinting (WFP)**.

## 📋 Descripción General

Este estado del arte incluye tres baselines clásicos para evaluar el impacto del Concept Drift en la clasificación de sitios web:

1. **SOTA_01_CUMUL_SVM.py** — Enfoque clásico basado en **Support Vector Machines (SVM)** sobre características estadísticas de tráfico de red a nivel **Macro** (enfoque CUMUL de Panchenko et al.).
2. **SOTA_02_DeepFingerprinting_CNN.py** — Enfoque de **Deep Learning** basado en **CNN 1D** aplicada directamente a los vectores de dirección de paquetes (Deep Fingerprinting).
3. **SOTA_03_Rimmer_LSTM.py** — Enfoque de **Deep Learning Secuencial** basado en **LSTM** para captura de patrones temporales en vectores de dirección (Vera Rimmer et al., 2018).

Ambos baselines comparten la misma metodologia de evaluacion: se entrenan con datos del **Pasado** y se evaluan sobre datos del **Futuro** (con Concept Drift) para cuantificar la degradacion del modelo.

## 📁 Estructura de Archivos

| Archivo | Descripción |
|---------|-------------|
| `SOTA_01_CUMUL_SVM.py` | Baseline Clásico — SVM (RBF) sobre features estadísticas Macro (enfoque CUMUL) |
| `SOTA_02_DeepFingerprinting_CNN.py` | Baseline Deep Learning — CNN 1D sobre vectores de dirección de paquetes (Deep Fingerprinting) |
| `SOTA_03_Rimmer_LSTM.py` | Baseline Deep Learning Secuencial — LSTM sobre secuencias de direcciones (Vera Rimmer et al., 2018) |
| `run_estado_del_arte.sh` | Script Bash para ejecución automatizada de los tres baselines |
| `README.md` | Esta documentación |
| `resultados/` | Directorio de salida (generado automáticamente) |

## 🎯 Objetivo

Evaluar el rendimiento de dos modelos baseline entrenados sobre características de tráfico de red, comparando su precisión entre:

- **Dataset del Pasado** (Train Accuracy) — Datos de referencia sin Concept Drift
- **Dataset del Futuro** (Concept Drift Accuracy) — Datos con evolución temporal del tráfico

La diferencia entre ambas métricas cuantifica el **impacto del Concept Drift** en la clasificación de sitios web.

## 🔬 Baselines

### 1. SOTA_01 — CUMUL SVM (Enfoque Clásico)

Utiliza **94 características estadísticas invariantes** de nivel Macro con un **SVM (RBF kernel)**. Ver la sección de detalles del SVM para mas información.

### 2. SOTA_02 — Deep Fingerprinting CNN (Enfoque Deep Learning)

Implementa una **Red Convolucional 1D (CNN)** aplicada directamente a los **vectores de direccion de paquetes** (direction_vector) de tamano fijo (3000 posiciones).

### 3. SOTA_03 — Rimmer LSTM (Baseline Temporal Secuencial)

Implementa una **Red Neuronal Recurrente (LSTM)** basada en el enfoque de Vera Rimmer et al. (2018) para captura de **patrones temporales/secuenciales** en vectores de dirección de paquetes. Este modelo actua como **baseline temporal** para medir la resiliencia estructural de los vectores de direccion ante el Concept Drift.

#### Arquitectura Rimmer LSTM (SOTA_03)

| Capa | Tipo | Detalles |
|------|------|----------|
| **Embedding** | `nn.Embedding(4, 32)` | Codifica direcciones {-1, 0, 1, PAD} → embeddings de 32D (+1 para mapear indices a {1,2,3,4}) |
| **LSTM** | `nn.LSTM(32, 128, 2, batch_first=True, dropout=0.2)` | 2 capas LSTM, hidden_size=128, dropout=0.2 entre capas |
| **Extracción de contexto** | `out[:, -1, :]` | Toma la salida del ultimo paso de tiempo del LSTM (secuencia completa) |
| **Dropout** | `nn.Dropout(0.2)` | Regularizacion antes del clasificador |
| **Clasificador** | `nn.Linear(128, 65)` | Capa lineal final de clasificacion |

#### Entrenamiento LSTM

- **Framework:** PyTorch 2.0 con Precision Mixta (`autocast` + `GradScaler`)
- **Optimizador:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Data Split:** 80% Train / 20% Validation (stratificado sobre dataset Pasado)
- **Early Stopping:** Paciencia de 5 épocas (evaluando validation loss) — los LSTMs sobreajustan rapido en secuencias largas (MAX_SEQ_LEN=3000)
- **Regularizacion:** Dropout 0.2 entre capas LSTM + Dropout 0.2 antes del clasificador linear

#### Hipotesis del Rimmer LSTM (Baseline Temporal)

- Una LSTM puede capturar **patrones temporales/secuenciales** del trafico que una CNN solo espacial puede pasar por alto.
- El modelo sera sensible al Concept Drift porque los cambios en las rutas de Tor alteran el **orden y direccion de los paquetes en el tiempo**, afectando la firma secuencial del sitio.
- Como **baseline temporal**, este modelo proporciona una linea base para comparar la resiliencia estructural frente a la CNN (SOTA_02) que opera sobre patrones puramente espaciales.


#### Entrenamiento CNN

- **Framework:** PyTorch 2.0 con Precision Mixta (`torch.autocast` + `GradScaler`)
- **Optimizador:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Data Split:** 80% Train / 20% Validation (stratificado)
- **Early Stopping:** Paciencia de 10 épocas (evaluando validation loss)
- **Data Augmentation:** Embedding de direcciones como tecnica de regularizacion

#### Hipotesis del Deep Fingerprinting

- Una CNN 1D sobre vectores de direccion puede capturar **patrones espaciales** del trafico que son informativos para la clasificacion de sitios web.
- El modelo sufrira degradacion significativa bajo Concept Drift debido al cambio en las rutas de Tor que alteran la firma espacial de los paquetes.

## 🔬 Metodología

### Enfoque Macro (SOTA_01 — CUMUL SVM)

- Se utilizan **94 características estadísticas invariantes** del tráfico de red (rama Macro).
- Estas características capturan patrones de comportamiento a nivel de flujo de paquetes.

#### Modelo: Support Vector Machine (SVM)

- **Kernel:** RBF (Radial Basis Function)
- **Parámetros:** `C=1.0`, `gamma='scale'`
- **Normalización:** StandardScaler ajustado exclusivamente sobre el dataset del Pasado (Cero Data Leakage)

### Enfoque Deep Fingerprinting (SOTA_02 — CNN)

- Se utilizan **vectores de dirección de paquetes** (direction_vector) de tamano fijo (3000 posiciones).
- Los vectores contienen: **+1** (paquete entrante), **-1** (paquete saliente), **0** (padding).
- La CNN transforma estas secuencias en representaciones de alta dimensión que capturan **patrones espaciales** del trafico.

### Filtrado de Paridad

- Se filtra el dataset para mantener únicamente las **65 clases de élite** definidas por el LabelEncoder de vectores (`ds3_label_encoder_vec_3000.joblib`).
- Esto garantiza la comparabilidad directa con otros experimentos del laboratorio.

### Flujo de Ejecución

1. **Carga de datos:** Dos CSVs — datos históricos (Pasado) y datos con Concept Drift (Futuro).
2. **Filtrado por paridad:** Se mantienen solo las 65 clases de élite.
3. **Extracción de features:** Se aislan las 94 características estadísticas.
4. **Limpieza:** Se eliminan filas con valores NaN.
5. **Codificación:** Se transforman las etiquetas usando `LabelEncoder.transform()`.
6. **Escalado:** StandardScaler ajustado SOLO sobre el dataset del Pasado (Zero Data Leakage).
7. **Entrenamiento:** SVM entrenado sobre datos del Pasado escalados.
8. **Evaluación:** Predicción sobre Pasado y Futuro.
9. **Reporte:** Accuracy, matriz de confusión normalizada, y gráfica visual.

## 🚀 Ejecución

### Prerrequisitos

Los paquetes Python necesarios incluyen:
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `joblib`
- `torch` (PyTorch 2.0+) — requerido por SOTA_02 y SOTA_03

### Ejecutar los tres baselines

```bash
chmod +x run_estado_del_arte.sh
./run_estado_del_arte.sh
```

### Ejecutar individualmente

```bash
# Baseline 1: CUMUL SVM (Clasico)
python3 SOTA_01_CUMUL_SVM.py

# Baseline 2: Deep Fingerprinting CNN (Espacial)
python3 SOTA_02_DeepFingerprinting_CNN.py

# Baseline 3: Rimmer LSTM (Temporal/Secuencial)
python3 SOTA_03_Rimmer_LSTM.py
```

### Salida

Al ejecutar los scripts se generará automáticamente el directorio `resultados/` con:

| Archivo | Descripción |
|---------|-------------|
| `log_SOTA_01_CUMUL_[timestamp].txt` | Log detallado de la ejecución SVM |
| `G_SOTA_01_CUMUL_Drift_[timestamp].png` | Gráfica de la Matriz de Confusión Normalizada (SVM) |
| `log_SOTA_02_DF_[timestamp].txt` | Log detallado de la ejecución CNN |
| `best_df_cnn.pth` | Mejores pesos del modelo CNN guardados |
| `G_SOTA_02_DF_Curvas_[timestamp].png` | Curvas de Train/Val Loss y Accuracy (CNN) |
| `log_SOTA_03_LSTM_[timestamp].txt` | Log detallado de la ejecución LSTM |
| `best_rimmer_lstm.pth` | Mejores pesos del modelo LSTM guardados |
| `G_SOTA_03_LSTM_Curvas_[timestamp].png` | Curvas de Train/Val Loss y Accuracy (LSTM) |

## 📊 Métricas de Salida

- **Accuracy de Entrenamiento (PASADO):** Precisión del SVM sobre datos de entrenamiento.
- **Accuracy frente al Concept Drift (FUTURO):** Precisión del SVM sobre datos futuros con drift.
- **Impacto del Concept Drift:** Diferencia en puntos porcentuales entre ambas métricas.
- **Matriz de Confusión Normalizada:** Por filas, mostrando la distribución de errores por clase.
- **Classification Report:** Precision, Recall, F1-Score por clase.

## 📊 Metricas de Salida (Comunes a ambos baselines)

- **Accuracy de Entrenamiento (PASADO):** Precisión del modelo sobre datos de entrenamiento.
- **Accuracy de Validacion (PASADO):** Precisión del modelo sobre datos de validacion (20% del pasado).
- **Accuracy frente al Concept Drift (FUTURO):** Precisión del modelo sobre datos futuros con drift.
- **Impacto del Concept Drift:** Diferencia en puntos porcentuales entre ambas metricas.
- **SVM:** Matriz de confusión normalizada, Classification Report.
- **CNN:** Curvas de entrenamiento (Train/Val Loss y Accuracy), Classification Report del drift.
- **LSTM:** Curvas de entrenamiento (Train/Val Loss y Accuracy), Classification Report del drift.

##

## 📝 Referencias Academicas (SOTA 01 - 06)

La seleccion de los modelos de este directorio está rigurosamente fundamentada en la literatura principal de Website Fingerprinting (WFP).

### Los Baselines Clasicos (Ya implementados)

*   **SOTA_01 (CUMUL SVM):**
    > Panchenek, A., Lanze, F., Pennekamp, J., Engel, T., Zinnen, A., Henze, M., & Wehrle, K. (2016). *Website Fingerprinting at Internet Scale*. In Proceedings of the 23rd Internet Society (NDSS) Symposium.

*   **SOTA_02 (Deep Fingerprinting - CNN):**
    > Sirinam, P., Imani, M., Juarez, M., & Wright, M. (2018). *Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning*. In Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security (CCS).

*   **SOTA_03 (Rimmer - LSTM):**
    > Rimmer, V., Preuveneers, D., Juarez, M., Van Goethem, T., & Joosen, W. (2018). *Automated Website Fingerprinting through Deep Learning*. In Proceedings of the 25th Network and Distributed System Security Symposium (NDSS).

### Los Baselines Modernos (Por implementar)

*   **SOTA_04 (Var-CNN):**
    > Bhat, S., Lu, D., Kwon, A., & Devadas, S. (2019). *Var-CNN: A Data-Efficient Website Fingerprinting Attack Based on Deep Learning*. Proceedings on Privacy Enhancing Technologies (PoPETs), 2019(4), 292-310.

*   **SOTA_05 (Transformer-WFP):**
    > *(Baseline generico representativo de la literatura de 2023-2024, inspirado en trabajos como TF-WFP y GANDALF)*. La arquitectura estándar implementada sera: un Positional Encoding absoluto seguido de multiples bloques Transformer Encoder (Self-Attention) para clasificacion de secuencias de red.

*   **SOTA_06 (Triplet / Metric Learning):**
    > Cherubin, G., Jansen, R., & Troncoso, C. (2022). *Online Website Fingerprinting: Evaluating Website Fingerprinting Attacks on Tor in the Real World*. In Proceedings of the 31st USENIX Security Symposium.

## ⚠️

- **Cero Data Leakage:** El StandardScaler se ajusta exclusivamente sobre el dataset del Pasado. Esta es una práctica esencial para evitar fugas de información en la evaluación de Concept Drift.
- **Modo Headless:** El script utiliza `matplotlib.use('Agg')` para ejecución en servidores sin interfaz gráfica.
- **Logger Dual:** Toda la salida de consola se registra simultáneamente en un archivo de texto para auditoría.