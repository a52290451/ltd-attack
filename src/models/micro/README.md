# Website Fingerprinting con Transformers (Deep Learning)

Este repositorio contiene una suite de experimentación en Deep Learning diseñada para resolver el problema de **Website Fingerprinting (WF)**. Utilizando arquitecturas basadas en **Transformers** (originalmente diseñadas para el Procesamiento de Lenguaje Natural), el sistema es capaz de identificar con precisión (+97%) a qué página web corresponde una captura de tráfico de red cifrado, analizando únicamente los metadatos de los paquetes.

## ¿Qué es el Website Fingerprinting?

El *Website Fingerprinting* es una técnica de análisis de tráfico donde un observador pasivo intenta deducir qué sitio web está visitando un usuario a través de un túnel cifrado (como VPNs, proxies o la red Tor). Dado que el contenido (payload) y las IPs destino están ocultos por el cifrado, el atacante (o el modelo de Machine Learning) se basa puramente en características de canal lateral (*side-channel*): 
* **Dirección:** Si el paquete entra (descarga) o sale (subida).
* **Tamaño:** Cuántos bytes pesa cada paquete.

Este proyecto aborda el WF como un problema de **Clasificación de Secuencias**. En lugar de palabras en una oración, el modelo lee la secuencia temporal de paquetes de red para extraer la "huella digital" única que genera la carga estructural de cada sitio web (HTML, imágenes pesadas, scripts de terceros, etc.).

---

## Sobre el Conjunto de Datos (Dataset)

Los datos utilizados para entrenar estos modelos provienen de un entorno de captura de tráfico de red real y automatizado. 

### Metodología de Captura
* **Frecuencia temporal:** Se registró el acceso a un listado específico de sitios web a razón de **1 acceso por hora** para cada sitio.
* **Ventana de observación:** Las capturas se realizaron de forma continua a lo largo de **60 días** aproximadamente.
* **Anatomía de un registro:** Cada fila (registro) en el dataset representa **una sesión de conexión completa** a una página web. Contiene el detalle secuencial absoluto de todos los paquetes de red generados durante la carga de dicho sitio (desde el saludo inicial TCP/TLS hasta la carga final de los recursos).

### Manejo de Ruido y Desbalanceo
En un entorno de red real (Internet), las capturas son susceptibles a fallos (tiempos de espera agotados, caídas de conexión o bloqueos de servidor). Esto genera un **desbalanceo natural** en el dataset, donde algunas páginas web tienen el registro completo de los 60 días (ej. ~1200 muestras), mientras que otras tienen menos debido a errores de captura.

Para mitigar esto, el pipeline implementa un **Filtro de Calidad**:
1. Se descartan anomalías conocidas (clases rotas como la 49 y la 5).
2. Se calcula la clase mayoritaria y se establece un **umbral estricto del 50%**. Si un sitio web no logró capturar al menos la mitad de las muestras que la clase mayoritaria, es descartado del entrenamiento para no sesgar a la red neuronal.

### Preparación y Partición de Datos (Data Splitting)
Antes de inyectar los datos al modelo Transformer:
1. **Estandarización de Longitud:** Dado que las páginas web varían en su peso, las secuencias de red tienen longitudes distintas. Se aplica un *Truncado* (corte) y un *Padding* (relleno con ceros) para que todas las sesiones midan exactamente lo mismo (ej. 1000, 3000 o 5000 paquetes).
2. **Normalización:** Los tamaños de los paquetes (en bytes) se escalan matemáticamente al rango `[0, 1]` usando *Min-Max Scaling* global.
3. **Split Estratificado:** El dataset se divide en un **80% para Entrenamiento (Train)** y un **20% para Validación (Test)**. Se utiliza el parámetro `stratify` para garantizar que la proporción exacta de cada sitio web se mantenga idéntica en ambos conjuntos, asegurando que el modelo se evalúe de manera justa.

---

## Arquitecturas de los Modelos

El proyecto compara dos enfoques principales para la extracción de características temporales, lidiando con secuencias de red que pueden superar los 5000 paquetes. Dado que el mecanismo de atención de un Transformer tiene una complejidad espacial y temporal de O(N^2), ambos modelos implementan una capa de convolución unidimensional (`Conv1d` con `stride=2`) como mecanismo de **Downsampling** para reducir la longitud de la secuencia a la mitad antes del Transformer, optimizando drásticamente el uso de VRAM sin perder información semántica.

### 1. Modelo Unimodal (Estrategia Base: `DirectionTransformer`)
Analiza exclusivamente el ritmo y patrón del tráfico mediante la secuencia de direcciones de los paquetes.
* **Entrada:** Vector discreto donde `+1` es saliente, `-1` es entrante, y `0` es padding.
* **Flujo de datos:**
  1. `Embedding` para proyectar el espacio discreto.
  2. `Conv1d` (Downsampling espacial).
  3. `Positional Encoding` (Seno/Coseno).
  4. `TransformerEncoder` (4 capas, 8 cabezas de atención).
  5. `Mean Pooling` + `Max Pooling` concatenados.
  6. Clasificador lineal final (MLP).

### 2. Modelo Multimodal (Fusión Temprana: `MultimodalTransformer`)
Combina de forma simultánea la dirección del paquete con su tamaño en bytes, permitiendo a la red entender el "volumen" real de las transferencias.
* **Entradas:** Vector discreto de direcciones + Vector continuo de tamaños.
* **Flujo de datos (Early Fusion):**
  1. Las direcciones pasan por un `nn.Embedding`.
  2. Los tamaños pasan por una proyección lineal (`nn.Linear`).
  3. **Fusión:** Ambos vectores se concatenan y se reducen mediante una capa lineal para unificar la representación.
  4. `Conv1d` (Downsampling espacial).
  5. `Positional Encoding` (Seno/Coseno).
  6. `TransformerEncoder` (4 capas, 8 cabezas de atención).
  7. `Mean Pooling` + `Max Pooling` concatenados.
  8. Clasificador lineal final (MLP).

---

## Estructura del Proyecto

### 1. Scripts Principales (Pipeline)

| Archivo | Descripción |
| :--- | :--- |
| `V_revision.py` | Herramienta de auditoría de datos. Compara los datasets de vectores vs features para garantizar integridad y visualizar el desbalanceo. |
| `VP2_Transformers_directions.py` | Script de entrenamiento para el **Modelo Unimodal** (Solo direcciones). |
| `VP1_Transformers_dir_size.py` | Script de entrenamiento para el **Modelo Multimodal** (Dirección + Peso). Implementa Fusión Temprana y normalización dinámica. |
| `VP_Test_modelos.py` | Script de Validación Masiva (Hold-Out). Permite cruzar todos los modelos entrenados contra un dataset crudo de un marco temporal distinto para evaluar la capacidad de generalización real y descartar *Data Leakage*. |
| `VP_Evaluar_ConceptDrift_Vectores.py` | Suite unificada que evalúa la resiliencia del modelo de vectores crudos frente al Concept Drift, generando métricas (.txt) y evidencias de autopsia visual (UMAP, Matriz de Dispersión, y Huella de Secuencia). |

### 2. Nomenclatura de Resultados (`/resultados`)

Los pesos, metadatos y reportes se guardan automáticamente con prefijos para identificar su origen y la longitud máxima de secuencia (`MAX_LEN`) configurada:

* **Familia `d_`** : Resultados del modelo Unimodal (Direcciones).
  * *Ejemplo:* `d_best_multimodal_transformer_vec_1000.pth`
* **Familia `ds2_`** : Resultados del modelo Multimodal (Direcciones + Tamaños).
  * *Ejemplo:* `ds2_resultado_final_5000.txt`

Cada ejecución exitosa genera 4 archivos:
1. `*.pth`: Pesos de la red neuronal en PyTorch.
2. `*label_encoder*.joblib`: El decodificador de clases de Scikit-Learn.
3. `*class_mapping*.joblib`: Diccionario de mapeo legible.
4. `*.txt`: Reporte final con Precision, Recall, F1-Score, Accuracy Global y tiempos de entrenamiento.

---

## Requisitos del Sistema

- **Python 3.8+**
- **Hardware:** Tarjeta Gráfica (GPU) con soporte CUDA altamente recomendada (Min. 12GB VRAM para procesar `MAX_LEN = 5000` con `BATCH_SIZE = 32`).
- **Dependencias principales:**
  ```bash
  pip install torch pandas numpy scikit-learn transformers joblib matplotlib