# LTD-Attack — SOTA S4, protocolo ex ante para DF-WFlib y VarCNN-WFlib

**Fecha:** 2026-11-02. **Rama:** `feature/macro-v2-historical-only`. **Estado:** CONFIGURACIÓN VERSIONADA, **SIN entrenamientos sobre Historical DEV**. Un archivo denominado `frozen_protocol` es el candidato de congelación y **no implica todavía** aprobación del gate; el hash se registra en el preflight antes del primer entrenamiento.

### Punto de partida verificado

- S1: 57.916 capturas Historical DEV, 65 clases, 41 fechas, Early=19.098 (14 días), Middle=20.160 (14), Late=18.658 (13). Archivo de asignaciones original perdido, splits reconstruidos desde Historical Macro con SHA y tablas congeladas verificadas. Su equivalencia UID a UID con el original debe revisarse si reaparece.
- S3: 37/37 pruebas; auditoría temporal completa de 57.916 UID y 0 inválidos, SHA256 PASS; adaptador DT2 aplica política WFlib de IAT negativo→0 con trazabilidad (2.209 trazas con retrocesos).
- DF y Var-CNN: fuente WFlib fijada en commit `537ec8e03c805540b04b95b144fa158ce50aa255`; adelante tipo **DF-WFlib-pinned**, no reproducción binaria del original Keras. Forward CPU/GPU ya PASS sobre RTX 5080, PyTorch 2.11.0+cu128. Ningún entrenamiento Historical.
- Almacenamiento histórico copiado a `/data/ltd-attack/storage` y chequeado con `rsync --checksum --dry-run`; el cambio físico de `~/ltd-storage` está **PENDIENTE** y no afecta esta fase. No volver a crear entornos PyTorch ni modificar cron. El entorno CUDA transitorio comprobado sigue en `/data/ltd-sota-bsierra/venv-cu128`, pendiente de posterior consolidación de infraestructura.

## Contrato de datos: desarrollo solamente

La única entrada científica autorizada para **entrenar, validar y elegir hiperparámetros** es el índice 09A Historical DEV de S1. Jamás abrir 09B INTERNAL_TEST, 10B Future-B o Dataset C en los scripts de entrenamiento. No usar los archivos SOTA heredados que aprovecharon selección Future-informed de 94 features.

Entradas por arquitectura: DF `DIR` [N,1,5000]; VarCNN `DT2` [N,2,5000] con timestamps observados, no inferidos, y retrocesos explícitos. La clase `site_label` tiene el orden de 65 labels **fijado por SHA S1**; el orden de logits debe ser mapeado a las 65 etiquetas canónicas de scores antes de la evaluación 09B/10B; ese gate G2 sigue pendiente, sin abrir scores para entrenamiento.

## Hipótesis y entrenamiento previamente declarado

Configuración machine-readable: [sota_s4_frozen_protocol_v1.json](../../configs/sota/sota_s4_frozen_protocol_v1.json). Selección principal de **una semilla, 42**, para ambos baselines; sensibilidad con semillas 11 y 73 solo en una fase separada y predeclarada, **sin adaptar la decisión a resultados 09B/10B**. No es obligatorio dedicar el mismo número de parámetros ni el mismo batch a arquitecturas distintas, pero sí informar presupuesto y coste por comparación.

| Modelo | Fuente WFlib | Optimizador | LR | Batch | Límite |
|---|---|---|---|---|---|
| DF-WFlib-pinned | `scripts/DF.sh` | Adamax | 0.002 | 128 | 30 épocas |
| VarCNN-WFlib-DT2 | `scripts/Var-CNN.sh` | Adam | 0.001 | 50 | 30 épocas |

**Estrategia cronológica sin fuga:**

1. **A, diagnóstico:** entrenar en DEV_EARLY, validar en DEV_MIDDLE. Métricas reportadas, **no** usar para seleccionar el número de épocas de refit final.
2. **B, selección ex ante:** entrenar en DEV_EARLY+DEV_MIDDLE, validar solo en DEV_LATE. **Elegir la época que maximiza Macro-F1 sobre las 65 clases**, empatando a favor de la época menor. Sin búsqueda posterior de LR, batch, features o arquitectura sobre Late; 30 épocas máximo, sin early stopping.
3. **C, refit:** entrenar desde inicialización nueva, misma semilla e hiperparámetros, en **los 57.916 DEV** exactamente la cantidad de épocas elegida en B. Congelar pesos y SHA **antes** de cualquier score externo. No usar INTERNAL_TEST como validación.
4. Evaluación posterior: primero 09B y 10B únicamente con checkpoint y protocolo congelados. A/B, C y baseline LTD se comparan sobre los mismos UID/fechas, orden de clases verificado, Accuracy, Macro-F1, Top-5 y MRR; bootstrap pareado por fecha 2.500, IC95 percentil y familias Holm preespecificadas. Futuro conocido: **análisis retrospectivo**, no ciego.

El modelo DF conserva el uso de direcciones; VarCNN agrega timestamps observados. La comparación es honesta sobre la misma adquisición/objetivo, aunque **no es un estudio de aislamiento de modalidades**; si DF y VarCNN difieren, informar el acceso adicional de VarCNN al tiempo, y no atribuir diferencias exclusivamente a la arquitectura.

## Gate S4 antes de cualquier entrenamiento Historical

```bash
cd ~/ltd-attack
git fetch origin feature/macro-v2-historical-only &&
git merge --ff-only FETCH_HEAD &&
python3 -m py_compile src/models/sota/v1/s4_protocol.py \
  scripts/sota/s4_protocol_preflight_v1.py \
  scripts/sota/s4_synthetic_optimizer_smoke_v1.py &&
python3 -m unittest discover -s tests -p 'test_sota_s4_protocol_v1.py' -v &&
python3 scripts/sota/s4_protocol_preflight_v1.py --save
```

Esta etapa trabaja **solo con metadatos** S1, no lee ningún vector, Future-B ni resultados de 09B. El informe queda bajo `/data/ltd-attack/sota/protocol`. Estado esperado `SOTA_S4_PREFLIGHT=PASS_PLAN_NO_TRAINING`.

Luego, **únicamente si ese gate pasa**, realizar una optimización con dos ejemplos artificiales y un paso de gradiente por modelo en la CUDA ya comprobada (sin entrenar datos Historical):

```bash
cd ~/ltd-attack
/data/ltd-sota-bsierra/venv-cu128/bin/python \
  scripts/sota/s4_synthetic_optimizer_smoke_v1.py
```

Este código no crea checkpoint, no escribe pesos y falla si falta CUDA, si el wheel no coincide o si hay menos de 5 GiB VRAM libre. Produce un informe pequeño bajo `/data/ltd-attack/sota/protocol`. **No ejecutar scripts de entrenamiento histórico** hasta confirmar el paso, revisar el plan completo de datos/batches y fijar el SHA definitivo del protocolo.

**Restricciones:** no modificar `~/ltd-venv`, no duplicar el entorno CUDA, no introducir codigo nuevo en `/data`, no borrar las copias de datos, no cambiar cron de Dataset C ni promoverla a evidencia científica antes de arreglar NTP.

## Ejecución real S4 — entrenador Historical DEV añadido

**El nuevo archivo `scripts/sota/s4_train_historical_dev_v1.py` implementa entrenamiento real, no una prueba sintética.** Consume solo Historical DEV etiquetado desde S1, lee el CSV de vectores en un único recorrido y construye tensores NumPy float32 una vez para cada modelo/etapa; no lee 09B/10B/C. Todos los índices y cohortes se cotejan contra S1 antes de crear pesos. Usa exactamente los optimizadores, batch sizes, longitudes y semilla del protocolo. Los resultados y checkpoints se guardan en `/data/ltd-attack/sota/training/s4_wflib_v1_seed42/<model>/<stage>` sin sobrescribir runs previos. Limita el uso a una GPU, dos hilos y DataLoader de cero workers (utiliza lotes directos).

El paso B entrena Early+Middle y valida Late; elige la época con máximo Macro-F1 de 65 clases (desempata por época menor), guarda los 30 valores y el número elegido. El paso C exige que `--epochs-from-B` coincida exactamente con ese valor y recalcula pesos nuevos con los 57.916 DEV. Se guardan la fuente, hash del protocolo, SHA del checkpoint, conteos y trazas de métricas.

**Primera orden de entrenamiento real propuesta, pendiente de ejecución y comprobación en Perseo:**

```bash
cd ~/ltd-attack
git fetch origin feature/macro-v2-historical-only &&
git merge --ff-only FETCH_HEAD &&
python3 -m py_compile scripts/sota/s4_train_historical_dev_v1.py &&
/data/ltd-sota-bsierra/venv-cu128/bin/python \
  scripts/sota/s4_train_historical_dev_v1.py --model DF --stage B
```

**Importante:** la implementación está publicada, pero **no se ha ejecutado aún un entrenamiento real en Perseo** ni se han medido pérdidas, Accuracy o Macro-F1 de DF. Una falla de compatibilidad debe detener la ejecución, no forzar cambios en S1 ni en los datasets. En particular, `B` produce selección de época, no evaluación externa. El orden de clases de scores congelados sigue bloqueado para el futuro gate de evaluación.
