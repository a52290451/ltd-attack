# LTD-Attack — protocolo de comparación SOTA / contrato preexperimental v1

**Checkpoint operativo S3 y Dataset C:** [CHECKPOINT_20261102_SOTA_S3_DATASET_C_PARALLEL.md](../research/CHECKPOINT_20261102_SOTA_S3_DATASET_C_PARALLEL.md) — incluye verificaciones ya realizadas, commits de referencia, pruebas locales pendientes y procedimiento de auditoría read-only de cron/capturas. Es el historial de decisiones de implementación en curso; no sustituye los manifiestos históricos congelados.

**Estado 2026-11-02:** preparación, sin ejecución de entrenamiento ni evaluación adicional. Reconciliación con documentación histórica: [protocolo de comparación 2026-10-30](../research/LTD_ATTACK_SOTA_FAIR_COMPARISON_PROTOCOL_DRAFT_20261030.md), [inferencia congelada V4](../research/LTD_ATTACK_STATISTICAL_INFERENCE_CLOSEOUT_V4_20261030.md), [linaje de features](../FEATURE_LINEAGE.md). Este es el **punto de entrada operativo único** para SOTA; conservar los documentos originales como evidencia histórica (no son basura).

## 0. Fronteras de datos y reproducibilidad

- **09A — Historical DEV**: 57.916 `pcap_uid` únicos, 65 etiquetas, 41 fechas. Partición fija: Early 19.098/14 días (2025-11-18 a 2025-12-01); Middle 20.160/14 (2025-12-02 a 2025-12-15); Late 18.658/13 (2025-12-16 a 2025-12-28). Solo este conjunto puede influir en entrenamiento, validación, preprocesamiento y selección de modelos.
- **09B — INTERNAL_TEST**: 14.687 `pcap_uid`, 65 etiquetas, 11 fechas (2025-12-29 a 2026-01-08). Validación retrospectiva, no fuente de hiperparámetros.
- **10B — Future-B DEV_FROZEN**: 18.543 `pcap_uid`, 65 etiquetas, 16 fechas. Misma versión de comparación contra Hybrid/Micro/Macro DEV_FROZEN, **no** `HISTORICAL_FINAL`. Future-B ya fue abierto; nuevas comparaciones son retrospectivas, no ciegas.
- **Dataset C**: QA recurrente y temporalmente no acreditada; no se usa en entrenamiento SOTA ni adaptación. NTP del host informado no sincronizado.
- **Manifiestos 09A/09B/10B/10C** y puntuaciones previas son inmutables. No reescribir resultados, particiones, pesos, etiquetas, experimentos históricos ni originales PCAP.
- Fuente de S1: `CLEAN_final_vectors_sites.csv` (Micro dirección) + `CLEAN_final_features_sites.csv` (metadatos de fecha y sitio) alineados **1:1 por `pcap_uid`**. Comprobar SHA256 respecto a los manifiestos congelados. El CSV de asignaciones original 09A debe cotejarse por SHA; si falta, reconstruir **solo** desde metadatos Historical verificados y fechas congeladas, dejar `RECONSTRUCTED`, comparar contra el original si reaparece y no afirmar identidad de sus bytes. Nada de muestreos aleatorios ni normalizadores ajustados con 09B/10B.
- Orden de clases: conservar el **mismo vector de 65 etiquetas, mismo orden** que los NPZ de puntuaciones canónicas. El orden alfabético solo puede aceptarse cuando se compruebe que coincide. Si no, crear un mapeo explícito antes de ejecutar.
- Si faltan muestras/representaciones: fallar con evidencia y calcular una **subcohorte de UIDs común**, por separado, recalculando *todos* los modelos sobre exactamente esos UIDs. No declarar comparables porcentajes sobre denominadores distintos.

## 1. Modelos y fidelidad

| Modelo | Prioridad | Track | Restricción |
|---|---|---|---|
| DF, original CCS 2018 | P1 | Frozen | Original 5.000 direcciones; comprobar si el extractor Historical permite 5.000. Si el archivo ya está truncado a 3.000, llamar a la ejecución `DF-adapted-3000` y no reclamar fidelidad original |
| Var-CNN | P1 | Frozen | Fijar commit de WFlib / original, entrada exacta (dirección / tiempo si disponible) y arquitectura contra paper |
| CUMUL/SVM reconstruido | P2 | Frozen | Features CUMUL originales derivadas solo de Historical; **prohibido** usar `new_features_invariantes_seguras.txt` (94 seleccionadas con Future) |
| Tik-Tok | P2 | Frozen | Solo si timestamps observables por paquete están presentes con fidelidad; no inventar tiempos |
| NetCLR | P2 | Frozen | Documentar arquitectura, aumentaciones, presupuesto de preentrenamiento y acceso a datos |
| Proteus, Aion | P3 | Adaptativo independiente | Diferentes hipótesis: tráfico futuro sin etiquetas o con supervisión. No puntuar como control congelado |

**Scripts históricos bloqueados como batería canónica:** `scripts/train/run_estado_del_arte.sh`, `src/models/sota/SOTA_01_CUMUL_SVM.py`, `SOTA_02_DeepFingerprinting_CNN.py`, `SOTA_03_Rimmer_LSTM.py`. Son referencias de ingeniería, **no ejecutables científicos aprobados**. Conservar en repositorio mientras se auditan dependencias; eliminar solo artefactos temporales demostrablemente redundantes. No sobrescribirlos con nuevas implementaciones.

## 2. Especificación común de evaluación (idéntica a LTD-Attack)

**Para cada cohorte y modelo**: guardar predicciones con `pcap_uid`, fecha, `site_label`, índice real `y_true`, **vector de 65 probabilidades ordenadas**, SHA del checkpoint, extractor/input, semilla y track. Validar unicidad de UID, no pérdida ni duplicación, probabilidades finitas/no negativas/suma≈1, correspondencia 1:1 de clases y misma representación del adversario.

| Métrica | Definición congelada | Interpretación |
|---|---|---|
| **Accuracy / Top-1** | Fracción de muestras cuyo máximo de probabilidad corresponde a su clase real | Endpoint principal del SOTA comparativo |
| **Macro-F1** | Media aritmética no ponderada de F1 de las 65 clases, con denominador `support_c + predicted_c`, recomputada tras cada remuestreo | Equidad entre clases, obligatoria |
| **Top-5 Accuracy** | Fracción de muestras con clase real dentro de los primeros cinco puestos | Recuperación en ranking |
| **MRR** | Media de `1 / rango(clase_real)`, rango basado en el orden completo de las 65 probabilidades | Calidad de ranking |

Indicadores complementarios **obligatorios**: N y días por cohorte, Top-1/Top-5/MRR/Macro-F1 por fecha y por clase, recall/F1 por clase, rescates/perjuicios vs Hybrid, dispersión temporal, coste de entrenamiento/inferencia, memoria/GPU, semillas, pérdidas por datos ausentes y características accesibles por baseline. No sustituir Macro-F1 por micro-F1 ni interpretar Top-5 como cinco modelos distintos.

**Inferencia común**: delta candidato − referencia sobre **los mismos UIDs y fechas**. Bootstrap pareado por día (09B: 11 bloques, 10B: 16), 2.500 remuestreos, IC percentil del 95 %; mismas muestras de bloques para ambos modelos. Para Top-1/Top-5/MRR puede realizarse intercambio condicional bilateral de etiquetas de modelo por día completo (11 días → 2^11, 16 días → 2^16); para Macro-F1 usar la prueba Monte Carlo por bloques, **no adjudicarle el p exacto aditivo**. Exponer supuestos de intercambiabilidad/dependencia entre días. La fracción bootstrap de diferencias positivas **NO es un valor p**.

**Corrección de múltiples comparaciones**: el expediente original distingue **Holm12** (3 pares × 4 métricas; Monte Carlo) y **Holm9** (3 pares × 3 métricas aditivas; intercambio exacto). **No fusionar familias ni reutilizar p ajustados para modelos nuevos**. Predefinir una familia independiente SOTA: por ejemplo 2 baselines principales (DF/Var-CNN) × 4 métricas, **Holm8 por cohorte** para pruebas exploratorias; sensibilidad de solo 3 métricas aditivas y sus métodos exactos en una familia distinta **Holm6**, sin confundirlas. Toda conclusión es retrospectiva dadas las aperturas previas de 09B/10B.

## 3. Entrenamiento y presupuesto — congelar ANTES de usar GPU

1. **S1 / gate de fuentes**: índice metadata-only 57.916 UIDs; 41 fechas/65 clases; hashes históricos, split, longitud de vectores, clases; obtener evidencia exacta del orden de `candidate_labels` de 09B y 10B **en un gate posterior, solo lectura** antes de inferencia. No abrir puntuaciones futuras en el entrenamiento.
2. **S2 / adaptadores**: crear implementación nueva en `src/models/sota/v1/`, tests unitarios de dirección/longitud/máscara, igualdad por UID, cronología y etiquetas. Revisar código external WFlib y fijar commit + licencia + representación; no ejecutar `run_estado_del_arte.sh`.
3. **S3 / plan de entrenamiento**: validación estricta Early→Middle y Early+Middle→Late. Selección de arquitecturas/hiperparámetros solo dentro de DEV, no test externo. Definir antes de lanzar: semillas, número de épocas, regla de early stopping, longitud, batch, GPU, tensor channels, número máximo de búsquedas, tiempo y política de comparación entre recursos. Para fidelidad del paper registrar desviaciones de las configuraciones originales.
4. **S4 / experimentos DEV**: guardar modelos y reportes con hash bajo `~/ltd-storage/artifacts/sota/<run_id>` y `~/ltd-storage/results/sota/<run_id>`. No escribir grandes modelos, datasets derivados ni NPZ al repositorio Git; sus manifiestos pequeños sí pueden versionarse cuando corresponda.
5. **S5 / evaluación 09B y 10B**: checkpoints seleccionados y congelados primero; exportar UIDs/probabilidades y comprobar resultados respecto a la población canónica. Comparar formalmente vs **LTD-HYBRID-FINAL**, MICRO-FINAL, MACRO-LTD-FINAL con misma 09A y cohortes pareadas. Reservar cualquier adaptación Proteus para otro track.
6. **S6 / reporte académico**: tablas con métricas, delta, IC, p por familia, coste y heterogeneidad por sitio/fecha; negativo y no concluyente incluidos; trazabilidad SHA/commit/semilla/protocolo.

## 4. Gates que bloquean la batería

- **G1:** S1 fuente/correspondencia/particiones reproducibles, incluyendo procedencia explícita de asignaciones reconstruidas.
- **G2:** exactamente 65 clases en el **mismo orden** que los scores congelados; mismos UID/fecha y definiciones de evaluación.
- **G3:** implementaciones fieles o correctamente identificadas como adaptadas, tests de tensor y aislamiento Future-B.
- **G4:** presupuesto/semillas/endpoint/comparaciones por cohorte congelados.
- **G5:** entorno ML y recursos evaluados para convivir con la captura QA de Dataset C sin saturar el disco/CPU/GPU.
- **G6:** solo después de G1–G5 iniciar entrenamiento DEV. Future-B no se toca hasta congelar checkpoints.

## 5. Higiene del repositorio

`.gitignore` ya excluye `*.pcap`, `*.pth`, `*.pt`, `*.ckpt`, `*.joblib`, `results/`, `data/local/`, `__pycache__` y entornos virtuales. **Auditar** antes de eliminar archivos: no borrar pruebas científicas congeladas, scripts históricos citados, documentación de hallazgos, datos A/B, artefactos C, ni reportes con SHA. Solo se eliminará basura comprobada (logs, cachés, temporales y duplicados sin referencias), mediante rutas precisas y una revisión explícita; nunca `git clean -fdx`, borrados masivos o `docker prune`.

**Situación actual:** S1 requiere superar fuentes y tests; no hay entrenamiento SOTA validado en esta versión.
