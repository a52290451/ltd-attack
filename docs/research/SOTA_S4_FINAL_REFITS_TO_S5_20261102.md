# LTD-Attack — cierre S4, entrada a S5 (2026-11-02)

## Evidencia real de entrenamiento refit C (Perseo)

| Modelo | DEV train UIDs | Épocas | Loss inicial/final | Checkpoint SHA256 |
|---|---:|---:|---|---|
| DF-WFlib-pinned | 57.916 | 28 | 1.78681699 → 0.11869485 | `0bb25cb4aa245af4ac545936b39bef73986cb4c034c9c31e853ca527fc3ceab3` |
| VarCNN-WFlib-DT2 | 57.916 | 30 | 1.10608922 → 0.02627860 | `642a944c8d751d5a9e4f81714cd2cc957dab6a1badfb0ed4e23627199ee8a54f` |

Ambos procesos terminaron con `SOTA_S4_REAL_TRAIN=PASS`, lectura de fuente Historical DEV y protocolo SHA256 `f3056b5ab9fd7a7e4abcdf2b59a350f7fd162e60bf941fc8b4b3d18a24dda4a0`. Ruta de resultados: `/data/ltd-attack/sota/training/s4_wflib_v1_seed42/<modelo>/C/`. Los valores de loss anteriores son **entrenamiento, no prueba externa**. Los mensajes de ejecución no prueban, por sí solos, los bytes persistidos del checkpoint: el gate S5 comprueba de nuevo sus SHA.

Selección previa en B: DF época 28, Accuracy DEV_LATE=0.9219101725801265 y Macro-F1=0.9202814193894742; VarCNN época 30, Accuracy=0.9505841998070532 y Macro-F1=0.9489230351808834. La validación de VarCNN fue altamente oscilante (p. ej. Accuracy 6,82 % en época 27, 94,80 % en 28). Conservar curva completa, no hacer reentrenamiento oportunista ni recalibrar hiperparámetros con test externo.

## Estado por hito

- S1–S3: cohortes DEV, canales DT2 y arquitectura pinneada, auditados.
- **S4: entrenamientos B y refit C completados; dos checkpoints definitivos bajo /data.**
- **S5: sin métricas externas nuevas todavía.** Los resultados 09B y Future-B de LTD ya existen, pero Future-B fue observado históricamente; esta nueva batería es **retrospectiva, no ciega**.
- S6: comparación pareada, IC95 por bloques de fecha (2.500) y multiplicidad preespecificada, pendiente.
- Dataset C: QA_ONLY; cron/entorno no se modifica. Migración `~/ltd-storage` a /data sigue en pausa (copia de datos verificada, original intacto).

## Gate inmediato de S5

Archivo versionado `scripts/sota/s5_freeze_and_label_alignment_v1.py`: sin entrenamiento ni inferencia, recalcula SHA256 de **ambos checkpoints C** y comprueba manifiestos, semilla/protocolo y conteos. Lee únicamente `candidate_labels` de los archivos NPZ canónicos 09B y 10B, con mapeos independientes por cohorte para remapear logits/probabilidades sin alterar entrenamiento ni etiquetas. **No usar datos 09B/10B para seleccionar arquitectura, hiperparámetros, época o umbral.** Este paso registra la SHA de ambos NPZ existentes sin modificar las entradas; no calcula resultados de evaluación.

```bash
cd ~/ltd-attack
git fetch origin feature/macro-v2-historical-only &&
git merge --ff-only FETCH_HEAD &&
/data/ltd-sota-bsierra/venv-cu128/bin/python scripts/sota/s5_freeze_and_label_alignment_v1.py
```

**Estado del gate:** script publicado, pendiente de ejecución real en Perseo. Requiere `S5_FREEZE_LABEL_GATE=PASS_NO_INFERENCE` para pasar al siguiente paso (inferencia de los checkpoints congelados en 09B y Future-B, exportación de 65 probabilidades alineadas por UID y métricas pareadas). La comparación será contra el track **DEV_FROZEN** de LTD en Future-B, no HISTORICAL_FINAL.

## Resultado S5 en Perseo y siguiente ejecución (2026-11-03)

El investigador ejecutó el gate S5 sobre los checkpoints definitivos. **Ambos SHA256 fueron verificados** y el orden de las 65 etiquetas es **IDENTICAL** en 09B y 10B DEV_FROZEN; las cohortes además comparten el mismo orden. El comando falló al final exclusivamente porque ya existía el archivo `frozen_s5_refits_and_labels_v1.json` y su código anterior rechazaba cualquier archivo previo, sin examinarlo. No es prueba de corrupción de datos ni de pesos. No se hizo inferencia.

**Corrección puntual:** `scripts/sota/s5_freeze_and_label_alignment_v1.py` ahora, si ya existe la evidencia, compara todos sus campos científicos con el resultado verificado, salvo la marca de tiempo de creación. Si coinciden, informa `S5_FREEZE_EVIDENCE=REUSED_VERIFIED_IDENTICAL` y `S5_FREEZE_LABEL_GATE=PASS_NO_INFERENCE`; si hay cualquier conflicto de pesos, protocolo, clases o fuentes, se detiene sin sobrescribir. Si no existe, lo crea una sola vez.

**Código de inferencia S5 ya publicado:** `scripts/sota/s5_infer_external_frozen_v1.py`, con opciones `--cohort 09B|10B_DEV_FROZEN` y `--model DF|VarCNN`. Cruza UID, `site_label`, etiquetas canónicas y representación `DIR`/`DT2` desde CSV originales con SHA congelado; usa exclusivamente los checkpoints C sin gradientes; exporta probabilidades 65-clases y métricas Accuracy, Macro-F1, Top-5 y MRR bajo `/data/ltd-attack/sota/s5/inference/<cohort>/<model>/`. El script no recalibra ni entrena y bloquea cambios de fuentes.

Primero ejecutar gate S5 corregido, seguido de 09B DF, VarCNN y 10B DF, VarCNN de forma **secuencial**. No interpretar la validación DEV_LATE como resultado de 09B/10B, ni comparar contra HISTORICAL_FINAL en Future-B. Los análisis futuros son retrospectivos por conocimiento anterior de las cohortes. Dataset C intacto.

## 2026-11-03 — S5 idempotencia de inferencia y SSH

El investigador ejecutó S5 con éxito: ambos checkpoints SHA congelados y `09B` / `10B_DEV_FROZEN` candidate_labels IDENTICAL; `S5_FREEZE_EVIDENCE=REUSED_VERIFIED_IDENTICAL`; `S5_FREEZE_LABEL_GATE=PASS_NO_INFERENCE`. El primer comando de inferencia real leyó los 14.687 UID de 09B/DF y abortó porque el directorio de destino ya existía. Ese error no demuestra ningún cambio de pesos o etiquetas; los resultados existentes deben preservarse, verificarse y reutilizarse si están completos. El `exit 1` en un bucle interactivo terminó la sesión SSH; no repetirlo.

Se actualizó `scripts/sota/s5_infer_external_frozen_v1.py` para resolver la existencia de salida **antes de leer de nuevo todos los vectores**: si carpeta inexistente, ejecutar; si carpeta vacía, continuar sin borrar; si contiene exactamente manifiesto y NPZ, validar hashes, checkpoint/protocolo, 09B/10B, UID/fechas/clases y probabilidades, informar `S5_INFERENCE_REUSED_VERIFIED`; si contiene archivos parciales/desconocidos o evidencia divergente, bloquear sin alterar nada. Se preservan resultados históricos y el contenido de `/data`. Nuevos resultados imprimirán `S5_INFERENCE_PASS`. El nuevo código aún necesita ser sincronizado en Perseo.

El bucle de ejecución será secuencial y con `break 2` al error dentro del shell interactivo, **nunca `exit 1`**. Por seguridad, S5 no permite overwrite de archivos ya creados. Esta modificación no reentrena y no accede a Dataset C.

## 2026-11-03 — S5 cuatro inferencias cerradas, S6 implementado y SOTA ampliado

**Perseo confirmó las cuatro evaluaciones externas (reutilización íntegra o inferencia nueva):** 09B DF Accuracy 0.9342275481718526 / Macro-F1 0.9323914139782679; 09B VarCNN 0.9729012051474093 / 0.9721640832020264; 10B DEV_FROZEN DF 0.3746427223210915 / 0.3619218200039363; 10B VarCNN 0.334466 / 0.311403. Cuatro NPZ con UID/fecha/clases/probabilidades bajo /data/ltd-attack/sota/s5/inference. S5 listo.

**Nuevo S6 código:** scripts/sota/s6_compare_frozen_sota_ltd_v1.py. Lee exclusivamente scores LTD 09B y 10B DEV_FROZEN y los cuatro S5 de DF/VarCNN; verifica SHA, UID, fecha, y_true, candidate_labels, checkpoint y protocolo. Reporta 5 modelos sobre las mismas 2 cohortes, Accuracy, Macro-F1, Top-5, MRR, desgloses diarios y por sitio, rescates/perjuicios; estadística pareada de bloques de fecha con 2500 bootstrap y Holm8 exploratorio por cohorte para DF/VarCNN vs Hybrid. Intercambio de bloques exacto para métricas aditivas y Monte Carlo para Macro-F1. No reentrena ni modifica evidencias congeladas. El código y las pruebas sintéticas fueron publicados, **todavía no se ha ejecutado S6 en Perseo**.

**Catálogo SOTA posterior:** docs/sota/SOTA_EXPANSION_ROADMAP_20261103.md. P1 CUMUL/SVM fiel sin features 94F contaminadas y Tik-Tok DT; P2 NetCLR/AWF/TF/RF/Holmes; P3 Proteus y Aion track ADAPTIVE independiente. No dar por terminado SOTA solo porque DF/VarCNN ya funcionan.

**Comando de S6** (no altera ni reentrena S4/S5):

```bash
cd ~/ltd-attack
git fetch origin feature/macro-v2-historical-only &&
git merge --ff-only FETCH_HEAD &&
/data/ltd-sota-bsierra/venv-cu128/bin/python \
  scripts/sota/s6_compare_frozen_sota_ltd_v1.py
```

Si se interrumpe por error no lanzar `exit 1` en la shell SSH. Si S6 reporta un fallo de integridad, corregir la fuente/bug mediante procedimiento controlado y conservar todos los datos; nunca overwrites.
