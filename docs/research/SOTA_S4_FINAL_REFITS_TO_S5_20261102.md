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
