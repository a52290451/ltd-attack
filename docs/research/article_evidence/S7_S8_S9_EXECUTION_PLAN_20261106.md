# LTD-Attack — siguiente batería SOTA: CUMUL/SVM, Tik-Tok, NetCLR

**Registro 2026-11-06.** Todos los resultados S6 están archivados en docs/research/article_evidence/S6_FROZEN_SOTA_LTD_RESULTS_20261106.md. No editar las predicciones congeladas anteriores. S6 muestra ventaja retrospectiva de LTD-Hybrid frente a DF y VarCNN en Future-B; no es una afirmación de superioridad general sobre todo SOTA.

## S7 — CUMUL-style 104 features + SVM-RBF

**Código implementado**: scripts/sota/s7_cumul_frozen_v1.py; modos --stage B, --stage C, --stage evaluate --cohort 09B|10B_DEV_FROZEN. **Todavía no ejecutado sobre Perseo.**

- Se comprobó en src/data/01_preprocesamiento.py que size_vector proviene de len(pkt) de paquetes IP/TCP seleccionados, con directions +1 saliente / -1 entrante. Se invierte el signo para la convención CUMUL de bytes entrantes positivos; se extraen 4 estadísticas (Nin,Nout,Sin,Sout) y 100 puntos de la curva suma de bytes firmados frente a suma absoluta interpolada. No usa features 94F seleccionadas con conocimiento futuro.
- Precaución: el paper CUMUL original trata tamaños TCP y detalles de normalización/padding propios. Hasta auditar cada transformación frente a la fuente original, reportar **CUMUL-style, 104 features**, nunca equivalencia exacta con implementación paper. scapy len(pkt) incluye cabeceras según capa disponible y no necesariamente equivalente a la representación original.
- Fase B entrena exclusivamente DEV_EARLY+MIDDLE (39.258), valida DEV_LATE (18.658), parámetros SVM-RBF C=1, gamma=scale, scaler solo train, probability=True, seed=42. No retoca parámetros desde DEV_LATE en v1: hay validación, no búsqueda de hiperparámetros. Fase C reentrena sobre las 57.916 capturas DEV usando configuración inmutable. Evaluación externa permitida solo con C congelado.
- SVC probability=True puede consumir muchos recursos y tardar; no ejecutar en paralelo con otros entrenamientos GPU o trabajos que compitan con Dataset C. En cada fase hay bloqueo de overwrite; no borrar carpetas ante error sin inspección.
- Los resultados deben ir a /data/ltd-attack/sota/s7_cumul_svm_rbf_seed42/{B,C,evaluation}/ con SHA de checkpoint, manifiesto y 65 probabilidades alineadas por UID.

## S8 — Tik-Tok: directional timing (DT)

**Implementado**: src/models/sota/v1/tiktok_adapter.py, representación de 5.000 elementos direction_i * timestamp relativo_i, padding derecho con cero, truncado por orden original, sin sumar un offset inventado. Se distingue explícitamente de VarCNN DT2 (dirección + inter-arrival). Datos históricos históricos contienen time_vector, pero el modelo y su entrenamiento aún no se han implementado/fijado.

**Código fuente conocido**: https://github.com/msrocean/Tik_Tok (PETS 2020), legado TF1/Keras. Una implementación PyTorch de Tik-Tok podría estar en el checkout WFlib ya existente; scripts/sota/s8_s9_source_gate_v1.py lista los archivos y blobs exactos del checkout actual antes de seleccionar arquitectura/hiperparámetros. No instalar TensorFlow 1.x ni cambiar la venv activa sin verificar fuente.

## S9 — NetCLR: preentrenamiento contrastivo

**Fuente inspeccionada públicamente**: https://github.com/FIND-Lab/Website-Fingerprinting-Library y https://github.com/jzx-bupt/NetRandAugment. La segunda contiene DF/models/NetCLR.py, CLR_pretrain.py y CLR_closed_world.py; su política de augmentación está modificada y **no representa automáticamente el NetAugment original**. Para reproducir NetCLR original, priorizar el código de WFlib de la familia NetCLR y fijar commit/blobs específicos antes de extraer pares contrastivos y preentrenar.

**Pendiente de implementar**: modelo, transformaciones originales NetAugment, preentrenamiento SimCLR solo sobre Historical DEV (o un subconjunto DEV permitido y registrado), fine-tuning supervisado DEV, checkpoint final, exportación 09B y Future-B. No es válido tomar DF ya entrenado y renombrarlo NetCLR ni introducir trazas Future-B no etiquetadas en pretraining.

## Procedimiento inmediato

```bash
cd ~/ltd-attack
git fetch origin feature/macro-v2-historical-only &&
git merge --ff-only FETCH_HEAD &&
/data/ltd-sota-bsierra/venv-cu128/bin/python -m unittest tests.test_sota_s7_s8_features_v1 &&
/data/ltd-sota-bsierra/venv-cu128/bin/python scripts/sota/s8_s9_source_gate_v1.py &&
/data/ltd-sota-bsierra/venv-cu128/bin/python scripts/sota/s7_cumul_frozen_v1.py --stage B
```

**Estado de la ejecución:** pendiente de Perseo. El comando inicia entrenamiento real B de CUMUL cuando pruebas y fuentes pasan; **no ejecuta C ni evaluación externa**, pues hay que conservar evidencia B y resolver primero cualquier problema de fidelidad/datos. No toca Dataset C, Docker, cron o TimeSync. Al publicar para artículo, reproducir resultados independientes adicionales y contrastes exploratorios ampliados separadamente de Holm8 original.
