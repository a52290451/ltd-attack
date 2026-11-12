# Protocolo de auditoría integral SOTA de LTD (12-11-2026)

## Estado científico
La batería original tiene **30/30 etapas finalizadas y verificadas**, pero esto no certifica que reproduzca exactamente los siete métodos publicados. El resultado BYOL de Swallow colapsó antes de validación histórica. WFCAT cambia fuertemente entre 09B y Future-B; la auditoría aún no atribuye causalidad ni valida todas las semánticas de timestamps.

La batería original, checkpoints, datos congelados, puntuaciones y manifiestos **no se modifican ni se sobrescriben**. Cualquier corrección que cambie entrenamiento/representaciones debe ejecutarse en un **experimento nuevo** identificado y compararse primero en Historical DEV sin consultar Future-B para seleccionar hiperparámetros.

## Flujo único

En Zeus, con la GPU ya libre y tras verificar que no quedan procesos de entrenamiento:

```bash
cd ~/ltd-attack || exit 1
set -euo pipefail
BASE="/data/datasets_brayan/ltd-sota-2026"
source "$BASE/host_manifests/host_environment.sh"
export LTD_SOTA_PYTORCH_PY="$HOME/modelos/venv/bin/python"
git fetch origin feature/macro-v2-historical-only
git merge --ff-only origin/feature/macro-v2-historical-only
"$LTD_SOTA_PYTORCH_PY" -m py_compile scripts/sota/*audit*py scripts/sota/run_full_method_certification_suite_v1.py
CUDA_VISIBLE_DEVICES=-1 nice -n 10 "$LTD_SOTA_PYTORCH_PY" \
  scripts/sota/run_full_method_certification_suite_v1.py \
  --repo "$HOME/ltd-attack" --base "$BASE" --audit-root "$BASE/audits"
```

La suite genera un directorio aislado con `certificate.json`, `forensic.json`, `chain.json`, `swallow_cause.json`, `swallow_trajectory.json`, y `seven_date_bootstrap.json`. No modifica `artifacts/sota`.

La suite conserva el resultado de cada control aunque otro falle; el certificado usa `PASS`, `FAIL`, `PENDING` y puede indicar **NOT CERTIFIED**. El certificado nunca declara una reproducción paper-exact basándose solo en hashes.

Si ya existen archivos NPZ de predicciones propias con esquema canónico `pcap_uid,query_date,y_true,candidate_labels,sota_probs`, añadir `--own-09b /ruta/09b.npz --own-future-b /ruta/future_b.npz`. No generar ni inferir un orden de clases a partir de una matriz sin metadatos.

## Criterios indispensables de validación

1. **B/C/selección**: 39.258 registros en B, validación DEV_LATE, 57.916 en C; época de C igual a la elegida en B; sin alterar checkpoints; SHA verificado.
2. **Cohortes**: 14.687 UID de 09B, 18.543 de Future-B, 65 etiquetas, mismas filas y fechas en los siete modelos, sin intersecciones entre tres poblaciones.
3. **Métricas**: accuracy independiente, macro-F1, Top-5, MRR; bootstrap pareado agrupado por fecha con 2.500 réplicas e intervalos percentiles, 11 fechas de 09B y 16 de Future-B.
4. **Fuentes**: HEAD de repositorios de autores, tracked tree limpio, hashes de módulos utilizados, transformaciones CIF/IAT con pruebas numéricas e historial de cambios. El pin de origen no certifica equivalencia metodológica.
5. **Swallow**: aleatorio vs 20/40/60/80/100 checkpoints BYOL, dispersión de las representaciones, calidad y variabilidad de vistas de augmentation, pérdida BYOL, B/C. La ausencia de diversidad entrenada invalida inferir una limitación intrínseca del método original.
6. **WFCAT**: unidades de time_vector, histogramas IAT, longitud/duración de capturas, orden temporal, colas y saturación. Las diferencias descriptivas no prueban concept drift causal.
7. **Paridad paper-exact**: tabla fuente/adapter por modalidad, arquitectura, init, loss, optimizer, epochs, batch, augmentation, feature timing/normalization y selección. **Sigue pendiente** una validación de equivalencia de código y configuración contra los scripts originales de cada paper; al menos Swallow y VarCNN direction-only tienen desviaciones materiales.
8. **Comparación con nuestros modelos**: requiere puntajes propios sobre las **mismas** capturas con UID, clases, fechas y probabilidades canónicas. Accuracy comparada entre datasets diferentes no es una comparación emparejada válida.

## Veredictos de la suite

- `AUDIT_FAILURES_REQUIRE_CORRECTION`: un control técnico falló; revisar `certificate.json`, corregir el auditor o crear una nueva ejecución del modelo si hay defecto de la implementación.
- `AUDIT_COMPLETED_METHOD_FIDELITY_AND_OWN_MODEL_PARITY_PENDING`: todos los controles ejecutados pasaron, pero no se incluyeron todavía las predicciones propias o paridad metodológica original.
- `AUDIT_TECHNICAL_GATES_PASS_METHODS_NOT_YET_CERTIFIED`: también pasó la alineación de puntajes propios; la fidelidad original todavía requiere evidencia externa específica.

**Ninguno de estos estados equivale por sí solo a CERTIFIED.**

## Evidencia del artículo

Para revisión por pares, archivar junto al manuscrito: Git commit de la suite, código y commit original de cada upstream, SHA de datos/manifiestos/checkpoints/NPZ, código concreto de conversión de features, pérdidas y configuraciones efectivas, reporte de paridad original, criterios cronológicos predefinidos, resultados de auditoría, IC pareados y decisiones sobre experimentos nuevos. Reportar `VarCNN direction branch` como tal, no el ensemble completo.

## Auditorías incorporadas a la suite

- `audit_pinned_upstream_training_parity_v1.py`: diferencias estáticas entre el código fuente original de Swallow y WFCAT y los adaptadores de entrenamiento (incluidos modo objetivo BYOL, épocas, batch, multiplicador de vistas). Confirma desviaciones, **no** su causalidad.
- `audit_swallow_pretrain_trajectory_v1.py`: representa checkpoints originales inmutables de épocas 20/40/60/80/100 y mide diversidad de vistas de augmentación.
- `audit_full_external_temporal_population_v1.py`: recorre todas las capturas congeladas de 09B/Future-B, comprueba SHA/UID/clases, distribución global, por fecha y por clase de paquetes, duración y retrocesos de tiempo.
- `audit_all_seven_paired_date_bootstrap_v1.py`: comprobación de poblaciones iguales entre los siete y diferencias pareadas de accuracy (2.500 réplicas por fecha).
- `audit_own_model_alignment_v1.py`: puerta opcional para importar nuestros propios scores NPZ sin inferir el orden de etiquetas.

La suite produce resultados forenses aun cuando no se adjunten scores propios. En ese caso reporta PENDING para la comparación y no emite certificado científico final.
