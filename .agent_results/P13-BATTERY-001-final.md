# P13-BATTERY-001 — Reporte final

## Estado

La batería P13-B1 quedó implementada y lista para ejecución posterior en
Zeus. No se ejecutaron entrenamientos pesados en local. No se hizo commit,
push, amend ni force.

Rama verificada: `feature/macro-v2-historical-only`  
Commit base verificado: `a8ca9a9 evidence: archive frozen temporal Micro ensemble 13A`

## Archivos creados

- `configs/experiments/PHASE13_BATTERY_V1.yaml`
- `configs/experiments/FUTURE_RANK_GAP_V1.yaml`
- `src/experiments/__init__.py`
- `src/experiments/phase13_battery.py`
- `src/analysis/__init__.py`
- `src/analysis/future_rank_gap_battery.py`
- `scripts/train/run_phase13_battery.py`
- `scripts/diagnostics/run_future_rank_gap_battery.py`
- `tests/test_phase13_battery.py`
- `tests/test_future_rank_gap_battery.py`
- `P13_B1_CHECKPOINT_REPORT.md`
- `docs/superpowers/specs/2026-10-04-P13-BATTERY-001-design.md`
- `docs/superpowers/plans/2026-10-04-P13-BATTERY-001.md`

## Archivos modificados

- `docs/EXPERIMENT_MAP.md`
- `docs/EXPERIMENT_MAP_EXEC.md`
- `docs/RESULTS_HISTORY.md`
- `docs/FEATURE_LINEAGE.md`
- `docs/catalog/RESEARCH_BRANCHES.csv`

La documentación histórica solo marca 13A como evidencia archivada, gate de
promoción pasado y Future-B no usado. También conserva Accuracy FAR:
ORIGIN14 72.79% → 75.97% (+3.17 pp), ORIGIN28 78.97% → 87.76% (+8.80 pp),
media aproximada +5.99 pp. No se escribieron resultados nuevos de 13B, 13C
ni 14A.

## Decisiones de diseño

- El loader reutiliza las implementaciones congeladas de 08B/13A/11H/11B/11L/
  12A mediante carga dinámica; no copia algoritmos históricos.
- `--dry-run` no carga entrenadores ni abre datasets.
- El cache exige coincidencia de hashes de configuración, código e inputs.
- Las matrices de predicción se guardan como `.npz` comprimido por
  `origin/temporal_mode/seed/window`.
- Future-B se procesa únicamente desde
  `10B_scores_DEV_FROZEN.npz`; si falta, 14A falla de forma clara y no llama
  a 10B.
- `fraction_delta_gt_0` se conserva como fracción bootstrap y nunca se llama
  p-value.
- La correlación Spearman descriptiva se calcula con rangos de pandas, sin
  añadir una dependencia nueva de SciPy.

## Matriz exacta

13B contiene:

- origins: ORIGIN14 y ORIGIN28;
- ventanas: NEAR, MID y FAR;
- modos: EARLY, UNIFORM y LATE;
- seeds: 11, 42 y 73;
- 18 entrenamientos Micro predeclarados;
- U1, U3, T3_S11, T3_S42, T3_S73 y T9.

13C contiene estos ocho productos fijos:

`U1xM0`, `U1xM1`, `U3xM0`, `U3xM1`, `T3_S42xM0`, `T3_S42xM1`, `T9xM0`,
`T9xM1`.

M0 es UNIFORM XGB + RECENT5 LTD; M1 es TEMPORAL_SYMMETRIC3 XGB +
MULTISCALE5 LTD. La fusión permanece Micro=0.45 y Macro=0.55.

14A analiza MICRO, MACRO_XGB, MACRO_LTD, MACRO_FINAL y HYBRID_FINAL.

## Outputs esperados

13B escribe en:
`result_path("micro", "MICRO-ROBUSTNESS-PHASE13-BATTERY")`:

- `13B_model_summary.csv`
- `13B_pairwise_deltas.csv`
- `13B_day_block_bootstrap.csv`
- `13B_expert_diversity.csv`
- `13B_training_summary.csv`
- `13B_rank_gap_historical.csv`
- `13B_manifest.json`

13C escribe en:
`result_path("hybrid", "LTD-ROBUSTNESS-PHASE13-BATTERY")`:

- `13C_factorial_summary.csv`
- `13C_factorial_effects.csv`
- `13C_pairwise_deltas.csv`
- `13C_day_block_bootstrap.csv`
- `13C_rank_gap_historical.csv`
- `13C_manifest.json`

14A escribe en:
`result_path("final", "FUTURE-RANK-GAP-PHASE14")`:

- `14A_rank_distribution.csv`
- `14A_per_site_rank_gap.csv`
- `14A_capture_groups.csv`
- `14A_sequence_size_summary.csv`
- `14A_per_site_size_metrics.csv`
- `14A_per_site_drift_metrics.csv`
- `14A_drift_similarity_pairs.csv`
- `14A_confusion_edges.csv`
- `14A_component_rescue.csv`
- `14A_correlations.csv`
- `14A_manifest.json`

## Cache y resume

Cada experto Micro usa el identificador inequívoco
`ORIGIN/MODE/SEED`, por ejemplo `ORIGIN28/LATE/73`. Se guardan checkpoint
PyTorch, min/max del scaler, metadata, curva y predicciones por ventana.
`--resume` reutiliza solo artefactos que tienen manifest compatible y
checkpoint presente; un hash incompatible produce error explícito.

El cache Macro de 13C se guarda por origin con manifest propio y hashes de
11L/12A, y se reutiliza sin duplicar entrenamientos cuando es compatible.

## Comandos exactos para Zeus

Desde la raíz del repositorio:

```bash
export LTD_ENV=zeus
export PYTHONPATH=.

python3 scripts/train/run_phase13_battery.py \
  --config configs/experiments/PHASE13_BATTERY_V1.yaml \
  --dry-run --only 13B

python3 scripts/train/run_phase13_battery.py \
  --config configs/experiments/PHASE13_BATTERY_V1.yaml \
  --resume --only 13B

python3 scripts/train/run_phase13_battery.py \
  --config configs/experiments/PHASE13_BATTERY_V1.yaml \
  --resume --only 13C

python3 scripts/diagnostics/run_future_rank_gap_battery.py \
  --config configs/experiments/FUTURE_RANK_GAP_V1.yaml \
  --dry-run

python3 scripts/diagnostics/run_future_rank_gap_battery.py \
  --config configs/experiments/FUTURE_RANK_GAP_V1.yaml
```

También se puede ejecutar la batería Historical completa con una sola orden:

```bash
LTD_ENV=zeus PYTHONPATH=. python3 scripts/train/run_phase13_battery.py \
  --config configs/experiments/PHASE13_BATTERY_V1.yaml --resume
```

## Estimación de entrenamientos

- Micro ORIGIN14: 9 entrenamientos; EARLY/UNIFORM/LATE × 11/42/73.
- Micro ORIGIN28: 9 entrenamientos; EARLY/UNIFORM/LATE × 11/42/73.
- Total Micro: 18.
- 13C reutiliza los Micro cacheados y ejecuta/reutiliza los componentes
  Macro 11H/11B/11L necesarios para M0/M1; no crea nuevas variantes Micro.
- 14A: 0 entrenamientos.

## Tests y validación ejecutados

Pasaron:

- `python3 -m compileall -q src/experiments src/analysis scripts/train/run_phase13_battery.py scripts/diagnostics/run_future_rank_gap_battery.py tests/test_phase13_battery.py tests/test_future_rank_gap_battery.py`
- `python3 -m pytest tests/test_phase13_battery.py tests/test_future_rank_gap_battery.py tests/test_paths.py -q` → `14 passed`.
- `python3 scripts/train/run_phase13_battery.py --dry-run --only 13B` → código 0; mostró 18 entrenamientos, derivados, factorial y rutas.
- `python3 scripts/diagnostics/run_future_rank_gap_battery.py --dry-run` → código 0; mostró NPZ, componentes, outputs y flags post-hoc.
- `git diff --check` → sin errores.

La suite completa `python3 -m pytest -q` no pudo recolectarse porque pytest
interpreta el script histórico `src/models/final/09B_evaluate_internal_test.py`
como módulo de tests y el entorno local no tiene `joblib`. No se instaló una
dependencia ni se modificó ese archivo; se ejecutó el alcance nuevo más
`tests/test_paths.py`.

## Riesgos y limitaciones

- La ejecución real depende de que Zeus tenga los datasets, label encoder y
  dependencias históricas de 08B/10A/11L disponibles.
- 13C falla explícitamente si no puede alinear las probabilidades Macro con
  las capturas Micro; no intenta corregir la alineación silenciosamente.
- 14A requiere el NPZ congelado y las fuentes Micro/Macro para tamaño y drift;
  su ausencia detiene el diagnóstico sin reabrir Future-B.
- ORIGIN14/ORIGIN28 siguen siendo desarrollo repetido, no validación
  independiente.
- El código no promueve automáticamente ningún Hybrid y el checkpoint report
  permanece sin resultados hasta Zeus.

## Incompatibilidades detectadas

- Los comandos `pytest` y `python` no existen como ejecutables directos en el
  entorno local; se usó `python3 -m pytest`.
- La suite completa tropieza con la dependencia histórica ausente `joblib`.
- Los wrappers directos necesitaban insertar la raíz del repositorio en
  `sys.path`; quedó corregido sin hardcodear rutas de máquinas.
- Los módulos históricos Macro importan `_common` localmente; el loader ahora
  inserta de forma controlada la carpeta del módulo.

## Diff y estado Git

Salida de `git diff --stat` para archivos tracked modificados:

```text
 docs/EXPERIMENT_MAP.md             | 74 +++++++++++++++++++++-----------------
 docs/EXPERIMENT_MAP_EXEC.md        | 18 +++++-----
 docs/FEATURE_LINEAGE.md            | 26 ++++++++++++++
 docs/RESULTS_HISTORY.md            | 29 +++++++++++++++
 docs/catalog/RESEARCH_BRANCHES.csv |  2 +-
 5 files changed, 107 insertions(+), 42 deletions(-)
```

El estado Git final debe conservar los archivos nuevos listados arriba y las
cinco modificaciones documentales; no se creó ningún commit.
