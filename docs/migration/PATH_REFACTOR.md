# Fase 3 — Normalización de rutas

Esta fase sustituye referencias relativas históricas del código activo por
resoluciones centralizadas en `src/utils/paths.py`. No se modificó ningún
archivo bajo `legacy/`, ni la lógica científica, los datasets, los splits,
los hiperparámetros o los nombres de resultados.

| original_reference | new_reference | file | reason | status |
|---|---|---|---|---|
| `../../output/<dataset>.csv` | `data_path('historical', '<dataset>.csv')` | `src/data/02_preprocesamiento_d_f_24.py`, `src/data/doc_mapeo_datos.py` | Resolver datasets desde `data_root` | completed |
| `output/preprocessed/<dataset>.csv` | `data_path('generated', 'preprocessed', '<dataset>.csv')` | `src/data/04_feature_aggregation.py`, `src/data/05_inter_day_dynamics_invariante.py`, `src/data/05_inter_day_dynamics_serie.py` | Separar datos generados de resultados | completed |
| `output/graficos_finales_masivos/` | `result_path('diagnostics', 'graficos_finales_masivos')` | `src/data/02_preprocesamiento_d_f_24.py` | Enviar figuras generadas al árbol de resultados | completed |
| `../vectores/resultados/ds3_label_encoder_vec_3000.joblib` | `artifact_path('ds3_label_encoder_vec_3000.joblib')` | `src/features/*`, modelos activos y evaluaciones | Centralizar artefactos/checkpoints | completed |
| `../vectores_features/resultados_analisis/new_features_invariantes_seguras.txt` | `result_path('macro', 'resultados_analisis', 'new_features_invariantes_seguras.txt')` | `src/features/*`, `src/models/*`, `src/evaluation/*` | Mantener la lista de features en resultados configurables | completed |
| `../../output/CLEAN_final_features_sites.csv` | `data_path('historical', 'CLEAN_final_features_sites.csv')` | `src/features/*`, macro, hybrid, DML, SOTA | Resolver el dataset sin depender del cwd | completed |
| `../../output/CLEAN_final_features_sites_concept_drift.csv` | `data_path('historical', 'CLEAN_final_features_sites_concept_drift.csv')` | `src/features/*`, macro, SOTA y evaluación | Resolver el dataset de drift sin rutas relativas | completed |
| `../../output/CLEAN_final_vectors_sites*.csv` | `data_path('historical', 'CLEAN_final_vectors_sites*.csv')` | micro, SOTA, diagnóstico y evaluación | Resolver vectores desde `data_root` | completed |
| `../../output/cached_vectors_with_dates.csv` | `data_path('historical', 'cached_vectors_with_dates.csv')` | `src/models/hybrid/*`, `src/models/dml/*` | Preservar el nombre del cache con ruta configurable | completed |
| `./resultados` | `result_path('<family>')` | micro, macro y SOTA | Separar salidas por línea de investigación | completed |
| `./resultados_94F3000V*` | `result_path('hybrid', 'resultados_94F3000V*')` | `src/models/hybrid/*`, `src/evaluation/*` | Mantener contexto y nombres de resultados | completed |
| `./resultados_DML*` | `result_path('dml', 'resultados_DML*')` | `src/models/dml/*`, `src/evaluation/*` | Centralizar resultados DML | completed |
| `./resultados_EXP7_Transfer_DML` | `result_path('dml', 'resultados_EXP7_Transfer_DML')` | `src/models/dml/EXP7_Train_Transfer_DML.py`, evaluación EXP7 | Compartir la salida del experimento entre train/eval | completed |
| `./graficas_tesis` | `result_path('features', 'graficas_tesis')` | `src/features/*` | Evitar salidas dependientes del cwd | completed |
| `$HOME/modelos/predicciones/...` | `PROJECT_ROOT` + invocación `python -m` | `scripts/evaluate/run_all.sh` | Ejecutar la batería desde cualquier cwd sin inventar almacenamiento | completed |
| invocación de scripts por nombre local | `python -m src.<area>.<modulo>` desde `PROJECT_ROOT` | `scripts/preprocess/run_features.sh`, `scripts/train/*`, `scripts/evaluate/run_all.sh` | Hacer los launchers independientes del cwd | completed |

## Convenciones

- `LTD_ENV=local` usa `configs/environments/local.yaml`.
- `LTD_ENV=zeus` usa las rutas Zeus conocidas de `configs/environments/zeus.yaml`.
- Las funciones de `paths.py` solo resuelven y, si se solicita, validan; no
  crean datasets ni artefactos faltantes.
- Los checkpoints y datasets locales siguen siendo externos/no versionados
  según las reglas existentes del repositorio.

La comprobación final de referencias prohibidas se realiza sobre `src/` y
`scripts/`, excluyendo documentación y `legacy/`.
