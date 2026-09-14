# Pending imports and paths after physical migration

This report is intentionally diagnostic. No script was edited to resolve these items.

## Launchers that now need path updates

| File | Pending reference | New canonical target |
|---|---|---|
| `scripts/preprocess/run_features.sh` | Invokes `EXP1_*`, `EXP2_XGBoost_*`, and `GEN_*` from its own directory | `src/models/macro/`, `src/evaluation/`, and `src/features/` |
| `scripts/train/run_baselines.sh` | Invokes `VP1_Transformers_dir_size.py` and `VP_Evaluar_ConceptDrift_Vectores.py` from its own directory | `src/models/micro/` and `src/evaluation/` |
| `scripts/train/run_estado_del_arte.sh` | Invokes the three `SOTA_*` files from its own directory | `src/models/sota/` |
| `src/models/sota/README.md` | Documents the runner as if it were beside the SOTA models | `scripts/train/run_estado_del_arte.sh` |
| `scripts/evaluate/run_all.sh` | Uses `$HOME/modelos/predicciones/vectores_features_v2` and `concept_drift` | External/Zeus orchestration path; reconcile before changing |

## Active source paths requiring later configuration work

The active files in `src/features/`, `src/models/{macro,hybrid,dml,sota}/`, and `src/evaluation/` still contain historical relative paths such as:

- `../vectores/resultados/`
- `../vectores_features/resultados_analisis/`
- `../vectores_features_v2/resultados*/`
- `../../output/`

These paths are embedded in the original scripts and may resolve only from their former working directories or from Zeus. They were left unchanged to comply with the no-refactor rule.

## Historical paths deliberately untouched

Scripts under `legacy/` retain their original relative paths, including the phased DML sibling imports. `legacy/dml/phased/fase4_entrenamiento.py` imports `fase2_dataloader` and `fase3_arquitectura`, and `fase5_evaluacion.py` imports `fase3_arquitectura`; the files remain together under `legacy/dml/phased/`. They should be tested from their historical execution context before any modernization.

## Import audit result

No broad repository-package imports were detected that required an automatic rewrite. The only non-third-party local imports found are the phased DML sibling imports listed above. No import or runtime path was modified in this migration.
