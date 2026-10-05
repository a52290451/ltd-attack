# P13-BATTERY-004 — Recovery después de 18/18 Micro trainings

## Estado

La implementación local está preparada para validar y reutilizar los 18
caches legacy de Perseo sin repetir entrenamiento. No se hizo commit ni push.

**LOCAL IMPLEMENTATION/TESTS = realizados**  
**REAL PERSEO CACHE VALIDATION = pendiente**

El checkout fue sincronizado antes de modificar código:

- rama: `feature/macro-v2-historical-only`
- HEAD: `7cbf0627334ee6c1be59a56199a5f2abc9bbd62d`
- los archivos untracked `docs/superpowers/plans/2026-10-04-P13-BATTERY-001.md`
  y `docs/superpowers/specs/2026-10-04-P13-BATTERY-001-design.md` fueron preservados.

## Causa raíz

`validate_expert_prediction_alignment` comparaba `origin`, `window`,
`temporal_mode` y `seed` contra el primer experto. Eso hacía fallar una
colección válida al llegar al experto con otra seed o modo temporal.

Para un mismo `origin/window`, `temporal_mode` y `seed` son dimensiones del
ensemble. Los nueve expertos esperados son exactamente:

`EARLY/11`, `EARLY/42`, `EARLY/73`, `UNIFORM/11`, `UNIFORM/42`,
`UNIFORM/73`, `LATE/11`, `LATE/42`, `LATE/73`.

## Cambio exacto

`validate_expert_prediction_alignment` ahora:

- exige igualdad de `origin` y `window`;
- exige igualdad posicional exacta de `pcap_uid`, `query_date`, `y_true` y
  `candidate_labels`;
- exige la misma shape de probabilidades, filas y número de clases;
- permite modos/seeds válidos diferentes;
- exige una sola vez cada identidad del producto `BATTERY_MODES × MICRO_SEEDS`;
- rechaza mode/seed inesperados, duplicados y expertos faltantes;
- no reordena capturas.

Se añadió `CacheState.COMPLETE_COMPATIBLE_LEGACY` y una validación explícita
que solo acepta el contrato histórico:

- training commit: `7cbf0627334ee6c1be59a56199a5f2abc9bbd62d`
- training fingerprint: `345f49ca1dfeddf93e8fa2ee4bf2b43a8f6990b10b9ae9f045586cd38535eeb8`
- code fingerprint: `7beffe2de894fb185bd334fca1909e809a23bbb3f43247da29ac87c31d8633a7`
- config SHA256: `f12bb55d513df8a81344a950a5d8f38e21a7085b03736a3215d05c4e4d8887c6`
- input fingerprint: `89f2b1fa344a1cb0e543583e36ee4ce437fb27e2096986078989bb3d5025f9cd`

La validación legacy comprueba identidad, arquitectura, SHA256 real de
`model.pt`, `scaler.json` y `NEAR/MID/FAR.npz`, además de metadata interna y,
cuando los datasets están disponibles, los arrays esperados de cada ventana.
Un cache parcial, ilegible, corrupto o con provenance distinta queda como
`COMPLETE_INCOMPATIBLE`. No hay reparación, overwrite ni reescritura de
checkpoints/NPZ históricos.

El manifest 13B generado después del postprocesamiento registrará:

- `training_cache_source_commit`
- `training_cache_source_fingerprint`
- `training_cache_reuse: VERIFIED_LEGACY_CACHE`
- `postprocessing_git_commit`

Por tanto separa el commit que produjo el entrenamiento del commit que
procesa ensembles, métricas y bootstrap.

El launcher exporta `PYTHONUNBUFFERED=1` antes de ejecutar Python.

## Evidencia de preservación de caches

El workspace local no contiene los caches reales de Perseo. La auditoría
local solo creó fixtures sintéticos bajo directorios temporales de pytest;
no se escribieron `artifacts/`, resultados 13B/13C ni manifests históricos.
No se ejecutó ningún entrenamiento Micro, Macro, 13B, 13C ni 14A.

## Tests

Comando ejecutado con `python3` porque el entorno no expone el alias
`python`:

```text
python3 -m pytest -q tests/test_phase13_battery.py tests/test_future_rank_gap_battery.py tests/test_paths.py
39 passed in 2.62s
```

Los tests cubren los nueve expertos válidos, seeds/modes distintos,
duplicado, faltante, mode/seed inesperados, shape distinta y permutaciones o
diferencias de UID, fechas, `y_true` y labels. Los fixtures legacy cubren
aceptación exacta, commit/config/input provenance distintos y corrupción de
checkpoint o NPZ.

También pasaron:

```text
bash -n scripts/run_p13_b1_background.sh
git diff --check
```

## Dry-run y entrenamientos

No se ejecutó el dry-run real contra Perseo porque los 18 caches no están en
este workspace. En Perseo, con los caches presentes y verificables, el estado
esperado para cada experto es `COMPLETE_COMPATIBLE_LEGACY — reutilizable bajo
--resume`, con `Entrenamientos Micro requeridos: 0`.

La confirmación operativa de esos 18 estados y del `0` real queda pendiente
de la ejecución posterior del dry-run en Perseo. La implementación no tiene
ruta de entrenamiento para un cache que haya sido aceptado como legacy.

## Diff y estado Git

`git diff --stat`:

```text
 scripts/run_p13_b1_background.sh   |   1 +
 src/experiments/phase13_battery.py | 261 +++++++++++++++++++++++++++++++------
 tests/test_phase13_battery.py      | 190 +++++++++++++++++++++++++--
 3 files changed, 399 insertions(+), 53 deletions(-)
```

`git status --short`:

```text
 M scripts/run_p13_b1_background.sh
 M src/experiments/phase13_battery.py
 M tests/test_phase13_battery.py
?? .agent_results/P13-BATTERY-004-recovery.md
?? docs/superpowers/plans/2026-10-04-P13-BATTERY-001.md
?? docs/superpowers/specs/2026-10-04-P13-BATTERY-001-design.md
```

READY FOR PERSEO LEGACY-CACHE VALIDATION
