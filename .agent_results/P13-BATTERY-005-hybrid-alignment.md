# P13-BATTERY-005 — Hybrid date alignment + pickle-free Macro cache

## Estado

**LOCAL FIX/TESTS = completed**  
**REAL PERSEO 13C RECOVERY = pending**

El workspace estaba en la rama `feature/macro-v2-historical-only` y en el
HEAD esperado:

`879fad4f21850de9b5758c29bdd5af7115ff270a`

Los archivos untracked existentes bajo `docs/superpowers/` fueron preservados.
No se hizo commit ni push.

## Causa raíz del `query_date` mismatch

`_array_equal()` convertía arrays a strings. Por eso un `datetime64[D]`
(`2025-12-02`) y un `datetime64[ns]`
(`2025-12-02T00:00:00.000000000`) no se trataban como la misma fecha de
captura, aunque representan el mismo día.

Se añadió `_dates_equal()` con canonicalización segura a `datetime64[D]`:

- preserva shape y orden posicional;
- rechaza `NaT` y fechas no convertibles;
- compara día por día sin ordenar ni reindexar.

Se usa exclusivamente para `query_date` en la alineación Micro y
Micro/Macro. `pcap_uid`, `y_true`, `candidate_labels`, `origin` y `window`
mantienen comparación estricta.

## Macro cache pickle-free

El writer Macro ahora normaliza antes de `np.savez_compressed`:

- probabilidades M0/M1: `float32`;
- `y`: `int64`;
- fechas: `datetime64[D]`;
- `pcap_uid` y `candidate_labels`: strings NumPy Unicode, nunca `object`.

Después de escribir, el NPZ se vuelve a abrir con `allow_pickle=False` y se
valida íntegramente.

El loader Macro exige fingerprint, identity, SHA256 real del NPZ, keys
completas, ausencia de `dtype=object`, shapes compatibles, filas alineadas,
labels válidas y fechas canonicalizables. No se creó ningún estado legacy
para Macro; el cache fallido de ORIGIN14 no se reutiliza y deberá archivarse
manualmente en Perseo antes de reconstruir ORIGIN14 y ORIGIN28.

## Resume directo desde 13C

`scripts/run_p13_b1_background.sh` acepta ahora `--from-13c` y
`--skip-14a` en cualquier orden.

Con `--from-13c` ejecuta únicamente:

```text
python3 scripts/train/run_phase13_battery.py \
  --config configs/experiments/PHASE13_BATTERY_V1.yaml \
  --only 13C \
  --resume
```

La ruta usa `load_13b_result()` para reconstruir `candidate_store` desde el
manifest 13B y los 18 caches Micro legacy verificados. No recalcula ni
modifica 13B. La validación final de outputs y el carril opcional 14A se
conservan.

## Preservación científica

- No se modificaron resultados 13B.
- No se modificaron los 18 caches Micro.
- No se cambiaron candidatos, métricas, bootstrap, gates ni pesos Hybrid.
- No se ejecutó Micro, Macro real, 13B real, 13C real ni 14A real.
- El dry-run local no entrenó: mostró caches `ABSENT` porque este workspace
  no contiene los artefactos de Perseo. Esa salida no representa el estado
  real de los caches en Perseo.

## Tests y verificación

Comando principal:

```text
python3 -m pytest -q tests/test_phase13_battery.py tests/test_future_rank_gap_battery.py tests/test_paths.py
52 passed in 2.96s
```

Los tests incluyen fechas `D`/`ns` y strings, `NaT` vía validación de fechas,
fechas distintas/permutadas, guardrails Hybrid, escritura/carga completa
pickle-free, dtypes, corrupción, SHA mismatch y parsing del launcher.

También pasaron:

```text
bash -n scripts/run_p13_b1_background.sh
git diff --check
```

Dry-run local ejecutado:

```text
LTD_ENV=local PYTHONPATH=. python3 scripts/train/run_phase13_battery.py \
  --config configs/experiments/PHASE13_BATTERY_V1.yaml --dry-run
```

Resultado relevante: `Estados Macro: ORIGIN14=ABSENT, ORIGIN28=ABSENT` y
`Entrenamientos Micro requeridos: 18`, debido exclusivamente a la ausencia
local de caches. La validación real y la recuperación 13C de Perseo siguen
pendientes.

## Diff y estado Git

`git diff --stat`:

```text
 scripts/run_p13_b1_background.sh   |  33 +++++++---
 src/experiments/phase13_battery.py | 124 +++++++++++++++++++++++++++++++++---
 tests/test_phase13_battery.py      | 135 +++++++++++++++++++++++++++++++++++--
 3 files changed, 269 insertions(+), 23 deletions(-)
```

`git status --short`:

```text
 M scripts/run_p13_b1_background.sh
 M src/experiments/phase13_battery.py
 M tests/test_phase13_battery.py
?? .agent_results/P13-BATTERY-005-hybrid-alignment.md
?? docs/superpowers/plans/2026-10-04-P13-BATTERY-001.md
?? docs/superpowers/specs/2026-10-04-P13-BATTERY-001-design.md
```

READY FOR PERSEO 13C-ONLY RECOVERY
