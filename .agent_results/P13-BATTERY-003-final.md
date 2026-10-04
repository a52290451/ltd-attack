# P13-BATTERY-003 — Corrección de BLOCKER/HIGH y hardening pre-Zeus

Fecha: 2026-10-04  
Repositorio: ~/dev/ltd-attack  
Rama: feature/macro-v2-historical-only  
Commit base: a8ca9a9 evidence: archive frozen temporal Micro ensemble 13A

## Resultado

Se implementaron las correcciones de P13-BATTERY-002 sin ejecutar
entrenamientos pesados, sin abrir Dataset C, sin modificar evidencia histórica,
sin commit y sin push.

La batería queda preparada para un preflight de Zeus. La decisión conservadora
se mantiene: los artefactos archivados de 13A no contienen checkpoints ni NPZ
reutilizables, por lo que P13-B1 ejecutará 18 entrenamientos Micro nuevos.

## Resolución de BLOCKER/HIGH originales

### B-01 — BLOCKER — alineación exacta de expertos 13B

Resuelto en src/experiments/phase13_battery.py.

Cada prediction NPZ guarda:

- pcap_uid;
- query_date;
- y_true;
- origin;
- window;
- temporal_mode;
- seed;
- candidate_labels;
- probs;
- fingerprints de checkpoint, scaler, configuración, inputs y código.

Antes de U1/U3/T3/T9 se valida igualdad exacta de shape, pcap_uid en orden,
fechas, etiquetas, candidate_labels y origin/window/mode/seed. No existe
reordenamiento correctivo silencioso.

### B-02 — HIGH — alineación Micro/Macro 13C

Resuelto.

validate_hybrid_alignment exige igualdad exacta de pcap_uid, query_date, y_true,
candidate_labels, origin y window antes de la fusión Hybrid. Un intercambio de
dos pcap_uid con las mismas etiquetas produce RuntimeError.

### C-01 — HIGH — caché parcial

Resuelto con CacheState:

- ABSENT;
- COMPLETE_COMPATIBLE;
- COMPLETE_INCOMPATIBLE;
- PARTIAL.

Los estados PARTIAL y COMPLETE_INCOMPATIBLE fallan explícitamente y nunca
sobrescriben ni reconstruyen silenciosamente. Se aplica a expertos Micro,
NPZ, caché Macro y outputs factoriales 13C.

### C-02 — HIGH — hashes y provenance

Resuelto.

13B hashea configuración, datos Micro/Historical y todos los módulos
científicos relevantes, incluyendo 08B, 13A, 10A, paths, common, 07A, 11B,
11F, 11H, 11L, 12A y el propio runner.

13C añade el manifest 13B, fuentes Historical y la misma cobertura científica.
Macro queda registrado como reconstrucción determinista bajo implementación y
configuración congeladas.

14A hashea configuración, NPZ Future-B, fuentes Micro/Macro Historical/Future,
paths, phase13 helpers, selector BASE-128 07A y el esquema de 10B.

Todos los manifests registran git_commit, config_sha256, inputs_sha256,
code_sha256 y outputs_sha256 cuando corresponde.

### C-03 — HIGH — checkpoint y scaler

Resuelto.

Cada experto guarda model.pt y scaler.json con metadata de origin, modo, seed,
arquitectura, epochs, max_len, d_model, heads, layers, ratio temporal,
configuración, inputs y código. El manifest registra SHA256 de ambos y los NPZ
quedan ligados a ellos.

--resume valida hashes, metadata, arquitectura, fingerprint, scaler y cada
prediction NPZ. Un checkpoint corrupto, intercambiado o incompatible produce
RuntimeError.

### C-04 — HIGH — YAML como fuente validada

Resuelto.

PHASE13_BATTERY_V1.yaml controla la matriz, modos, seeds, epochs, ratio,
definiciones U1/U3/T3/T9, candidatos 13C, pesos, métricas, bootstrap,
guardrails y rutas lógicas.

FUTURE_RANK_GAP_V1.yaml controla componentes, MAX_LEN, métricas, reglas,
guardrails y rutas lógicas.

Ambas configuraciones se parsean mediante esquema explícito y se validan
contra los guardrails científicos congelados. Cambiar origins, windows, seeds,
pesos, ratio, componentes o rutas requeridas produce error antes de ejecutar.

## Decisión 13A

No se reutilizan checkpoints ni CSV agregados de 13A como predicciones. La
evidencia histórica queda intacta.

Matriz nueva exacta:

- ORIGIN14: EARLY/UNIFORM/LATE × 11/42/73;
- ORIGIN28: EARLY/UNIFORM/LATE × 11/42/73.

Total: 18 entrenamientos Micro nuevos. Los seis expertos seed 42 de 13A solo
habrían reducido el trabajo a 12 si existieran artefactos criptográficamente
verificables; no existen.

## Macro 13C

M0 continúa siendo UNIFORM XGB + RECENT5 LTD.

M1 continúa siendo TEMPORAL_SYMMETRIC3 XGB + MULTISCALE5 LTD.

Si no existe caché verificable, se reconstruyen mediante las implementaciones
históricas congeladas. Los CSV de 11L no se tratan como predicciones. El
caché Macro contiene probabilidades, pcap_uid, fechas, etiquetas, identidad,
inputs, código, configuración y hashes.

## Outputs protegidos

13B:

- 13B_model_summary.csv
- 13B_pairwise_deltas.csv
- 13B_day_block_bootstrap.csv
- 13B_expert_diversity.csv
- 13B_training_summary.csv
- 13B_rank_gap_historical.csv
- 13B_manifest.json

13C:

- 13C_factorial_summary.csv
- 13C_factorial_effects.csv
- 13C_pairwise_deltas.csv
- 13C_day_block_bootstrap.csv
- 13C_rank_gap_historical.csv
- 13C_manifest.json

14A:

- 14A_rank_distribution.csv
- 14A_per_site_rank_gap.csv
- 14A_capture_groups.csv
- 14A_sequence_size_summary.csv
- 14A_per_site_size_metrics.csv
- 14A_per_site_drift_metrics.csv
- 14A_drift_similarity_pairs.csv
- 14A_confusion_edges.csv
- 14A_component_rescue.csv
- 14A_correlations.csv
- 14A_manifest.json

Los outputs factoriales 13C tienen también estado de caché. Un conjunto parcial
o incompatible falla; uno completo compatible solo se reutiliza con --resume y
se verifican sus hashes CSV.

## Tests añadidos

Se cubren:

- pcap_uid permutado entre expertos;
- fechas, etiquetas y candidate_labels incompatibles;
- origin, window y seed incorrectos;
- Micro/Macro con pcap_uid intercambiado y misma etiqueta;
- los cuatro estados de caché;
- hashes de checkpoint y scaler;
- divergencia de YAML;
- definición exacta de M0 y M1;
- las ocho combinaciones factoriales;
- bootstrap determinista;
- clasificación de estados en dry-run;
- flags Future-B selection_allowed=false y training_allowed=false;
- manifiesto 14A con git_commit.

El bloque F13 nuevo de FEATURE_LINEAGE fue traducido al español sin tocar el
contenido histórico previo.

## Background de Zeus

Se creó:

scripts/run_p13_b1_background.sh

Características:

- exige LTD_ENV=zeus;
- exporta PYTHONPATH=.;
- acepta --skip-14a;
- ejecuta Phase13 completa con --resume;
- solo ejecuta 14A si Phase13 termina correctamente;
- valida outputs finales;
- escribe timestamps;
- informa STATUS SUCCESS o STATUS FAILED;
- no inicia nohup por sí mismo;
- no contiene rutas absolutas de Mac, Zeus ni /home.

Comandos exactos:

    export LTD_ENV=zeus
    export PYTHONPATH=.
    bash scripts/run_p13_b1_background.sh

Para omitir únicamente 14A:

    export LTD_ENV=zeus
    export PYTHONPATH=.
    bash scripts/run_p13_b1_background.sh --skip-14a

Ejecución directa equivalente:

    LTD_ENV=zeus PYTHONPATH=. python3 scripts/train/run_phase13_battery.py \
      --config configs/experiments/PHASE13_BATTERY_V1.yaml --resume

    LTD_ENV=zeus PYTHONPATH=. python3 scripts/diagnostics/run_future_rank_gap_battery.py \
      --config configs/experiments/FUTURE_RANK_GAP_V1.yaml

## Validación ejecutada

Correcto:

    python3 -m compileall -q src/experiments/phase13_battery.py \
      src/analysis/future_rank_gap_battery.py \
      scripts/train/run_phase13_battery.py \
      scripts/diagnostics/run_future_rank_gap_battery.py \
      tests/test_phase13_battery.py \
      tests/test_future_rank_gap_battery.py

    python3 -m pytest -q tests/test_phase13_battery.py \
      tests/test_future_rank_gap_battery.py \
      tests/test_paths.py

    bash -n scripts/run_p13_b1_background.sh

    python3 scripts/train/run_phase13_battery.py \
      --config configs/experiments/PHASE13_BATTERY_V1.yaml \
      --dry-run --only 13B

    python3 scripts/diagnostics/run_future_rank_gap_battery.py \
      --config configs/experiments/FUTURE_RANK_GAP_V1.yaml \
      --dry-run

    LTD_ENV=zeus python3 scripts/train/run_phase13_battery.py \
      --config configs/experiments/PHASE13_BATTERY_V1.yaml \
      --dry-run --only 13B

    LTD_ENV=zeus python3 scripts/diagnostics/run_future_rank_gap_battery.py \
      --config configs/experiments/FUTURE_RANK_GAP_V1.yaml \
      --dry-run

    git diff --check

Resultado de la suite solicitada: 27 passed.

Los dry-runs muestran los 18 expertos como ABSENT, los seis ensembles Micro,
las ocho combinaciones 13C, los estados Macro y output factorial, además de las
rutas Zeus correctas. No se ejecutó ningún entrenamiento.

La suite completa python3 -m pytest -q sigue sin poder recolectar por una
limitación histórica del entorno: src/models/final/09B_evaluate_internal_test.py
importa joblib, que no está instalado. No se instaló ninguna dependencia.

## Git

git diff --stat:

    docs/EXPERIMENT_MAP.md             | 74 +++++++++++++++++++++-----------------
    docs/EXPERIMENT_MAP_EXEC.md        | 18 +++++-----
    docs/FEATURE_LINEAGE.md            | 27 ++++++++++++++
    docs/RESULTS_HISTORY.md            | 29 +++++++++++++++
    docs/catalog/RESEARCH_BRANCHES.csv |  2 +-
    5 files changed, 108 insertions(+), 42 deletions(-)

git status --short:

    M docs/EXPERIMENT_MAP.md
    M docs/EXPERIMENT_MAP_EXEC.md
    M docs/FEATURE_LINEAGE.md
    M docs/RESULTS_HISTORY.md
    M docs/catalog/RESEARCH_BRANCHES.csv
    ?? .agent_results/
    ?? P13_B1_CHECKPOINT_REPORT.md
    ?? configs/experiments/FUTURE_RANK_GAP_V1.yaml
    ?? configs/experiments/PHASE13_BATTERY_V1.yaml
    ?? docs/superpowers/
    ?? scripts/diagnostics/run_future_rank_gap_battery.py
    ?? scripts/run_p13_b1_background.sh
    ?? scripts/train/run_phase13_battery.py
    ?? src/analysis/
    ?? src/experiments/
    ?? tests/test_future_rank_gap_battery.py
    ?? tests/test_phase13_battery.py

No se hizo commit ni push.

READY FOR COMMIT AND ZEUS PREFLIGHT
