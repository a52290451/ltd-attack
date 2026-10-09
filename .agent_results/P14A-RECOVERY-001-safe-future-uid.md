# P14A-RECOVERY-001 — Recuperación segura de metadata Future-B

## Resultado

Implementada la recuperación posthoc de `pcap_uid` para 14A sin deserializar
pickle y sin modificar el NPZ congelado. El loader valida el contrato completo
del artefacto antes de exponer el mapping a los diagnósticos existentes.

## Causa raíz

`10B_scores_DEV_FROZEN.npz` contiene `pcap_uid` como `dtype=object`. El loader
anterior abría el archivo con `allow_pickle=False`, pero después intentaba
leer `payload["pcap_uid"]`; NumPy rechaza esa lectura porque requiere
pickle. Los otros ocho campos del NPZ son arrays seguros y no necesitaban
recalcularse.

La inspección de `src/models/final/10B_evaluate_future_concept_drift.py`
confirmó que 10B construyó la identidad como
`macro_future["pcap_uid"].astype(str).to_numpy()`. `macro_future` conserva el
orden del CSV después de: mapping condicional `site -> site_label`,
`dropna(site_label)`, filtrado por los sitios élite y `reset_index(drop=True)`.
No hay ordenación posterior por UID. Las fechas se derivan con
`07A_candidate_conditioned_temporal_encoder.derive_dates()`.

## Recuperación y controles

`_load_npz()` ahora:

1. Comprueba la existencia y SHA256 del NPZ antes de abrirlo.
2. Inspecciona las cabeceras NPY dentro del ZIP sin leer los payloads object.
   Exige `pcap_uid` object unidimensional y rechaza cualquier campo object
   adicional o estructura inesperada.
3. Abre el NPZ únicamente con `allow_pickle=False` y carga solo
   `query_date`, `y_true`, `candidate_labels` y las cinco matrices de
   probabilidades. Nunca accede a `payload["pcap_uid"]` en esa ruta.
4. Reconstruye los UID desde el CSV Macro Future-B en su orden original,
   usando el mapping histórico únicamente si falta `site_label`.
5. Reconstruye las fechas con la misma función de 10B y codifica las etiquetas
   con el orden congelado de `candidate_labels`.
6. Falla explícitamente si no coinciden fila a fila `y_true` o `query_date`, si
   hay UID nulos, vacíos o duplicados, si el CSV está permutado de forma
   incompatible, o si la cobertura UID del dataset Micro Future-B no es
   íntegra.

Los defaults de producción exigen exactamente 18.543 capturas y 65 clases.
Las cinco matrices se copian sin transformaciones; los tests comprueban
igualdad bit a bit con las matrices originales del fixture. No se usa ninguna
ordenación heurística para alinear filas.

## Hash y preservación del artefacto

Hash SHA256 aprobado y exigido por el loader:

```text
d0edcd2780454b5cd39ef962ec45375d8822dc26b4b3c0b02cf4c06b0b29c7bb
```

El NPZ original y los CSV Future-B no están montados en este workspace local;
por tanto, no se reabrieron ni se reescribieron aquí. El hash conocido queda
codificado como guardrail y se registra en este reporte para la revalidación
en Perseo. Los fixtures sintéticos verifican que la lectura no altera el
archivo: el SHA256 antes y después permanece idéntico.

## Pruebas

Casos añadidos en `tests/test_future_rank_gap_battery.py`:

- recuperación positiva de UID object desde CSV;
- coincidencia exacta de clases y fechas;
- mapping histórico solo cuando falta `site_label`;
- discrepancias de `y_true` y `query_date`;
- UID duplicados, cobertura Micro incompleta y orden CSV permutado;
- cabecera `pcap_uid` inesperada y campo object adicional;
- conservación bit a bit de las cinco matrices;
- ausencia de modificación del NPZ.

Comandos ejecutados:

```text
python3 -m pytest -q tests/test_future_rank_gap_battery.py tests/test_phase13_battery.py tests/test_paths.py
60 passed in 2.91s

git diff --check
```

No se ejecutaron entrenamientos, evaluaciones de modelos, búsquedas de
hiperparámetros ni 10B. No se modificaron `phase13_battery.py`, archivos de
13B/13C, scripts de entrenamiento, checkpoints, datasets ni el NPZ.

## Archivos modificados

- `src/analysis/future_rank_gap_battery.py`
- `tests/test_future_rank_gap_battery.py`
- `.agent_results/P14A-RECOVERY-001-safe-future-uid.md`
- `docs/superpowers/plans/2026-10-08-P14A-RECOVERY-001.md`

No se hizo commit ni push.

READY FOR PERSEO 14A REVALIDATION
