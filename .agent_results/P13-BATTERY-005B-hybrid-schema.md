# P13-BATTERY-005B — Fix Hybrid in-memory schema mismatch

## Causa raíz

`candidate_store` conserva el esquema interno Micro con `y` y `dates`, pero
`validate_hybrid_alignment()` exige `y_true` y `query_date`. `run_13c()` pasaba
el registro Micro directamente al validador, por lo que `micro.get("y_true")`
era `None` y el guardrail fallaba con `Micro/Macro desalineados: y_true`.

## Evidencia de alineación científica

La auditoría directa reportada sobre el cache real de ORIGIN14 para NEAR, MID y
FAR mostró en las tres ventanas:

- mismo número de filas Micro y Macro;
- `pcap_uid`, `query_date` y `candidate_labels` exactos;
- mismo conjunto de labels;
- `y_true` exacto;
- cero filas con `y` distinto;
- clase verdadera semánticamente exacta, con cero diferencias.

Por tanto, el fallo era exclusivamente de esquema interno y no de datos ni de
alineación científica.

## Cambio aplicado

Antes de llamar al validador en `run_13c()`, se construye explícitamente
`micro_alignment` con exactamente estos campos:

```text
pcap_uid, query_date, y_true, candidate_labels, origin, window
```

`micro_record["y"]` se asigna únicamente a `micro_alignment["y_true"]`.
`validate_hybrid_alignment()` no fue modificado y continúa exigiendo esos seis
campos sin aceptar aliases silenciosamente.

El test negativo confirma que una diferencia real entre Micro `y` y Macro
`y_true` sigue produciendo `Micro/Macro desalineados: y_true`.

No se modificaron datasets, caches Micro/Macro, resultados 13B, candidatos,
probabilidades, métricas, bootstrap, pesos Hybrid ni el formato del cache
Macro. No se ejecutó ningún experimento.

## Tests y verificaciones

Tests añadidos:

- registro estilo `candidate_store` alineado: el adapter produce el payload
  estricto y la validación pasa;
- `y` Micro distinto de `y_true` Macro: la validación falla.

Comando solicitado:

```text
python3 -m pytest -q tests/test_phase13_battery.py tests/test_future_rank_gap_battery.py tests/test_paths.py
```

Resultado: `54 passed in 3.00s`.

`bash -n scripts/run_p13_b1_background.sh`: OK.

`git diff --check`: OK, sin salida ni errores.

## Estado del árbol

`git diff --stat`:

```text
 src/experiments/phase13_battery.py | 10 ++++-
 tests/test_phase13_battery.py      | 88 ++++++++++++++++++++++++++++++++++++++
 2 files changed, 97 insertions(+), 1 deletion(-)
```

`git status --short`:

```text
 M src/experiments/phase13_battery.py
 M tests/test_phase13_battery.py
?? .agent_results/P13-BATTERY-005B-hybrid-schema.md
?? docs/superpowers/
```

No se hizo commit ni push.
