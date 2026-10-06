# P13-BATTERY-005A — Portable legacy checkpoint fixture

## Causa

El fixture `_write_legacy_cache()` escribía siempre `model.pt` como texto JSON.
En el entorno local de Codex no está instalado Torch, por lo que
`_load_checkpoint_metadata()` usa su fallback JSON y el test pasaba. En Perseo
Torch sí está instalado y el loader usa `torch.load()`, que rechaza ese JSON
como checkpoint Torch válido.

## Corrección

Se modificó únicamente el fixture de `tests/test_phase13_battery.py`:

- intenta importar Torch;
- usa `torch.save(checkpoint, model.pt)` cuando está disponible;
- conserva JSON cuando Torch no está disponible;
- calcula `checkpoint_sha256` después de escribir el archivo real.

No se modificó código científico en `src/experiments/phase13_battery.py`, ni
scripts, protocolo, fingerprints, caches o resultados. No se ejecutó ningún
experimento.

## Verificación

Comando solicitado:

```text
python3 -m pytest -q tests/test_phase13_battery.py tests/test_future_rank_gap_battery.py tests/test_paths.py
```

Resultado: `52 passed in 2.61s`.

La ruta de escritura compatible con Torch también se comprobó con un
serializador temporal equivalente; el cache resultante fue clasificado como
`COMPLETE_COMPATIBLE_LEGACY`.

`git diff --check`: OK, sin salida ni errores.

## Estado del árbol

`git diff --stat`:

```text
 tests/test_phase13_battery.py | 10 ++++++++--
 1 file changed, 8 insertions(+), 2 deletions(-)
```

`git status --short`:

```text
 M tests/test_phase13_battery.py
?? .agent_results/P13-BATTERY-005A-test-portability.md
?? docs/superpowers/
```

No se hizo commit ni push.
