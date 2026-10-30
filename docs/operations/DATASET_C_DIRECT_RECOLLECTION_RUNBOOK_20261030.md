# Dataset C — Purga obfs4 y reconstrucción Tor directa (30-10-2026)

**Orden expreso del investigador:** desechar las capturas previas de Dataset C recolectadas mediante bridges/obfs4 porque no se consideran equivalentes a A/B; volver a utilizar las denominaciones de cuatro contenedores Docker, pero **sin** su antiguo `torrc-obfs4`, `LD_PRELOAD` ni el ajuste de reloj `-2963`. Esta nueva decisión **reemplaza la instrucción operativa de conservar datos antiguos**, pero NO modifica informes históricos, checkpoints, archivos de A/B ni resultados de modelos.

**Estado al crear este documento: NO se ha eliminado nada del servidor ni se han reemplazado contenedores; todo está preparado en GitHub para ejecución por el operador en Perseo.** La conexión directa Tor fue validada en un contenedor descartable: `TOR_DIRECT_BOOTSTRAP_100=YES`, `TOR_DIRECT_ISTOR=TRUE`. El reloj anfitrión sigue `NTPSynchronized=no` y ~49 min adelantado respecto a HTTP externo; el test de conectividad no valida timestamps de PCAP. Nunca usar `CAP_SYS_TIME` para corregir un reloj supuestamente aislado: el tiempo real del kernel está compartido.

## A — Inventario de datos y actores (NO destructivo)

Desde Perseo, tras git fast-forward controlado:

```bash
bash -n scripts/dataset_c/retire_bridge_capture_data_v1.sh
bash -n scripts/dataset_c/rebuild_direct_tor_stack_v1.sh
python3 -m py_compile scripts/dataset_c/three_instance_direct_qa_v1.py

bash scripts/dataset_c/retire_bridge_capture_data_v1.sh --inventory
bash scripts/dataset_c/rebuild_direct_tor_stack_v1.sh --status
```

El inventario lista candidatos **exactos**: `/data/ltd-c-qa-bsierra`, `~/ltd-storage/dataset-c-smoke-obfs4` y `~/ltd-storage/dataset-c-smoke`. No deducir que esos tres abarcan todo Dataset C: las rutas adicionales encontradas o enlaces simbólicos se revisan antes de borrarlos. Revisar propiedad, tamaño, rutas y contenido de primer nivel. **No borrar** `~/ltd-attack`, `~/ltd-storage/datasets` (A/B), `~/ltd-storage/results` (evidencia científica), `~/ltd-storage/artifacts`, modelos, entorno virtual, ni código `~/ltd-collector-c-smoke-v22`. No ejecutar `docker system prune`, `docker volume prune`, ni `rm -rf` de rutas genéricas.

## B — Purga explicitamente autorizada de raíces antiguas

Solo después de inspeccionar el inventario y confirmar que los candidatos contienen exclusivamente la data obfs4 antigua:

```bash
LTD_C_PURGE_CONFIRMED=YES_DELETE_ONLY_OLD_OBFS4_C \
  bash scripts/dataset_c/retire_bridge_capture_data_v1.sh --purge
```

El script exige cron antiguo ausente, contenedores C detenidos, rutas exactas, directorios reales del operador y no montajes de filesystem. Escribe una bitácora de rutas, tamaños y archivos borrados en `~/ltd-ops/dataset-c-bridge-retirement/`, **fuera de la data eliminada**. Este borrado es irreversible; preserva solo la trazabilidad resumida, no los PCAP anteriores. Si aparecen otros depósitos o symlinks, inventariar manualmente y documentar una nueva lista segura, nunca ampliar automáticamente el patrón.

## C — Reconfiguración con sustitución de contenedores

No se puede cambiar con `docker restart` el `Entrypoint` original que contiene obfs4/shim. Se crean y comprueban **cuatro contenedores nuevos** sobre las redes preexistentes, y solo al superar el test se les transfieren los nombres usados por el recolector:

```bash
LTD_C_DIRECT_REBUILD_CONFIRMED=YES_STAGE_DIRECT_C \
  bash scripts/dataset_c/rebuild_direct_tor_stack_v1.sh --stage

# La primera comprobación puede hacerse antes de que Tor termine el bootstrap.
bash scripts/dataset_c/rebuild_direct_tor_stack_v1.sh --verify-stage

LTD_C_DIRECT_CUTOVER_CONFIRMED=YES_SWITCH_C_NAMES \
  bash scripts/dataset_c/rebuild_direct_tor_stack_v1.sh --cutover

bash scripts/dataset_c/rebuild_direct_tor_stack_v1.sh --verify-current
```

Reusa redes `isolated_net_1site_inst1/2/3` y `bridge` para `ltd-c-manual-tor`. El estado nuevo vive en `/tmp` (no se reutilizan volúmenes ni state obfs4). `UseBridges 0`, `SocksPort 127.0.0.1:9050`, `CAP_SYS_TIME` no concedido, `--restart=no`, sin puertos publicados, sin cron/captura automática. Los antiguos contenedores parados se renombran con sufijo `obfs4-retired-...` para rollback de **configuración**; no contienen la data PCAP eliminada y no se arrancan. Al confirmar sistema nuevo pueden eliminarse explícitamente por nombre, si procede, tras auditar sus volúmenes de estado.

## D — Piloto QA directo sin promover a ciencia

```bash
install -d -m 0700 /data/ltd-c-direct-v1
python3 scripts/dataset_c/three_instance_direct_qa_v1.py check
python3 scripts/dataset_c/three_instance_direct_qa_v1.py smoke
```

El piloto recoge **12 visitas QA** (2 por categoría en 3 instancias) en `/data/ltd-c-direct-v1/qa_only/`, con `metadata.json` `QA_ONLY_NOT_FOR_MODEL_TRAINING_OR_INFERENCE`, transport `DIRECT_TOR_NO_BRIDGES`, reloj y origen del colector. No instalar cron ni reutilizar `three_hourly_qa.sh` o `run_hourly_qa.sh` antiguos (integran o etiquetan explícitamente obfs4).

## E — Cuándo habilitar captura C_DIRECT_V1 científica/recurrente

**Bloqueos actuales**: (1) tiempo UTC real no calibrado; (2) contrato original A/B (observación de la red, versión Tor/browser, BPF, sesiones, filtrado, 65 etiquetas canónicas y correspondencia UID) sin verificar; (3) colector histórico C toma 120 slots / 118 URLs, NO necesariamente 65 sitios canónicos. `IsTor=true` no prueba equivalencia.

Opciones técnicamente defendibles para el tiempo: NTP del host sincronizado; o validación y calibración auditable de offset en cada lote mediante al menos dos fuentes externas, con tolerancia preestablecida y corrección determinista de metadatos y PCAP, conservando crudos y controlando drift/error. **Esa calibración todavía no está implementada ni validada**. Ejecutar el piloto QA sin NTP es válido solo para evaluar funcionalidad del colector, no para inferencia longitudinal. No conceder `CAP_SYS_TIME` al contenedor ni usar `LD_PRELOAD` de reloj del puente como sustituto silencioso.

Después de aprobar A/B, 65 sitios y reloj, implementar un nuevo runner y scheduler `C_DIRECT_V1` con bloqueo de concurrencia, límites de disco, retries seguros, SHA256 y manifiestos. No iniciar colección recurrente antes de esos gates.

## Comandos de cierre

```bash
git status --short
bash scripts/dataset_c/rebuild_direct_tor_stack_v1.sh --status
df -h /data
```

El usuario aporta la salida del inventario y de cada gate; el conector GitHub **no ejecuta comandos sobre Perseo**.

**SOTA y checkpoint de evidencia científica continúan inalterados.**
