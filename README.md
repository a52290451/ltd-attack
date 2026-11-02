# LTD-Attack

Repositorio científico **único y paraguas** para LTD-Attack: análisis histórico,
Micro/Macro/Hybrid, comparación SOTA, Dataset C, pruebas de calidad, automatización
operacional y documentación de evidencia.

La organización de código versionado se mantiene dentro de este repositorio:

- `src/`: implementación científica, extracción y arquitecturas/adaptadores.
- `scripts/`: operaciones de Perseo, captura de Dataset C, auditorías y SOTA.
- `configs/`: configuración reproducible y rutas según entorno.
- `tests/`: pruebas unitarias y de integración.
- `docs/`: protocolos, runbooks, decisiones, checkpoints y trazabilidad.
- `experiments/`: metadatos de experimentos.
- `legacy/`: código histórico solo de referencia.

## Separación física exigida en Perseo

| Tipo | Ubicación objetivo | Política |
|---|---|---|
| Repositorio, scripts, documentación y código | `/home/bsierra/ltd-attack` | Única fuente Git de LTD; no crear proyectos paralelos |
| Dependencias y virtualenv de LTD | Dentro de `/home/bsierra/ltd-attack`, fuera de Git | **Reutilizar primero**; creación/migración solamente después de inventario y aprobación de espacio |
| Código upstream WFlib fijado por commit | Subdirectorio ignorado dentro del paraguas LTD | No duplicar repositorios ni romper referencias existentes sin migración verificada |
| Capturas, datasets, artefactos grandes, checkpoints y resultados | Almacenamiento `/data` | Mantener bases históricas y Dataset C intactos; configurar rutas centralmente |
| Logs/evidencia operativa preexistentes | Rutas actuales hasta inventario y migración segura | No borrar, renombrar ni cambiar cron automáticamente |

**Estado transitorio identificado el 2026-11-02:** un entorno PyTorch
`/data/ltd-sota-bsierra/venv-cu128` fue instalado y validado en CPU/CUDA;
es una **excepción a la política**, no una estructura nueva aprobada. También
existen `~/ltd-ops/sota` (informes y checkout upstream) y
`~/ltd-ops/dataset-c-direct120` (logs cron), mientras
`configs/environments/perseo.yaml` declara
`/home/bsierra/ltd-storage`. Todavía **no se ha verificado** si
`ltd-storage` es enlace, montaje o directorio físico. No mover nada sin
comprobar sus referencias y el sistema de archivos real.

**No confundir GPU con disco:** el valor 96 % observado corresponde a la
partición raíz `/`; una lectura anterior de `nvidia-smi` registró 0 %
de uso GPU y 15.827 MiB VRAM libre, sin identificar procesos ajenos.

Consulta el [procedimiento de organización y diagnóstico S3](docs/sota/SOTA_S3_RUNTIME_GPU_V1.md)
y el [checkpoint de S3 y Dataset C](docs/research/CHECKPOINT_20261102_SOTA_S3_DATASET_C_PARALLEL.md).
Las rutas físicas reales de Perseo, la ocupación por componente y cualquier
plan de migración necesitan una **auditoría local de solo lectura** antes de
modificarse. La instalación del entorno de SOTA y forwards sintéticos **no
autorizan entrenamiento**.

Para datos locales fuera de Git, ver también `docs/migration/` y
`data/local/`.
