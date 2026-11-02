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
| Dependencias y virtualenv de LTD | `/home/bsierra/ltd-venv` (preexistente) | **Evaluar y reutilizar antes de instalar nada**; no modificarlo sin verificar compatibilidad con sus usos anteriores |
| Código upstream WFlib fijado por commit | Subdirectorio ignorado dentro del paraguas LTD | No duplicar repositorios ni romper referencias existentes sin migración verificada |
| Capturas, datasets, artefactos grandes, checkpoints y resultados | Almacenamiento `/data` | Mantener bases históricas y Dataset C intactos; configurar rutas centralmente |
| Logs/evidencia operativa preexistentes | Rutas actuales hasta inventario y migración segura | No borrar, renombrar ni cambiar cron automáticamente |

**Corrección confirmada con captura de terminal:** existe el entorno Python `/home/bsierra/ltd-venv` con `bin/`, `include/`, `lib/`, `lib64/` y `pyvenv.cfg`. La búsqueda anterior solo inspeccionaba `~/ltd-attack` y, por su alcance, lo omitió. **Verificación posterior:** el entorno anterior ocupa **167 MB**, utiliza Python **3.12.3**, NumPy **2.5.4** y Pandas **3.1.0**, y no tiene PyTorch instalado. El entorno CUDA temporal ocupa **6,7 GB**. Mantener ambos intactos hasta medir el consumo de `~/ltd-storage` y los otros componentes de HOME; **no instalar de nuevo, no crear `~/ltd-attack/.venv` automáticamente ni modificar `~/ltd-venv` sin análisis de compatibilidad**.

**Estado transitorio identificado el 2026-11-02:** un entorno PyTorch
`/data/ltd-sota-bsierra/venv-cu128` fue instalado y validado en CPU/CUDA;
es una **excepción a la política**, no una estructura nueva aprobada. También
existen `~/ltd-ops/sota` (informes y checkout upstream) y
`~/ltd-ops/dataset-c-direct120` (logs cron), mientras
`configs/environments/perseo.yaml` declara
`/home/bsierra/ltd-storage`. La terminal de Perseo confirma que esta
ruta resuelve sobre sí misma y está en la partición raíz, **96 % ocupada**;
su tamaño y dependencia exactos todavía necesitan inventario. No mover
nada sin comprobar referencias y espacio.

**No confundir GPU con disco:** el valor 96 % observado corresponde a la
partición raíz `/`; una lectura anterior de `nvidia-smi` registró 0 %
de uso GPU y 15.827 MiB VRAM libre, sin identificar procesos ajenos.

Consulta primero el [inventario físico y plan de consolidación de Perseo](docs/operations/LTD_PERSEO_LAYOUT_CONSOLIDATION_20261102.md).

Consulta el [procedimiento de organización y diagnóstico S3](docs/sota/SOTA_S3_RUNTIME_GPU_V1.md)
y el [checkpoint de S3 y Dataset C](docs/research/CHECKPOINT_20261102_SOTA_S3_DATASET_C_PARALLEL.md).
Las rutas físicas reales de Perseo, la ocupación por componente y cualquier
plan de migración necesitan una **auditoría local de solo lectura** antes de
modificarse. La instalación del entorno de SOTA y forwards sintéticos **no
autorizan entrenamiento**.

Para datos locales fuera de Git, ver también `docs/migration/` y
`data/local/`.
