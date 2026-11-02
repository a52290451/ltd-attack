# SOTA S3 — entorno aislado y gates RTX 5080 (2026-11-02)

**Situación:** GPU NVIDIA RTX 5080 (16303 MiB VRAM reportada), controlador 595.99.02, 16 CPU, RAM disponible 30014960 KiB. El Python de sistema carece de torch/numpy; no modificarlo. Dataset C QA continúa por cron en paralelo. **Este procedimiento solo autoriza instalación aislada, auditoría e inferencia sintética; no autoriza entrenamiento.**

## Fuente y versiones

PyTorch 2.11.0, wheel Linux CUDA 12.8 publicado por el proyecto: https://pytorch.org/get-started/previous-versions/ . Fuente única de torch: https://download.pytorch.org/whl/cu128 . NumPy 2.2.6. No instalar torchvision/torchaudio porque DF y Var-CNN WFlib no lo necesitan para forward. Fijar versiones efectivas en el informe local.

Arquitecturas de origen **único** y checkout fijo: FIND-Lab/Website-Fingerprinting-Library commit 537ec8e03c805540b04b95b144fa158ce50aa255, ubicado en ~/ltd-ops/sota/upstream/WFlib, sin duplicarlo en Git científico. Preflight comprueba git HEAD y blobs DF/VarCNN antes de importarlos.

## Paso A. Capacidad y aislamiento en /data (revisado después de auditoría real)

El 2 de noviembre, Perseo reportó **38 GB libres en /** (96 % de ocupación) y **578 GB libres en /data**. Aunque la reserva anterior de 20 GiB en HOME se cumpliría nominalmente, no se recomienda instalar varios gigabytes de bibliotecas CUDA en la partición raíz casi llena. Mantener el Python del sistema, el repositorio, los datos A/B y Dataset C sin modificaciones. Se reserva exclusivamente un directorio independiente, **/data/ltd-sota-bsierra** (fuera de los árboles de Dataset C). Nunca crear otra ubicación ni modificar permisos con sudo si no existe acceso.

```bash
cd ~/ltd-attack
SOTA_ROOT="/data/ltd-sota-bsierra"
umask 077
mkdir -p "$SOTA_ROOT/tmp" || { echo "BLOCKED: sin permisos para /data/ltd-sota-bsierra"; false; }
export TMPDIR="$SOTA_ROOT/tmp"
python3 - <<'PY'
from pathlib import Path
import shutil
p=Path("/data/ltd-sota-bsierra")
if not p.is_dir() or not p.joinpath("tmp").is_dir() or not p.joinpath("tmp").stat().st_mode:
    raise SystemExit("BLOCKED: directoria SOTA no disponible")
free=shutil.disk_usage(p).free
print("SOTA_DATA_FREE_GIB",round(free/1024**3,1))
if free < 160*1024**3:
    raise SystemExit("BLOCKED: reserva de 160 GiB para instalar sin comprometer Dataset C")
PY
df -h "$SOTA_ROOT" "$HOME"
```

Si el gate falla, **no** instalar; no borrar datos de ningún proyecto, no tocar NTP, Docker o cron. Asegurar que TMPDIR apunta a /data antes de usar pip evita descargar wheels gigantes en /tmp del sistema.

## Paso B. Virtualenv independiente en /data (sin sudo)

Ejecutar solamente si Paso A muestra permisos y reserva satisfactoria. No reutilizar entornos ni descargar dependencias al HOME.

```bash
SOTA_ROOT="/data/ltd-sota-bsierra"
export TMPDIR="$SOTA_ROOT/tmp"
VENV="$SOTA_ROOT/venv-cu128"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --no-cache-dir \
  torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/python" -m pip install --no-cache-dir numpy==2.2.6
"$VENV/bin/python" -m pip freeze > "$HOME/ltd-ops/sota/env_lock_torch211_cu128.txt"
df -h "$SOTA_ROOT" "$HOME"
```

Si falta el módulo venv/ensurepip, detenerse y registrar el error. No usar sudo, --break-system-packages ni pip de sistema. Instalar wheel con no-cache para evitar llenar el disco con duplicados descargados. La disponibilidad de rueda/compatibilidad RTX5080 solo se confirma durante la prueba real.

## Paso C. Forwards sintéticos, cero datos reales

```bash
cd ~/ltd-attack
VENV="/data/ltd-sota-bsierra/venv-cu128"
"$VENV/bin/python" scripts/sota/s3_reference_architecture_preflight_v1.py --forward-smoke
"$VENV/bin/python" scripts/sota/s3_cuda_synthetic_preflight_v1.py
```

Se esperan logits finitos **[2,65]** para DF con tensor **[2,1,5000]** y Var-CNN DT2 con **[2,2,5000]**, primero CPU y luego CUDA. El CUDA smoke valida runtime, compute capability SM120, commit de WFlib y cantidad de parámetros; registra evidencia JSON en ~/ltd-ops/sota. Si falla, **no entrenar**, conservar errores y revisar compatibilidad del wheel.

## Paso D. Auditoría total del canal temporal Historical DEV

```bash
cd ~/ltd-attack
python3 -m unittest discover -s tests -p 'test_sota_s3_full_timing_audit.py' -v
ionice -c3 nice -n 10 \
  python3 scripts/sota/s3_full_dev_timing_audit_v1.py --max-uids 150 --save
ionice -c3 nice -n 10 \
  python3 scripts/sota/s3_full_dev_timing_audit_v1.py --max-uids 0 --save
```

La ejecución total comprueba exactamente 57.916 UIDs DEV, 65 clases, splits, alineación dirección-tiempo, regresiones e intervalos negativos usados por DT2. **Reverifica el SHA256** del archivo Historical al completar la corrida; esto implica una lectura adicional de disco a baja prioridad. No abre Future-B, no entrenan modelos, no altera muestras. Cualquier entrada inválida o diferencia SHA bloquea entrenamiento Var-CNN.

## Gate científico posterior (NO aprobado)

Una vez los controles técnicos den PASS: fijar semillas, epochs/batch/optimizador y regla de selección temporal sin fuga de futuro; controlar presupuesto GPU y no interferir con cron; verificar el orden de 65 etiquetas con NPZ congelados solo para evaluación y no para tuning; asegurar mismos UID/fechas/etiquetas para comparaciones con LTD-Hybrid/Micro/Macro; exportar cuatro métricas Top-1, Macro-F1, Top-5, MRR y test estadístico pareado por fecha/IC95/Holm. La reconstrucción de 09A aún tiene cotejo original UID↔split pendiente. Dataset C sigue QA_ONLY por NTP no sincronizado.
