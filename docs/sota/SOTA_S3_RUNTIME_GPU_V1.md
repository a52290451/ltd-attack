# SOTA S3 — auditoría del entorno, GPU y organización unificada de LTD

**Estado real a 2026-11-02.** La instalación aislada de PyTorch y el forward
sintético **YA se ejecutaron** en Perseo. Este documento sustituye la orden
anterior de instalar entornos nuevos en `/data`; **NO volver a ejecutar la
instalación** ni crear copias hasta cerrar el inventario físico.

## 1. Resultados reproducibles confirmados

- PyTorch `2.11.0+cu128`, NumPy `2.2.6`, CUDA runtime `12.8`.
- GPU: NVIDIA RTX 5080, compute capability `(12,0)`, controlador `595.99.02`.
- DF-WFlib: 3.699.393 parámetros, entrada `[2,1,5000]`,
  logits `[2,65]` finitos; CPU y CUDA **PASS**.
- VarCNN-WFlib: 8.806.209 parámetros, entrada `[2,2,5000]`,
  logits `[2,65]` finitos; CPU y CUDA **PASS**.
- Checkout WFlib fijado y verificado:
  `537ec8e03c805540b04b95b144fa158ce50aa255`.
- Evidencias actuales:
  `~/ltd-ops/sota/s3_reference_20261102T151428Z.json` y
  `~/ltd-ops/sota/s3_cuda_smoke_20261102T151430Z.json`.
- **0 pasos de entrenamiento;** no se abrieron Future-B ni Dataset C para
  estas inferencias.
- Historical DEV: **37/37 tests**, 57.916/57.916 UIDs auditados,
  `INVALID=0`, `SHA256=PASS`. Ver checkpoint científico.

**Confusión corregida:** `/` estaba al 96 % de **capacidad de disco**
(38 GB disponibles), NO la GPU. Lectura de recursos anterior a los forwards:
`nvidia-smi` GPU `utilization.gpu=0` y 15.827 MiB VRAM libres de
16.303 MiB totales. La diferencia de 476 MiB VRAM **no identifica** un proceso
o propietario. Solo `nvidia-smi` y los PIDs en tiempo real pueden hacerlo.

## 2. Regla de arquitectura física — LTD es un solo paraguas

- Código versionado: **`~/ltd-attack`**. Contiene scripts de Dataset C,
  modelos SOTA, Micro/Macro/Hybrid, pruebas, documentos y configuración.
- Dependencias de ejecución: **ya existe `~/ltd-venv`** bajo
  `/home/bsierra` (terminal con `pyvenv.cfg`). Su presencia fue omitida por
  una búsqueda anterior limitada a `~/ltd-attack`. **Inspeccionar primero
  este entorno**; no crear `~/ltd-attack/.venv`, no reemplazar sus paquetes
  ni suponer que contiene CUDA hasta medirlo.
- Upstream WFlib: fuente externa fijada dentro del área del proyecto, tras
  verificar que cambiarla no rompe el loader. No clonar duplicados.
- **`/data` solo para datos y resultados:** PCAP, CSV, matrices derivadas,
  reportes, checkpoints y artefactos. No instalar paquetes ni guardar código
  fuente allí.
- Las rutas previas bajo `~/ltd-ops` y
  `/home/bsierra/ltd-storage` deben **inventariarse antes de reorganizarse**.
  La configuración actual `configs/environments/perseo.yaml` todavía apunta a
  `/home/bsierra/ltd-storage`; puede ser un enlace, montaje o carpeta física.
  **No cambiar esa configuración a ciegas**.

**Entorno previo recuperado de la evidencia del investigador:** la captura
de terminal muestra `/home/bsierra/ltd-venv/{bin,include,lib,lib64,pyvenv.cfg}`.
No estaba bajo `~/ltd-attack`; la inspección anterior no buscó los
directorios hermanos del proyecto. Antes de cualquier nueva instalación o
limpieza, verificar versión de Python, PyTorch/CUDA y paquetes existentes,
sin ejecutar `pip install` ni escribir en el entorno. Una copia preexistente
puede soportar dependencias de otros pasos históricos; preservarlas.

**Excepción temporal detectada:** el entorno PyTorch instalado en
`/data/ltd-sota-bsierra/venv-cu128` funciona y es el único que actualmente
tiene CUDA 12.8 verificado para SOTA. Su ubicación no cumple la política
objetivo. Mantenerlo **sin usar para entrenamiento ni moverlo** hasta disponer
del inventario y de una migración probada. Un `venv` normalmente contiene
rutas absolutas y **no debe trasladarse con `mv` entre montajes** suponiendo
que seguirá funcionando. Su eventual reemplazo debe probarse primero y solo
después evaluar la eliminación exacta de la copia anterior, con autorización.

## 3. Única auditoría de inventario (SOLO LECTURA)

Ejecutar desde Perseo; no se crean ni modifican archivos:

```bash
cd ~/ltd-attack

echo "=== GPU ACTUAL Y PROCESOS ==="
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits || true
ps -eo pid,user,comm,%cpu,%mem --sort=-%cpu | head -n 18

echo "=== SISTEMAS DE ARCHIVOS ==="
df -h "$HOME" "$HOME/ltd-attack" "$HOME/ltd-storage" /data
for p in "$HOME/ltd-attack" "$HOME/ltd-storage" \
         "$HOME/ltd-ops" "$HOME/ltd-ops/sota/upstream/WFlib" \
         "/data/ltd-sota-bsierra" "/data/ltd-c-qa-bsierra"; do
  if [ -e "$p" ]; then
    printf 'PATH %s -> %s\n' "$p" "$(readlink -f "$p")"
    ls -ld "$p"
  fi
done

echo "=== TAMAÑO DE INSTALACIONES NO CIENTIFICAS ==="
for p in "/data/ltd-sota-bsierra/venv-cu128" \
         "$HOME/ltd-ops/sota/upstream/WFlib" \
         "$HOME/ltd-attack/.venv"; do
  if [ -d "$p" ]; then du -sh "$p"; fi
done
echo "=== VENV EXISTENTES EN EL PARAGUAS ==="
find "$HOME/ltd-attack" -maxdepth 4 -name pyvenv.cfg -print
echo "=== CONFIGURACION PERSEO ACTUAL ==="
cat configs/environments/perseo.yaml
```

Los PIDs de `nvidia-smi` permiten consultar `ps -p PID -o
pid,user,comm,args` para conocer el propietario si aún siguen vivos. Si no
hay procesos y la GPU está ociosa, **no inferir uso por terceros** de la mera
VRAM reservada. Esta auditoría puede mostrar procesos que corresponden a
containers del propio usuario.

### Comprobación específica del entorno `~/ltd-venv` (sin modificaciones)

```bash
cd ~/ltd-attack
printf '=== EXISTENTE ===\n'
cat "$HOME/ltd-venv/pyvenv.cfg"
du -sh "$HOME/ltd-venv" "/data/ltd-sota-bsierra/venv-cu128"
"$HOME/ltd-venv/bin/python" - <<'PY'
import sys
from importlib import metadata
print("PYTHON", sys.executable, sys.version.split()[0])
print("PREFIX", sys.prefix)
for package in ("torch", "numpy", "pandas", "scipy", "scikit-learn"):
    try: print(package, metadata.version(package))
    except metadata.PackageNotFoundError: print(package, "NOT_INSTALLED")
PY
```

No usar `source`, `pip install` ni `mv` hasta comparar resultados y
confirmar qué scripts históricos utilizan ese entorno.

## 4. Gate de reorganización: pendiente de inventario

No instalar de nuevo, no borrar el directorio de `/data` ni realizar
`mv`, `rm`, `pip install`, `docker prune`, `crontab` o cambios de
`perseo.yaml` durante el diagnóstico. Una vez confirmemos sistemas de
archivos, entornos existentes y tamaños:

1. Auditar y priorizar el **entorno preexistente `~/ltd-venv`**. Si sus
   dependencias no son compatibles con PyTorch 2.11/CUDA 12.8, **no actualizarlo
   automáticamente**, porque otros procesos LTD podrían depender de sus versiones.
   Analizar opciones sin añadir copias ni ocupar espacio antes de decidir.
2. Verificar CPU y CUDA, hashes de WFlib y reproducibilidad en el destino
   antes de retirar la copia del venv mal ubicado.
3. Reubicar el checkout upstream al área LTD sin duplicarlo cuando el loader
   y los reportes puedan seguir resolviendo rutas; actualizar código,
   documentación, tests y SHA de procedencia en **un único commit coherente**.
4. Unificar futuras salidas **en /data**, a través de
   `src/utils/paths.py` y la configuración Perseo, sin alterar ni perder
   informes anteriores ni las capturas/cron de Dataset C.

**No entrenar:** siguen pendientes presupuesto, semillas, validación temporal,
contrato de etiquetas/probabilidades e integridad del protocolo estadístico
SOTA. Dataset C permanece QA_ONLY por NTP no sincronizado.
