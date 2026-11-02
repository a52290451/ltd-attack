# LTD-Attack — inventario y consolidación física de Perseo (2026-11-02)

**Ámbito:** un único paraguas versionado en `/home/bsierra/ltd-attack`; `/data` exclusivamente para PCAP, datasets, modelos, checkpoints, métricas y resultados. No crear un segundo proyecto ni un segundo entorno de PyTorch sin revisar primero las instalaciones existentes.

**Estado del documento:** observaciones obtenidas de la terminal Perseo compartida por el investigador, más inspección del repositorio GitHub en `feature/macro-v2-historical-only`. No hay acceso SSH ni intervención física en Perseo desde GitHub. Los cambios de código aquí referenciados se deben sincronizar mediante `git fetch/merge` en Perseo antes de probarlos.

## 1. Hechos verificados de Perseo

| Control | Observación | Interpretación |
|---|---|---|
| GPU RTX 5080 | `GPU-Util=0%`, `15 MiB/16303 MiB` VRAM usada, 33 °C, 9 W | No hay trabajos GPU en la instantánea |
| Proceso gráfico GPU | `Xorg` PID 1635, 4 MiB | No es evidencia de entrenamiento ni de uso por terceros |
| CPU y procesos | aparecen `chrome`, `docker-init` y `tcpdump` con UID numérico 10042 | UID sin resolver: no asignar identidad ni terminar procesos; pueden relacionarse con Dataset C |
| Partición `/` | 824G tamaño, 745G usados, 38G libres, **96 %** | **Disco**, no GPU; no instalar/copiar paquetes CUDA allí antes de revisar ocupación |
| Partición `/data` | 2,7T, 2,0T usados, 571G libres, **79 %** | Espacio para capturas/resultados, preservar umbral de Dataset C |
| Venv SOTA | `/data/ltd-sota-bsierra/venv-cu128`, **6,7G** | PyTorch CUDA instalado y probado, pero fuera del destino acordado |
| Checkout WFlib | `~/ltd-ops/sota/upstream/WFlib`, **1,9M** | Commit y blobs verificados, fácil consolidación tras comprobar compatibilidad |
| Entornos LTD | `find ~/ltd-attack -maxdepth 4 -name pyvenv.cfg` no encontró ninguno | No demuestra ausencia de todos los entornos Python de HOME ni de contenedores |
| `~/ltd-storage` | `readlink -f` apunta a sí mismo y `df` informa partición `/` | Histórico, artefactos y resultados configurados aún en disco del sistema; **volumen desconocido** |
| SOTA S3 | `37/37` tests, `57916/57916` DEV, 0 invalid, SHA256 PASS y forwards DF/VarCNN CPU/CUDA PASS | Gates de preprocesamiento/ejecución sintética cerrados, sin entrenamiento |

## 2. Dependencias reales que impiden mover carpetas a ciegas

Detectadas revisando código versionado:

- `src/utils/paths.py` centraliza `data_root`, `artifact_root`, `result_root`, pero `configs/environments/perseo.yaml` **aún** apunta a `~/ltd-storage/{datasets,artifacts,results}` (en partición raíz). Cambiarlo sin mover físicamente las fuentes históricas puede inutilizar los pipelines.
- `scripts/dataset_c/final29_recovery_qa_v1.py` lee `APP=~/ltd-ops/collector-qa-recovery-v25` (scripts de navegador protegidos por SHA), y datos de `/data/ltd-c-qa-bsierra/C_DIRECT_V1`.
- `scripts/dataset_c/run_direct120_recurrent_qa_v1.py` consulta listas en `~/ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2/site_lists`, guarda logs operativos en `~/ltd-ops/dataset-c-direct120`, con locks en `/data`.
- `scripts/dataset_c/run_direct120_halfsplit_qa_v2.py` construye entradas cron con ruta de Python y script existentes, apoyándose en los directorios anteriores. **No modificar cron ni retirar dependencias externas durante la captura.**
- `src/models/sota/v1/upstream_models.py` antes dependía exclusivamente de `~/ltd-ops/sota/upstream/WFlib`; ahora tiene una **resolución compatible** que prefiere `~/ltd-attack/.vendor/WFlib` cuando exista y, si no, conserva el checkout ya validado. No crea ni copia carpetas.
- `~/ltd-ops/sota` contiene manifiestos S1/S2/S3 y resultados reproducibles de los gates; no borrarlos hasta migración con comprobación de hashes y actualización de referencias.

## 3. Objetivo de organización

```text
/home/bsierra/ltd-attack/        # Único paraguas científico versionado
  src/                           # Algoritmos Micro/Macro/Hybrid/SOTA, utilidades
  scripts/                       # Captura Dataset C, pruebas, auditorías, automatización
  configs/                       # Rutas por entorno
  tests/                         # Contratos de integridad
  docs/                          # Evidencia y decisiones
  .vendor/WFlib/                 # Checkout externo fijado, ignorado por Git
  # Entorno preexistente: /home/bsierra/ltd-venv; evaluar antes de crear .venv

/data/                            # Datos y productos de investigación
  ltd-c-qa-bsierra/               # Dataset C, no alterar
  [raíces LTD A/B y resultados]    # DESTINOS NO FIJADOS: verificar inventario
```

`ltd-ops`, `ltd-storage`, `ltd-collector-c-smoke-v22` y las rutas SOTA existentes se consideran **fuentes protegidas** hasta verificar referencias y espacio. Este diagrama no autoriza crear, mover o eliminar ninguna ruta. La aplicación de paths debe ser una migración por fases, no un reemplazo ciego de strings.

## 4. Próxima auditoría de espacio y dependencias — solo lectura

**No ejecutar** `du /` ni borrar cachés sin atribuir volumen. Estos comandos delimitan el costo del escaneo al área del investigador y reducen prioridad de I/O durante Dataset C:

```bash
cd ~/ltd-attack
echo "===== RESUMEN HOME / STORAGE ====="
for p in "$HOME/ltd-storage/datasets" "$HOME/ltd-storage/artifacts" \
         "$HOME/ltd-storage/results" "$HOME/ltd-attack" \
         "$HOME/ltd-ops" "$HOME/ltd-collector-c-smoke-v22"; do
  if [ -d "$p" ]; then
    echo "TAMAÑO $p"
    timeout 120s ionice -c3 nice -n 10 du -xsh "$p" 2>&1
  fi
done

echo "===== SISTEMAS DE ARCHIVOS ====="
findmnt -T "$HOME/ltd-storage"
findmnt -T "$HOME/ltd-attack"
findmnt -T /data

echo "===== ENTORNOS PREEXISTENTES EN HOME (LIMITADO) ====="
find "$HOME" -maxdepth 4 -name pyvenv.cfg -print 2>/dev/null

echo "===== FUENTES EXTERNAS UTILIZADAS POR CRON ====="
crontab -l | grep -E 'LTD_C_DIRECT120|ltd-attack|ltd-ops' || true
test -e "$HOME/ltd-ops/collector-qa-recovery-v25/capture_dom.js" && echo "C_SCRIPT_PRESERVED=yes"
test -e "$HOME/ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2/site_lists" && echo "C_SITE_LISTS_PRESERVED=yes"
git status --short
```

**Nota:** si uno de los escaneos llega al límite de 120 s, su tamaño no queda establecido; no interpretarlo como 0. No ejecutar migración si hay rutas originales que no pudieron inventariarse. Cualquier salida del comando `crontab -l` se usa exclusivamente como evidencia, sin actualizar el planificador.

## 4.1. Auditoría real de `~/ltd-venv`: se encontró y es diferente a SOTA

**Salida comprobada en Perseo tras revisar el directorio hermano del repositorio:**

| Elemento | `~/ltd-venv` preexistente | `/data/ltd-sota-bsierra/venv-cu128` |
|---|---|---|
| Python | `3.12.3`, ejecutable `/home/bsierra/ltd-venv/bin/python` | PyTorch wheel `cp312`, forward CPU/CUDA comprobado |
| Tamaño físico | **167 MB** | **6,7 GB** |
| NumPy | **2.5.4** | **2.2.6** |
| Pandas | **3.1.0** | No inventariado |
| PyTorch | **No instalado** | **2.11.0+cu128** |
| SciPy / scikit-learn | **No instalados** | No inventariado |

El archivo `~/ltd-venv/pyvenv.cfg` declara `home = /usr/bin`, `version = 3.12.3`, `executable = /usr/bin/python3.12` y `include-system-site-packages = false`. Por tanto, el entorno anterior fue un aislamiento intencional de LTD y **no se debe actualizar o borrar sin revisar sus dependencias y consumidores**. Un solo proyecto paraguas **no obliga** a mezclar librerías incompatibles; pero tampoco justifica copias o entornos redundantes sin necesidad.

**Decisión de seguridad y almacenamiento:** el entorno PyTorch nuevo ocupa 6,7 GB y ya está instalado (no repetir descargas); mantenerlo sin modificación como excepción temporal hasta decidir migración. La raíz conserva ~38 GB libres y 96 % de uso, por lo que **no instalar otros 6,7 GB sobre `~/ltd-venv` todavía**. La próxima investigación deberá responder qué parte de `~/ltd-storage` y otros directorios de HOME consume espacio, qué contenidos realmente deberían estar en `/data`, y qué proyectos usan `~/ltd-venv`. Ninguna migración sin copia controlada, verificación de hash y pruebas de rutas antes de retirar el origen.

### Medición siguiente (solo lectura, sin copiar ni instalar)

```bash
cd ~/ltd-attack
echo "=== OCUPACION FISICA DIRECTA EN HOME ==="
ionice -c3 nice -n 10 du -xhd1 "$HOME" 2>/dev/null | sort -h
echo "=== STORAGE HISTORICO, SIN TOCAR DATOS ==="
ionice -c3 nice -n 10 du -xhd1 "$HOME/ltd-storage" 2>/dev/null | sort -h
echo "=== CODIGO / OPERACION LTD ==="
for d in "$HOME/ltd-attack" "$HOME/ltd-ops" "$HOME/ltd-venv" \
         "$HOME/ltd-collector-c-smoke-v22"; do
  [ ! -d "$d" ] || ionice -c3 nice -n 10 du -xsh "$d"
done
echo "=== PARTICIONES ==="
df -h "$HOME" /data
echo "=== CONSUMIDORES ACTIVOS DEL ENTORNO EXISTENTE ==="
ps -eo pid,user,args | grep -F "$HOME/ltd-venv" | grep -v grep || true
```

**No usar `du /`, `rm`, `pip install`, `mv`, `crontab -`, `docker prune` ni cambiar config durante el diagnóstico.** El total de HOME puede tardar según el número de archivos; `du` no accede a /data por ser otro montaje con `-x`. Si alguna ruta impide lectura, reportar limitación en vez de inferir un tamaño nulo.

## 5. Plan de consolidación sujeto a evidencia

1. **No duplicar la instalación CUDA de 6,7G.** Está viva y ya pasó el forward; conservarla hasta acreditar espacio suficiente bajo HOME y un procedimiento de reconstrucción (no usar `mv` entre montajes con un venv, pues contiene rutas absolutas). Una reubicación segura puede requerir reconstruir temporalmente en `.venv/` y solo después, con aprobación expresa, retirar el antiguo; **no realizar esta operación todavía**.
2. **Consolidar WFlib como único checkout**, con migración controlada a `~/ltd-attack/.vendor/WFlib`, commit/hashes revisados. El loader ya acepta ambas ubicaciones sin duplicar el repositorio. El traslado físico aún no ha ocurrido.
3. **Liberar espacio del root solo con un inventario causal.** Posibles raíces `ltd-storage` deben cuantificarse antes de tomar decisiones. Los datasets históricos no se borran, ni se mueven sin verificación SHA + actualización de rutas + prueba de lectura del modelo.
4. **Revisar Dependencias de Dataset C** antes de consolidar su aplicación de QA y listas en el paraguas LTD; no cambiar cron, imágenes Docker, locks, UID o directorios durante la recolección.
5. **Separar resultados futuros de SOTA/C** bajo carpetas aprobadas en `/data` y documentar la asignación centralizada, preservando rutas de manifiestos S1/S2/S3 congeladas.

**Gates científicos todavía pendientes:** semillas, presupuesto de epochs, comparación de arquitectura DF-WFlib/VarCNN-WFlib, orden exacto de 65 clases y protocolo de validación temporal. Internal 09B y Future-B no se usan para entrenar o seleccionar hiperparámetros. Dataset C continúa QA por NTP no sincronizado.
