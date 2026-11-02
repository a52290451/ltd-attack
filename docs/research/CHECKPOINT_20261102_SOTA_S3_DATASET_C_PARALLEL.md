# Checkpoint 2026-11-02 — SOTA S3 y verificación operacional de Dataset C

**Alcance:** preservar conocimiento experimental, estado real vs pendiente, decisiones de arquitectura y auditoría separada de recolección. Todo lo aquí indicado como **confirmado** se apoya en registros de Perseo suministrados por el investigador o en inspección de código GitHub; ninguna tarea remota fue ejecutada desde este documento.

## 1. Estado y resultados recibidos

**SOTA S0:** inventario ejecutado sobre `feature/macro-v2-historical-only`, 605 archivos versionados inicialmente. Falta entorno ML; `/usr/bin/python3` no encontró paquetes `torch`, `numpy`, `sklearn` o `xgboost` en esa verificación.

**SOTA S1:** **PASS** el 2 de noviembre (Perseo). Índice `/home/bsierra/ltd-ops/sota/s1_dev_frozen_20261102T130232Z/s1_manifest.json`, 57.916 UIDs DEV / 65 clases / 41 días y hashes verificados: vectors `a8a88041c44e5627e126e2c29c7e068b17c8598a0b01c63a5ff1cd3105c2c7c9`, Macro `cadfc0848caca0b2556e03b9061f842e19e1d99d862bc39dc7c0bf56169375ce`. El CSV histórico original `01_temporal_split_assignments.csv` **no está** en la ruta registrada; los splits se reconstruyeron desde metadatos Historical cuyo SHA coincide, filtrar por las 65 clases originales y comprobando 52 días, tabla por sitio/split y recuento por fecha. 5.082 registros fuera de la cohorte excluidos. **`SOTA_SPLIT_REVIEW_PENDING True`**: si aparece CSV original, cruzar UID a UID; la coincidencia de marginales no demuestra igualdad exhaustiva de pares UID/split.

**SOTA S2:** **PASS** en Perseo: 12/12 tests, 2 capturas de cada bloque DEV, tensor de dirección `6 × 5000`, direcciones ±1/0. `time_vector` y `size_vector` están **presentes como columnas**, pero presencia no demuestra validez física/temporal de contenido. Fuente Historical contiene vectores de hasta 27.720 posiciones; 27.019/57.916 superan 3.000 y 14.092/57.916 superan 5.000. No confundir entrada no truncada con equivalencia de adquisición de paquetes/celdas. Informe: `~/ltd-ops/sota/s2_adapter_smoke_20261102T133306Z.json`.

**Dataset C, último estado aportado:** ronda QA inicial **120/120 intentos** y **71/120 PCAP válidos**, con 43/65 cohortes congeladas válidas en esa ronda; cobertura QA acumulada de recuperación anterior **78/120** (47/65) no equivale a la última ronda. Existe cron HOST `:00` y `:30`, 3 contenedores × 20 posiciones/categoría, antiguo cron horario eliminado, sin evidencia aún en la conversación de ambos ciclos `half60` completos. Cron usa un bloqueo compartido: los ciclos solapados se **omiten**, no se encolan. NTP `no`; datos C = **QA_ONLY**. Espacio medido por último `df -h /data`: aproximadamente 579 GB libres (78 % ocupados), reserva operativa 120 GB, sin retención automática autorizada.

## 2. Fuentes externas fijadas — modelos WFlib, sin duplicar código

- WFlib: <https://github.com/FIND-Lab/Website-Fingerprinting-Library>, **commit** `537ec8e03c805540b04b95b144fa158ce50aa255`.
- WFlib `WFlib/models/DF.py`: git **blob SHA1** `8e084f556dd8dadac8fc5a7260b96d6de9f517f8`.
- WFlib `WFlib/models/VarCNN.py`: git **blob SHA1** `c93fb5d73497fabc9e5065d9ca82ea6d881298db`.
- DF autor original (Keras) <https://github.com/deep-fingerprinting/df>, **commit** `38df0c15a089f13228e4df06bd5269c8a37340fa`, fuente `src/Model_NoDef.py`, blob `a44f47de333566c8f927a60f26c46ee7f5b2dc54`.

**Diferencia comprobada:** DF original usa convoluciones Keras con padding `same`, max pooling `same`, y WFlib PyTorch emplea `padding=kernel_size//2` + maxpool padding=0. En WFlib DF la entrada es dirección `[N,1,5000]` y salida logits `[N,65]`; NO llamar a esta arquitectura reproduccion binaria exacta del modelo Keras, sino **`DF-WFlib-pinned`** con salvedades documentadas.

**Var-CNN WFlib fijado**: clase `VarCNN`, 2 encoders 1D residuales dilatados; concat 1024 → salida 65 clases, `[N,2,5000]`; su script `scripts/Var-CNN.sh` usa `--feature DT2 --seq_len 5000 --train_epochs 30 --batch_size 50 --optimizer Adam --learning_rate 1e-3`. WFlib genera canal `DT2` a partir de **un mismo vector de timestamp con signo**: direcciones `sign(X)`, tiempos `diff(abs(X))` con negativos fijados a 0 y padding/truncamiento. Nuestra `time_vector` Historical **aún no está auditada** para reconstruir esta representación sin inventar tiempos ni perder su vínculo de dirección. `VarCNN-DT2` permanece bloqueado hasta comprobar contenidos, precisión, orden y linaje del extractor.

WFlib DF `scripts/DF.sh`: DIR 5000, Adamax LR 0.002, batch 128, 30 épocas, selección F1. **No asumir** que ese entrenamiento fue ejecutado; config final, presupuesto/semillas/validación temporal todavía requieren congelación. No atribuir exactitud DF original al port WFlib.

El código de tercero **NO se copia** al repositorio LTD-Attack. Checkout fijado exclusivamente en `~/ltd-ops/sota/upstream/WFlib` y verificaciones de commit, árbol limpio y hashes blob antes de importar. Importación de modelos requiere PyTorch; solo preflight sintético, sin entrenamiento.

## 3. Código agregado en esta fase

| Archivo | Qué valida o implementa | Efectos |
|---|---|---|
| `src/models/sota/v1/upstream_models.py` | Modelo DF/VarCNN cargado desde WFlib exacto; HEAD/blob de ambos verificados | Solo importación, no datos |
| `scripts/sota/s3_reference_architecture_preflight_v1.py` | Gate estático y forward sintético opcional `[2,1,5000]` / `[2,2,5000]` → `[2,65]` | Modelo sin gradiente, sin train |
| `scripts/sota/s3_time_channel_audit_v1.py` | Hasta 2 vectores Historical por bloque DEV; longitudes, no-negatividad, monotonía del `time_vector` | Solo lectura, **DT2 no aprobado automáticamente** |
| `scripts/dataset_c/audit_halfsplit_operations_v1.py` | Cron y servicio, logs, rondas completas, 3×20, identidad de URL original, QA PCAP/JSON/HTML, disco, NTP y solapamientos | Solo lectura; `--save` escribe informe nuevo fuera del repo |
| `tests/test_sota_s3_and_c_operations.py` | Contratos de fuentes fijadas, manejo de valores temporales, lotes 3×20 y errores | Datos sintéticos |
| `docs/research/CHECKPOINT_20261102_SOTA_S3_DATASET_C_PARALLEL.md` | Este checkpoint; referencias, evidencias, advertencias y próximos pasos | Documentación |

**Registro de cambios S3/C (commits secuenciales, rama científica):** `dd0ed835` auditor QA, `6a4b0321` loader WFlib, `052e5166` preflight de arquitectura, `88e2a26a` auditoría de tiempos, `e1730656` tests, `9d3abba9` identidad de URL y artefactos de C, `518548d6` creación de este checkpoint, `60a2359a` vínculo en protocolo SOTA. Revisar el historial Git para SHAs completos y diff exacto. **Estos cambios son implementación y revisión estática; ningún PASS local de Perseo S3/C se declara antes de recibir su salida.**

Los scripts anteriores de S0/S1/S2 y los modelos `src/models/sota/SOTA_01/02/03` se conservan; estos últimos **no** son la batería canónica. No borrar evidencia ni A/B/C. `.gitignore` ya ignora logs, cachés, artefactos pesados y temporales; el preflight anterior detectó **0 archivos generados indebidamente versionados**. La higiene se hará solo tras una auditoría de dependencias, sin limpieza destructiva.

## 4. Ejecución segura — herramientas nuevas en Perseo

```bash
cd ~/ltd-attack
git fetch origin feature/macro-v2-historical-only &&
git merge --ff-only FETCH_HEAD

python3 -m py_compile \
  src/models/sota/v1/upstream_models.py \
  scripts/sota/s3_reference_architecture_preflight_v1.py \
  scripts/sota/s3_time_channel_audit_v1.py \
  scripts/dataset_c/audit_halfsplit_operations_v1.py

python3 -m unittest discover -s tests -p 'test_sota_s3_and_c_operations.py' -v
python3 scripts/sota/s3_reference_architecture_preflight_v1.py --selftest
ionice -c3 nice -n 10 python3 scripts/sota/s3_time_channel_audit_v1.py --per-split 2
python3 scripts/dataset_c/audit_halfsplit_operations_v1.py --save
```

Luego en checkout de WFlib separado:

```bash
UP="$HOME/ltd-ops/sota/upstream/WFlib"
mkdir -p "$(dirname "$UP")"
if [ ! -d "$UP/.git" ]; then
  git clone --no-checkout https://github.com/FIND-Lab/Website-Fingerprinting-Library.git "$UP"
fi
git -C "$UP" fetch origin 537ec8e03c805540b04b95b144fa158ce50aa255
git -C "$UP" checkout --detach 537ec8e03c805540b04b95b144fa158ce50aa255
cd ~/ltd-attack
python3 scripts/sota/s3_reference_architecture_preflight_v1.py
```

La prueba opcional `--forward-smoke` requiere **entorno SOTA con torch instalado**. No ejecutar instalación ni entrenamientos hasta inventario de recursos de Perseo y aprobación de presupuesto/semillas, para convivir con Dataset C.

### Precisión adicional en el auditor de cron

Se agregó comprobación de **pareja horaria real**: el auditor ya no considera suficientes dos informes de 60 posiciones producidos en horas diferentes. Extrae las marcas UTC de los `run_id`, busca inicios en minutos `:00` a `:10` y `:30` a `:40` de una misma hora y verifica 120 slots distintos sin duplicados. Solo informa `RECENT_PAIRED_HOUR_CONFIRMED_QA_ONLY` si existe una pareja completada dentro de las últimas tres horas según el reloj del host; no considera este reloj científicamente sincronizado. La ausencia de pareja queda como advertencia de operación. Commits `5886487e` y tests ampliados a **13** casos con `13d2b222`. Esta prueba de cadencia sigue siendo evidencia de QA, no sustituto de la sincronización NTP o de una política de no solapamiento con garantías de completitud.

## 5. Gates aún pendientes

1. **SOTA G2:** cotejar exactamente el orden de 65 `candidate_labels` original 09B/10B con el orden del índice S1 *antes de evaluar*, sin usar etiquetas futuras para ajustar modelos. El `label_order.json` alfabético no es evidencia de identidad; cabe crear mapeo explícito de scores.
2. **SOTA G3:** pruebas de arquitectura WFlib en CPU o GPU, protocolos de entrada DF y VarCNN; validar `time_vector` y el esquema firmado DT2; versión/semillas/presupuesto.
3. **SOTA G4/G5:** presupuesto GPU/CPU y RAM, entorno torch aislado, validación temporal Early→Middle / Early+Middle→Late; no reentrenar ni abrir Future-B.
4. **Dataset C:** esperar informes verdaderos para primera y segunda mitad; verificar timestamps :00/:30, duración por lote, cron daemon activo, brechas por `SKIPPED_OVERLAP`, uso y crecimiento real del almacenamiento. Cron instalado ≠ recolección exitosa. Resolver NTP y paridad de metodología A/B–C antes de elevar QA a ciencia.
5. **Informe de comparabilidad:** para todos los modelos Accuracy/Top-1, Macro-F1, Top-5, MRR sobre los mismos UIDs. Bootstrap pareado por fecha (09B 11 días, 10B 16, 2.500), IC95 percentil, pruebas condicionales de intercambio y corrección Holm con familias declaradas (no confundir Holm9 y Holm12). Future-B ya fue visto: conclusiones retrospectivas.

**Ningún paso de esta fase autoriza entrenar ni acepta datos C como observaciones científicas.** Conservar outputs de scripts localmente y adjuntar un checkpoint posterior con los valores reales para cerrar los gates.

## 6. Validación real S3 y Dataset C — Perseo 2026-11-02 13:51 (hora del servidor)

Resultados **confirmados por salida de Perseo** recibida tras el checkpoint inicial:

- **S3 pruebas**: 13/13 unit tests `OK`; `SOTA_S3_CONTRACT_SELFTEST=PASS`; no training; Future-B cerrado. Auditor de tiempos en **6 muestras DEV**: `TIME_DIRECTION_ALIGNMENT=True`, `TIME_NONNEGATIVE_MONOTONIC=True`, `TIME_NONZERO=True`. Reporte: `/home/bsierra/ltd-ops/sota/s3_time_channel_20261102T135133Z.json`.
- **S3 fuente externa**: checkout WFlib fijo en `/home/bsierra/ltd-ops/sota/upstream/WFlib`, `SOTA_S3_WFLIB_PIN_VERIFIED=537ec8e03c805540b04b95b144fa158ce50aa255`, hash blob de DF/VarCNN verificado por `verify_checkout()`, informe `/home/bsierra/ltd-ops/sota/s3_reference_20261102T135158Z.json`. Forward con PyTorch **no ejecutado**; entrenamiento **no autorizado**.
- **Dataset C**: `C_HALFSPLIT_STATUS=RECENT_PAIRED_HOUR_CONFIRMED_QA_ONLY`; cron `:00`/`:30` exacto, daemon activo, cero `blocked` / `overlaps_skipped` / errores. 2026-11-02 12 UTC: **120/120 intentos, 66/120 QA válidos**. 13 UTC: **120/120, 71/120 válidos** (36/60 primera mitad, 35/60 segunda). **240 visitas intentadas y 137 PCAP válidos en esas dos horas**; no confundir con 137 sitios únicos: el inventario sigue siendo 120 posiciones, 118 URLs únicas. Informes de primera/segunda mitad bajo `/data/ltd-c-qa-bsierra/C_DIRECT_V1/diagnostics/direct120_half_*/half60_summary.json`. Auditoría `/home/bsierra/ltd-ops/dataset-c-direct120/audit_halfsplit_20261102T135133Z.json`.
- **Disco**: `C_DISK_FREE_GIB=577.6` y `C_DISK_RESERVE_HEALTHY=True`. Reserva 120 GB, sin limpieza destructiva/retención automática. **NTP continúa `no`**; `SCIENTIFIC_CAPTURE=NOT_AUTHORIZED` sin cambio. La hora usada en los informes depende del host y no constituye aún timestamp científico validado.

### Evidencia de semántica temporal del extractor

La función `process_pcap_with_features` en `src/data/01_preprocesamiento.py` utiliza `PcapReader` de Scapy, filtra paquetes TCP/IPv4, preserva el orden del PCAP, obtiene `float(pkt.time)`, lo alinea con dirección/tamaño para cada paquete aceptado y posteriormente aplica `times = [float(t-t0) for t in times]` con `t0=times[0]`; serializa `direction_vector`, `time_vector` y `size_vector` en el mismo registro `pcap_uid`. Por ello **el código conservado documenta unidades relativas en segundos y origen temporal 0**, coherentes con la auditoría de seis muestras. **Limitación:** falta demostrar que la versión de ese script fue precisamente la que produjo los CSV congelados; no basta examinar el HEAD actual para probar reproducibilidad histórica del extractor. Comprobar su linaje y eventualmente valores PCAP ↔ CSV con evidencia inmutable.

WFlib VarCNN `DT2` espera dirección y diferencias de tiempos entre paquetes, obtenidas por `sign(X)` y `diff(abs(X))` con reducción de diferencias negativas a cero. Como nuestros tiempos son relativos y empiezan en 0, formar `X_i = d_i*t_i` perdería la **primera dirección** al evaluar `sign(0)`. Una representación equivalente es `X_i = d_i*(t_i+1)`: preserva todos los signos y deja los `diff(abs(X))` invariantes. La traducción debe validarse **antes de usar VarCNN**, en float64 previo a la conversión a tensor float32 y con tests de longitudes 5000/5001 para el último IAT tras truncamiento. La auditoría de seis muestras no acredita todavía el conjunto de 57.916 registros.

## 7. Avance de ingeniería posterior a validación (código publicado, pruebas Perseo pendientes)

Se incorporó una **representación temporal WFlib-DT2 desde Historical DEV** sin duplicar WFlib en Git y sin modificar los vectores congelados:

- `src/models/sota/v1/timing_adapter.py` (`befd61a1`): interpreta listas de tiempos relativos en segundos y direcciones ±1, valida longitudes/monotonía/valores finitos, produce `(direction, interarrival)` con **2 canales × 5000** float32, calculando IAT en float64 y rellenando/truncando a la derecha. Reproduce `sign(d*(t+1))=d` y `diff(abs(d*(t+1)))=diff(t)`; corrige pérdida del primer signo si `t0=0`. Para **≥5001** paquetes usa el timestamp 5001 para el IAT de índice 4999, igual que `WFlib` (que hace `diff` antes de `length_align`). El adaptador **rechaza** duraciones negativas o conteos desalineados, no inventa datos temporales. Es una equivalencia de *transformación de datos*, no prueba de equivalencia de origen de adquisición.
- `tests/test_sota_s3_timing_adapter.py` (`e8190d4`): **12 tests nuevos** para dirección del primer paquete, correspondencia `DT2` con timestamp firmado +1, umbrales de longitud 5000/5001, padding, rechazo de tiempos inválidos/NaN/infinito y prohibición de sintetizar tiempo desde tamaños o direcciones. **Aún no reportados como ejecutados en Perseo**.
- `scripts/sota/s3_time_channel_audit_v1.py` (`8bb8a0a`): amplió auditoría sobre hasta 2–5 capturas por partición DEV y deriva `DT2` real. Emite `SOTA_S3_DT2_TENSOR_SHAPE [2,5000]` y `SOTA_S3_DT2_SMOKE_SAMPLES`. No habilita entrenamiento automáticamente.
- `scripts/sota/s3_runtime_resources_v1.py` (`c9818bf` y `cb5409b`): auditor **solo lectura** de intérprete Python/venv, paquetes ML, `nvidia-smi` con GPU/VRAM/controlador, RAM, CPU, disco libre y reserva operativa antes de planificar entrenamiento sin afectar la captura C. Sin instalaciones, inferencia ni modificaciones al cron.

**Secuencia de ejecución recomendada en Perseo:**

```bash
cd ~/ltd-attack
git fetch origin feature/macro-v2-historical-only && git merge --ff-only FETCH_HEAD
python3 -m py_compile src/models/sota/v1/timing_adapter.py \
  scripts/sota/s3_time_channel_audit_v1.py \
  scripts/sota/s3_runtime_resources_v1.py \
  tests/test_sota_s3_timing_adapter.py
python3 -m unittest discover -s tests -p 'test_sota_s3_timing_adapter.py' -v
ionice -c3 nice -n 10 python3 scripts/sota/s3_time_channel_audit_v1.py --per-split 5
python3 scripts/sota/s3_runtime_resources_v1.py --save
```

**Puertas restantes:** 1) 12 tests de DT2 y muestra extendida en Perseo; 2) hardware GPU/VRAM/driver, Python ML aislado; 3) **forward sintético real** de DF/VarCNN con PyTorch `--forward-smoke`; 4) validación exhaustiva de 57.916 tiempos/UIDs integrada en extracción para entrenamiento; 5) revisión científica de representación/origen y paridad A/B; 6) declarar y congelar semillas, budget, early stopping y orden de probabilidades/etiquetas. **Nunca ejecutar batch hasta cerrar estas puertas.**

## 8. Nuevo incidente reproducible S3: timestamps no monótonos (2 de noviembre)

**Resultado observado y aportado por Perseo (antes de estas correcciones):**

- Tras actualizar a \`aa4fcaa5\`, se ejecutó \`python3 -m unittest discover -s tests -p 'test_sota_s3*.py' -v\`: **25/25 PASS** (13 pruebas de arquitectura/operación + 12 originales de DT2). No hubo entrenamiento ni acceso Future-B.
- La ampliación \`scripts/sota/s3_time_channel_audit_v1.py --per-split 5\` **falló con** \`SOTA_S3_TIME_AUDIT_ABORTED ValueError time_vector not monotonic\`. Por el fallo y el encadenamiento \`&&\`, **no se ejecutó** en ese bloque \`scripts/sota/s3_runtime_resources_v1.py --save\`; el estado GPU/RAM/PyTorch continúa pendiente.
- \`timedatectl status\` de Perseo: \`System clock synchronized: no\`, \`NTP service: active\`, reloj de host 2026-11-02T15:05:45+01:00. El servicio activo **NO** significa reloj sincronizado; hay que consultar administrador para corregir origen NTP/red/offset. Dataset C sigue QA_ONLY, independientemente de la batería SOTA.

**Diagnóstico de raíz:** el extractor registrado en \`src/data/01_preprocesamiento.py\` conserva el orden físico de paquetes TCP en el PCAP y extrae \`float(pkt.time)\`, luego hace \`t_i-t_0\`. Un PCAP puede presentar marcas temporales regresivas en ese mismo orden; una muestra no monótona no significa por sí sola un fallo de alineación dirección↔tiempo. La implementación **real fijada** de WFlib para DT2 utiliza \`np.diff(np.abs(X))\` y convierte **intervalos negativos en cero** sin reordenar paquetes; la versión inicial del adaptador LTD era más restrictiva e interrumpía ante una regresión.

**Decisión técnica correctiva, aún pendiente de comprobar en Perseo:** mantener fuentes Historical inmutables, no ordenar/truncar/eliminar PCAP por esta causa, usar \`max(0, t[i+1]-t[i])\` para la segunda entrada (semántica WFlib DT2) y **registrar explícitamente** número total de retrocesos, retrocesos dentro de la ventana de 5000 IAT y magnitud máxima por muestra. Si hay tiempos relativos anteriores a \`t0\`, el modelo sintético de timestamp firmado usa un desplazamiento positivo por traza para mantener dirección y conservar diferencias; las entradas calculadas directamente no cambian el tiempo físico. Una decisión posterior sobre legitimidad científica de ese canal necesita **auditoría de cohorte completa, frecuencia/magnitud y linaje exacto del extractor congelado**. **No declarar fidelidad original o integridad del 100% del canal temporal solo por pasar 15 muestras**.

**Cambios realizados en GitHub:** \`src/models/sota/v1/timing_adapter.py\` (\`001784d3\`), pruebas actualizadas \`tests/test_sota_s3_timing_adapter.py\` (\`02f90aab\`, \`c07008ab\`; ahora 14 tests DT2), y \`scripts/sota/s3_time_channel_audit_v1.py\` (\`2e34904b\`) para preservar diagnósticos y no bloquear por regresiones explicables por la semántica original de WFlib. Repetir validación: 13 + 14 = **27 pruebas** S3, luego \`--per-split 5\`, después auditor GPU/recursos **por comando independiente** para que errores en un control no oculten los otros.

**Estado al publicar esta nota:** implementación corregida y revisada en GitHub, pero **todavía sin ejecución runtime validada de la corrección en Perseo**. Entrenamiento DF/VarCNN no autorizado, 09B/10B conservados.

## 9. Validación S3 sobre Perseo y siguientes gates (2026-11-02)

**Resultados reales comprobados por terminal del investigador:** 27/27 tests S3 PASS. Las 15 muestras Historical DEV (5 por partición) mantuvieron alineación dirección-tiempo. Hubo 1 muestra con 29 retrocesos temporales en la ventana DT2, máximo 3.886222839355469e-05 segundos (38,86 microsegundos). Se contabilizan y se convierten a IAT=0 según la transformación fijada WFlib; los PCAP y CSV originales no se reordenan, eliminan ni alteran. Estos 15 ejemplos no demuestran integridad de 57.916 muestras.

**Hardware real:** 16 CPU lógicas, RAM disponible 30014960 KiB, RTX 5080 con 16303 MiB VRAM (15827 MiB libres) y driver 595.99.02; Python sistema sin torch ni NumPy. Disco /data 577.4 GiB libres, reserva de 120 GB satisfecha. Reporte operativo: /home/bsierra/ltd-ops/sota/s3_runtime_resources_20261102T141746Z.json. Ningún entrenamiento iniciado.

**Dataset C QA:** 12:00 y 13:00 UTC dos rondas horarias completas (240 intentos, 137 PCAP QA válidos, no 137 sitios distintos); se verificaron cero omisiones por solapamiento. La ronda de primera mitad de 14:00 produjo 36/60 PCAP válidos; la segunda mitad de esa hora no había finalizado en la última auditoría. NTP sigue no sincronizado: no promover Dataset C a observaciones científicas.

**Nuevo gate exhaustivo:** scripts/sota/s3_full_dev_timing_audit_v1.py (commits 10cf5219 y b73ee3c9) procesa por streaming exactamente las 57.916 capturas de Historical DEV del índice S1. Produce estadísticas de retrocesos y magnitud por fecha/clase/partición y cuenta diferencias negativas dentro/fuera de 5000 posiciones. Rechaza UIDs duplicados, entradas temporales malformadas y desajustes dirección-tiempo. Nunca abre Future-B y no cambia datos. Tests sintéticos: tests/test_sota_s3_full_timing_audit.py (commit 1191f40e), ocho casos. Ejecutar primero con --max-uids 150 y, solo si pasa, con --max-uids 0; modo completo con ionice -c3 nice -n 10 y --save. Informe bajo ~/ltd-ops/sota. Cualquier fila inválida mantiene bloqueado VarCNN.

**Entorno GPU:** PyTorch publica torch==2.11.0 para Linux CUDA 12.8, instalación oficial desde https://download.pytorch.org/whl/cu128 (https://pytorch.org/get-started/previous-versions/). Proponemos venv aislado en ~/ltd-ops/sota/venv-cu128, previa comprobación de 20 GiB libres en su partición. Instalar únicamente torch==2.11.0 desde índice oficial y numpy==2.2.6, sin sudo ni modificar Python de sistema, repositorio o contenedores. Compatibilidad de RTX 5080 se aceptará solo tras ejecutar CUDA y forward sintético de los dos modelos. El nuevo scripts/sota/s3_cuda_synthetic_preflight_v1.py (commit 04634717) comprueba GPU SM120 y logits finitos [2,65] para DF [2,1,5000] y VarCNN [2,2,5000] sobre 2 entradas artificiales, sin entrenar, sin tocar Historical/09B/10B/C, guardando informe fuera de Git.

**Pendientes antes de batería:** completar full DEV timing audit; forward CPU/CUDA; declarar semillas, duración, selección cronológica y presupuesto GPU; cotejar el orden de 65 clases con etiquetas de los scores congelados solo en gate de evaluación; comparar Top-1, Macro-F1, Top-5, MRR sobre mismos UIDs, IC95 pareados por día y Holm con familia predeclarada. La asignación original 09A está ausente y la comparación con el split reconstruido sigue pendiente. La ejecución de esta fase no autoriza entrenamiento ni evaluación Future-B.

**Trazabilidad adicional del gate S3:** script de auditoría de todos los tiempos históricos scripts/sota/s3_full_dev_timing_audit_v1.py, creado en 10cf5219, corrección forense b73ee3c9 y comprobación final SHA256 6540b279; ocho pruebas sintéticas tests/test_sota_s3_full_timing_audit.py commit 1191f40e. Smoke de inferencia GPU RTX 5080 sobre datos exclusivamente sintéticos scripts/sota/s3_cuda_synthetic_preflight_v1.py, commit 04634717. Procedimiento reproducible de entorno virtual PyTorch 2.11.0 CUDA 12.8 y evaluación CPU/CUDA en docs/sota/SOTA_S3_RUNTIME_GPU_V1.md, commit 038fcd5a; enlazado al protocolo canónico SOTA. **Estado: archivos publicados y pendientes de ejecución en Perseo; no declarar PASS ni iniciar training antes de resultados reales.**

## 10. Validación exhaustiva real 57.916 DEV y defecto unitario corregido (2026-11-02)

**Evidencia Perseo aportada por el investigador:**
- Checkout de rama actualizado a commit base `0673aec`; `py_compile` PASS, **34/35 unit tests S3 PASS** y **una prueba FAIL**: `test_nonfinite_timestamp_is_invalid`, porque el método puro `summarize_trace([1,-1],[0.,float("nan")])` no rechazaba NaN. Se detectó inconsistencia defensiva entre `numbers()` (sí rechazaba no finitos al leer CSV) y `summarize_trace()` (no); no es evidencia de que existan NaN en Historical.
- **Auditoría completa** en `/home/bsierra/ltd-ops/sota/s3_full_time_quality_20261102T143857Z.json`: `PASS_FULL_POPULATION`, `57916/57916`, 19098 Early, 20160 Middle, 18658 Late, `SOURCE_SHA256=PASS`, `INVALID=0`, `NEGATIVE_TRACES=2209`, `NEGATIVE_STEPS=195691`, `CLIPPED_IN_5000=145589`, mayor retroceso `0.000244140625` segundos (244.14 µs). Aproximadamente **3.81 % de las capturas** presenta uno o más retrocesos; los intervalos negativos se mapean a cero conforme a WFlib DT2 sin modificar fuentes. **Estos datos ya demuestran la cobertura completa en el pipeline CSV**, pero el conjunto de pruebas de seguridad de la función pura debía corregirse antes de aprobar la compuerta general.
- Prueba limitada `150/57916`, `150` válidos, `7` trazas con retrocesos, `75` pasos negativos; solo Early 74 y Late 76 porque `--max-uids` limita el orden del archivo y no realiza muestra representativa ni estratificada. **No extraer inferencias de esa muestra**.
- Montajes: raíz `/dev/nvme0n1p1` con **38 GB libres, 96 % utilizado**, y `/data` con **578 GB libres, 78 % utilizado**. **No instalar CUDA/PyTorch en HOME o /tmp de /**; usar directorio aislado `/data/ltd-sota-bsierra` con TMPDIR en ese mismo montaje y comprobación de reserva de disco, y nunca limpiar proyectos de terceros ni datos científicos.

**Correcciones GitHub posteriores a esta observación:** `scripts/sota/s3_full_dev_timing_audit_v1.py` commit `87709efe`: `summarize_trace()` valida tipos y `math.isfinite` para todos los timestamps; además, la auditoría ahora puede generar el informe forense completo aunque haya filas inválidas, bloqueando expresamente la liberación. `tests/test_sota_s3_full_timing_audit.py` commit `ef6f3c68`: dos casos nuevos que comprueban infinito, booleanos y texto. Resultado esperado **37/37 tests**, aún **PENDIENTE de confirmación en Perseo**. En el pase anterior la lectura de Historical ya aplicaba `numbers()` y rechazaba no finitos antes del cálculo; ninguna evidencia de modificación de cohortes o métricas. Repetir tests y corrida full bajo commit corregido para cerrar trazabilidad.

**Decisión de avance:** SOTA preparación S3 aceptable a nivel datos y linaje; **NO liberar entrenamiento** hasta 37/37 PASS, auditoría total de código corregido con SHA, y dos forwards sintéticos con PyTorch. El runbook GPU corregido es `docs/sota/SOTA_S3_RUNTIME_GPU_V1.md`, commit `52b76a78`, con montaje `/data` separado de Dataset C. No acceder a Future-B ni ajustar hiperparámetros a 09B.

## 11. Cierre confirmado de auditoría completa Historical DEV — 2026-11-02

**Fuente de verdad: salida completa de Perseo aportada por el investigador (sustituye el registro inicialmente truncado).** Bajo commit local `e48934a`, `python3 -m py_compile` pasó y `python3 -m unittest discover -s tests -p 'test_sota_s3*.py' -v` finalizó **37/37 OK**. El mismo bloque continuó y ejecutó íntegramente `scripts/sota/s3_full_dev_timing_audit_v1.py --max-uids 0 --save`, con salida:

- `SOTA_S3_FULL_SOURCE_SHA256=PASS`, `SOTA_S3_FULL_STATUS PASS_FULL_POPULATION`.
- `SOTA_S3_FULL_UIDS 57916/57916`, particiones `DEV_EARLY=19098`, `DEV_MIDDLE=20160`, `DEV_LATE=18658`, `SOTA_S3_FULL_TEMPORAL_VALID 57916`, `SOTA_S3_FULL_INVALID 0`.
- `SOTA_S3_FULL_NEGATIVE_TRACES 2209` (3,81 % de DEV), `SOTA_S3_FULL_NEGATIVE_STEPS 195691`, `SOTA_S3_FULL_CLIPPED_IN_5000 145589`, retroceso máximo `0.000244140625` segundos (244,14 µs). Son **eventos de intervalos negativos, no capturas descartadas**. Conservan orden de paquetes y siguen política explícita WFlib DT2.
- `FUTURE_B_UNTOUCHED=YES`; reporte de evidencia `/home/bsierra/ltd-ops/sota/s3_full_time_quality_20261102T145706Z.json`.

**Decisión:** gate de integridad temporal de Historical DEV **CERRADO/PASS para los datos y el código de auditoría vigente**. No se requiere ejecutar nuevamente esta auditoría salvo cambio de fuentes, herramientas o necesidad de nueva evidencia. Continúan gates independientes: (a) entorno PyTorch aislado en /data, (b) forward sintético DF/VarCNN en CPU y RTX 5080, (c) presupuesto/semillas/validación cronológica, (d) mapeo exacto de 65 labels para evaluación, (e) cotejo del CSV original de particiones 09A si reaparece. **Ningún entrenamiento autorizado todavía.** La raíz de Perseo tiene 96 % de uso (38 GB libres); el montaje /data reportó 578 GB libres, suficiente para estudiar instalación aislada sujeto a permiso y reserva. Dataset C QA y sus cron se mantienen intactos.

## 12. Rectificación de arquitectura, ocupación del host y cierre de forwards (2026-11-02)

**Evidencia de Perseo aportada en archivo de terminal:** se completó la instalación de PyTorch 2.11.0+cu128, NumPy 2.2.6, bibliotecas CUDA 12.8 en `/data/ltd-sota-bsierra/venv-cu128`, tras verificar almacenamiento 576,9 GiB libres en /data. CPU `SOTA_S3_FORWARD_PASS DF` (3.699.393 parámetros) y `VarCNN` (8.806.209 parámetros). GPU `SOTA_CUDA_SMOKE=PASS_NO_TRAINING`, NVIDIA RTX 5080 `compute capability (12,0)`, logits finitos [2,65] para ambos, con [2,1,5000] DF y [2,2,5000] VarCNN; `training_steps=0`. Informes `~/ltd-ops/sota/s3_reference_20261102T151428Z.json` y `~/ltd-ops/sota/s3_cuda_smoke_20261102T151430Z.json`. **Integridad S3 data, contrato 37/37 y forward CPU/CUDA: comprobados.** No se autoriza todavía entrenamiento porque falta protocolo temporal/configuración.
 
**Corrección conceptual inmediata:** la ocupación **96 %** informada por `df -h` pertenece a la partición de disco del sistema `/`, **no** a la RTX 5080. Un muestreo anterior `nvidia-smi` mostraba utilización GPU **0 %**, VRAM libre 15.827/16.303 MiB; no permite saber qué procesos/usuarios usan la GPU *ahora*. Solo `nvidia-smi` con procesos y consulta de sus PID en tiempo real puede determinarlo. No afirmar actividad ajena sin evidencia.

**Instrucción arquitectónica expresa del investigador:** LTD-Attack debe ser el **único paraguas** de código, dependencias y documentación en `/home/bsierra/ltd-attack`; `/data` **solo** para capturas, datos, checkpoints y resultados, no para código ni virtualenv. La creación de `/data/ltd-sota-bsierra/venv-cu128` fue una decisión inadecuada: existe como **excepción temporal** y ha pasado los tests, pero **no se duplicará, migrará ni eliminará** sin inventario físico de discos, entornos existentes y llamadas de código que dependan de rutas. WFlib fijado existe además bajo `~/ltd-ops/sota/upstream/WFlib`; informes bajo `~/ltd-ops/sota`; Dataset C QA usa `~/ltd-ops/dataset-c-direct120` para logs operacionales. `configs/environments/perseo.yaml` aún apunta a `~/ltd-storage`; verificar si es symlink/montaje antes de asumir ubicación. El repositorio README y `docs/sota/SOTA_S3_RUNTIME_GPU_V1.md` quedaron corregidos a esta política y a un diagnóstico **de solo lectura**. No modificar cron, Docker, ruta de Dataset C ni los datasets A/B/09B/10B/C.

**Estado físico pendiente:** cuánto ocupa el entorno instalado, si existe un virtualenv previo compatible bajo `~/ltd-attack`, si el HOME puede alojar sin riesgo ese entorno considerando / al 96 %, si hay procesos GPU de otros usuarios y cuáles de las carpetas preexistentes son enlaces. **No avanzar hacia entrenamientos, nuevas instalaciones o consolidaciones destructivas** hasta recoger ese inventario. El hecho de pasar los forwards no prueba reproducibilidad de entrenamiento ni equidad del benchmark.

## 13. Rectificación de inventario: entorno LTD preexistente (2026-11-02)

**Nueva evidencia visual aportada por el investigador:** la carpeta `/home/bsierra/ltd-venv` ya existía antes del nuevo entorno SOTA y contiene `bin`, `include`, `lib`, `lib64` y `pyvenv.cfg`. No se descubrió en el primer barrido porque el comando `find "$HOME/ltd-attack" -maxdepth 4 -name pyvenv.cfg` solo recorría el directorio del repositorio y no sus hermanos. La afirmación anterior de que no había ningún entorno LTD reutilizable **carecía de fundamento** y queda rectificada aquí.

**Instrucción efectiva de consolidación:** primero inventariar versiones de Python, torch/CUDA y demás paquetes de `~/ltd-venv`, sin escribir ni modificar el entorno existente. No crear otro virtualenv, no actualizar paquetes del entorno histórico a ciegas, no duplicar PyTorch ni mover/eliminar el entorno SOTA en `/data` hasta comprobar compatibilidad, uso por otros scripts y estrategia segura. El SOTA GPU ya pasó CPU/CUDA sintético, y no es necesario repetirlo únicamente por descubrir esta carpeta. El paraguas de **código y documentación** continúa siendo `~/ltd-attack`, con **un entorno del proyecto preexistente** en `~/ltd-venv`; `/data` se destina exclusivamente a datos, capturas y resultados.

Los documentos README y `docs/sota/SOTA_S3_RUNTIME_GPU_V1.md` incorporan esta corrección y un comando de auditoría de solo lectura. **Datos aún desconocidos:** tamaño de `~/ltd-venv`, versiones de Python/torch instaladas y qué procesos LTD lo usan. No declarar que este venv sea compatible hasta inspeccionarlo.

## 14. Inventario exacto del entorno preexistente de LTD (Perseo, 2026-11-02)

Resultados de comando **de solo lectura** compartido por el investigador: `cat ~/ltd-venv/pyvenv.cfg` confirma que el entorno nació con `/usr/bin/python3.12` (Python 3.12.3, `include-system-site-packages = false`). `du -sh` registra **167 MB en `/home/bsierra/ltd-venv`**, frente a **6,7 GB en `/data/ltd-sota-bsierra/venv-cu128`**. El primero tiene **NumPy 2.5.4**, **Pandas 3.1.0**, y **no tiene PyTorch, SciPy ni scikit-learn**; PyTorch CUDA 2.11 ya funciona solo en el segundo. No hay dos copias de PyTorch en esta evidencia; sí dos entornos distintos, uno de ellos creado innecesariamente en el volumen destinado a datos.

**Decisión:** no instalar torch en el entorno original de LTD hasta auditar espacio real (raíz 96 %, 38 GB libres) y sus usos; no copiar ni mover un venv existente entre particiones; no crear más venv ni borrar el CUDA temporal. Priorizar el inventario físico de `~/ltd-storage` y HOME y la atribución de consumo, con `ionice -c3 nice -n 10 du -xhd1`; mantener intactos Docker, cron y capturas Dataset C, así como splits/manifiestos SOTA. Consultar `docs/operations/LTD_PERSEO_LAYOUT_CONSOLIDATION_20261102.md` para el siguiente diagnóstico. **Los gates científicos y de GPU de S3 ya están PASS; el bloqueo actual es organizacional + protocolo de entrenamiento, no rendimiento de GPU.**

## 15. Decisión acelerada de organización (Perseo, 2026-11-02)

Inventario de ocupación recibido: `~/ltd-storage=27G` (datasets `26G`, artifacts 694M, results 178M, audits 158M), `~/ltd-attack=35M`, `~/ltd-ops=4M`, `~/ltd-venv=167M`, `~/ltd-dataset-c-recovery=206M`. La raíz tiene 38G libres (96 % ocupada) y /data 571G libres (79 % ocupado). El directorio LTD `~/ltd-attack` está limpio (`git status --short` vacío); no hay procesos referenciando `~/ltd-venv` en el muestreo. **No se ha causado este uso de 96 % por SOTA ni GPU**; gran parte de los datos de LTD siguen guardados en el volumen del sistema.

**Decisión:** en vez de una nueva instalación o más inventarios repetidos, se añadió bajo el único paraguas LTD `scripts/ops/stage_historical_storage_perseo_v1.py`, que copia `~/ltd-storage` a `/data/ltd-attack/storage`, conservando la fuente. Realiza una comprobación de diferencias completa con `rsync --checksum --dry-run` (sin cambios al cron o rutas preexistentes) y reserva de /data. Código publicado; **no ejecutado todavía en Perseo**. Cualquier cambio de nombres, links o borrado para liberar espacio del origen exige PASS de esa verificación y plan de rollback. Runbook: `docs/operations/LTD_PERSEO_LAYOUT_CONSOLIDATION_20261102.md`. El entorno CUDA 6,7G en /data se mantiene intacto por ahora; no crear ni instalar otro. Integridad y forwards SOTA S3 siguen PASS, entrenamiento aún no autorizado.

## 16. Almacenamiento histórico staged y cutover reversible preparado (2026-11-02)

**Perseo real:** 28.138.252.513 bytes de ltd-storage copiados a /data/ltd-attack/storage; 457 entradas (348 regulares, 99 directorios, 10 enlaces), comparación checksum dry-run PASS_IDENTICAL_CHECKSUM_DRY_RUN. /data tiene 543,8 GiB libres después de la copia. Fuente histórica conservada, sin cambios de cron, entornos ni repositorio. Informe de la comprobación: /data/ltd-attack/storage/.ltd_storage_verify_v1.txt.

**GitHub posterior a ese resultado:** script nuevo scripts/ops/cutover_historical_storage_perseo_v1.py + tests/test_ltd_storage_cutover_v1.py (6 tests sintéticos) para fase de cambio de ruta reversible: revalidación checksum, preservación íntegra del original bajo HOME como ltd-storage.precutover-TIMESTAMP, symlink histórico hacia /data, validación y rollback condicionado a ausencia de escrituras nuevas. No se ejecutó el cutover en Perseo todavía. Los tests requieren ejecución real; sin esa evidencia no declarar PASS.

**Punto crítico:** la fase 2 NO libera espacio de la partición raíz. El directorio original de 27 GB permanece en el mismo filesystem hasta la fase 3, que requiere comprobaciones de funcionamiento y conservación de un plan de restauración. Tampoco se actualiza perseo.yaml, la captura Dataset C, el cron, los modelos, las métricas o ningún entorno virtual. SOTA aún sin entrenamiento por condiciones científicas pendientes. Runbook: docs/operations/LTD_PERSEO_LAYOUT_CONSOLIDATION_20261102.md.

## 17. Rectificación de bloqueo de cutover: diferencia no científica de mtime

El investigador reportó 6/6 unit tests de almacenamiento PASS, pero `--activate` bloqueado sin cambios por la única línea rsync `.d..t...... ./` (exit 0). Es la fecha de modificación del directorio raíz, afectada por los reportes de staging; las dos rutas históricas aún apuntaban a HOME y los CSV y configuración funcionaban. No hubo cutover ni borrados. La corrección mínima `material_differences` solo ignora esa línea exacta, dejando todas las demás diferencias como bloqueos. Se agregaron 4 tests, total 10. Los nuevos tests y la nueva activación aún **no han sido ejecutados en Perseo**, por lo que el estado es **CUTOVER_PENDING**. El SOTA S3 de datos y GPU sigue verificado; los entrenamientos todavía no comienzan. Tras un cutover satisfactorio, volver al diseño de protocolo S4 en lugar de prolongar la reorganización.

## 18. Retorno al objetivo científico: SOTA S4 ex ante (2026-11-02)

A petición del investigador, la reorganización de almacenamiento se **pausa**: la copia staging Historical en `/data/ltd-attack/storage` está validada con checksum, pero el cutover `~/ltd-storage` **no se completó** (bloqueo inocuo por mtime de directorio raíz). Se corrigió la comparación en GitHub y se agregaron tests, pero **no se han ejecutado ni la corrección ni el cutover en Perseo**; las rutas originales siguen operativas. No borrar backups ni bloquear la ingeniería científica por este asunto.

**Entregables S4 publicados en GitHub** (aún sin prueba de ejecución en Perseo):

- `configs/sota/sota_s4_frozen_protocol_v1.json`: protocolo **propuesto ex ante** de DF-WFlib-pinned y VarCNN-WFlib-DT2, una semilla principal 42, máximo 30 épocas, LR y batch exactos de `WFlib/scripts/DF.sh` y `Var-CNN.sh`. Selección de checkpoint por `Macro-F1` de 65 clases sobre DEV_LATE tras entrenar en DEV_EARLY+DEV_MIDDLE, y refit en todo DEV con número seleccionado; diagnóstico Early→Middle. Sin ajuste por scores externos.
- `src/models/sota/v1/s4_protocol.py`: validador estricto de ventanas cronológicas, clases, presupuesto, optimizador, salida solo a /data, fuga de cohortes, y elección determinista de época (máximo Macro-F1, desempate por época menor).
- `tests/test_sota_s4_protocol_v1.py`: 12 tests stdlib, incluidos escenarios de contaminación de Future-B y solapamiento train/valid, recomputación Macro-F1 sobre las 65 clases y población S1 simulada de 57.916 UIDs.
- `scripts/sota/s4_protocol_preflight_v1.py`: verifica SHA y metadatos de S1, fechas y contrato, sin leer vectores ni datos externos; informe pequeño en `/data/ltd-attack/sota/protocol`, declara `ready_to_train=False`.
- `scripts/sota/s4_synthetic_optimizer_smoke_v1.py`: una actualización de pesos con **dos ejemplos artificiales** y un solo paso sintético por DF/VarCNN, para comprobar gradientes y optimizador en la RTX5080 usando el venv CUDA ya comprobado. **No es entrenamiento sobre Historical**; no genera checkpoint científico.
- `docs/sota/SOTA_S4_EXANTE_TRAINING_V1.md`: runbook, decisiones, salvedades de comparabilidad y próximos gates. Vinculado desde el protocolo SOTA común.

**Estado al finalizar este cambio:** S4 **implementado en su fase de protocolo y prueba sintética, pendiente de PASS Perseo**. No se han lanzado entrenamientos ni abierto INTERNAL_TEST, Future-B o Dataset C. Tras recibir esos PASS, el siguiente desarrollo será el motor de entrenamiento Historical DEV A/B/C con pesos/checkpoints SHA congelados, sin selección de hiperparámetros posterior. Orden de etiquetas de 65 puntuaciones aún pendiente del gate de evaluación G2 y ausencia de CSV 09A original documentada. Todo código queda bajo el único paraguas `~/ltd-attack`; no instalar más entornos.
