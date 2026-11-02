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
