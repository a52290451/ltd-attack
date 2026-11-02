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

## 5. Gates aún pendientes

1. **SOTA G2:** cotejar exactamente el orden de 65 `candidate_labels` original 09B/10B con el orden del índice S1 *antes de evaluar*, sin usar etiquetas futuras para ajustar modelos. El `label_order.json` alfabético no es evidencia de identidad; cabe crear mapeo explícito de scores.
2. **SOTA G3:** pruebas de arquitectura WFlib en CPU o GPU, protocolos de entrada DF y VarCNN; validar `time_vector` y el esquema firmado DT2; versión/semillas/presupuesto.
3. **SOTA G4/G5:** presupuesto GPU/CPU y RAM, entorno torch aislado, validación temporal Early→Middle / Early+Middle→Late; no reentrenar ni abrir Future-B.
4. **Dataset C:** esperar informes verdaderos para primera y segunda mitad; verificar timestamps :00/:30, duración por lote, cron daemon activo, brechas por `SKIPPED_OVERLAP`, uso y crecimiento real del almacenamiento. Cron instalado ≠ recolección exitosa. Resolver NTP y paridad de metodología A/B–C antes de elevar QA a ciencia.
5. **Informe de comparabilidad:** para todos los modelos Accuracy/Top-1, Macro-F1, Top-5, MRR sobre los mismos UIDs. Bootstrap pareado por fecha (09B 11 días, 10B 16, 2.500), IC95 percentil, pruebas condicionales de intercambio y corrección Holm con familias declaradas (no confundir Holm9 y Holm12). Future-B ya fue visto: conclusiones retrospectivas.

**Ningún paso de esta fase autoriza entrenar ni acepta datos C como observaciones científicas.** Conservar outputs de scripts localmente y adjuntar un checkpoint posterior con los valores reales para cerrar los gates.
