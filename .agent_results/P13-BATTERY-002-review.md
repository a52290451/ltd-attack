# P13-BATTERY-002 — Revisión pre-commit

Fecha: 2026-10-04  
Repositorio: ~/dev/ltd-attack  
Rama: feature/macro-v2-historical-only  
Commit base: a8ca9a9 evidence: archive frozen temporal Micro ensemble 13A

## Veredicto

**NO LISTO PARA COMMIT NI PREFLIGHT DE ZEUS.**

La matriz científica está correctamente predeclarada, pero hay hallazgos
BLOCKER/HIGH en alineación y procedencia de caché. Ejecutar Zeus antes de
corregirlos podría producir ensembles plausibles asociados a capturas,
ventanas u origins incorrectos.

Esta revisión fue de solo lectura. No se modificaron código, configuración,
tests ni documentación de P13-BATTERY-001.

## Hallazgos

### BLOCKER — B-01: 13B no verifica la alineación de los nueve expertos

Evidencia: src/experiments/phase13_battery.py:330-340 y :493-505.

Cada NPZ guarda probs, y_true, dates y un fingerprint global, pero no pcap_uid,
origin, modo, seed ni ventana. Al construir U3, T3 y T9 se toman y y dates de
UNIFORM/42, sin comprobar que los otros ocho expertos tengan exactamente las
mismas filas y el mismo orden. Un archivo de igual shape y fingerprint puede
ser reordenado o copiado desde otro artefacto y entrar en la media geométrica.

Corrección propuesta: guardar pcap_uid y la identidad completa
origin/window/temporal_mode/seed en cada NPZ; validar igualdad exacta de
pcap_uid, fechas y etiquetas para todos los expertos antes de derivar un
candidato. Cualquier diferencia debe producir error, sin reordenamiento
silencioso. Añadir un test con filas permutadas.

### HIGH — B-02: 13C valida etiquetas, pero no identidad de captura

Evidencia: src/experiments/phase13_battery.py:657-665.

La comprobación Micro/Macro solo compara longitud y y. No compara pcap_uid ni el
orden de las fechas. Dos capturas distintas del mismo sitio pueden ocultar una
desalineación.

Corrección propuesta: conservar la lista de pcap_uid de cada ventana en los
resultados Micro y Macro y exigir igualdad exacta antes de fuse/Hybrid. Añadir
un test específico de misalignment.

### HIGH — C-01: un caché incompleto se vuelve a entrenar silenciosamente

Evidencia: src/experiments/phase13_battery.py:461-479 y :609-612.

Cuando --resume encuentra solo parte de los artefactos esperados, la condición
de reutilización es falsa y el flujo entra en entrenamiento o reconstrucción.
El requisito exige que un caché incompleto falle explícitamente; la
reconstrucción puede además sobrescribir un directorio parcial. El caché Macro
presenta el mismo patrón.

Corrección propuesta: distinguir ausente de parcial; si existe cualquier
artefacto de una identidad esperada pero falta otro, lanzar RuntimeError con la
identidad y los archivos faltantes. Solo crear un caché cuando no exista ningún
artefacto de esa identidad.

### HIGH — C-02: cobertura de hashes insuficiente

Evidencia: src/experiments/phase13_battery.py:449-452, :599-608 y
src/analysis/future_rank_gap_battery.py:339-355.

El fingerprint 13B no incluye 10A_refit_full_historical.py, 11H,
src/features/macro_v2/_common.py, src/utils/paths.py ni todo el código que
determina carga, fechas y alineación. El manifest 13C tampoco cubre todo el
cierre científico que ejecuta. El manifest 14A solo hashea el NPZ Future-B y
el módulo de análisis, aunque usa las tres fuentes CSV de tamaño/drift y el
selector BASE-128 07A. Además, el manifest 14A no registra git_commit.

Corrección propuesta: definir por etapa una lista explícita de todos los
módulos científicos, utilidades y entradas que afectan el resultado; guardar
sus SHA256, incluir las fuentes auxiliares de 14A y registrar siempre el
commit Git. El mismo conjunto debe formar el fingerprint de resume.

### HIGH — C-03: model.pt se exige por existencia, pero no se valida

Evidencia: src/experiments/phase13_battery.py:461-479.

En --resume el checkpoint solo debe existir. No se comprueba SHA256,
fingerprint interno, origin, modo, seed, scaler/min-max ni arquitectura.
Tampoco se carga: la inferencia procede únicamente de los NPZ. Un checkpoint
corrupto o intercambiado puede pasar mientras el manifest coincida.

Corrección propuesta: registrar hash del checkpoint y scaler, validar metadatos
internos completos y rechazar discrepancias. Si los NPZ son la fuente canónica
de resume, debe declararse y validarse igualmente el checkpoint asociado.

### HIGH — C-04: los YAML no son la fuente efectiva de verdad

Evidencia: ambos YAML y las constantes de
src/experiments/phase13_battery.py:27-58 y
src/analysis/future_rank_gap_battery.py:31-43.

Los YAML se leen y hashean, pero la ejecución usa constantes codificadas para
matriz, pesos, epochs, ratio, reglas, componentes, rutas y guardrails. Cambiar
un YAML cambia el hash pero no cambia el experimento ejecutado.

Corrección propuesta: usar un esquema tipado que alimente la ejecución y
valide los guardrails congelados; alternativamente, fallar si el YAML difiere
de las constantes. No aceptar silenciosamente dos fuentes de protocolo.

### MEDIUM — P-01: los artefactos de 13A no son reutilizables de forma segura

La evidencia archivada de 13A contiene CSV y manifest, pero no checkpoint ni
matrices NPZ. El resumen confirma solo seed 42 para EARLY/UNIFORM/LATE en ambos
origins. El dry-run detecta cero modelos reutilizables y planifica 18.

Conclusión: el número seguro de entrenamientos nuevos en Zeus es 18, no 12. Si
existieran artefactos verificables de 13A para los seis expertos seed 42, serían
12 nuevos; con los artefactos presentes no se debe reutilizar nada. No se
propone una conversión insegura.

### MEDIUM — P-02: 13C puede reentrenar Macro aunque exista evidencia histórica

Evidencia: src/experiments/phase13_battery.py:613-636.

Si no existe su caché propio, el runner llama a prepare_origin y
train_multiscale_ltd. La evidencia 11L no contiene checkpoints o matrices NPZ
verificables para este runner. No abre Future-B ni cambia pesos, pero puede
repetir entrenamiento Macro en Zeus.

Corrección propuesta: declarar que Macro se reconstruye bajo implementaciones
congeladas, o crear un adaptador solo con artefactos por origin/ventana y hashes
completos. No tratar CSV agregados como predicciones.

### MEDIUM — C-05: --resume de 13C no reutiliza el resultado factorial

run_13c reutiliza cachés Micro/Macro, pero vuelve a calcular y escribir los CSV
de 13C; no valida ni reutiliza un manifest/output cache 13C existente.

Corrección propuesta: añadir manifest de ejecución 13C con fingerprint de
entradas y outputs, y fallar ante outputs parciales o incompatibles antes de
recalcular.

### MEDIUM — C-06: el dry-run clasifica reutilización por presencia, no por hash

Evidencia: src/experiments/phase13_battery.py:343-369.

Marca un modelo como reutilizable si existen manifest, checkpoint y tres NPZ;
el hash se valida solo durante la ejecución. Puede mostrar reutilizable un
caché incompatible.

Corrección propuesta: calcular compatibilidad en dry-run y distinguir
reutilizable, incompatible, incompleto y ausente.

### MEDIUM — T-01: faltan tests de rutas de fallo críticas

Los 14 tests pasan y cubren media geométrica, pesos temporales, matriz, rank
2..5, entropía, margen, confusion edges, flags Future-B y dry-run. Faltan:

- pcap_uid permutado entre expertos 13B;
- pcap_uid Micro/Macro desalineado en 13C;
- origin/window/seed incorrecto en un caché;
- caché parcial que debe fallar;
- checkpoint incompatible o corrupto;
- M0/M1 y las ocho combinaciones factoriales;
- configuración YAML divergente;
- determinismo del bootstrap.

El test actual de caché solo cubre fingerprint distinto en el manifest.

### MEDIUM — D-01: el bloque nuevo de FEATURE_LINEAGE está en inglés

El texto añadido por P13-BATTERY-001 en docs/FEATURE_LINEAGE.md:2293-2317
está en inglés, aunque la tarea exige que la documentación nueva esté en
español.

Corrección propuesta: traducir únicamente el bloque nuevo F13; no alterar el
contenido histórico anterior.

## Comprobaciones positivas

### A. 13B

- **OK:** exactamente 2 origins × 3 modos × 3 seeds = 18 entrenamientos.
- **OK:** windows y origins coinciden con 13A: ORIGIN14/28 y NEAR/MID/FAR.
- **OK:** EARLY, UNIFORM y LATE parten del mismo micro_df y train_idx.
- **OK:** ratio 4.0; UNIFORM usa unos y EARLY/LATE pesos de entrenamiento.
- **OK:** U3 usa UNIFORM 11/42/73; T3_S42 usa E42/U42/L42; T3_S11 y
  T3_S73 replican; T9 contiene nueve expertos.
- **OK:** media geométrica con pesos iguales; no hay búsqueda de pesos ni
  selección adaptativa.
- **OK:** Accuracy, Macro-F1, Top-5, MRR y mean true rank están implementados;
  Accuracy y Macro-F1 son co-principales en el protocolo.
- **OK:** bootstrap por bloques de día y fraction_delta_gt_0, no p-value.
- **BLOCKER:** la conclusión práctica queda condicionada a B-01.

### B. 13C

- **OK:** exactamente U1/U3/T3_S42/T9 × M0/M1.
- **OK:** Micro=0.45 y Macro=0.55 congelados.
- **OK:** M0 es XGB UNIFORM + RECENT5 LTD; M1 es XGB
  TEMPORAL_SYMMETRIC3 + MULTISCALE5 LTD.
- **OK:** 13C reutiliza el candidate_store de 13B y no reentrena Micro.
- **HIGH:** la alineación efectiva queda afectada por B-02.

### C. 14A

- **OK:** no importa entrenadores, no llama a 10B y falla si falta el NPZ
  10B_scores_DEV_FROZEN.npz.
- **OK:** selection_allowed y training_allowed son siempre false.
- **OK:** usa la fuente Micro Future-B correcta, conserva MAX_LEN=3000 y
  calcula tamaño/truncamiento por captura y sitio.
- **OK:** BASE-128 se escala solo con medianas/MAD Historical; las categorías
  son reglas predefinidas y descriptivas.
- **HIGH:** falta procedencia completa y git_commit en manifest; aplica C-02.

### D. Paths y documentación

- **OK:** los archivos nuevos usan src.utils.paths; no tienen rutas absolutas
  de Mac o Zeus.
- **OK:** LTD_ENV=zeus resolvió correctamente destinos de artifacts, results y
  datasets mediante src.utils.paths.
- **OK:** 13A figura archivada, con gate pasado y Future-B unused; están los
  FAR Accuracy solicitados y no hay resultados inventados de 13B/13C/14A.
- **MEDIUM:** el bloque nuevo de FEATURE_LINEAGE incumple idioma; aplica D-01.

## Validación ejecutada

    python3 -m compileall -q src/experiments/phase13_battery.py src/analysis/future_rank_gap_battery.py scripts/train/run_phase13_battery.py scripts/diagnostics/run_future_rank_gap_battery.py tests/test_phase13_battery.py tests/test_future_rank_gap_battery.py
    python3 -m pytest -q tests/test_phase13_battery.py tests/test_future_rank_gap_battery.py tests/test_paths.py
    python3 scripts/train/run_phase13_battery.py --dry-run --only 13B
    python3 scripts/diagnostics/run_future_rank_gap_battery.py --dry-run
    git diff --check
    git status --short

Resultado: **14 passed**. Ambos dry-runs no entrenaron, mostraron los 18
modelos, los seis ensembles Micro, las ocho combinaciones 13C, las rutas y las
flags 14A. El dry-run confirmó cero cachés reutilizables.

La suite histórica completa tiene una limitación local conocida: la colección
incluye src/models/final/09B_evaluate_internal_test.py, que requiere joblib no
instalado. No se instalaron dependencias ni se ejecutaron entrenamientos.

## Correcciones requeridas antes de Zeus

1. Resolver B-01 y B-02 con identidad pcap_uid y tests de permutación.
2. Resolver C-01, C-02 y C-03 antes de habilitar --resume.
3. Hacer los YAML autoritativos o validar explícitamente su igualdad con el
   protocolo ejecutado.
4. Documentar la decisión conservadora de ejecutar 18 entrenamientos Micro; no
   reutilizar 13A sin checkpoint/predicción verificable.
5. Añadir los tests críticos de T-01 y traducir el bloque nuevo de
   FEATURE_LINEAGE.
6. Repetir compileall, tests, dry-runs, git diff --check y revisión de estado
   después de las correcciones.

## Estado Git

No se hizo commit ni push. El reporte es el único archivo creado durante
P13-BATTERY-002; los demás cambios pertenecen a P13-BATTERY-001.

