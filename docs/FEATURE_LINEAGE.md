# LTD-Attack — Feature Engineering Lineage

Última actualización: 2026-09-18

## Objetivo

Este documento registra la genealogía de las representaciones estadísticas
Macro utilizadas en LTD-Attack, incluyendo sus criterios de selección,
scripts productores, uso de datos futuros y validez metodológica.

La finalidad es impedir que resultados pertenecientes a generaciones
experimentales diferentes sean tratados como directamente comparables.

---

## F0 — Extracción estadística inicial

Origen:
- Capturas PCAP de tráfico Tor.
- Vectores de dirección, tiempo y tamaño.
- Extracción de características estadísticas por captura.

Universo inicial documentado:
- 328 características aproximadamente.

Estatus:
- HISTORICAL.
- Debe recuperarse el extractor/productor exacto para completar trazabilidad.

---

## F1 — Reducción inicial

Input:
- 328 características aproximadamente.

Metodología documentada:
- eliminación de colinealidad;
- ranking de importancia;
- eliminación de variables redundantes.

Output:
- 230 características.

Rol:
- Primer conjunto estadístico Macro empleado sistemáticamente.

Resultados históricos asociados:
- Micro/vectorial: 97.26% histórico / 33.08% Future.
- Micro + 230F: 97.01% histórico / 22.78% Future.

Advertencia:
Estos resultados pertenecen a generaciones históricas anteriores y no deben
considerarse equivalentes al benchmark CLEAN actual.

---

## F2 — MACRO-45

Artefacto:
- features_invariantes_seguras.txt

Productor identificado:
- VF6_Analisis_Covariate_Shift.py / generaciones equivalentes.

Metodología:
- comparación de distribuciones Historical vs Future;
- selección de variables consideradas invariantes ante covariate shift.

Output:
- 45 características.

Resultados históricos asociados:
- Hybrid 45F neutral: 95.70% / 30.06%.
- Hybrid 45F Micro-biased: 96.42% / 34.98%.

Validez metodológica:
- FUTURE-INFORMED FEATURE SELECTION.
- Adicionalmente, algunos trainers 45F ajustaban StandardScaler antes del split.

Estatus:
- EXPLORATORY / HISTORICAL.
- No utilizar como baseline canónico static/unseen-future.

---

## F3 — Rama de Meta-Features Temporales

Metodología documentada:
- agrupación por sitio y día;
- dinámica intra-día;
- mean;
- std;
- range;
- skew;
- diff_std;
- autocorrelation lag 1;
- Zero Crossing Rate;
- FFT energy.

Universo documentado:
- alrededor de 540 meta-dimensiones en una generación histórica.

Posteriormente:
- filtrado de valores extremos;
- reducción de colinealidad mediante Pearson;
- conjunto documentado de 79 variables ortogonales.

Importante:
No existe evidencia de que 79 -> 94 sea una transformación directa.
Las generaciones 79 y 94 deben tratarse como ramas experimentales diferentes
hasta que código adicional demuestre lo contrario.

---

## F4 — MACRO-94

Artefacto:
- new_features_invariantes_seguras.txt

Productor confirmado:
- concept_drift/Reduccion_Features.py

Dataset Historical:
- CLEAN_final_features_sites.csv

Dataset Future:
- CLEAN_final_features_sites_concept_drift.csv

Clases:
- filtro de paridad con las 65 clases de la rama Micro.

Universo evaluado:
- 310 características numéricas válidas.

Método:
- Kolmogorov-Smirnov two-sample test.
- Se compara directamente cada feature en Historical vs Future.
- Variables de varianza cero son descartadas.
- Se conserva una feature si KS_Distance <= 0.15.

Resultado:
- 94 / 310 variables.

Archivo generado:
- new_features_invariantes_seguras.txt

Validez metodológica:
- FUTURE-INFORMED FEATURE SELECTION.

El modelo puede entrenarse exclusivamente sobre Historical, pero el conjunto
de variables ya incorpora información sobre la distribución Future.

Por tanto:
- training leakage: NO necesariamente;
- preprocessing leakage en reproducciones R: NO;
- feature-selection leakage respecto a Future: SÍ;
- canonical unseen-future benchmark: NO.

---

## F5 — Reproducción CLEAN con MACRO-94

Resultados reproducidos 2026-09-18:

| Experimento | Historical validation | Future | Fusión |
|---|---:|---:|---|
| HYB-003-R | 89.56% | 14.82% | Neutral |
| HYB-004-R | 90.51% | 20.28% | Feature-biased |
| HYB-005-R | 85.65% | 17.82% | Micro-biased |
| HYB-006-R | 95.98% | 32.97% | Cross-Attention |

Interpretación:
- la selección 94F no es canónica porque vio Future;
- el entrenamiento de estas reproducciones sí evita Future;
- HYB-006-R muestra señal complementaria importante entre Micro y Macro;
- estos resultados se conservan como controles experimentales.

---

## F6 — MACRO-V2-HIST

Estado:
- PLANNED.

Regla central:
Future no participará en:
- generación de candidatos;
- feature selection;
- estabilidad;
- thresholds;
- ranking;
- normalización;
- arquitectura;
- hyperparameters;
- checkpoint selection.

Fuente permitida:
- Historical exclusivamente.

Metodología:
- particiones temporales internas;
- discriminabilidad;
- estabilidad longitudinal;
- persistencia por sitio;
- redundancia.

Output previsto:
- MACRO-V2-HIST.txt
- MACRO-V2-HIST.yaml
- manifest.json
- métricas completas de selección.


---

## F6 — MACRO-V2-HIST: auditoría inicial y split temporal

Fecha: 2026-09-18

### Universo Historical

Dataset:
CLEAN_final_features_sites.csv

SHA256:
cadfc0848caca0b2556e03b9061f842e19e1d99d862bc39dc7c0bf56169375ce

Población:
- 65 sitios
- 72,603 capturas
- 52 días
- 2025-11-18 a 2026-01-08

Features:
- 320 columnas numéricas candidatas
- 9 de varianza cero
- 311 candidatas no constantes

Las 9 variables constantes son:
- times_min
- cumul_in_diffs_p75
- cumul_in_diffs_p90
- cumul_out_diffs_p25
- cumul_interp_in_len
- cumul_interp_out_len
- ngrams_2_topk_cumfrac
- ngrams_2_top5_count
- ngrams_2_top5_frac

### Comparación con selección MACRO-94 histórica

El selector Historical/Future anterior evaluó 310 variables.

Las 310 están contenidas dentro de las 311 candidatas Historical actuales.

La única candidata Historical adicional es:
- burst_durations_min

Esto confirma que el universo antiguo dependía también de las propiedades
estadísticas del dataset Future.

### Split temporal congelado

DEV_EARLY:
2025-11-18 -> 2025-12-01
14 días
19,098 capturas

DEV_MIDDLE:
2025-12-02 -> 2025-12-15
14 días
20,160 capturas

DEV_LATE:
2025-12-16 -> 2025-12-28
13 días
18,658 capturas

INTERNAL_TEST:
2025-12-29 -> 2026-01-08
11 días
14,687 capturas

Los 65 sitios tienen representación en todos los bloques.

Regla:
Los stages 02-06 pueden usar exclusivamente DEV_EARLY, DEV_MIDDLE y DEV_LATE.
INTERNAL_TEST permanece oculto hasta stage 07.
Future permanece fuera de toda la pipeline de selección.

