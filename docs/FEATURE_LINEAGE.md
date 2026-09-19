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


---

## F6.2 — Discriminabilidad Historical-only

Stage:
02_discriminability

Universo:
311 features no constantes.

Datos utilizados:
DEV_EARLY + DEV_MIDDLE + DEV_LATE.

Datos prohibidos:
- INTERNAL_TEST
- Future

Métricas:
- Mutual Information normalizada por entropía de clase.
- Eta squared entre sitios.
- Kruskal-Wallis epsilon squared.

Ranking:
Cada métrica se transforma en rango percentil por periodo.
El score de cada periodo es la media de los tres percentiles.

Criterio primario de ranking:
mínimo score obtenido entre DEV_EARLY, DEV_MIDDLE y DEV_LATE.

Esto favorece características consistentemente discriminativas en el tiempo.

Hallazgos principales:

1. sizes_sum ocupa el primer lugar y mantiene alta discriminabilidad
   en los tres periodos.

2. Varias features relacionadas con packet count / trace length /
   n-gram windows presentan métricas casi idénticas, indicando fuerte
   redundancia estructural.

3. spectral_energy_low/mid/high presentan elevada discriminabilidad y
   deben ser analizadas especialmente en estabilidad temporal.

4. Varias características seleccionadas por MACRO-94 debido a estabilidad
   Historical/Future resultan prácticamente no discriminativas.

Ejemplos:
- sizes_min: rank 305/311
- ngrams_3_unique: rank 306/311
- burst_durations_p25: rank 307/311
- ngrams_2_unique: rank 308/311
- burst_size_p25: rank 309/311
- sizes_p25: rank 310/311

Conclusión:
La estabilidad marginal de distribución no es suficiente para construir
un fingerprint longitudinal útil.

No se selecciona todavía ninguna feature.


---

## F6.3 — Estabilidad temporal Historical-only

Stage:
03_temporal_stability

Datos:
DEV_EARLY + DEV_MIDDLE + DEV_LATE.

Prohibidos:
- INTERNAL_TEST
- Future

Metodología:
- balanceo por sitio;
- KS ponderado;
- Wasserstein normalizada por escala robusta;
- desplazamiento normalizado de mediana.

Comparaciones:
- Early vs Middle
- Middle vs Late
- Early vs Late

El ranking utiliza el peor drift observado entre periodos.

Hallazgos:

1. Las features más estables no coinciden necesariamente con las más
   discriminativas.

2. Varias features del antiguo MACRO-94 son extremadamente estables pero
   presentan muy poca capacidad discriminativa.

3. sizes_sum:
   rank discriminabilidad = 1
   rank estabilidad = 196.
   Es un fingerprint potente pero temporalmente menos robusto.

4. ngrams_2_sparsity:
   rank discriminabilidad = 3
   rank estabilidad = 49.

5. ngrams_3_sparsity:
   rank discriminabilidad = 21
   rank estabilidad = 46.

6. ngrams_4_sparsity:
   rank discriminabilidad = 26
   rank estabilidad = 48.

7. spectral_energy_low/mid/high combinan alta discriminación con
   estabilidad temporal intermedia-alta.

8. Existe fuerte redundancia entre contadores de paquetes, longitudes,
   ventanas posibles y ocurrencias de n-grams.

No se selecciona ninguna feature todavía.


---

## F6.4 — Persistencia longitudinal por sitio

Stage:
04_site_persistence

Datos:
DEV_EARLY + DEV_MIDDLE + DEV_LATE.

Prohibidos:
- INTERNAL_TEST
- Future

Representación:
mediana de cada feature por sitio y periodo temporal.

Métricas:
- Spearman entre fingerprints de sitios;
- desplazamiento normalizado intra-site;
- Mean Reciprocal Rank para recuperación de identidad;
- Top-1 / Top-5 de identidad;
- ratio temporal drift / inter-site separation.

Hallazgos:

1. Las features longitudinalmente persistentes no coinciden necesariamente
   con las simplemente estables a nivel global.

2. ngrams_3_min_count obtiene rank 1 en persistencia:
   Spearman worst = 0.9829,
   MRR worst = 0.4006,
   Top-5 worst = 72.31%.

3. ngrams_4_sparsity presenta una combinación especialmente interesante:
   discriminabilidad rank 26,
   estabilidad rank 48,
   persistencia rank 18.

4. cumul_interp_out_diffs_p90 presenta uno de los perfiles más equilibrados:
   D = 0.8092,
   T = 0.7653,
   P = 0.8288,
   DTP mínimo = 0.7653.

5. Las características espectrales mantienen alta discriminabilidad y
   persistencia/estabilidad intermedias.

6. Varias features relacionadas con longitud total, packet count,
   possible windows y total n-gram occurrences presentan resultados casi
   idénticos, confirmando redundancia estructural.

7. Las características con drift/separation > 1 presentan desplazamiento
   temporal comparable o superior a su separación entre sitios y son
   candidatas débiles para fingerprint longitudinal.

No se selecciona todavía ninguna feature.


---

## F6.5 — Redundancia dual-space Historical-only

Stage:
05_redundancy

Datos:
DEV_EARLY + DEV_MIDDLE + DEV_LATE.

Prohibidos:
- INTERNAL_TEST
- Future

Espacios analizados:
1. 57,916 capturas DEV.
2. 195 perfiles longitudinales site-period.

Regla canónica de redundancia:
- |Spearman sample| >= 0.95
- |Spearman longitudinal| >= 0.95
- signo consistente
- clustering complete-linkage.

Resultados:
- 311 features iniciales.
- 2,701 pares con redundancia fuerte.
- 148 clusters efectivos.
- 47 clusters redundantes.
- 101 singletons.
- 210 features pertenecen a clusters redundantes.
- cluster máximo: 43 features.
- reducción potencial a un representante por cluster: 52.4%.

Hallazgo principal:
Muchas features inicialmente interpretadas como señales distintas resultan
altamente redundantes tanto por captura como longitudinalmente.

El cluster principal agrupa, entre otras:
- packet/trace length;
- cumulative descriptors;
- n-gram counts/windows/sparsity;
- sizes_sum;
- size histograms;
- spectral_energy_low/mid/high.

Se identificaron además numerosas equivalencias exactas o casi exactas,
incluyendo median/p50, varias definiciones de duración, cumulative last/max,
packet counts y possible n-gram windows.

Stage 05 no elimina ni selecciona ninguna feature.
La selección del representante de cada cluster se difiere a Stage 06.


---

## F6.6 — Representantes multiobjetivo y Pareto

Stage:
06A_cluster_representatives_pareto

Entradas:
artefactos Historical DEV de stages 02-05.

Resultados:
- 148 clusters.
- 148 representantes.
- 18 Pareto fronts.
- Pareto Front 1: 16 variables.

Mejores representantes por robustez:

1. cumul_interp_out_diffs_p90
   D=0.8092 T=0.7653 P=0.8288
   min=0.7653

2. ngrams_4_sparsity
   D=0.9041 T=0.7519 P=0.7986
   min=0.7519

El representante del cluster principal de 43 variables es
ngrams_4_sparsity.

Interpretación de Pareto:

Pareto Front 1 contiene tanto soluciones equilibradas como soluciones
extremas de un único objetivo. Por ello los fronts se conservan para
describir trade-offs, pero no se usarán directamente como subsets
acumulativos de features.

El número final de variables aún no está seleccionado.

INTERNAL_TEST y Future permanecen cerrados.


---

## F6.7 — Freeze MACRO-V2-BASE-128

Stages:
06B_dev_model_selection
06B_R1_dev_model_selection

Entrada:
148 representantes no redundantes de Stage 06A.

Evaluación:
- EARLY -> MIDDLE;
- EARLY+MIDDLE -> LATE;
- Logistic Regression;
- Random Forest;
- XGBoost.

La curva predictiva temporal alcanza una meseta entre 128 y 148 features.

06B-R1 elimina la incertidumbre de convergencia observada en Logistic
Regression durante 06B:
- max_iter=3000;
- 0 ConvergenceWarnings en las 54 ejecuciones;
- TOP-128 vuelve a ser seleccionado.

Resultado congelado:

MACRO-V2-BASE-128

128 features provenientes de 128 dimensiones de redundancia distintas.

INTERNAL_TEST no ha sido utilizado.
Future-B no ha sido utilizado.


---

## F6.8 — Daily BASE Profiles

Stage:
06C_daily_base_profiles

Entrada:
MACRO-V2-BASE-128 congelado.

Datos:
DEV_EARLY + DEV_MIDDLE + DEV_LATE.

Representación:
cada sitio-día se representa mediante la mediana por feature de las
capturas BASE-128 disponibles en ese día.

Resultados:
- 57,916 capturas DEV;
- 65 sitios;
- 41 fechas;
- 2,653 perfiles site-day;
- máximo teórico: 2,665 perfiles;
- cobertura site-day aproximada: 99.55%;
- 57 celdas BASE ausentes;
- missingness: 0.0168%;
- 0 perfiles diarios completamente ausentes.

Cobertura temporal:
- DEV_EARLY: 900 perfiles / 14 fechas;
- DEV_MIDDLE: 908 perfiles / 14 fechas;
- DEV_LATE: 845 perfiles / 13 fechas.

No se entrenó ningún modelo.
No se realizó nueva selección de features.
INTERNAL_TEST permanece cerrado.
Future-B permanece cerrado.

