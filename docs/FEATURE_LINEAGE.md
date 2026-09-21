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


---

## F6.9 — Historical Prototype Baseline

Stage:
06D_longitudinal_prototype_baseline

Objetivo:
evaluar si el contexto histórico de los sitios contiene señal útil para
identificar observaciones temporalmente posteriores.

Evaluación:
- Fold A: DEV_EARLY -> DEV_MIDDLE
- Fold B: DEV_EARLY + DEV_MIDDLE -> DEV_LATE
- 65 prototypes candidatos por query
- sin uso de la etiqueta real para seleccionar contexto
- INTERNAL_TEST cerrado
- Future-B cerrado

Mejor configuración para capturas individuales:
capture_median + Euclidean.

Fold A:
- Accuracy: 0.2104
- Macro-F1: 0.2049
- Top-5: 0.5372
- MRR: 0.3634

Fold B:
- Accuracy: 0.2261
- Macro-F1: 0.2224
- Top-5: 0.5904
- MRR: 0.3919

Para perfiles diarios, la identidad del sitio es considerablemente más
fuerte, alcanzando en Fold B:
- Accuracy: 0.6663
- Macro-F1: 0.6428
- Top-5: 0.8864
- MRR: 0.7674

Sin embargo, daily_profile corresponde a una unidad de consulta agregada
y no es directamente comparable con el ataque de captura individual.

daily_balanced_median fue consistentemente inferior a capture_median.

Interpretación:
el contexto histórico contiene señal discriminativa, pero una agregación
orderless mediante mediana no modela dinámica longitudinal explícita.

Siguiente etapa:
06E — evaluar prototypes sensibles a recencia y tendencia temporal sobre
queries de captura individual.


---

## F6.10 — Dynamic / Recency Prototype Baseline

Stage:
06E_dynamic_prototype_baseline

Objetivo:
evaluar si recencia o tendencia temporal mejoran el matching de una
captura individual frente al prototype histórico estático.

Unidad de consulta:
single capture.

Métrica de distancia:
Euclidean, congelada después de 06D.

Fold A — DEV_EARLY -> DEV_MIDDLE:
- static Macro-F1: 0.2049
- recent-7 Macro-F1: 0.2002
- recent-3 Macro-F1: 0.1974
- last-day Macro-F1: 0.1792
- linear-trend Macro-F1: 0.1852

Fold B — DEV_EARLY+DEV_MIDDLE -> DEV_LATE:
- static Macro-F1: 0.2224
- recent-7 Macro-F1: 0.2498
- recent-3 Macro-F1: 0.2288
- last-day Macro-F1: 0.2160
- linear-trend Macro-F1: 0.2404

En Fold B, recent-7 mejora frente a static:
- Accuracy: +0.0300
- Macro-F1: +0.0274
- Top-5: +0.0155
- MRR: +0.0271
- mean true rank: 8.72 -> 7.22

Interpretación:
la recencia contiene señal, pero las reglas manuales no son
consistentemente superiores entre folds. Esto justifica evaluar un
encoder temporal aprendido capaz de utilizar la secuencia completa sin
reducirla previamente a una mediana o tendencia lineal.

INTERNAL_TEST permanece cerrado.
Future-B permanece cerrado.

Siguiente etapa:
07A — Candidate-Conditioned Temporal Encoder.



---

## F7.1 — Candidate-Conditioned Temporal Encoder

Stage:
07A_candidate_conditioned_temporal_encoder

Objetivo:
evaluar si un contexto longitudinal aprendido de hasta 7 días mejora la
clasificación single-capture frente a un control BASE-only neuronal pareado.

Fold A — DEV_EARLY -> DEV_MIDDLE

BASE-MLP:
- Accuracy: 0.6156
- Macro-F1: 0.6061
- Top-5: 0.8890

LTD:
- Accuracy: 0.6107
- Macro-F1: 0.5974
- Top-5: 0.8920

Fold B — DEV_EARLY+DEV_MIDDLE -> DEV_LATE

BASE-MLP:
- Accuracy: 0.6813
- Macro-F1: 0.6688
- Top-5: 0.9323
- MRR: 0.7901

LTD:
- Accuracy: 0.6820
- Macro-F1: 0.6641
- Top-5: 0.9519
- MRR: 0.7963

Interpretación:

LTD no mejora de manera robusta Top-1/Macro-F1 frente al control BASE
pareado.

Sin embargo, mejora la señal de ranking, particularmente Top-5 en Fold B
(+1.96 pp), además de MRR y mean true rank.

La variabilidad entre semillas es considerablemente mayor en LTD que en
BASE-MLP. Por tanto, no se justifica aumentar todavía la complejidad del
Temporal Encoder.

Decisión:

La hipótesis "LTD reemplaza directamente al clasificador BASE" se cierra
en su forma actual.

La señal longitudinal se conserva como fuente auxiliar para evaluar
complementariedad y posterior fusion/reranking con BASE-128 + XGBoost.

Status:
07A CLOSED.

INTERNAL_TEST permanece cerrado.
Future-B permanece cerrado.

Next:
07B-1 Complementarity Audit.



---

## F7.2 — BASE-XGB / LTD Complementarity Audit

Stage:
07B_01_complementarity_audit

Objetivo:
determinar si LTD proporciona información complementaria cuando el
clasificador BASE-128 + XGBoost se equivoca.

Control:
BASE-XGB reproduce exactamente 06B-R1 en ambos folds.

Fold A — EARLY -> MIDDLE:

BASE-XGB:
- Accuracy: 0.7369
- Macro-F1: 0.7311
- Top-5: 0.9328

LTD ensemble:
- Accuracy: 0.6491
- Macro-F1: 0.6358
- Top-5: 0.8975

Complementariedad:
- both correct: 11,487
- XGB-only correct: 3,369
- LTD-only correct: 1,598
- both wrong: 3,706
- oracle union accuracy: 0.8162
- oracle gain vs XGB: +0.0793
- LTD rescues 30.13% of XGB errors

Fold B — EARLY+MIDDLE -> LATE:

BASE-XGB:
- Accuracy: 0.7674
- Macro-F1: 0.7536
- Top-5: 0.9510

LTD ensemble:
- Accuracy: 0.7158
- Macro-F1: 0.6979
- Top-5: 0.9583

Complementariedad:
- both correct: 11,918
- XGB-only correct: 2,401
- LTD-only correct: 1,438
- both wrong: 2,901
- oracle union accuracy: 0.8445
- oracle gain vs XGB: +0.0771
- LTD rescues 33.14% of XGB errors

Interpretación:

LTD no sustituye al clasificador BASE-XGB, pero contiene una señal
claramente complementaria.

En ambos folds LTD resuelve aproximadamente 30-33% de las observaciones
incorrectas de XGB.

El oracle union muestra un margen potencial de aproximadamente +7.7 a
+7.9 puntos porcentuales de accuracy.

Existen además fuertes diferencias por sitio, confirmando que las dos
representaciones capturan patrones distintos.

Decisión:

07B-1 PASSES the complementarity gate.

No se realizará una fusión arbitraria 50/50.

La siguiente etapa evaluará primero una fusión probabilística geométrica
con un único peso alpha seleccionado exclusivamente en Fold A y transferido
sin cambios a Fold B.

Status:
07B-1 CLOSED — POSITIVE.

INTERNAL_TEST permanece cerrado.
Future-B permanece cerrado.

Next:
07B-2A Geometric Probability Fusion.



---

## F7.3 — Geometric BASE-XGB / LTD Fusion

Stage:
07B_02_geometric_probability_fusion

Objetivo:
convertir la complementariedad observada entre BASE-XGB y LTD en una
mejora efectiva de clasificación single-capture.

Fusión:

score(c) =
(1-alpha) * log(P_XGB(c))
+
alpha * log(P_LTD(c))

Selección:

alpha se seleccionó exclusivamente sobre Fold A.

Grid:
0.000 -> 1.000, step 0.025.

Alpha seleccionado:
0.375

Fold B no participó en la selección de alpha.

Fold A — selection:

XGB:
- Accuracy: 0.7369
- Macro-F1: 0.7311

Fusion:
- Accuracy: 0.7451
- Macro-F1: 0.7350

Delta:
- Accuracy: +0.0082
- Macro-F1: +0.0039
- MRR: +0.0019
- Top-5: -0.0119
- net Top-1 correct: +165

Fold B — temporal transfer:

XGB:
- Accuracy: 0.7674
- Macro-F1: 0.7536
- Top-5: 0.9510
- MRR: 0.8463

Fusion:
- Accuracy: 0.7946
- Macro-F1: 0.7793
- Top-5: 0.9693
- MRR: 0.8683

Delta:
- Accuracy: +0.0272
- Macro-F1: +0.0257
- Top-5: +0.0183
- MRR: +0.0220
- fusion-only correct: 848
- XGB-only correct: 341
- net Top-1 gain: +507

Interpretación:

La información longitudinal complementaria detectada en 07B-1 puede
convertirse en una mejora real mediante una fusión probabilística simple.

La transferencia a Fold B es positiva simultáneamente en Accuracy,
Macro-F1, Top-5 y MRR.

La curva Fold-A presenta una región relativamente estable aproximadamente
entre alpha 0.30 y 0.40, reduciendo evidencia de una selección extremadamente
frágil.

Advertencia:

Fold B es DEV y ya había sido observado durante el desarrollo. Por tanto,
estos resultados no constituyen todavía una estimación independiente de
generalización.

Decisión:

07B-2A CLOSED — POSITIVE.

Candidate final architecture:

MACRO-V2-BASE-128
+
XGBoost BASE
+
07A LTDPairScorer
+
7-day candidate context
+
3-seed LTD probability ensemble
+
geometric fusion alpha=0.375

No se modifica esta arquitectura durante el siguiente audit.

INTERNAL_TEST permanece cerrado.
Future-B permanece cerrado.

Next:
07B-2R Fusion Robustness Audit.



---

## F7.4 — MACRO-LTD Fusion Robustness and Freeze

Stage:
07B_02R_fusion_robustness_audit

Objetivo:
validar la robustez temporal de la fusión MACRO-LTD candidata sin realizar
ningún nuevo ajuste de arquitectura ni de alpha.

Configuración auditada:
- MACRO-V2-BASE-128
- BASE classifier: XGBoost
- LTDPairScorer
- context window: 7 días
- LTD seeds: 11, 42, 73
- LTD ensemble: media de probabilidades
- geometric probability fusion
- alpha: 0.375

Fold A:
- delta Accuracy: +0.0082
- delta Macro-F1: +0.0039
- 14/14 días mejoran Accuracy

Day-block bootstrap:
- Accuracy IC95%: +0.0049 a +0.0114
- P(delta Accuracy > 0): 1.000
- P(delta Macro-F1 > 0): 0.962

Fold B:
- delta Accuracy: +0.0272
- delta Macro-F1: +0.0257
- delta Top-5: +0.0183
- delta MRR: +0.0220
- 13/13 días mejoran Accuracy

Day-block bootstrap:
- Accuracy IC95%: +0.0238 a +0.0311
- Macro-F1 IC95%: +0.0220 a +0.0301
- P(delta Accuracy > 0): 1.000
- P(delta Macro-F1 > 0): 1.000

Alpha stability:
- selected alpha: 0.375
- near-optimal region: 0.30–0.40

Resultado:
ROBUSTNESS PASS.

Interpretación:

La mejora longitudinal no depende de un subconjunto aislado de días.
La fusión mejora Accuracy en todos los días de ambos folds.

Existen diferencias por clase, por lo que MACRO-LTD no debe interpretarse
como solución completa del problema. Estas diferencias son especialmente
relevantes para estudiar complementariedad posterior con la representación
Micro.

Decisión:

Se congela la rama Macro como:

MACRO-LTD-V1

MACRO-LTD-V1 =
MACRO-V2-BASE-128
+ XGBoost
+ LTDPairScorer
+ 7-day context
+ seeds [11,42,73]
+ probability ensemble
+ geometric fusion alpha=0.375

Status:
MACRO-LTD-V1 FROZEN.

Importante:
INTERNAL_TEST NO se abre todavía.

Motivo:
el objetivo final es construir un modelo híbrido Micro + Macro-LTD.
INTERNAL_TEST se preserva como holdout para la arquitectura híbrida final,
no para una rama intermedia.

Future-B permanece cerrado.

Next:
07C-1 — Confidence-Stratified Fusion Audit.

Research policy:
MACRO-LTD-V1 remains frozen and immutable, but does not terminate the Macro
research line. New improvements are developed as MACRO-LTD-V2+ without
altering the V1 reference.

The Micro/Hybrid branch remains planned for a later phase.



---

## F7.5 — Confidence-Stratified Fusion Audit

Stage:
07C_01_confidence_stratified_fusion_audit

Objective:
determine whether the value of longitudinal LTD information depends on
the confidence of BASE-XGB.

No model training.
No alpha tuning.
Frozen alpha: 0.375.

Confidence metric:
XGB Top-1 minus Top-2 probability margin.

Thresholds were defined exclusively on Fold A:
- q20: 0.2644217
- q40: 0.6064769
- q60: 0.8667137
- q80: 0.9716550

Fold A fusion delta vs XGB:
- Q1 lowest confidence: +4.54 pp Accuracy
- Q2: +0.20 pp
- Q3: -0.55 pp
- Q4: -0.10 pp
- Q5: 0.00 pp

Fold B fusion delta vs XGB:
- Q1 lowest confidence: +10.77 pp
- Q2: +2.30 pp
- Q3: +0.76 pp
- Q4: +0.11 pp
- Q5: +0.05 pp

Model disagreement:

Fold A:
- XGB Accuracy: 0.4800
- Fusion Accuracy: 0.5035
- net correct: +165

Fold B:
- XGB Accuracy: 0.4373
- Fusion Accuracy: 0.5297
- net correct: +507

Interpretation:

The contribution of LTD is strongly confidence-dependent.

Most useful longitudinal information appears when BASE-XGB is uncertain.
When XGB confidence is high, fixed fusion contributes almost no additional
Top-1 information and can slightly damage some DEV observations.

This supports testing a query-adaptive fusion policy while preserving
MACRO-LTD-V1 as an immutable reference.

Status:
07C-1 CLOSED — ADAPTIVE SIGNAL CONFIRMED.

INTERNAL_TEST remains closed.
Future-B remains closed.

Next:
07C-2 — Confidence-Gated Fusion.



---

## F7.6 — Confidence-Gated Fusion

Stage:
07C_02_confidence_gated_fusion

Hypothesis:
a hard confidence gate can improve MACRO-LTD-V1 by applying LTD fusion
only when BASE-XGB confidence is low.

Selection protocol:
- candidate thresholds derived exclusively from Fold A;
- alpha remained frozen at 0.375;
- Fold B did not participate in gate selection.

Selected policy:
Q20.

Fold A:
- fusion coverage: 20.00%
- V2 vs V1 Accuracy: +0.0009
- V2 vs V1 Macro-F1: +0.0039
- V2 vs V1 Top-5: +0.0075
- V2 vs V1 MRR: +0.0023
- net correct: +18

Fold B temporal transfer:
- fusion coverage: 19.25%
- V2 vs V1 Accuracy: -0.0064
- V2 vs V1 Macro-F1: -0.0053
- V2 vs V1 Top-5: -0.0089
- V2 vs V1 MRR: -0.0069
- net correct: -120

Interpretation:

XGB confidence is diagnostically associated with LTD usefulness, but a
hard confidence threshold selected on Fold A is not temporally robust.

The absolute confidence regime shifts enough across temporal folds that
the selected Q20 rule does not transfer.

Decision:

07C-2 CLOSED — NEGATIVE.

The hard confidence-gating hypothesis is rejected in its current form.

No alternative threshold will be selected using Fold-B results.

MACRO-LTD-V1 remains the frozen reference.

INTERNAL_TEST remains closed.
Future-B remains closed.

Next:
07D-1 — LTD Context-Length Ablation.



---

## F7.7 — LTD Context-Length Ablation

Stage:
07D_01_context_length_ablation

Objective:
test whether the 7-day longitudinal context frozen in MACRO-LTD-V1 is
the most useful context length for the final fused Macro classifier.

Candidate windows:
1, 3, 5, 7 days.

Protocol:
- all candidates evaluated only on Fold A;
- primary selection: fused Macro-F1;
- secondary: fused Accuracy;
- only the selected context transferred to Fold B;
- fusion alpha remained frozen at 0.375;
- Fold B did not participate in context selection.

Selected context:
5 days.

Fold A, W=5 vs V1 W=7:
- Accuracy: +0.00094
- Macro-F1: +0.00042
- Top-5: +0.00392
- MRR: +0.00174

Fold B temporal transfer, W=5 vs V1:
- Accuracy: +0.00338
- Macro-F1: +0.00414
- Top-5: +0.00005
- MRR: +0.00180

W=5 Fold-B absolute performance:
- Accuracy: 0.797995
- Macro-F1: 0.783429
- Top-5: 0.969343
- MRR: 0.870105

Important observation:

All context candidates skipped the same seven earliest training dates.
Therefore the W=5 improvement is not caused by recovering additional
training queries. It is attributable to the context representation itself.

Another relevant result:

The strongest standalone LTD on Fold A remained W=7, while the strongest
fused classifier used W=5.

This supports optimizing longitudinal information for complementarity
with BASE-XGB rather than standalone LTD accuracy.

Decision:

07D-1 PASSES the context gate.

W=5 is a MACRO-LTD-V2 candidate, not yet a frozen V2.

The improvement over V1 is modest and therefore requires a paired
robustness audit before promotion.

INTERNAL_TEST remains closed.
Future-B remains closed.

Next:
07D-1R — Paired Context Robustness Audit.



---

## F7.8 — MACRO-LTD-V2 Promotion

Source:
07D_01R_context_robustness_audit

Candidate:
MACRO-LTD-V1 with context reduced from 7 to 5 site-days.

No alpha change.
No feature change.
No XGBoost change.
No seed change.

Fold A paired V2 vs V1:
- Accuracy: +0.00094
- Macro-F1: +0.00042
- Top-5: +0.00392
- MRR: +0.00174

Fold-A Accuracy and Macro-F1 bootstrap intervals cross zero.
This fold was used for context selection and is not treated as an
independent confirmation of the selected context.

Fold B temporal transfer:

- Accuracy delta: +0.00338
- Macro-F1 delta: +0.00414
- Top-5 delta: +0.00005
- MRR delta: +0.00180
- V2-only correct: 242
- V1-only correct: 179
- net correct: +63
- improved dates: 11
- worse dates: 1
- tied dates: 1

Day-block bootstrap Fold B:

Accuracy:
- IC95%: +0.00209 to +0.00462
- P(delta > 0): 1.000

Macro-F1:
- IC95%: +0.00260 to +0.00557
- P(delta > 0): 1.000

MRR:
- IC95%: +0.00113 to +0.00246
- P(delta > 0): 1.000

Top-5 is effectively unchanged.

Decision:

07D-1R PASSED the predeclared promotion gate.

MACRO-LTD-V2 is promoted as a new immutable DEV milestone.

MACRO-LTD-V1 remains preserved.

MACRO-LTD-V2 configuration:
- BASE-128
- XGBoost
- LTDPairScorer
- context = 5 site-days
- ordered sinusoidal positional encoding
- seeds = [11,42,73]
- mean probability ensemble
- geometric fusion alpha = 0.375

Fold-B DEV performance:
- Accuracy: 0.797995
- Macro-F1: 0.783429
- Top-5: 0.969343
- MRR: 0.870105

INTERNAL_TEST remains closed.
Future-B remains closed.

Next:
07E-1 — Temporal Order Ablation.



---

## F7.9 — Temporal Order Ablation

Stage:
07E_01_temporal_order_ablation

Question:
does ordered temporal information contribute beyond the same set of
five longitudinal daily profiles?

Control:
same architecture, same trainable parameter count, same W=5 and same
training protocol, without positional encoding.

Trainable parameters:
116865.

Fold A, Ordered - NoPos fusion:
- Accuracy: -0.00045
- Macro-F1: -0.00090
- Top-5: +0.00273
- MRR: +0.00009
- net ordered correct: -9

Fold B, Ordered - NoPos fusion:
- Accuracy: +0.00525
- Macro-F1: +0.00587
- Top-5: +0.00193
- MRR: +0.00314
- net ordered correct: +98

Fold-B day-block bootstrap:
- Accuracy IC95%: +0.00268 to +0.00760
- Macro-F1 IC95%: +0.00301 to +0.00851
- Top-5 IC95%: +0.00075 to +0.00321
- MRR IC95%: +0.00188 to +0.00419

Interpretation:

Temporal order provides strong positive evidence in Fold B, but the effect
does not reproduce in Fold A for Top-1 Accuracy or Macro-F1.

Therefore the order-specific causal claim is not considered established.

Decision:
ORDER_EFFECT_INCONCLUSIVE.

MACRO-LTD-V2 remains the frozen DEV milestone because the ablation does not
constitute a replacement-selection experiment.

Current scientific claim:
longitudinal historical context contributes complementary information.

Current scientific claim NOT yet supported:
temporal ordering alone is responsible for the gain.

INTERNAL_TEST remains closed.
Future-B remains closed.

Next:
07F-1 — Query-Age-Aware LTD.



---

## F7.10 — Query-Age-Aware LTD

Stage:
07F_01_query_age_aware_ltd

Hypothesis:
absolute query-to-profile staleness may contain useful information beyond
ordinal temporal position.

Design:
- reference: MACRO-LTD-V2;
- context: 5 site-days;
- same 116865 trainable parameters;
- same seeds [11,42,73];
- same fusion alpha=0.375;
- ordinal sinusoidal encoding replaced by actual profile age in days.

Result:
NEGATIVE.

Fold A — Age minus V2 fusion:
- Accuracy: -0.00243
- Macro-F1: -0.00208
- Top-5: -0.00585
- MRR: -0.00273
- net correct: -49

Fold B — Age minus V2 fusion:
- Accuracy: -0.00407
- Macro-F1: -0.00551
- Top-5: -0.00381
- MRR: -0.00315
- net correct: -76

Fold-B day-block bootstrap:
- Accuracy IC95%: -0.00577 to -0.00226
- Macro-F1 IC95%: -0.00762 to -0.00342
- Top-5 IC95%: -0.00650 to -0.00130
- MRR IC95%: -0.00457 to -0.00181

Decision:
07F-1 CLOSED — FAILED PROMOTION.

MACRO-LTD-V2 remains the frozen reference.

Important failure-mode observation:

During causal training, candidate context is predominantly fresh.
Typical five-token context ages are approximately 1--5 days.

During frozen-history evaluation, the same historical context becomes
progressively stale. Test token ages extend to approximately 18--19 days.

Therefore replacing ordinal position by absolute age requires extrapolation
to staleness regimes that are poorly represented during model training.

This motivates one final targeted Macro hypothesis:

07G-1 — Stale-Context Training.

The architecture remains MACRO-LTD-V2. Only the training context distribution
is modified so the model is explicitly trained with historical candidate
snapshots of different ages.

Macro stop rule:

07G is the final planned Macro hypothesis.

If stale-context training does not produce robust temporal transfer,
MACRO-LTD-V2 becomes MACRO-FINAL and research proceeds to Micro.

If it succeeds, the candidate is subjected to one robustness audit and,
if confirmed, becomes MACRO-FINAL.

INTERNAL_TEST remains closed.
Future-B remains closed.



---

## F7.11 — Macro Final Freeze

07G-1 tested the final prospectively allowed Macro hypothesis:
stale-context training.

Result:
NEGATIVE.

Fold A — STALE vs MACRO-LTD-V2 fusion:
- Accuracy: -0.00491
- Macro-F1: -0.00612
- Top-5: -0.00174
- MRR: -0.00361
- net correct: -99

Fold B:
- Accuracy: -0.00263
- Macro-F1: -0.00308
- Top-5: -0.00236
- MRR: -0.00148
- net correct: -49

Fold-B bootstrap intervals are below zero for all four metrics.

Decision:

07G-1 CLOSED — FAILED PROMOTION.

The prospectively defined Macro stop rule is activated.

MACRO-LTD-V2 becomes:

MACRO-LTD-FINAL

Frozen architecture:
- BASE-128
- XGBoost
- LTDPairScorer
- context = 5 site-days
- ordinal sinusoidal temporal encoding
- seeds = [11,42,73]
- mean probability ensemble
- geometric fusion alpha = 0.375

DEV Fold-B reference:
- Accuracy: 0.797995
- Macro-F1: 0.783429
- Top-5: 0.969343
- MRR: 0.870105

No additional Macro architecture search is permitted before holdout
evaluation.

The final all-DEV refit is still pending and will occur only after the
hybrid architecture has been frozen.

INTERNAL_TEST remains closed.
Future-B remains closed.

Next:
Phase P7 — Clean Temporal Micro.



---

## F8.1 — Micro/Macro Capture Alignment

Stage:
08A_micro_temporal_alignment_audit

Purpose:
establish an exact capture-level bridge between frozen Macro experiments
and the clean temporal Micro phase.

Result:
PASS.

Canonical Micro source:
historical/CLEAN_final_vectors_sites.csv

Join key:
pcap_uid

Canonical 65-site population:
72,603 captures.

Coverage:
- DEV_EARLY: 19,098 / 19,098 = 100%
- DEV_MIDDLE: 20,160 / 20,160 = 100%
- DEV_LATE: 18,658 / 18,658 = 100%
- INTERNAL_TEST metadata: 14,687 / 14,687 = 100%

Integrity:
- duplicate Macro keys: 0
- duplicate Micro keys: 0
- label mismatches: 0

Both available vector sources covered the canonical Macro population.

CLEAN_final_vectors_sites.csv was selected because it is the canonical
legacy Micro source and its 65-site row count exactly matches the current
Macro population.

cached_vectors_with_dates.csv contains additional observations and is not
used in the canonical 08B experiment.

08A read metadata only.

INTERNAL_TEST vector values remain unused.
Future-B remains closed.

Next:
08B — Clean Temporal Micro Baseline.

