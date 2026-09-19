# LTD-Attack — Research Decisions

## 2026-09-18 — MACRO-45 y MACRO-94 reclasificados

### Evidencia

La auditoría de código confirmó que:
- MACRO-45 fue seleccionada utilizando comparación Historical/Future.
- MACRO-94 fue generada mediante KS entre Historical y Future.
- MACRO-94 seleccionó 94 variables de 310 evaluadas usando KS <= 0.15.

### Decisión

MACRO-45 y MACRO-94 se conservan como generaciones históricas
FUTURE-INFORMED.

No se eliminan resultados, código ni artefactos.

No serán considerados conjuntos de features canónicos para un protocolo
Static / Unseen Future.

---

## 2026-09-18 — Reproducciones HYB-003-R ... HYB-006-R

Los cuatro experimentos se consideran:

TRAINING:
CLEAN / HISTORICAL ONLY

SCALER:
TRAIN-ONLY

FEATURE SET:
FUTURE-INFORMED (MACRO-94)

ROL:
CONTROL EXPERIMENTAL / EXPLORATORY

Mejor resultado:
HYB-006-R Cross-Attention
Historical validation = 95.98%
Future = 32.97%

---

## 2026-09-18 — Creación de MACRO-V2-HIST

Se decide reconstruir la ingeniería de características utilizando
exclusivamente la campaña Historical.

Future queda prohibido para:
- feature engineering;
- feature selection;
- ranking;
- thresholds;
- hyperparameter tuning;
- architecture selection;
- model selection;
- early stopping.

El número final de variables no será prefijado.

La selección deberá optimizar simultáneamente:
1. discriminabilidad entre sitios;
2. estabilidad temporal;
3. persistencia por sitio;
4. baja redundancia.


---

## 2026-09-18 — Freeze TEMPORAL-SPLIT-V1

Se congela el primer split temporal de MACRO-V2-HIST.

Development:
41 días históricos distribuidos cronológicamente en:
DEV_EARLY / DEV_MIDDLE / DEV_LATE.

Internal test:
últimos 11 días Historical.

Los stages 02-06 no pueden utilizar INTERNAL_TEST.

El Future externo no participa en ninguna etapa de ingeniería o selección.


---

## 2026-09-18 — Stage 06A: representantes y uso de Pareto

Se valida Stage 06A.

Resultados:
- 311 features originales.
- 148 clusters redundantes/no redundantes.
- 148 representantes, uno por cluster.
- 18 Pareto fronts.
- Pareto Front 1 contiene 16 features.

La selección de representantes queda congelada con la regla:

1. maximizar min(D,T,P);
2. maximizar media geométrica;
3. maximizar media aritmética;
4. minimizar imbalance max-min;
5. desempate alfabético.

Decisión metodológica:

Los Pareto fronts NO se utilizarán directamente como conjuntos acumulativos
para Stage 06B.

Motivo:
un frente no dominado puede contener soluciones extremadamente buenas en
un único eje pero muy débiles en los demás.

Ejemplo:
burst_size_p25 pertenece a Pareto Front 1 debido a su alta estabilidad,
pero presenta discriminabilidad muy baja.

Pareto se conserva como análisis de trade-off multiobjetivo.

Para la validación temporal 06B se construirán conjuntos anidados de los
148 representantes ordenados por robustez:

primary = min(D,T,P)
secondary = geometric_mean(D,T,P)
tertiary = arithmetic_mean(D,T,P)

INTERNAL_TEST y Future permanecen cerrados.


---

## 2026-09-18 — Compuerta 01A: elegibilidad DEV-only

Se auditó retrospectivamente el universo de features utilizado en stages
02-06A para comprobar si la determinación original de features no constantes,
realizada sobre todo Historical, había permitido que INTERNAL_TEST afectara
la selección de candidatas.

Resultado:

- candidatos estructurales: 320
- nonconstant Historical-wide: 311
- elegibles calculadas exclusivamente sobre DEV: 311
- diferencia legacy-only: 0
- diferencia DEV-only-only: 0
- igualdad de conjuntos: True
- igualdad de orden: True

Conclusión:

El uso descriptivo inicial de todo Historical no produjo ningún cambio en
el universo de 311 features utilizado por stages 02-06A.

Por tanto, los resultados de stages 02-06A permanecen válidos y no requieren
recomputación.

A partir de esta auditoría, la lista canónica de elegibilidad es:

01A_candidate_features_dev_eligible.txt

INTERNAL_TEST y Future permanecen cerrados.

