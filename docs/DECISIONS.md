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

