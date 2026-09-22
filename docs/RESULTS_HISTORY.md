# LTD-Attack — Results History

Los resultados de este documento no implican automáticamente validez
canónica. Cada generación conserva su contexto metodológico.

## Historical exploratory generations

| Generación | Historical | Future | Estado |
|---|---:|---:|---|
| Micro legacy | 97.26% | 33.08% | Historical generation |
| Micro + 230F | 97.01% | 22.78% | Historical generation |
| Hybrid 45F neutral | 95.70% | 30.06% | Future-informed features |
| Hybrid 45F Micro-biased | 96.42% | 34.98% | Future-informed + preprocessing issues |
| HYB-003 old strong | 96.02% | 37.02% | Non-canonical |
| HYB-005 old strong | 95.15% | 41.38% | Non-canonical |
| HYB-006 old strong | 95.02% | 40.65% | Non-canonical |
| DML old strong | 95.40% | 43.37% | Non-canonical |

## Current reproduced baselines

| Modelo | Historical | Future | Estado |
|---|---:|---:|---|
| MICRO-002-R | 95.94% | 22.97% | Clean reproduced |
| Deep Fingerprinting | 93.17% | 34.85% | Clean reproduced |
| Rimmer/LSTM | 84.22% | 9.49% | Clean reproduced |
| MACRO-94 | 86.76% | 15.34% | Future-informed feature set |

## Clean-training 94F hybrids

| Modelo | Historical | Future | Feature provenance |
|---|---:|---:|---|
| HYB-003-R | 89.56% | 14.82% | Future-informed |
| HYB-004-R | 90.51% | 20.28% | Future-informed |
| HYB-005-R | 85.65% | 17.82% | Future-informed |
| HYB-006-R | 95.98% | 32.97% | Future-informed |

Current clean static reference:
Deep Fingerprinting = 34.85% Future.



---

## Future-B — Frozen Final Evaluation

Stage:
10B_future_concept_drift_evaluation

Population:
- 18,543 captures
- 65 sites
- 2026-03-23 to 2026-04-07

Primary confirmatory track:
DEV_FROZEN.

Same-checkpoint INTERNAL_TEST -> Future-B:

MICRO-FINAL:
- Internal Accuracy: 0.950024
- Future Accuracy: 0.431160
- Accuracy retention: 0.453841
- Future Top-5: 0.723885

MACRO-LTD-FINAL:
- Internal Accuracy: 0.840675
- Future Accuracy: 0.223049
- Accuracy retention: 0.265321
- Future Top-5: 0.579949

LTD-HYBRID-FINAL:
- Internal Accuracy: 0.974399
- Future Accuracy: 0.494526
- Accuracy retention: 0.507519
- Future Top-5: 0.805587

Hybrid vs Micro on Future-B:
- Accuracy: +0.063366
- Macro-F1: +0.051259
- Top-5: +0.081702
- MRR: +0.069147

All corresponding day-block bootstrap intervals are positive.

Interpretation:

Concept Drift remains severe.

The frozen Micro/Macro hybrid significantly mitigates the degradation
relative to Micro alone but does not eliminate it.

The explicit LTD branch itself exhibits severe Future-B degradation.

Future-B is now OPEN and permanently frozen as confirmatory evidence.

No model or hyperparameter may be selected using Future-B.

Any subsequent analyses using stored Future-B predictions are POST-HOC
MECHANISTIC ANALYSES and must not be represented as independent model
validation.

Historical-final operational track:

- Micro Accuracy: 0.222024
- Macro Final Accuracy: 0.235507
- Hybrid Accuracy: 0.302432

The full-Historical Micro refit unexpectedly degraded Future-B
performance and requires separate analysis.


---

## 10C — Future-B Mechanistic Ablation

Status:
POST-HOC MECHANISTIC ANALYSIS.

No training.
No hyperparameter search.
No alpha search.
No Future-B raw CSV reread.

DEV_FROZEN:

- Micro: 43.12%
- Micro + XGB: 49.70%
- Micro + LTD: 36.23%
- Hybrid without LTD, renormalized: 49.16%
- Hybrid without XGB, renormalized: 44.66%
- Actual Hybrid: 49.45%

Interpretation:

The static XGB Macro component is currently the dominant source of
Future-B robustness.

Removing XGB causes a large and temporally consistent degradation.

Removing LTD produces only a small/non-conclusive Top-1 change in the
DEV_FROZEN track and improves Top-5.

However, LTD is not completely redundant.

In HISTORICAL_FINAL:

- Micro: 22.20%
- Micro + LTD: 24.86%
- Hybrid without LTD: 27.91%
- Actual Hybrid: 30.24%

Here the LTD component provides a measurable complementary gain.

Conclusion:

The current longitudinal encoder contains useful identity information,
but its contribution is not stable across long temporal gaps and
training regimes.

Phase 11 will therefore focus on learning temporally stable Macro
representations using Historical data only.

Future-B remains frozen.
No new model may be selected using Future-B.


---

## 11A — Historical Temporal-Stable Macro Basis

Status:
CLOSED — FAILED PROMOTION.

Data:
Historical only.
72,603 captures.
65 sites.
52 dates.
Future-B not used.

Fixed-origin BASE128:

NEAR:
- Accuracy: 0.7715
- Macro-F1: 0.7676

MID:
- Accuracy: 0.6561
- Macro-F1: 0.6285

FAR:
- Accuracy: 0.6380
- Macro-F1: 0.6112
- Top-5: 0.8614

Best temporal-stability-only candidate:
STABLE_TOP128.

STABLE_TOP128:

NEAR Macro-F1:
0.6771

MID Macro-F1:
0.5310

FAR Macro-F1:
0.5324

Decision:

The temporal-stability-only feature basis does not improve
long-horizon robustness.

BASE128 remains the canonical Macro representation.

Interpretation:

Low individual temporal drift is insufficient for feature selection.
Some temporally variable features retain important cross-site
discriminative information.

No Stable-K feature subset is promoted.

Next:
11B — Multiscale Longitudinal Memory using BASE128.

