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

