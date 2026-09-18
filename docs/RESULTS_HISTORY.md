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

