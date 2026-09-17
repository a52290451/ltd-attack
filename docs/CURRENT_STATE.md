# LTD-Attack — Current Experimental State

Last consolidated: 2026-09-17

## Research objective

Evaluate Website Fingerprinting models under longitudinal Concept Drift and develop a hybrid architecture combining:

- Micro-level session representation
- Macro-level longitudinal/statistical representation

The current target is to improve robustness against temporal drift.

---

# Current datasets

## Historical

Original processed collection:
- 118 classes/sites
- 133,119 observations

Cleaned dataset:
- 69 sites
- 77,685 observations

Current comparable experimental subset:
- 65 sites
- 72,603 historical observations

## Future / Concept Drift

Cleaned:
- 69 sites
- 20,363 observations

Comparable 65-site subset:
- 18,543 observations

Approximate temporal gap:
- 74 days

---

# Verified current results

| Model | Historical | Future | Status |
|---|---:|---:|---|
| MICRO-002-R | 95.94% | 22.97% | VERIFIED |
| MICRO legacy evaluator | 95.94% | 24.83% | VERIFIED LEGACY |
| MACRO-002-R (94F) | 86.76% | 15.34% | VERIFIED |
| CUMUL / SVM | 63.45% train | 25.15% | PARTIAL |
| Deep Fingerprinting CNN | 93.17% val | 34.85% | VERIFIED |
| Rimmer / LSTM | 84.22% val | 9.49% | VERIFIED |

Current strongest reproduced future baseline:

Deep Fingerprinting CNN = 34.85%

---

# Historical hybrid results requiring reproduction

Historical experiments reported:

| Variant | Historical | Future |
|---|---:|---:|
| Micro vector baseline | 97.26% | 33.08% |
| Micro + 230F | 97.01% | 22.78% |
| Micro + 45F neutral | 95.70% | 30.06% |
| Micro + 45F Micro-biased | 96.42% | 34.98% |

These results belong to an earlier experimental generation and must not yet be considered equivalent to the current 65-class reproduced pipeline.

---

# Pending current hybrid reproduction

- HYB-003 — 94F neutral
- HYB-004 — 94F feature-biased
- HYB-005 — 94F Micro-biased
- HYB-006 — Cross-Attention
- DML-001
- DML-002

Historical 45F experiments also require reproduction.

---

# Current experimental questions

1. Is 45F more useful than 94F for Micro-Macro fusion?
2. Can Macro improve a strong Micro representation even though Macro alone performs poorly?
3. Can the Deep Fingerprinting embedding be strengthened using Macro context?
4. Can learned fusion outperform manually weighted fusion?
5. Can metric/contrastive objectives preserve class identity under temporal displacement?

---

# Next experimental phase

## Phase A — Existing hybrids

Reproduce:

- 45F neutral
- 45F Micro-biased
- 94F neutral
- 94F feature-biased
- 94F Micro-biased
- 94F Cross-Attention
- DML hybrid
- DML transfer

## Phase B — Expanded SoTA

Add candidates:

- Var-CNN
- Tik-Tok
- k-Fingerprinting
- Robust Fingerprinting / TAM

## Phase C — New LTD-Attack candidates

Evaluate progressively:

1. Late Fusion
2. Learnable Gated Fusion
3. Cross-Attention
4. Deep Fingerprinting + Macro
5. Drift-conditioned gating
6. Metric / contrastive alignment

---

# Current performance target

Strongest reproduced temporal baseline:

Deep Fingerprinting = 34.85%

Immediate objective:

Hybrid future accuracy > 34.85%

Research objective:

Increase temporal retention while preserving strong historical discrimination.
