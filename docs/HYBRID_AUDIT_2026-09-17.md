# Hybrid / DML Audit — 2026-09-17

## Main findings

### 45F generation

Available trainers:
- EXP2 Neutral
- EXP3 Feature-biased

Available evaluators:
- EXP2 Neutral
- EXP3 Feature-biased
- EXP4 Vector/Micro-biased
- EXP5 Cross-Attention

Important:
- EXP2/EXP3 trainers fit StandardScaler before the temporal split.
- Therefore these historical results contain preprocessing leakage.
- The historically reported 34.98% corresponds to the EXP4 Micro/Vector-biased branch.
- The EXP4 trainer is not currently present in the identified legacy/45F trainer set.
- EXP5 Cross-Attention references 34.98% as a prior threshold, not as its own verified result.

## 94F generation

Current trainers:
- HYB-003 Neutral
- HYB-004 Feature-biased
- HYB-005 Micro-biased
- HYB-006 Cross-Attention

Current 94F trainers fit StandardScaler using training indices only.

Historical runs show multiple generations:

HYB-003:
- strong generation: 96.02% historical / 37.02% future
- later generation: 69.99% historical / 13.72% future
- later training encountered NaNs

HYB-004:
- future results observed: 29.42% and 20.37%

HYB-005:
- strong generation: 95.15% historical / 41.38% future
- later generation: 87.08% historical / 21.78% future

HYB-006:
- strong generation: 95.02% historical / 40.65% future
- later generation: 94.69% historical / 29.69% future

These generations must not be mixed.

## DML

DML-001:
- Joint Cross-Entropy + Supervised Contrastive Learning
- PK sampling
- historical future result found: 8.33%
- scaler fitted before split

DML-002:
- Two-stage transfer DML
- pretrained from HYB-005 / EXP4
- frozen/pretrained encoder + SupCon projection
- Nearest-Centroid evaluation

Historical generations:
- generation A: 95.40% historical / 43.37% future
- generation B: 82.76% historical / 23.24% future

Important:
- scaler fitted before split
- sampling contains NumPy random operations
- one historical evaluator run also failed because of state_dict mismatch

## Decision

1. Reproduce current 94F hybrids first.
2. Do not use old 40–43% results as canonical.
3. Compare current clean reproductions with older strong generations.
4. Identify architectural/data/preprocessing changes responsible for regression.
5. Correct DML preprocessing before considering DML canonical.
