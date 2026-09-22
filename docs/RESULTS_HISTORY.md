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


---

## 11B — Multiscale Longitudinal Memory

Status:
CLOSED — NOT PROMOTED.

Data:
Historical only.
Future-B not used.

Control:
RECENT5_SEQUENCE.

Candidate:
MULTISCALE5:
- all-history median
- last-10 median
- last-5 median
- last-3 median
- last observed day

Same BASE128 representation.
Same five-token context budget.
Same LTD architecture.
Same seeds and training configuration.

Result:

MULTISCALE5 improved LTD Macro-F1 in all six
origin/window evaluations.

Most relevant FAR results:

ORIGIN14:
- LTD Macro-F1: +0.94 pp
- Fusion Macro-F1: +0.29 pp

ORIGIN28:
- LTD Macro-F1: +2.00 pp
- Fusion Macro-F1: +0.38 pp

The FAR LTD Macro-F1 bootstrap intervals were positive
for both temporal origins.

The gain became strongest in the later/staler ORIGIN28
FAR condition, supporting the hypothesis that multiscale
history helps preserve longitudinal identity.

However, the mean FAR Fusion Macro-F1 improvement was
approximately +0.33 pp, below the predeclared +0.50 pp
promotion threshold.

In addition, ORIGIN28 NEAR and MID fusion metrics showed
small degradations.

Decision:

MULTISCALE5 is not promoted as the canonical memory.

RECENT5 remains the control.

The multiscale result is retained as positive mechanistic
evidence and may be reconsidered only after an independently
successful representation improvement.

Next:

11C — Temporal Contrastive Representation.

BASE128 remains canonical.
XGB remains unchanged.
11C will modify only the representation used by the LTD branch.


---

## 11C — Temporal Contrastive Representation

Status:
CLOSED — FAILED PROMOTION.

Data:
Historical only.
Future-B not used.

Control:
BASE128 + RECENT5.

Candidate:
TCL128 + RECENT5.

Temporal contrastive pools were constructed independently per site
using class-specific earliest/latest thirds of available training dates.

Pool audit:
- 65/65 sites valid in ORIGIN14.
- 65/65 sites valid in ORIGIN28.
- strictly positive temporal separation.

The temporal contrastive objective optimized successfully, but the
learned representation degraded downstream temporal identification.

FAR deltas vs RAW128:

ORIGIN14:
- LTD Macro-F1: -2.07 pp
- Fusion Macro-F1: -0.24 pp

ORIGIN28:
- LTD Macro-F1: -4.60 pp
- Fusion Macro-F1: -0.69 pp

ORIGIN28 FAR bootstrap intervals were fully negative for all principal
LTD and Fusion metrics.

Decision:

TCL128 is rejected.

BASE128 remains canonical.
RECENT5 remains canonical.

Interpretation:

Explicitly forcing same-site embeddings from separated Historical
periods to become invariant removed or distorted temporally varying
information that remains useful for Website Fingerprinting.

Combined with 11A and 11B, current evidence suggests that preserving
temporal variation while modeling it across multiple temporal scales
is preferable to removing temporal variation.

Next:
11D — long-term prototype / drift-aware candidate scoring.


---

## 11D — Long-Term / Drift-Aware Candidate Scoring

Status:
CLOSED — FAILED PROMOTION.

Data:
Historical only.
Future-B not used.

Control:
BASE128 + RECENT5.

Candidates:
LONGTERM and DUALANCHOR prototype scores.

Candidate selection:
ORIGIN14 only.

Transfer:
exactly one candidate to ORIGIN28.

Selected candidate:
LONGTERM_BETA0.25.

ORIGIN14 FAR:
- Branch Macro-F1 delta: -0.34 pp
- Fusion Macro-F1 delta: -0.05 pp

ORIGIN28 FAR:
- Branch Macro-F1 delta: -0.38 pp
- Fusion Macro-F1 delta: +0.05 pp

The ORIGIN28 Fusion FAR increase was not statistically stable and
its bootstrap interval crossed zero.

NEAR and MID performance degraded in ORIGIN28.

Increasing prototype weight progressively degraded performance.

Interpretation:

Static long-term identity prototypes do not solve stale-history drift.

The evidence from 11A–11D suggests that robustness is not obtained by:

- removing temporally varying features,
- forcing temporal invariance,
- or anchoring queries to static long-term identities.

The strongest positive evidence remains 11B, where preserving BASE128
while changing the temporal representation improved LTD performance,
especially at FAR horizons.

Next:

11E — explicit longitudinal trajectory / velocity scoring.

The new hypothesis is that site-specific direction of temporal change,
rather than static temporal identity, carries useful longitudinal
information.


---

## 11E — Explicit Longitudinal Trajectory Scoring

Status:
CLOSED — FAILED PROMOTION.

Data:
Historical only.
Future-B not used.

Control:
BASE128 + RECENT5 LTD + canonical XGB.

Selected on ORIGIN14:
RAY_DISTANCE_BETA0.25.

FAR:

ORIGIN14:
- Branch Macro-F1 delta: -0.02 pp
- Fusion Macro-F1 delta: approximately 0.00 pp

ORIGIN28:
- Branch Macro-F1 delta: -0.04 pp
- Fusion Macro-F1 delta: +0.21 pp

The ORIGIN28 Fusion improvement did not have a fully positive
bootstrap confidence interval.

Trajectory audit:

The historical velocity assumption was not supported.

ORIGIN14:
- 63/65 sites had negative early-to-middle vs middle-to-late
  directional consistency.
- median directional consistency was approximately -0.47.

ORIGIN28:
- 40/65 sites had negative directional consistency.
- median directional consistency was approximately -0.26.

Interpretation:

Website Macro evolution is not well represented by a persistent
linear direction in BASE128 space.

The evidence is consistent with non-monotonic, recurrent, abrupt,
or feature-dependent temporal change.

This explains why stability selection, static prototypes,
contrastive invariance and explicit linear trajectory extrapolation
did not improve robustness.

The strongest Phase-11 positive evidence remains 11B:
preserving several historical scales improved the LTD branch,
particularly at FAR horizons.

Decision:

Stop hand-designed frozen trajectory/prototype search.

Next:
11F — causal oracle memory refresh feasibility.

11F will test whether fresh longitudinal context itself can recover
performance before implementing any deployable pseudo-label
adaptation.


---

## 11F — Causal Oracle Memory Refresh

Status:
CLOSED — FEASIBILITY PASSED.

Scientific status:
NON-DEPLOYABLE MECHANISM DIAGNOSTIC.

Data:
Historical only.
Future-B not used.

Question:

Would correctly refreshing the recent longitudinal history after each
already-scored day improve stale-context robustness?

Protocol:

- predict current day using only previous history;
- current-day ground-truth labels cannot affect current-day predictions;
- after current day has been completely scored, construct true-label
  site-day profiles;
- make those profiles available only to subsequent days;
- each evaluation window starts from the original frozen training history.

Causal verification:

For all six origin/window evaluations:

first_day_max_probability_difference = 0.0

Therefore the oracle update cannot affect the first/current day's
prediction.

FAR results:

ORIGIN14:

LTD:
- Macro-F1: 0.5111 -> 0.5580
- delta: +4.69 pp

Fusion:
- Macro-F1: 0.6117 -> 0.6333
- delta: +2.16 pp
- Accuracy delta: +2.57 pp
- Top-5 delta: +4.49 pp
- MRR delta: +2.83 pp

ORIGIN28:

LTD:
- Macro-F1: 0.6479 -> 0.6743
- delta: +2.63 pp

Fusion:
- Macro-F1: 0.7601 -> 0.7714
- delta: +1.13 pp
- Accuracy delta: +1.07 pp
- Top-5 delta: +0.92 pp
- MRR delta: +1.06 pp

Day-block bootstrap:

ORIGIN14 FAR Fusion Macro-F1:
95% interval approximately [+1.18, +3.09] pp.

ORIGIN28 FAR Fusion Macro-F1:
95% interval approximately [+0.63, +1.63] pp.

fraction_delta_gt_0 = 1.0 in both origins.

Decision:

The feasibility gate PASSES.

Freshness of longitudinal candidate history is a real contributor to
temporal robustness.

This explains why frozen static prototypes and trajectory extrapolation
failed while MULTISCALE5 showed a weaker positive effect.

Interpretation:

The current evidence supports tracking evolving site states rather than
forcing invariant or deterministic temporal representations.

Next:

11G — deployable causal pseudo-label self-updating memory.


---

## 11G — Pseudo-label Self-Updating Memory

Status:
CLOSED — FAILED PROMOTION.

Scientific decision:
The online/self-updating adaptation branch is stopped.

Research objective clarification:

The target of LTD-Attack is NOT continual model or memory adaptation.

The objective is to train a frozen model whose performance degrades
as slowly as possible under longitudinal concept drift, extending the
useful interval between retraining events.

11F remains a mechanism diagnostic only:

It demonstrated that stale context contributes to degradation in the
current LTD architecture, but it does not imply that online refreshing
is the desired solution.

11G evaluated a deployable approximation using XGB/LTD consensus
pseudo-labels.

Selected policy:
CONSENSUS_ONLY.

ORIGIN14 FAR:
- LTD Macro-F1 delta: -1.84 pp
- Fusion Macro-F1 delta: -0.31 pp

ORIGIN28 FAR:
- LTD Macro-F1 delta: +0.48 pp
- Fusion Macro-F1 delta: -0.37 pp

ORIGIN28 FAR Fusion Macro-F1 bootstrap interval was entirely negative.

Despite relatively high pseudo-label precision and substantial update
coverage, memory contamination degraded overall performance.

Decision:

Reject online pseudo-label memory adaptation.

No further self-updating, continual-learning or test-time adaptation
will be pursued in Phase 11.

Revised frozen-model hypothesis:

Temporal robustness should be obtained during training by exposing the
model to multiple temporal distributions/states and learning a decision
function that remains discriminative across them.

Evidence motivating this direction:

- temporal stability selection failed;
- forced temporal invariance failed;
- static prototypes failed;
- linear trajectories failed;
- MULTISCALE memory improved LTD because it preserved multiple temporal
  states instead of collapsing them;
- XGB remains the strongest Macro source of Future-B robustness.

Next:

11H — Frozen Temporal Environment Ensemble.


---

## 11H — Frozen Temporal Environment Ensemble

Status:
CLOSED — POSITIVE SIGNAL, FAILED FORMAL PROMOTION.

Scientific status:
FROZEN MODEL TEMPORAL GENERALIZATION.

No online adaptation.
No parameter updates at inference.
No memory updates at inference.
Future-B not used.

Control:
UNIFORM XGB + canonical RECENT5 LTD.

Candidate selected on ORIGIN14:
TEMPORAL_SYMMETRIC3.

The candidate combines:
- EARLY-weighted XGB
- UNIFORM XGB
- LATE-weighted XGB

All models use BASE128 and the same XGB architecture.

Transfer result:

ORIGIN14:

NEAR:
- XGB Macro-F1: +0.59 pp
- Macro Fusion F1: +0.17 pp

MID:
- XGB Macro-F1: +0.69 pp
- Macro Fusion F1: +0.66 pp

FAR:
- XGB Macro-F1: +0.61 pp
- Macro Fusion F1: +0.47 pp

ORIGIN28:

NEAR:
- XGB Macro-F1: +0.74 pp
- Macro Fusion F1: +0.38 pp

MID:
- XGB Macro-F1: +0.49 pp
- Macro Fusion F1: +0.34 pp

FAR:
- XGB Macro-F1: +0.72 pp
- Macro Fusion F1: +0.27 pp

The Macro-F1 improvement is positive in all six temporal
origin/window evaluations.

FAR Macro-F1 bootstrap:

ORIGIN14:
- delta: +0.47 pp
- 95% interval fully positive
- fraction_delta_gt_0 = 0.998

ORIGIN28:
- delta: +0.27 pp
- fraction_delta_gt_0 = 0.941
- confidence interval slightly crosses zero.

Mean FAR Macro-F1 gain:
approximately +0.371 pp.

Promotion threshold:
+0.500 pp.

Decision:

Formal promotion gate fails only because the mean FAR effect does not
reach the predeclared +0.50 pp threshold.

TEMPORAL_SYMMETRIC3 is therefore NOT promoted, but it is retained as
the strongest frozen temporal-generalization candidate of Phase 11.

Interpretation:

Training across multiple temporal distributions is a promising
direction for reducing longitudinal degradation without adaptation.

Next:

11I — cross-fitted temporal worst-environment reweighting.

Objective:

Train one frozen XGB by assigning greater training importance to
temporal environments that are hardest to generalize to from the
other historical environments.


---

## 11I — Cross-fitted Temporal Worst-Environment XGB

Status:
CLOSED — FAILED PROMOTION.

Scientific status:
FROZEN MODEL TEMPORAL GENERALIZATION.

No Future-B.
No test-time adaptation.
No parameter or memory update during inference.

Method:

Historical training dates were divided into three contiguous temporal
environments.

Leave-one-environment-out cross-fitting estimated the difficulty of
each temporal environment.

A single final XGB was trained using group-balanced weights derived
from those cross-fitted losses.

Results:

ORIGIN14 FAR:
- XGB Macro-F1 vs UNIFORM: -0.18 pp
- Macro Fusion F1 vs UNIFORM: -0.13 pp

ORIGIN28 FAR:
- XGB Macro-F1 vs UNIFORM: +0.11 pp
- Macro Fusion F1 vs UNIFORM: +0.30 pp

Mean FAR Macro-F1 gain vs UNIFORM:
approximately +0.084 pp.

Mean FAR Macro-F1 relative to 11H:
approximately -0.287 pp.

Decision:

11I is rejected.

TEMPORAL_SYMMETRIC3 from 11H remains the leading frozen
temporal-generalization candidate.

Interpretation:

Temporal environment difficulty is measurable, but collapsing that
information into a single one-step reweighted XGB does not reproduce
the benefit of maintaining multiple temporal hypotheses.

Technical note:

The provisional log-loss calculation emitted probability-normalization
warnings because clipped probability vectors were not explicitly
renormalized. This does not affect the reported transfer metrics.
Future DRO experiments will explicitly renormalize probabilities
before log-loss calculation.

Next:

11J — iterative cross-fitted temporal GroupDRO-style weighting.

11J is the final planned experiment in the temporal-reweighting family.

