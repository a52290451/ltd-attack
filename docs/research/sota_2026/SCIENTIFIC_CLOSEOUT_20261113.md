# LTD-ATTACK — Scientific closeout of the seven adapted SOTA baselines

Date: 2026-11-13. Branch: feature/macro-v2-historical-only. Scope: retrospective source-adapted benchmarks.

## Scientific decision

The experiment is complete as a comparison between LTD Micro, Macro and Hybrid and seven identifiable source-adapted SOTA variants. It is NOT a paper-exact reimplementation of the complete original published benchmarks. The original Swallow v1 collapsed and is excluded as a comparator. Swallow BNtrain v2 passed pretraining (100 epochs), chronological B/C, both frozen evaluations, release certification and paired scoring.

On Zeus the established evidence is:

- /data/datasets_brayan/ltd-sota-2026/audits/SOTA_ARTICLE_FINAL_20261113T172826Z/
- /data/datasets_brayan/ltd-sota-2026/audits/SOTA_STATISTICAL_FINAL_20261113T175918Z/

## Cohorts and comparison

Historical 09B contains 14,687 captures on 11 dates. Future-B (10B_DEV_FROZEN) contains 18,543 captures on 16 dates. The 65 monitored class labels follow a canonical fixed order. Training and epoch selection are based on Historical DEV: B trained on 39,258 EARLY+MIDDLE examples and selected with DEV_LATE; C refit used 57,916 historical examples. Future-B was not used for fitting or selecting model hyperparameters.

Four metrics are used: accuracy, macro-F1, top-5 accuracy, MRR. Existing uncertainty intervals come from 2,500 paired date-cluster bootstrap resamples (95% percentile). P values were computed using 2,500 paired whole-date label swaps for each of the 168 contrasts, with a plus-one Monte Carlo correction. Holm correction is reported for the 168-test exploratory family and for a 14-test Hybrid accuracy-focused family. The 14-test family was identified DURING analysis and must be explicitly labeled posthoc, not pre-registered.

## Accuracy results (percentage points)

| Model | Historical 09B | Future-B | Future minus historical |
|---|---:|---:|---:|
| LTD Micro | 95.00 | 43.12 | -51.88 |
| LTD Macro | 84.07 | 22.30 | -61.77 |
| LTD Hybrid | 97.44 | 49.45 | -47.99 |
| DF adapted | 93.65 | 39.23 | -54.42 |
| VarCNN direction only | 91.37 | 44.55 | -46.82 |
| Tik-Tok adapted | 93.99 | 17.68 | -76.31 |
| NetCLR adapted | 93.03 | 28.65 | -64.38 |
| RoFiRe DyWin 10x | 94.21 | 40.57 | -53.64 |
| WFCAT IAT | 94.08 | 3.72 | -90.36 |
| Swallow BNtrain v2 adapted | 92.61 | 26.62 | -65.99 |

The cross-cohort declines here are DESCRIPTIVE. Do not claim statistical significance of temporal change: no corresponding cross-cohort hypothesis test was run. WFCAT's dramatic temporal performance loss is observational, not a validated causal mechanism; timestamp collector provenance remains a methodological caveat.

## Primary accuracy comparisons

- Historical 09B: Hybrid versus strongest SOTA, RoFiRe adapted. Delta +3.234 pp, paired 95% CI [+2.826, +3.569] pp, Monte Carlo two-sided p 0.001200, Holm-14 posthoc p 0.007197, Holm-168 exploratory p 0.098361.
- Future-B: Hybrid versus strongest SOTA, VarCNN direction only. Delta +4.902 pp, paired 95% CI [+3.469, +6.237] pp, Monte Carlo two-sided p 0.000400, Holm-14 posthoc p 0.005598, Holm-168 exploratory p 0.067173.

All 14 Hybrid-versus-SOTA accuracy contrasts are positive and significant under the 14-comparison posthoc Holm correction of the original Monte Carlo analysis. None of those 14 is significant at p less than 0.05 under the global Holm-168 correction. These are different multiple-testing families and must NOT be conflated. Absence of Holm-168 significance does not establish equivalence.

The dedicated scientific closeout generator independently computes EXACT two-sided date-block sign-flip p values for these 14 focal accuracy comparisons, enumerating 2^11=2,048 or 2^16=65,536 possible day-level swaps, and adjusts the exact p values with Holm-14. This avoids Monte Carlo uncertainty for the 14 focused tests. The existing 168 Monte Carlo p-values and bootstrap CIs remain frozen and unmodified.

## Model fidelity / scope limitations

- DF: author-sourced DFNet adapted to 65 classes, Keras compatibility and common chronology.
- VarCNN: author directional ResNet branch only; not its full published multimodal architecture/ensemble.
- Tik-Tok: author architecture and directional-time representation adapted to the LTD traces/protocol.
- RoFiRe: author classifier and DyWin augmentation, but 10x not author 200x and altered augmentation source quantity.
- NetCLR: author pretraining modules, but changed epochs, batch size and burst-CDF implementation.
- WFCAT: author network and IAT features; adapted epochs/batch and unresolved acquisition-time-unit provenance.
- Swallow BNtrain v2: author CIF/ResNet and original target training mode restored, but 100 rather than 200 epochs, batch32 rather than batch512, and no full original 6x augmentation multiplier. Original collapsed v1 must never enter published model rankings.

Only a single seed is available per adapted source model. Date-level resampling does not quantify seed-to-seed variation.

## Supported claim

On 09B and Future-B frozen retrospective cohorts, LTD Hybrid obtained higher ACCURACY than each of seven named source-adapted SOTA implementations. The paired accuracy differences to the strongest adapted competitor have bootstrap CIs excluding zero. P-value conclusions must be qualified by the chosen posthoc/exploratory family, exchangeability assumptions, and number of date clusters.

## Unsupported claims

- Universal/prospective superiority beyond the frozen 65-site cohorts.
- Faithful numerical reproduction or full author-paper methods.
- Hybrid dominance in every metric (Future-B VarCNN direction Top-5 is 80.98%, versus Hybrid 80.56%; raw p for that Top-5 contrast was 0.4574).
- Statistical significance of cross-cohort temporal degradation or causal explanation of WFCAT's drop.
- Pre-registered or prospectively confirmatory status of the Holm-14 family.

## Publication handoff

Run scientific_closeout_seven_sota_v1.py with the existing article and statistical evidence. It produces English Markdown analysis, two main LaTeX tables, a manuscript-results LaTeX section, a CSV of 14 exact p-values/ICs, the unchanged 168 p/CI CSV, 10-model metrics and changes over time, and a provenance certificate. Compare exact decisions with the 2,500-draw Monte Carlo decisions, then incorporate tables and limitations into the paper. Do not restart training or modify frozen checkpoints for this publication scope.

Scientific benchmark phase: COMPLETE as adapted-source comparison. Scientific manuscript package generation and incorporation: next actions.
