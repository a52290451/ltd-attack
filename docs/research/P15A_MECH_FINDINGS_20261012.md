# P15A-MECH — validated historical paired fusion findings

Status: **execution completed on Perseo**, 18/18 pairs, 13B and 13C frozen hashes checked by code, per-pair metrics cross-checked with existing results. Console ended in `P15A_MECH_COMPLETE`. Four derived CSVs and SHA manifest written to `/home/bsierra/ltd-storage/results/diagnostics/P15A-MECH/`. **Manifest/output hashes have not yet been independently rechecked**; use separate read-only verification before scientific closeout.

## Results
Across 2 origins × 3 windows × 3 Micro candidates (U1, U3, T3_S42), all 18 Hybrid x M0 combinations improve Top-1 against the paired Micro. Day-block bootstrap 95% intervals for Top-1 delta reported wholly positive in all 18 cases.

However:
- **ORIGIN14:** 9/9 Top-5 differences negative; Top-1 improvements positive.
- **ORIGIN28:** 9/9 Top-5 differences positive; Top-1 improvements positive.
- FAR (pp), origin14 U1 +7.106 Top-1 / -0.487 Top-5; U3 +1.723 / -1.723; T3_S42 +2.656 / -2.109.
- FAR (pp), origin28 U1 +14.778 / +2.078; U3 +4.731 / +0.387; T3_S42 +4.543 / +0.564.
- **Paired** FAR U3 origin14: 577 Micro errors rescued, 407 correct Micro predictions harmed, net +170; origin28: 532 rescued, 104 harmed, net +428.
- ORIGIN14 TOP5 decrease persists in all NEAR/MID/FAR × U1/U3/T3_S42, while ORIGIN28 TOP5 improvement persists in all corresponding comparisons. These data support an *origin-dependent effect*, not a causal explanation of why.

## Scientific interpretation and limitations
- Distinct origin cohorts and windows are **not randomized**; comparing ORIGIN14 vs ORIGIN28 does not isolate history length from composition, temporal separation, or environment.
- Confidence/margin/entropy diagnostics by outcomes and within-day/site stability are in derived CSVs, but have **not yet been numerically inspected/reported**. No claims about these statistics beyond previously validated 15A baseline outputs.
- Site-level examples from Future-B must not be used for choosing experiments or fitting thresholds. The historical dataset alone is eligible for candidate design and gate evaluation.
- The site-level CSV may identify composition-driven effects; distinguish class-weighted from capture-weighted analyses.
- Positive descriptive day-block bootstrap CIs are conditional on the observed folds/days and are not a confirmation of causal transportability.
- Prior historical 08F/08G failure implies do not simply retry confidence gating or generic reranking. Target a new independently justified mechanism only if Historical diagnostics indicate it.
- Dataset C continues isolated collection/QA.

## Next gate
1. Independently verify manifest hashes and reconcile all 18 count-derived metrics against 13B/13C.
2. Summarize 15A day-block Top-5 CIs, site-level rescue/harm distribution, and confidence in `harmed_by_hybrid` vs `rescued_by_hybrid` using **only** derived Historical CSVs.
3. Decide whether to preregister 15B rank-specific or 15C temporal-memory work. No promotion, no Future-B tuning.
