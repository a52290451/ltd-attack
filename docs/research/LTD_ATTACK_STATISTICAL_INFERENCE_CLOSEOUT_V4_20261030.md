# LTD-Attack — Statistical inference closeout V4 (2026-10-30)

**Status:** COMPLETE for the retrospective paired 09B/10B inference audit; **NOT** prospective independent confirmation. Archived model scores and Future-B labels were not altered. No retraining, selection, or tuning was performed.

## Sources and reproducibility

- Parent frozen V1 manifest SHA256: `5d38ffb7d3a4ec2d6a01913019141250786e1f9af9f2a2d310e3748054f968c8`.
- Parent V2 manifest SHA256: `3224d634527e9df03d4d6d784cce56f47cf019feb7d9d307751b39ccb2ecc2a6`.
- Original 09B scores SHA256: `5b13d0c16bf4a2d418eddf2dbd11ffca42b5f45485d731d9b9680f5c5b69d0f3`.
- Original 10B DEV_FROZEN scores SHA256: `d0edcd2780454b5cd39ef962ec45375d8822dc26b4b3c0b02cf4c06b0b29c7bb`.
- Source program: `scripts/diagnostics/audit_frozen_pairwise_stats.py`; V2 sensitivity `scripts/diagnostics/closeout_frozen_evidence_v2.py`.
- NEW V4 Git evidence: [42 independently reproduced exact paired swaps](./evidence/FROZEN-STATISTICAL-INFERENCE-V4_20261030.csv). All 42/42 results and observed effects match V2 to numerical tolerance. The local V4 reproducibility bundle also checks V1/V2 output hashes and retains the original 10C CI-only exports.

## Explicit hypotheses and test procedure

For an evaluation cohort with identical `pcap_uid` across models, define `delta = metric(candidate) - metric(reference)`. The **substantive research hypothesis** is that Hybrid improves the relevant metric, i.e. delta > 0. The **actual p-value test** is **two-sided**: under the *conditional null of whole-day model-label exchangeability*, swapping the candidate and reference labels within an entire day should not change the test-statistic distribution. This is stronger/more specific than merely asserting a population mean difference of zero.

For Top-1, `d_i = I(candidate correct) - I(reference correct)`; aggregate `S_day=sum(d_i within day)`. For each whole-day sign assignment `s_day in {-1,+1}`, compute `T_s = sum(s_day*S_day)/N`. The two-sided conditional p-value is `count(|T_s| >= |T_observed|)/2^D`, with the same construction for additive Top-5/MRR. Exchangeability/symmetry **across whole-day contributions** is essential; a small p does not itself prove that the assumption holds. Independence across adjacent days may fail.

V1 used 9,999 Monte Carlo whole-day label swaps for **all four metrics** and adjusted 12 primary audit tests per cohort (3 Hybrid comparisons x 4 metrics) with **Holm**; this was defined retrospectively, not prospective preregistration. The V1 floor is p=0.0001 (1/(9999+1)); it is not an exact probability. V2/V4 exhaustively enumerate **2^11=2,048** assignments for 09B and **2^16=65,536** assignments for Future-B for the **three additive metrics only**. They apply Holm to 9 additive primary tests per cohort. **Do not mix Holm9 and Holm12 as if they were one family.** Macro-F1 is nonlinear in confusion counts and its p remains the original V1 Monte Carlo+Holm12, not an exact-enumerated p.

Bootstrap *95% confidence intervals* are a separate quantity: V1 draws **2,500** date blocks with replacement (retaining all captures, both models, and class-level counts); the 2.5th/97.5th percentiles form a paired percentile CI. The bootstrap fraction of positive replicates is **not** a p-value. V2 checked 20,000 circular moving-block draws of 1–4 consecutive days for additive outcomes.

### Primary Top-1 comparisons

| Cohort | Candidate − reference | Delta (pp) | Bootstrap CI95 (pp) | Exact two-sided conditional p | Holm9 exact post-hoc | Holm12 Monte Carlo post-hoc V1 |
|---|---|---:|---|---:|---:|---:|
| 09B | Hybrid − Micro | +2.4375 | [+2.0786,+2.8104] | 0.0009765625 | 0.0087890625 | 0.0096 |
| 09B | Hybrid − Macro | +13.3724 | [+12.5832,+14.0975] | 0.0009765625 | 0.0087890625 | 0.0096 |
| 09B | Hybrid − XGB | +14.9588 | [+14.4688,+15.4718] | 0.0009765625 | 0.0087890625 | 0.0096 |
| Future-B | Hybrid − Micro | **+6.3366** | [+5.8663,+6.7497] | **0.0000305176** | **0.0002746582** | 0.0012 |
| Future-B | Hybrid − Macro | **+27.1477** | [+26.2219,+28.0356] | **0.0000305176** | **0.0002746582** | 0.0012 |
| Future-B | Hybrid − XGB | +26.1177 | [+25.1097,+27.0053] | 0.0000305176 | 0.0002746582 | 0.0012 |

For Future-B Hybrid−Micro the contingency is **1,865 rescued / 690 harmed = +1,175 net correct** on 18,543 captures, with day gains on 16/16 dates. For Future-B Hybrid−Macro: **5,821 rescued / 787 harmed = +5,034 net correct**, also 16/16 dates. These direct paired observations plus the CIs and conditional p-values support within-benchmark advantages of the original Hybrid.

**Macro-F1 support:** V1 reports positive delta and CI95 excluding zero for all three primary Hybrid comparisons in both periods, with retrospective Monte Carlo Holm12 adjusted p=0.0096 (09B) and 0.0012 (Future-B). The exact additive p-values cannot be assigned to Macro-F1.

## The four original research questions

1. **Hybrid vs Micro, matched captures:** favorable positive and statistically supported *within 09B/10B*, not generalization guaranteed outside them.
2. **Hybrid vs Macro, matched captures:** favorable positive and statistically supported under the same assumptions and limitations.
3. **XGBoost added contribution:** in **10C Future-B post-hoc renormalized ablation**, Hybrid−without-XGB Top1 +4.7943 pp CI95 [+4.2763,+5.2801]; Macro-F1 +3.4715 pp CI95 [+2.8854,+4.0168]; Top5 +8.3158 pp CI95 [+7.8053,+8.8937]; MRR +0.057373 CI95 [+0.054203,+0.060405]. The archived 10C V1 extracts contain **CIs but no independent day-swap p** for this ablation. It is not a causally isolated XGB-only effect because the remaining weights were renormalized.
4. **Does LTD add a real gain?** Macro−XGB Top1 +1.5864 pp [1.0471,2.1228] internal, **−1.0300 pp** [−1.5447,−0.5148] Future-B. In the **10C Hybrid vs no-LTD renormalized** ablation, Top1 +0.2912 pp CI95 [−0.1569,+0.7114] crosses zero, Macro-F1 +0.1538 pp CI95 [−0.3547,+0.6445] crosses zero, and Top5 **−0.7011 pp** CI95 [−0.9636,−0.4813]. No robust incremental LTD benefit demonstrated for Future-B in this configuration.

## What remains to make a stronger publication claim

- Future-B was already opened and analyzed repeatedly; V1/V2/V4 are **retrospective reanalyses**. The 65-site set was partly conditioned on B coverage. The observed p-values do not repair model/feature/site-selection bias or constitute preregistered confirmation.
- Only 11/16 dates and possible cross-day dependence; day-level label-swapping exactness is **conditional on its null assumptions**, not a universal frequentist guarantee. Report block-length sensitivity and seek an independent statistical review for high-stakes inferential wording.
- Source acquisition contract A/B (Tor version, transport, observation point, filter, capture handling) and canonical site-ID mapping still require primary evidence.
- For 10C's own randomization p-values, rerun a **read-only** conditional test using preserved per-day paired prediction contrasts / per-class counts or UID-level scores, retaining SHA; do **not** derive a p-value from `fraction_delta_gt_0`.
- External blind prospective assessment and SOTA with a single frozen, matched protocol are still missing.

**Status:** V4 closes the **already-computed p-value provenance, exactness checks, Holm families, explicit null and statistical reporting** for primary frozen comparisons. It **does not** certify an external blind SOTA superiority or a fully submission-ready paper.

References: MacKinnon, Nielsen & Webb, J. Econometrics (2023), https://doi.org/10.1016/j.jeconom.2022.04.001; `https://pmc.ncbi.nlm.nih.gov/articles/PMC10066020/`; SciPy paired permutation documentation.
