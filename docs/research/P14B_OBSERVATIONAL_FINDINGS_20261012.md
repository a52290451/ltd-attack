# P14B-OBS — site-level observational evidence (2026-10-12)

**Status:** 14A diagnostic files verified 10/10 on Perseo; global sample and per-site summaries reproduced. No model fit, no parameter selection. Source: `/home/bsierra/ltd-storage/results/final/FUTURE-RANK-GAP-PHASE14`.

## Frozen outcomes
- 18,543 Future-B captures, 65 canonical classes. Hybrid Top-1 9,170 (49.4526%). Rank 2–5: 5,768 (31.1061pp). Beyond Top-5 3,605 (19.4413%).
- 61.5385% of the 9,373 Top-1 errors have the true class at ranks 2–5 (oracle headroom **not** deployable performance).
- Fixed predeclared `14A` category rule: `high_top1` (11 classes), `recoverable_gap` (41), `lost_beyond_top5` (13). **The final label is a fallback category**, NOT proof that a site's Top-5 accuracy is low; e.g. label 43 has Top-5 96.15%.
- Opposing examples: numeric label 97 Top-1 5.3191%, Top-5 100%; numeric label 77 Top-1 1.5564%, Top-5 5.0584%. These demonstrate distinct observed rank profiles; no causal interpretation.
- Confusion 97→86 occurred 237 times with true site Top-5 fraction 1.0 for this edge. Confusion 14→24 131 times with true site Top-5 fraction 0.007634. These are not validated external mechanisms.

## Methodological caveat
`14A_per_site_drift_metrics.csv` has **69 rows**, whereas the canonical holdout has **65 classes**. Do not interpret aggregate correlations or rankings as canonical without exact mapping/filtering. The prior `14A_correlations.csv` should remain descriptive and flagged as subject to this cohort-alignment audit. Large shift magnitudes should be investigated for scale/outlier dominance before an explanation is asserted.

## Immediate follow-up
- `scripts/diagnostics/report_p14b_site_domains.py` verifies 10/10 SHA256 entries and maps site_label→site from the frozen input CSV; it prints a 65-site annotated summary and top confusion edges. It makes no changes.
- Next, analyze per-site confusion and rescue counts without selecting candidate models, thresholds or architecture from B.
- Build **15A-HIST** Historical-only evidence and freeze criteria in a YAML before any newly trained 15B study. Keep 08F/08G failures as mandatory negative controls, matched budget, two temporal origins, day-block bootstrap.
- Dataset C stays operationally independent and QA-only during its 24-hour pilot; NTP not synchronized.

## Provenance
Perseo outputs provided in the ChatGPT log on 2026-10-12; this GitHub document contains a textual summary, not copies of the frozen CSVs. `14A_INTEGRITY_VERIFIED 10/10`, `EXCLUSIVE_RANK_BUCKET_SUM 18543` were observed in console output. Raw input and generated diagnostics are not committed to Git.
