# P14B — Interpretation of class-wise rank degradation

Date: 2026-10-12. Source: 14A verified 10/10 manifest outputs and 65-domain read-only report on Perseo. **Observational, Future-B OPEN/FROZEN; NO model selection.**

## Observations
18,543 samples / 65 canonical websites. Hybrid 9,170 rank-1 (49.4526%); 5,768 ranks 2–5 (31.1061pp); 3,605 ranks >5 (19.4413%). Of 9,373 Top-1 errors 61.5385% retain true rank 2–5. This is oracle rank headroom only.

Class categories from frozen 14A rule: 41 recoverable_gap, 11 high_top1, 13 residual lost_beyond_top5. **The residual label is NOT a strict class-level low-Top-5 indicator.**

Selected descriptive extremes (site, n, top1, top5):
- transfermarkt.es: 282, 5.32%, 100.00%; 267 recoverable rank2–5; repeatedly confused as bwin.es (237).
- senado.es: 289, 29.41%, 100.00%; 204 recoverable; confused as bwin.es (177).
- saludcastillayleon.es: 276, 19.57%, 97.46%; 215 recoverable; confused as makeup.es (111).
- druni.es: 338, 5.33%, 72.19%; 226 recoverable; confused as marvimundo.com (245).
- primor.eu: 306, 13.40%, 17.97%; 251 outside Top-5; confused as buscounchollo.com (131).
- segurcaixaadeslas.es: 257, 1.56%, 5.06%; 244 outside Top-5.
These sites are *illustrations* of observed failure modes, not permissible Future-selected training targets.

## Candidate explanations NOT YET established
1. Rank inversion under drift while site identity remains represented in top-k.
2. Genuine loss of discriminative signal for specific captures.
3. Stale macro temporal context and component disagreement.
4. Truncation/sequence length and browser/load artifacts.
5. Systematic score bias toward certain predicted classes unrelated to semantic similarity.
These are competing hypotheses, not demonstrated mechanisms. Per-site examples cannot establish cause.

## Outstanding evidence caveats
- 14A drift metric table covers 69 raw classes; canonical cohort comprises 65. Filter by verified canonical mapping before any aggregate drift analysis.
- `14A_component_rescue.csv` has 9,373 rows for Hybrid Top-1 errors. Distinguish component Top-1 actual rescue from top-3/top-5 possible rescue; cannot claim a deployable improvement from oracle statistics.
- Prior 08F confidence gate and 08G learned reranker failed in Historical development; never re-run an equivalent scheme without a new preregistered mechanism and negative controls.
- No Future-B-selected architecture, domain targets, parameters, score transformations or thresholds permitted.

## Decision
Phase 15A-HIST: measure rank reversals and information loss in Historical development using pre-existing scores and class-balanced temporal windows. Follow 15A with separately frozen matched-budget 15B only if warranted. 15C investigates temporal-memory mechanisms if Historical corroborates. Preserve Dataset C as external QA-only until collection quality, clock and transport comparability meet gates.

### Traceability
- `docs/research/P13_B1_CLOSEOUT_20261012.md`
- `docs/research/P14B_PRESENTATION_EVIDENCE_INDEX.md`
- `docs/research/P14_TOP5_GAP_RESEARCH_ROADMAP_20261012.md`
- `scripts/diagnostics/report_p14b_site_domains.py`
