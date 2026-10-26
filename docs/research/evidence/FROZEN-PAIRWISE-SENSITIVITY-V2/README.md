# FROZEN-PAIRWISE-SENSITIVITY-V2 — audit supplement

Data: frozen V1 aggregated outputs, never raw PCAP/NPZ. No fitting/selection, no FUTURE tuning.
Input V1 manifest SHA256: `5d38ffb7d3a4ec2d6a01913019141250786e1f9af9f2a2d310e3748054f968c8`.
Bootstrap: circular day-block lengths 1–4, 20000 draws, seed base 20261026.
Block-bootstrap intervals are exploratory sensitivity checks; few (11/16) possibly dependent dates.
Exact day-label sign-flips enumerate all 2^D patterns for accuracy/Top-5/MRR. They are
exact only CONDITIONAL ON WHOLE-DAY MODEL-LABEL EXCHANGEABILITY; temporal serial dependence
is a threat. Holm9 uses 3 primary comparisons × 3 additive metrics; it DOES NOT replace
V1 Holm12 (which also includes nonlinear Macro-F1). Never call these confirmatory p-values.
Per-class deltas are descriptive marginals, NOT paired site×date uncertainty or causal attribution.

## Future B Hybrid vs Micro exact conditional sign flip

- accuracy: delta=+0.06336623, exact conditional p=3.0517578e-05, Holm9=0.0002746582; 65536 permutations
- top5: delta=+0.08170199, exact conditional p=3.0517578e-05, Holm9=0.0002746582; 65536 permutations
- mrr: delta=+0.06914653, exact conditional p=3.0517578e-05, Holm9=0.0002746582; 65536 permutations

## Scope and unfinished evidence
- Macro-F1 cannot be exactly recomputed from per-day macro-F1 averages: requires classwise TP, predicted totals and support in original NPZ.
- Neither paired site×date errors nor cross-site temporal interaction is recoverable from these marginal CSVs.
- Site label to domain mapping must be validated against the canonical frozen encoder; legacy site_dictionary.csv is not proven identical.
- A/B acquisition equivalence and B-conditioned 65-site coverage must be independently documented.
- 10C Historical-Final rescue table is a different checkpoint; no interval for it in the V1 10C bootstrap subset.
- Training/model/weight promotion is forbidden on opened Future-B; next blinded set requires prospective protocol.
- The corrected V2 output manifest records script hash and V1 SHA; preserves all V1 files unchanged.
