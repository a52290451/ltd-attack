# LTD-Attack — Evidence freeze and statistically defensible paper (2026-10-13)

**Decision:** STOP new model experiments, weight searches, temporal-memory variants, and Dataset-C-based comparisons while the paper-evidence audit is conducted. Historical and frozen Future-B artifacts MUST NOT be overwritten. Post-hoc statistical extraction from frozen predictions is allowed with explicit labels, SHA256 provenance and no parameter/model selection. Next research stages require an explicit new protocol and approval.

## Scientific questions, not stage numbers
RQ1 — How much does frozen Micro+Macro improve generalisation under longitudinal drift relative to frozen Micro on exactly matched Future-B captures?
RQ2 — Which component is associated with the observed gains (Micro, static XGBoost aggregate features, or LTD longitudinal memory)?
RQ3 — How reproducible and heterogeneous are the gains across dates and 65 websites, and which failure modes remain (rank inversion, outside Top-5, rescue vs harm)?
RQ4 — What has *not* been established: causal attribution to a specific feature, independent validation of post-hoc-selected variants, complete mitigation of concept drift, or comparability of the obfs4 Dataset C pilot.

## Canonical evidence already archived
- 10B frozen DEV_FROZEN, same 18,543 Future-B captures / 65 sites (2026-03-23..04-07):
  - Micro Top1 0.431160; Hybrid Top1 0.494526; paired delta **0.063366 = +6.3366 pp**.
  - Hybrid–Micro Macro-F1 delta **0.051259 = +5.1259 pp**, Top5 delta **0.081702 = +8.1702 pp**, MRR delta **0.069147**.
  - 10B day-block bootstrap intervals for these four differences reported positive. The EXACT CI endpoints must be extracted from archived files before writing inferential claims; do not invent a p-value or a CI.
  - Hybrid Internal Test accuracy 0.974399 vs Future 0.494526; absolute observed drop ~47.99 pp (distinct cohorts, NOT a paired effect).
- 10C **post-hoc Future-B** component ablation (not model selection):
  - Micro 43.12%; Micro+XGB branch substitution 49.70%; Micro+LTD branch substitution 36.23%;
  - Hybrid without LTD, renormalized 49.16%; Hybrid without XGB, renormalized 44.66%; actual 49.45%.
  - Static XGB is empirically the stronger source of Future-B complementarity in this comparison; LTD's marginal contribution is regime-dependent. Note different fusion-weight structures in branch substitutions vs ablations.
  - HISTORICAL_FINAL separate regime: actual Hybrid 30.24% versus without LTD 27.91%, so it is INCORRECT to claim LTD is always useless.
- 11A–11L: documented Historical feature/memory controls and failures; specifically multiscale-5 memory improved LTD-only FAR in 11B but missed downstream promotion; do not omit negative studies.
- 12A robust Hybrid integration missed formal gate; 13B equal-budget temporal T3 vs U3 missed FAR origin consistency; 13C nine-expert T9 descriptive, NOT equal compute.
- 14A/14B frozen post-hoc Future rank positions: 9,170 rank1, 5,768 ranks2–5, 3,605 beyond Top5; the 31.11 pp gap is an ORACLE ranking opportunity, not deployable accuracy.
- 15A-MECH frozen Historical paired Micro vs Hybrid: Top1 gain in all 18 comparisons; ORIGIN14 Top5 decreases in 9/9; ORIGIN28 Top5 increases in 9/9. ORIGIN14/FAR U3 rescued 577 harmed 407; ORIGIN28/FAR rescued 532 harmed 104. Mechanistic DESCRIPTIVE, not feature-level causal evidence.
- 15B geometric vs arithmetic/RRF fixed fusion: both fail preregistered FAR Macro-F1 promotion; 15C/D frozen M0/M1 shows no consistent Hybrid gain and decreases Top5 in both FAR origins; P15D 12/12 comparisons, 5/5 SHA, 2/2 math tests verified on Perseo 2026-10-12.
- Dataset C initial three Tor-instances pilot with bridge/obfs4 is **quarantined / non-comparable to original A/B acquisition** until a new documented provenance and direct-Tor equivalence audit. DO NOT use C for fitting, selecting, testing model candidates, or claiming generalisation.

## Required statistical evidence ledger — READ-ONLY, no new modelling
Create `PAPER_EVIDENCE_LEDGER.csv` with columns:
claim_id, hypothesis, population, n, days, sites, model_A, model_B, feature_contract, statistic, observed_A, observed_B, paired_delta, unit, ci_method, ci95_low, ci95_high, preregistered_or_posthoc, source_file, source_sha256, experiment_commit, guardrail_status, limitations, paper_location.
Each empirical claim in manuscript must cite an artifact SHA and distinguish training/internal vs future/heldout, DEV_FROZEN vs HISTORICAL_FINAL, and original vs post-hoc.

1. **Integrity and matching:** verify original manifests and SHA; same trace UID, query date, 65-label ordering and n for each compared row; ensure no leakage from selected Future-B results. Validate denominators and all per-site totals sum to 18,543.
2. **Primary paired metrics:** observed ΔTop1 (pp), ΔMacroF1 (pp), ΔTop5 (pp), ΔMRR for frozen Micro vs frozen Hybrid; reproduce day-block paired bootstrap percentile 95% CI already archived (2,000 repetitions / fixed seed), and assess sensitivity to cluster assumptions. Keep original CIs unchanged; new reanalysis must be clearly labelled.
3. **Count-based paired correctness:** compute a 2x2 confusion-of-correctness table: both right, Micro-only right (Hybrid harms), Hybrid-only right (rescues), both wrong. Verify ΔTop1=(rescues−harms)/n. The raw paired counts / ratio are essential to explain complementarity, and can be calculated from frozen scores without retraining. Avoid naive independent-proportion tests.
4. **Robustness:** day-level time series, site-level deltas for all 65 sites, day-block bootstrap for paired metrics; possibly site-level signed-rank and hierarchical/two-way cluster sensitivity where defensible. Explicitly state only 16 future dates and dependence among repeated visits.
5. **Macro-component attribution:** start from frozen 10C original ablations, contrasts at identical traces. Inspect 10C per-day, per-class, rescue and bootstrap artifacts. Compare standalone and the *renormalized component-deletion* variants; do not infer causal mechanisms from branch substitutions with different weights. Do NOT fit or optimise any new ensemble on Future-B.
6. **Selective inference:** predeclare primary endpoint for evidence synthesis (the frozen 10B Top1 delta), secondary endpoints MacroF1, Top5, MRR; distinguish 10B confirmatory from 10C/14A posthoc, 11–15 historical development. Multiple exploratory comparisons do not inherit a confirmatory p-value.
7. **Reproducibility:** publish exact configs, hashes, code commits, metadata schema, limitations (Tor setup, temporal distribution, class set, closed-world scope, ethics and artifact release). A descriptive non-causal result must not be worded as proven causal feature importance.

## Hard manuscript gates before venue selection
- Numeric results independently cross-checked against immutable 10B, 10C, 13–15 reports; unresolved discrepancies flagged instead of guessed.
- Main claim is truthful: *frozen complementary representations improve observed performance relative to Micro*, **not** “LTD memory solely causes +6.34 pp”.
- Distinguish matched compute from T9 nine-model budget and same-cohort comparisons.
- Provide complete 65-site table with aggregate reconciliation and appropriately qualified site-level findings.
- Related work review must include contemporary concept-drift website-fingerprinting defenses, timing/acquisition bias, study ethics and attack validity.
- Dataset C bridge pilot documented separately; replacement direct capture is NEW cohort, not retroactive evidence in this paper.
- Paper outline: Motivation/privacy implications; research questions & threat model; A/B capture methodology; architecture (Micro, XGB, LTD, fusion); frozen experiments & statistics; component attribution; temporal/site heterogeneity; negative controls; threats/limitations; related work; ethics/reproducibility; conclusion.
- Before submission obtain co-author/doctoral-supervisor approval on originality, contribution level (diagnosis vs mitigation), and venue's publication category.

## Source provenance
`docs/RESULTS_HISTORY.md`; `src/models/final/10C_future_mechanistic_ablation.py`; `docs/research/P15A_MECH_CLOSEOUT_20261012.md`; `docs/research/P15B_INTERPRETATION_AND_P15C_DECISION_20261012.md`; `scripts/diagnostics/run_p15d_and_milestone.py` and Perseo result manifests.
