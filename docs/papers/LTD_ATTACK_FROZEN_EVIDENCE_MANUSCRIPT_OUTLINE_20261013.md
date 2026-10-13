# LTD-Attack — Evidence-grounded manuscript working outline
**Status:** working manuscript foundation, NOT submission-ready, created 2026-10-13. No new experiments permitted during evidence freeze. Source-of-truth: frozen records, not reconstructed recollections.

## Working title
**Measuring Longitudinal Drift and Complementary Traffic Representations in Tor Website Fingerprinting**
The title deliberately does NOT claim robust temporal memory mitigation until specific evidence supports it.

## 0. Central bounded claim
In a previously frozen evaluation of 18,543 Future-B traces from 65 websites, a fixed probabilistic fusion of sequence-based Micro and aggregated-traffic Macro attains +6.3366 percentage points Top-1, +5.1259 pp Macro-F1 and +8.1702 pp Top-5 relative to frozen Micro, with archived positive day-block bootstrap 95% intervals. **Place EXACT interval endpoints from artifact extraction here; do not invent.**
The model is imperfect: 49.4526% Future-B Top-1 vs 97.4399% frozen Internal Test; it mitigates relative degradation but does not solve concept drift.
Post-hoc ablations suggest static XGBoost contributes more to Future-B gain than LTD memory; memory contribution changes with training regime. Post-hoc is NOT a separate independent confirmation.

## 1. Abstract (to draft once evidence ledger is complete)
Problem and threat model: Tor website fingerprinting models drift under evolution of content/network conditions.
Approach: frozen 65-site sequence Transformer (Micro), aggregate XGBoost and longitudinal representation (Macro), fixed weighted geometric score fusion.
Methodology: immutable Historical development and frozen Future-B temporal evaluation. Paired units = capture; correlation/time uncertainty = query-day blocks. Ethics of security/attack study acknowledged.
Key results: quantified frozen paired deltas with *exact* CIs; component/failure modes.
Limitation: no proven causal effect of LTD memory on 10B +6.34pp; no claims from bridged C.
Consequence: informs tradeoffs of complementarity and drift under bounded closed-world threat model.

## 2. Introduction (logic, not marketing)
- Why longitudinal shifts matter for encrypted Tor traffic inference and privacy.
- Which prior website-fingerprinting drift studies exist; what they already solved and their settings (blind evaluation, adaptation vs frozen inference). Do not claim priority without literature review.
- What our work contributes that is not generic model ensembling: a specific empirically audited longitudinal benchmark, methodologically valid fixed-score complementarity decomposition and heterogeneity/negative controls, **if** distinct from related work.
- Contributions numbered and traced to evidence IDs from ledger, not cherry-picked gains.
- Explicit attacker perspective and privacy implications: stronger attacks may worsen user anonymity.

## 3. Threat model and research questions
RQ1 Does the frozen hybrid outperform its frozen Micro component on the same later-collected Tor traces?
RQ2 Are errors corrected because of complementary aggregated features or longitudinal memory; what can we establish observationally?
RQ3 How do improvements and failures vary across time and website and which tradeoffs affect Top1 vs Top5?
Setting: closed world of 65 sites, no live adaptation, no knowledge of future labels during inference, fixed acquisition conditions for A/B as verified from original collector contracts.
Separate questions of direct Tor versus bridge/obfs4 transport; bridged pilot C excluded.

## 4. Materials and methods
- A Historical 72,603 samples, 52 days; DEV 57,916 / 41 days and internal test 14,687 / 11 days; clarify canonical partitions and refit vs DEV_FROZEN.
- B Future 18,543 samples from 2026-03-23 to 2026-04-07, 65 sites; exact acquisition details from original raw metadata, not memory.
- Micro architecture Transformer direction+size sequential, 3,000 positions, 256 hidden, 8 heads, 4 layers, 30 epochs in relevant frozen configuration.
- Macro: feature extraction and XGBoost, LTD memory branch; fusion original 0.45 Micro / 0.55 Macro, internal Macro XGB 0.625 / LTD 0.375 (effective Micro .45, XGB .34375, LTD .20625 in log score).
- Cross-entropy training and model selection strictly before future opening; exact seeds/code version/manifest.
- Ethical protocol, artifacts, reproducibility, privacy consequences and closed-world scope.

## 5. Statistical analysis plan
**Primary confirmatory contrast:** 10B DEV_FROZEN Micro vs exact original Hybrid on matched Future-B captures. ΔTop1 as percentage-point paired estimate; exact 95% day-block bootstrap from 10B/10C archival evidence; report raw captures and number of days and site distribution. Bootstrap is a dependence-aware uncertainty estimate, not magical guarantee of independent samples. Cross-check paired rescue/harm table.
**Secondary endpoints:** MacroF1, Top5, MRR; their archived day-block intervals; site- and day-level heterogeneity descriptive with uncertainty and no multiple-testing false discoveries. Confidence fraction>0 is NOT a frequentist p-value.
**Exploratory:** 10C XGB vs LTD ablations, 14A/B rank gap, 15A paired rescue/harm, 15B–D negative results; clearly distinguish from 10B confirmation, control for multiplicity where inference is made.
**Threats:** short 16-day future horizon, between-day dependence, site-class imbalance, unmeasured collection confounders; no claims of causal mechanism from pure ablations with renormalized weights.
No new training or model selection as part of evidence closeout.

## 6. Results — proposed figures and tables
Figure 1: data collection timeline and A/B frozen split (C bridge quarantine separately).
Figure 2: architecture diagram Micro / XGB / LTD and exact fusion coefficients.
Figure 3: 10B matched Top1/Top5/MacroF1 with 95% paired date-block CI (no false scale).
Figure 4: 10C frozen component effect, using **renormalized deletions** distinct from standalone branch substitutions.
Figure 5: paired rescued/harmed matrix; 15A historical temporal heterogeneity.
Figure 6: complete 65-site rank1 vs rank2–5 vs beyond Top5 distributions; no interpretation as oracle deployable gain.
Table 1: site counts/capture distributions and dates.
Table 2: all 10B exact deltas and CIs including MRR.
Table 3: 11–15 successes, failures, equal-budget caveats and gates.
Supplement: complete 65-site breakdown with class IDs, definitions, SHA and demographics of observed capture dates.

## 7. Discussion
Answer each RQ using its precise support and uncertainty. Static XGBoost complements Micro under Future-B; LTD memory is regime-dependent, not demonstrated to account for the entire improvement. Explain why temporally reweighted Micro T3 does not beat equal-budget U3 reliably; why fusion can lower Top5 while improving Top1. Do not confuse ORIGIN14/28 with randomized memory-age experiment.

## 8. Related work
Original Tor WF attacks; defenses and open-world constraints; website/content drift, temporal evolution, measurement reproducibility; prior adaptive/online drift defenses, especially contemporary 2025/2026 research. Build reference matrix with exact bibliographic details, threat models and matching/nonmatching datasets BEFORE novelty claim.

## 9. Conclusion
Report measured performance improvement and remaining severe temporal decay without claiming solved drift. Propose direct-Tor matched C cohort only after acquisition QA; future independent confirmation, not current evidence.

## Acceptance checklist / blockers
- [ ] Audit extract complete: exact CI endpoints and source SHAs, observed Δ reconstruction, all 65 classes.
- [ ] Original A/B acquisition equivalence contract verified and archived.
- [ ] Component ablation interpretations reviewed (different effective weights / noncausal).
- [ ] Full literature novelty matrix built and co-authors sign off.
- [ ] Ethics and artifact-release plan approved.
- [ ] Supervisor confirms PETS/PoPETs vs conference proceedings eligibility and submission target.
- [ ] Manuscript text fully drafted and peer-reviewed by collaborators.

## Required source links
`docs/RESULTS_HISTORY.md`, `docs/research/PAPER_EVIDENCE_FREEZE_AND_STATISTICAL_AUDIT_20261013.md`, `docs/research/ICORE2026_VENUE_FIT_FOR_FROZEN_EVIDENCE_20261013.md`, `src/models/final/10C_future_mechanistic_ablation.py`, P15D Perseo manifests.
