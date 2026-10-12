# LTD-Attack — Evidence-led research roadmap after 49.45% / 80.56%

## Research question
Why does frozen Hybrid retain 80.56% true-site Top-5 while only 49.45% reaches Top-1 under Future-B drift? Is the gap caused by loss of calibration/rank order, confusion among similar websites, component disagreement, stale longitudinal memory, or genuine disappearance of class signal?

**Diagnostic vs confirmatory firewall:** Future-B may be used for **post-hoc explanation only** (14A), never for ranking feature selection, fitting parameters, thresholds, architecture search, promotion, or repeated optimization against B. Historical development only for new hypotheses and preregistration. Dataset C remains external QA/holdout until scientifically usable.

## Existing evidence
- 10B fixed Hybrid: 49.45% Top-1, 80.56% Top-5, N=18,543, 65 sites. `14A_rank_distribution.csv`: 5,768 samples true rank 2–5 and 3,605 beyond top five.
- 14A per-site and per-sample diagnostics exist locally (see P13_B1_CLOSEOUT); numerical **site-specific causal interpretations not yet audited or committed**.
- 13B equal-budget control T3_S42 vs U3 fails preregistered gate; T3_S73 positive exploratory.
- 13C T9xM0 wins FAR descriptively but with nine-expert budget; no promotion.
- Prior 08F confidence gate and 08G candidate reranker failed to improve on Historical Fold-B. Any new ranking experiment must change a justified mechanism and include these controls; do **not** just retry the failed recipes.

## Stage 14B-OBS — diagnostic inventory only, NO experiments on Future
Write a reproducible, read-only report from existing 14A CSV and manifests:
- per-site n, Top-1, Top-5, Top5-minus-Top1, MRR, true rank median, count lost vs recoverable.
- compare labels high_top1/recoverable_gap/lost_beyond_top5 fixed by `FUTURE_RANK_GAP_V1.yaml` and count sites by bucket;
- confusion edges (true site -> predicted site) and errors clustered by category; note ties/sample-size and top-5 headroom;
- characterize sequence lengths/truncation, calibrated confidence proxies and component rescue conditional on rank class;
- site features/drift similarity: descriptive only; prohibit choosing targets to tune based on observed Future gains.
- audit checks: all 65 classes represented, metrics weighted correctly, numerical agreement with published 10B and 14A rank distribution, SHA hashes and source provenance.
Output descriptive tables / plots for paper; no new model outputs, no tuning, no test-selection.

## Stage 15A-HIST — preregistered matched-budget rank mechanism diagnostics
Data: Historical DEV ORIGIN14/ORIGIN28 NEAR/MID/FAR. No read of Future-B data by code.
Reproduce Top-1/2–5/>5 distribution for frozen U1/U3/T3_S42 and baseline Hybrid at each Historical window.
Required stratification: predicted confidence, top1-top2 margin, entropy, component agreement/disagreement, class/category, capture length and truncation, memory age and temporal gap. Fix bins using Historical EARLY (or prespecify fixed bins) and report sample counts.
Question: Is class still in top-k but ranking reverses as temporal gap increases, controlling class composition?

## Stage 15B-HIST — candidate discrimination ablation (preregister BEFORE run)
Three fixed variants with equal compute and identical candidate lists/splits:
- **B0:** current frozen Hybrid geometric score, same rank baseline.
- **B1:** Hist-only out-of-fold rank calibration using small monotonic/temperature family, strictly fitted on earlier Historical DEV dates, tested on later dates (no Future/C). Note uniform temperature does not change ranking — include a rank-preserving negative control.
- **B2:** Hist-only pairwise top-k score disambiguation using only pretrained frozen component scores and Historical features, train on earlier Historical and evaluate later ORIGIN14/28; strict site-specific leakage and future-date guards; matched candidate count and compute.
Primary metrics: accuracy, macro-F1 and conditional rank-2–5 rescue *with empirical net top-1 loss and newly broken examples*, not oracle headroom. Secondary Top-5, MRR, ECE/Brier and latency.
Stop/go: require gains on both FAR origins for macro-F1 and accuracy, day-block bootstrap positive fractions >.90; decision threshold min macro-F1 gain >= +0.005 mean FAR; no degrading >.005 Top-5 or either origin by >.0025 Macro-F1. Gate thresholds are a **proposal and must be frozen before execution**; do not retroactively tune them.
Explicit controls: compare against generic ensemble U3 and failed 08G reranker; document why B2 architecture/regularization differs; keep number of parameters, trainable data and inference latency in budget.

## Stage 15C-HIST — temporal memory and drift ablation, only if 15A justifies
Fixed XGB-only vs LTD-only vs original Macro vs Micro-only vs Hybrid, memory W=1/3/5, and stale/fresh History-only contexts; do not search by Future. Build matched time gap, site and traffic-duration strata and paired day-block bootstrap. Mechanism endpoints: class Top-5 retention, top-1 rank reversal, memory staleness and site heterogeneity. Gate predeclared before evaluation.

## Decision after 15A
If mostly rank misordering: run 15B. If mostly Top-5 signal destruction: prioritize robust representation or acquisition/alignment diagnostics (15C or subsequent preregistered representation study). If signal varies by class: preregister stratified robustness on Historical, not the high-error Future sites. Never declare a rank improvement from oracle rank headroom.

## Reproducibility/engineering contract
Each stage requires YAML config, immutable input SHA256, code SHA256/git commit, fixed candidate matrix, output manifest with SHA256, 65-class mapping, paired UIDs, date-separated train/validation, seeds, bootstrap details, negative controls, and an independent QC gate.
Keep frozen models, Future-B NPZ and Dataset C untouched. Do not run new experiments before Historical-only preflight and protocol freeze.
