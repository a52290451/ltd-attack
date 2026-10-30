# LTD-Attack — SOTA fair-comparison protocol (DRAFT, 2026-10-30)

**Status: DESIGN ONLY; NOT EXECUTED.** No baseline has been retrained here. No change to the canonical 09A, 09B, 10B, 10C scores or preapproved presentation.

## Audit of already present SOTA scripts

The repo already includes `scripts/train/run_estado_del_arte.sh` and:
- `src/models/sota/SOTA_01_CUMUL_SVM.py`: loads `new_features_invariantes_seguras.txt`, the old **94-feature future-informed** lineage (see `docs/FEATURE_LINEAGE.md`). Its scaler may fit past data only, but feature selection was not past-only. **Do not treat as a valid unseen-future baseline**, and do not label 94 hand-selected features a faithful CUMUL reproduction without checking the original feature pipeline.
- `src/models/sota/SOTA_02_DeepFingerprinting_CNN.py`: uses 3,000 direction entries and old past/future CSVs with random stratified past split; original DF documentation uses 5,000 entries. Verify architecture and name it *DF-adapted* unless faithful reproduction is demonstrated.
- `src/models/sota/SOTA_03_Rimmer_LSTM.py`: old 3,000-direction LSTM and random past split; source-paper fidelity and strictly chronological validation are not established.

**DO NOT run the old SOTA launcher as a canonical comparison without remediation.** Train vs Future accuracy from these legacy scripts is not the same as the fixed 09B/10B benchmark.

## Candidate baseline families and references

| Candidate | Purpose | Source | Track |
|---|---|---|---|
| Classical CUMUL/SVM | hand-crafted baseline | check the original article and implement past-only features | FROZEN |
| Deep Fingerprinting (DF) | CNN baseline | https://github.com/deep-fingerprinting/df | FROZEN |
| Var-CNN | data-efficient CNN | https://github.com/FIND-Lab/Website-Fingerprinting-Library | FROZEN |
| Tik-Tok | directional-timing approach, only if original timing available | https://github.com/msrocean/Tik_Tok | FROZEN |
| Proteus (NDSS 2026) | unsupervised/target-unlabelled adaptation | https://github.com/Xinhao-Deng/Adaptive-WF-Attack | **SEPARATE ADAPTIVE** |
| Aion (IEEE TDSC 2026) | drift-aware adaptation with limited target labels | https://doi.org/10.1109/TDSC.2026.3723526 | **SEPARATE ADAPTIVE** |

Use paper-matching single-tab closed-world methods as fair direct baselines. Multi-tab/open-world/hidden-service settings are NOT directly comparable. Do not copy their published accuracies as if they arose on our captures.

## Pre-execution gate (must be approved and frozen)

1. Fix the **same 65 labels**, `pcap_uid`, ordering, splits and capture filters: Historical DEV 57,916 / 41 dates; Internal 09B 14,687 / 11; Future-B 10B 18,543 / 16. Verify original A/B acquisition conditions and class mappings from primary logs/collector contracts. If any baseline requires filtering, report the intersection-of-UID cohort explicitly and rerun ALL comparators on it.
2. Train/validate/select hyperparameters on Historical DEV using **chronological** partitions. No Future-B label, feature ranking, distributional threshold, normalization or early stopping during model construction. Future-B has already been inspected: new results are **retrospective** rather than independent blind confirmation.
3. Before executing: pin external repo commits, model architecture/config/seed budget, input representation/direction/time/size and padding length (DF original 5000 vs legacy 3000), training schedule, GPU/CPU time and memory. Match compute budgets for mechanism claims; document unequal budgets otherwise. No adaptive future data unless in a distinct and explicitly declared adaptive track.
4. For every evaluated UID save class-ordered 65-probability vector, date, label, top-k, training checkpoint SHA and input lineage. Reconcile 65 classes, duplicate/missing IDs, sums to 1, matching captures and source hashes against frozen 09B/10B manifests.
5. Predeclare comparison family and primary endpoint (Top-1 or Macro-F1) BEFORE model runs; report Top-1, Macro-F1, Top-5 and MRR; paired rescue/harm, date/site heterogeneity, bootstrap day-block CI, conditional day-swap p with its assumptions and Holm; report performance by seed and training cost.
6. Keep **frozen vs adaptive** track separate: Proteus uses unlabeled target traffic adaptation; Aion includes limited labeled target-domain supervision. They answer different threat models and are not direct frozen-deployment comparators.
7. A **new blind cohort with precommitted protocol** is required to claim independent prospective generalization or confirmatory SOTA superiority. Dataset C bridge/obfs4 pilot is currently not interchangeable with A/B.

## Acceptance before starting any SOTA training

- [ ] Historical preprocessing/label map provenance verified.
- [ ] A/B transport and capture conditions reconciled.
- [ ] 94-feature future-informed selection removed from canonical baseline.
- [ ] Data split and accessible representation identical and leakage checked.
- [ ] Candidate model commits and budget frozen; one comparison family defined.
- [ ] Frozen benchmark remains unchanged; baseline predictions archived by UID.
- [ ] Statistical/ethical reporting and limitations documented.

**Output only after implementation/execution:** `SOTA-FAIR-COMPARISON-V1` with model/config manifests, run logs, comparable scores and statistical contrasts. This file documents the protocol; it is not evidence that SOTA was trained or benchmarked.
