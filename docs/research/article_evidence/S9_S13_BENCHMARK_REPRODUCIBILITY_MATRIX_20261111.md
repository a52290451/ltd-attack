# SOTA S9–S13 execution and reproducibility matrix (2026-11-11)

## Status — evidence, not forecasts

> **Author-source-first correction (11 Nov 2026).** DF, VarCNN, Tik-Tok and NetCLR have publicly available implementation artifacts released by their authors. Their current WFlib results remain frozen *adapted/ported baselines*, not preferred primary original-source replications. CUMUL has publicly available third-party reproduced source; original author source availability has not been independently established as a maintained GitHub repository. Refer to `docs/research/article_evidence/S3_S9_ORIGINAL_SOTA_SOURCE_AUDIT_20261111.md`. The WFlib-NetCLR stage runner now requires `LTD_ALLOW_WFLIB_ADAPTED_S9=YES` explicitly to prevent accidental relabeling; no primary original-NetCLR benchmark has been run.

| ID | Model | Original scientific implementation used | Local adaptation | Execution gate |
|---|---|---|---|---|
| S3 | Deep Fingerprinting (DF, 2018) | Pinned [WFlib DF](https://github.com/FIND-Lab/Website-Fingerprinting-Library), not the original authors' unmodified reproduction | Historical DEV chronological B/C and frozen UID-aligned evaluation | Completed |
| S4 | VarCNN (2019) | Pinned [WFlib VarCNN](https://github.com/FIND-Lab/Website-Fingerprinting-Library) | Historical chronological B/C, adaptation for 65 classes | Completed |
| S7 | CUMUL (2016) | **Local original-style feature engineering**, 104 features + scikit-learn RBF-SVM; *not* the authors' original code or exact packet-size pipeline | Historical split and frozen scores; label **CUMUL-style** | Completed |
| S8 | Tik-Tok (2020) | Pinned [WFlib TikTok](https://github.com/FIND-Lab/Website-Fingerprinting-Library); temporal signed-direction inputs derived locally | Custom input adapter + training/evaluation drivers | Completed |
| S9 | NetCLR (2023) | Pinned [WFlib NetCLR and DFsimCLR](https://github.com/FIND-Lab/Website-Fingerprinting-Library) | Custom SimCLR training with batch 64, augmentation and source fixes; own classification B/C and external drivers | Pretrain B completed; B smoke passed; supervised B/C/external **not executed** |
| S10 | RoFiRe (2026) | Official https://github.com/AISec-EWHA/RoFiRe at commit 7ef45ee7c9af9ac04048456992d3d96398791bbc | No LTD 65-class chronological integration yet | Source fetch/audit ready; **training blocked** |
| S11 | Swallow (2025) | Official https://github.com/wujinhe0814/Swallow at commit c6dc304bca6e61b10b36b00178ee84e38c17961b | No LTD source-equivalent pretraining and historical fine-tuning yet | Source fetch/audit ready; **training blocked** |
| S12 | WFCAT (online 2025; 2026 journal issue) | Official https://github.com/websitefingerprinting/wfcat at commit cd382d30f7d33714fa4d025cddb3daaf2ffb0f09 | No validated IAT .cell temporal adapter and chronological evaluation yet | Source fetch/audit ready; **training blocked** |
| S13 | Proteus (NDSS 2026) | Official https://github.com/Xinhao-Deng/Adaptive-WF-Attack at commit 4cdab4163bf3de7036a2498dac533c888b97664d | None; continual adaptation uses unlabeled future data and is **not equivalent** to the no-future-tuning LTD protocol | Related work / separate protocol |

### Why some implementations are adaptations

- S3/S4/S8/S9: WFlib supplies publicly available model implementations with pinned source commit and blob hashes. Our runner changes the training protocol, epochs/selection, input normalization and output schema to enable evaluation on the same held-out site labels. Source-pin ≠ original-paper exact replication.
- S7: CUMUL is based on hand-implemented signed-packet statistics and cumulative features; original differences in length of TCP payload/packet and normalization are explicitly disclosed.
- S9: Original WFlib NetCLR implements the encoder and SimCLR projection head, but the LTD variant uses a modified contrastive pipeline with batch 64, an integer burst/CDF alignment fix and historical-only data. Do **not** call paper-exact.
- S10: original RoFiRe code uses TensorFlow/Keras with pre-generated augmented NPZ and an 8:1:1 split. Running its default experiment would mix Historical DEV and external test definitions; a new chronological adapter must be reviewed.
- S11: authors' code runs pretraining and few-shot fine-tuning with their own scenario files. Their published 65-label adaptation requires verified sampling and UID chronological partition.
- S12: authors' code uses .cell timestamp/direction traces and stratified folds by default. Do not run default training as an LTD result: chronological DEV train/valid and frozen cohort scoring require integration.
- S13: Proteus adapts with unlabeled drift traffic; it **must not** access 09B/Future-B unlabeled traffic under the LTD primary benchmark.

## S9 completed evidence

Pretrain-B: 39,258 Historical DEV_EARLY/MIDDLE, 100 epochs, verified SHA-256:
\`121851a09f2046ea550b8655d31e2af2b8d1d66324b19e57b233d8263d362523\`
Details: \`docs/research/article_evidence/S9_NETCLR_PRETRAIN_B_V2_COMPLETED_20261111.md\`.
No NetCLR accuracy exists until B, C and both external evaluations run.

## Perseo commands

\`\`\`bash
cd ~/ltd-attack
git fetch origin feature/macro-v2-historical-only
git merge --ff-only FETCH_HEAD
PY=/data/ltd-sota-bsierra/venv-cu128/bin/python

# No GPU required. Executes CPU model/signed checkpoint smoke and syntax checks.
bash scripts/sota/run_s9_and_next_sota_v1.sh preflight

# Source checkout: clone official upstream code to separate ~/ltd-ops/sota/upstream_2026
# This accesses network and writes only new source checkouts; no pip install or training.
bash scripts/sota/run_s9_and_next_sota_v1.sh fetch-upstream

# Execute in ORDER when NVIDIA compute process has vacated the GPU:
bash scripts/sota/run_s9_and_next_sota_v1.sh B
bash scripts/sota/run_s9_and_next_sota_v1.sh C
bash scripts/sota/run_s9_and_next_sota_v1.sh 09B
bash scripts/sota/run_s9_and_next_sota_v1.sh Future-B
\`\`\`

Each stage is exclusive (will refuse existing evidence directories). B chooses epoch by DEV_LATE macro-F1; C refits classifier on complete Historical DEV with the historical-only contrastive backbone; external evaluation only reads, emits UID-aligned \`scores_sota_aligned.npz\` with 65 probabilities and hashes. **C is a historical supervised refit, not a second contrastive pretraining pass**. This distinction is required in Methods.

## Gates before attempting S10–S12 training

1. Source pin audit passes on Perseo for all selected upstream repositories.
2. Requirements checked against isolated environment (RoFiRe uses TensorFlow/Keras; Swallow lists Python 3.10; WFCAT uses PyTorch).
3. An audited transformation derives each model's exact feature representation from the frozen vector source and class order, with UID/date sidecars. No random train/test folds.
4. Training uses only DEV_EARLY + DEV_MIDDLE for B, DEV_LATE for model selection, and complete Historical DEV for frozen C.
5. External evaluation preserves exactly 14,687 09B and 18,543 Future-B UIDs and exports full 65-class probabilities, metrics and checksums.
6. No output directory overwrite, external tuning, dataset C access or hidden use of target cohort unlabeled traffic.
7. Memory/compute budget and seed logged. Advance by explicit audit PASS; a pinned repo alone is not a runnable benchmark.

Dataset C QA is temporarily suspended; do not treat its artifacts as labeled evaluation data. No experiment in this list opens Dataset C.
