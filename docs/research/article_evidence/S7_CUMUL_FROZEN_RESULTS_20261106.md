# LTD-Attack — S7 CUMUL-style frozen evaluation, 2026-11-06

## Evidence provenance
Source: Perseo terminal output supplied by researcher in Chat 4. This document records observed console values; independent review of on-server artifacts and recomputation of paired statistics remain pending. No historical checkpoints or S6 reports modified.

## S7 run and settings
- Branch: feature/macro-v2-historical-only; script: scripts/sota/s7_cumul_frozen_v1.py.
- B trained on Historical DEV_EARLY + DEV_MIDDLE (39,258), evaluated on DEV_LATE (18,658).
- B accuracy: 0.6436917140100761; macro-F1: 0.6206611019033133.
- B checkpoint SHA256: 538de70ce894cdd0344c95a28b3b62ee29c528aa0c38e47abb7e4dad6ceef6d8.
- C refit on Historical DEV 57,916, checkpoint SHA256: 9243a97a6e96d3ecfea6c33235b98164c9c9f096f4c67f5820570fed0c12d532.
- CUMUL-style 104 features, SVM RBF; *not* established paper-exact CUMUL; packet sizes from len(pkt) with possible header overhead.
- scikit-learn 1.9.1 warns probability=True is deprecated; not a training error. Do not change estimator inside current frozen experiment.

## Frozen external evaluation
| Cohort | N | Accuracy | Macro-F1 | Top5 | MRR |
|---|---:|---:|---:|---:|---:|
| 09B | 14,687 | 0.6999387213181726 | 0.6894399001221707 | 0.9325253625655342 | 0.8023257616126463 |
| Future-B DEV_FROZEN | 18,543 | 0.07733376476298334 | 0.028486931696973666 | 0.33646119829585286 | 0.2265473064590403 |

Console status: S7_CUMUL_REAL_TRAIN_PASS C; S7_CUMUL_EXTERNAL_PASS 09B; S7_CUMUL_EXTERNAL_PASS 10B_DEV_FROZEN; S7_COMPLETION_PASS. SHA256 of C model matches both evaluation manifests.

## Descriptive comparison against LTD-Hybrid, frozen S6
- 09B: LTD-Hybrid Accuracy 0.97440; CUMUL-style 0.6999387; delta Hybrid minus CUMUL = +27.4461 percentage points.
- Future-B: LTD-Hybrid 0.49453; CUMUL-style 0.0773338; delta Hybrid minus CUMUL = +41.7196 percentage points.
- This comparison is descriptive. Do **not** recycle S6 Holm8 p-values for S7, claim an independently tested difference, or infer a causal mechanism from the observed temporal drop. New paired per-UID tests and correction family must be computed separately after SHA/UID verification.
- S7 is an additional baseline, not proof of general superiority across all SOTA.

## S8/S9 source pinned inspection
WFlib repo FIND-Lab/Website-Fingerprinting-Library at commit 537ec8e03c805540b04b95b144fa158ce50aa255.
- TikTok model blob: 8736ef4d7d4c00bb1559def532f511cb8e142625; WFlib/models/TikTok.py. PyTorch 1D CNN accepts (N,1,5000), outputs 65 logits if instantiated with 65 classes. Upstream training recipe: Adamax lr 2e-3, 30 epochs, batch 128, feature DT, selection metric F1. The dataset and temporal freeze policy must use LTD Historical DEV, not WFlib CW.
- NetCLR model blob f63cdf80046220baf0544e967ce7a6077a7096e4; augmenter blob e28a587c6caa31b39397354316a7d22edeb4b414; pretrain blob 8d728f5ce324d0f6b59aa293e831a358e312c815.
- NetCLR upstream recipe: 100 pretraining epochs, batch 256, Adam lr 3e-4; supervised training 30 epochs, batch 256, Adam lr 3e-4. Its original shell script uses NCDrift_sup and NCDrift_inf: for LTD, both training and supervised selection must be restricted to allowed Historical DEV, and these are thus modified experimental conditions.
- NetCLR pretrainer hardcodes .cuda() and positive-pair label tensors sized from self.batch_size: handle a shortened last minibatch explicitly (drop_last=True or adapt code), without silently modifying pinned upstream files. Source also needs contrastive augmentation distribution parameters derived solely from Historical DEV.
- A CPU-only source/blob/forward smoke script is supplied at scripts/sota/s8_s9_architecture_smoke_v1.py. Its execution on Perseo is **pending**; do not report PASS without actual console evidence.

## Next steps
1. Execute new smoke with dedicated SOTA Python and inspect output.
2. Implement training-time historical-only dataloader and validate all UID/class/date split contracts, DT semantic mapping and NetCLR augmentor statistics.
3. Train TikTok B on DEV_EARLY+MIDDLE, select epoch by DEV_LATE without external tuning, refit C with fixed recipe, then evaluate 09B and Future-B.
4. NetCLR contrastive pretrain + supervised B and frozen C, then external evaluation with fully traced checkpoints; document modifications from upstream.
5. Compare all models via aligned UID/day paired inference; keep the existing S6 Holm8 separate.

## Safety and reproducibility
No Dataset C access, no changes to C acquisition cron/time sync, no deletion or rewriting experimental evidence, no external evaluation before frozen final checkpoint. Protect storage and GPU resources; avoid simultaneous heavy jobs.
