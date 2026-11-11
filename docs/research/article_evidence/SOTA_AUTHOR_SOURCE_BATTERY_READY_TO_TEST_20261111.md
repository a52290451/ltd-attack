# LTD-Attack — Author-source-first SOTA battery (11 Nov 2026)

Branch: feature/macro-v2-historical-only. Status: scripts published, but new training/evaluation stages have NOT run on Perseo. No new author-source SOTA scores are claimed.

## Source provenance and methodology

| ID | Upstream author code | LTD implementation | Status |
|---|---|---|---|
| S3 DF | deep-fingerprinting/df @ 38df0c15 | Authors' DFNet extracted from original Keras source; chronological 65-class B/C/eval | Unverified runtime |
| S4 VarCNN | sanjit-bhat/Var-CNN @ e5db76a8 | Authors' ResNet18 direction BRANCH only, not original full ensemble | PARTIAL architecture |
| S7 CUMUL | lsvih/CUMUL @ da1297a5 | Public third-party code incomplete: utils/data.py get_feature ends after four simple features, without interpolation or return | BLOCKED original implementation; old CUMUL-style preserved |
| S8 TikTok | msrocean/Tik_Tok @ 0aa56660 | Authors' ConvNet, historical DT feature representation | Unverified runtime |
| S9 NetCLR | SPIN-UMass original @ c53ca10d | Authors' DFNet, DFsimCLR and NetAugment source code; Historical DEV-only pretrain B, supervised B/C, UID evaluation | Unverified new execution |
| S10 RoFiRe | AISec-EWHA/RoFiRe @ 7ef45ee7 | Authors' DFVarNet and RMSNorm plus DyWin from authors' feature code | Adapted augmentation rate and multiplier |
| S11 Swallow | wujinhe0814/Swallow @ c6dc304b | Authors' CIF, ResNet18, MLPHead, augmentation and BYOL; explicit upstream infinite-loop fix | Unverified new execution |
| S12 WFCAT | websitefingerprinting/wfcat @ cd382d30 | Authors' IAT feature code and WFCAT architecture | Unverified new execution |
| S13 Proteus | Xinhao-Deng/Adaptive-WF-Attack @ 4cdab416 | Pinned original reference only | External unlabelled drift adaptation incompatible with primary frozen evaluation |

All model code is read from pinned clean author repositories at runtime. Compatibility fixes and deviations are declared; this is not a claim of paper-exact replication. Existing frozen WFlib/CUMUL-style results remain unchanged.

## Perseo: no-GPU phase

1. Update feature/macro-v2-historical-only using git fetch and git merge --ff-only.
2. Run bash scripts/sota/run_original_sota_battery_v1.sh cpu-preflight .
3. Create isolated Keras/TensorFlow environment when necessary: bash scripts/sota/setup_isolated_authors_keras_v1.sh install . This does not replace the existing PyTorch environment.
4. Run bash scripts/sota/setup_isolated_authors_keras_v1.sh check .
5. Repeat cpu-preflight so it also tests original Keras models.
6. Run bash scripts/sota/run_original_sota_battery_v1.sh cpu-prepare to build historical Swallow CIF and WFCAT IAT data caches on CPU (can be disk/CPU intensive).

## When GPU is free

Run bash scripts/sota/run_original_sota_battery_v1.sh gpu-all .

This sequentially executes S9 NetCLR, S11 Swallow, S12 WFCAT, then original DF, VarCNN direction branch, TikTok, RoFiRe; the GPU guard refuses concurrent training. All stages fail on preexisting result outputs instead of overwriting prior data. Final independently checks exact UID/date/labels, 65-class probabilities, accuracy, macro-F1, top5, MRR and source hashes for 09B (14,687) and Future-B (18,543).

Run the CPU preflight first. It may expose compatibility failures; the new scripts were written and published without direct execution in Perseo. No accuracy or successful end-to-end training is asserted yet.

## Scientific gates

- Never permit DEV_LATE for B pretraining; DEV_LATE is validation for supervised B and all Historical DEV is allowed for frozen C.
- 09B/Future-B only for frozen evaluation; never for training, unsupervised fitting or hyperparameter choice.
- CUMUL original public code remains incomplete. Do not relabel old CUMUL-style score as original.
- Current VarCNN version is direction-only, not the original full direction/timing/metadata ensemble.
- Swallow authors' aggregation has an infinite-loop bug with initial handshake; isolated in-memory compatibility patch is marked explicitly.
- RoFiRe DyWin multiplier reduced from the original example; NetCLR has explicit 100-epoch/batch64 and CDF compatibility adaptations.
- TensorFlow CUDA support on RTX5080 must be checked after isolated environment install before any GPU Keras training.
- Dataset C remains suspended and is not used by these pipelines.
