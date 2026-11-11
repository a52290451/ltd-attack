# Audit: authoritative/public source code for already evaluated SOTA baselines

**Date:** 2026-11-11. **Policy from PI:** When original author implementation is publicly available, it is the **primary reproducibility target**. Do not prefer paper-inspired recreations or third-party ports solely because they run in the current venv. Preserve the current frozen outputs as *ported/adapted baselines*, rerun the authors' source under the same Historical DEV chronological protocol as **new independent experiments**, and avoid retroactive score changes.

## Verifiable source provenance

| LTD ID | Work | Upstream author implementation / archive | Pinned commit | Historical LTD results currently use | Action |
|---|---|---|---|---|---|
| S3 | Deep Fingerprinting (2018) | https://github.com/deep-fingerprinting/df , authors' release; e.g. \`src/Model_NoDef.py\` | \`38df0c15a089f13228e4df06bd5269c8a37340fa\` | WFlib PyTorch port | New run using original-author model, adapted 65-site inputs and chronological train/validation. |
| S4 | VarCNN (2019) | https://github.com/sanjit-bhat/Var-CNN , authors' release (archived/read-only); \`var_cnn.py\`, \`run_model.py\` | \`e5db76a86fcbf8839f764b9aca241749d1c1f700\` | WFlib PyTorch port | New run using original architecture and paper-specified branches/ensemble; no default random split. |
| S7 | CUMUL (2016) | https://github.com/lsvih/CUMUL , third-party *public reimplementation*, describes authors' historical code as its source; \`utils/data.py\` contains cumulative signed-byte features. Additional public historical wrapper: https://www.cs.sfu.ca/~taowang/wf/attacks/ | \`da1297a5b3bd8ba0a138e666f55686c109ff69c1\` | Local CUMUL-style 104-feature extractor + RBF-SVM | Compare exact feature vectors/sign conventions/packet sizes and hyperparameter search against cited original before choosing an implementation; do not label the GitHub implementation as author-official. |
| S8 | Tik-Tok (2020) | https://github.com/msrocean/Tik_Tok , authors' release; \`DL_Experiments/cw_attack.py\` | \`0aa56660b435d88f3542939b3f08c0ba1409dbf5\` | WFlib PyTorch TikTok model + local DT adapter | New source-equivalent run using official DT and classifier, isolated legacy runtime/port with verified layer parity. |
| S9 | NetCLR (2023) | https://github.com/SPIN-UMass/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces , authors' release; \`artifacts/src/NetCLR/pre-training.ipynb\`, \`fine-tuning-cw.ipynb\` | \`c53ca10df0cad6bd5c8b27301a10dcc464278187\` | WFlib DFsimCLR/NetCLR adapted to batch 64 and modified augmentor, 100-epoch pretrain B; no 09B/Future-B classification | Convert authors' notebooks into an auditable chronological Historical DEV procedure using original augmentor and encoder/head. This is distinct from the existing pretrain checkpoint. |

### Key distinctions

1. **Published public source does not mean source can run unchanged.** DF/TikTok/VarCNN have TensorFlow/Keras/CUDA assumptions from 2018–2020; original loaders may silently random-split public datasets. Adapt only the I/O, fixed 65-class classifier output, protocol and environment while retaining layer/augmentation parity. If modernization changes topology, explicitly call it a port and demonstrate layer-by-layer equivalence.
2. **WFlib is a public collection of independent PyTorch implementations, not a guarantee that each one is the authors' original implementation.** Its useful results are retained as adapted baselines; they do **not** become historical "original source" results by renaming them.
3. **CUMUL is not a learned deep architecture.** The scientifically significant implementation details are signed-size representation, feature extraction, scaling and SVM-RBF selection. The currently used \`len(pkt)\`-derived sizes may differ from original TCP representations. Source audit must settle this discrepancy before publication.
4. **NetCLR:** The SPIN-UMass repository contains its own original authors' pre-training and closed-world fine-tuning notebooks. Its pre-training uses NetAugment. The currently completed 100-epoch WFlib-adapted pretraining is *not equivalent by default*. Even if the authors' implementation turns out technically incompatible with modern GPU/venv, preserve documented layer, policy and optimizer parity before accepting a faithful port.
5. **Results cannot be substituted.** All previously frozen WFlib/CUMUL-style results remain unchanged and can be labeled adapted. New original-source results must use new paths and SHA/UID manifests. 09B/Future-B are used only for frozen evaluation, never selection, unsupervised adaptation or initial checkpoint choice. Dataset C is excluded entirely.
6. **Prefer a genuinely original publication-associated source** whenever available; the CUMUL third-party source is an interim comparator, not proof of an author release.

### Source reproducibility gate (safe to run while GPU is busy)

\`\`\`bash
cd ~/ltd-attack
git fetch origin feature/macro-v2-historical-only
git merge --ff-only FETCH_HEAD
python3 scripts/sota/s3_s9_original_sources_gate_v1.py --stage fetch
python3 scripts/sota/s3_s9_original_sources_gate_v1.py --stage inspect
\`\`\`

The gate downloads no models/datasets, installs no dependencies, does not execute training, and checks commit + tracked-file state + required source files. Exact provenance is pinned independently of WFlib and the existing S10–S13 official-repository gate.

### Strict execution readiness

- \`S9_WFlib-adapted\` pretrain 100/100 and supervised B smoke are validated; its B/C/evaluation scripts exist but are **not author-original**. Do not start new WFlib-adapted B/C as the **primary** NetCLR benchmark.
- \`S3/S4/S7/S8/S9 original source\`: official/public code identified and pin gate ready. **Chronological original-source trainers, environment compatibility and reproducibility parity tests remain to be implemented and executed.** No original-source result is claimed.
- \`S10/S11/S12\`: authors' source already fetched on Perseo, pins PASS. Their chronological adapters and independent evaluations are still pending.
- Primary next implementation order: **NetCLR original (S9-O)**, DF original (S3-O), VarCNN original (S4-O), TikTok original (S8-O), CUMUL extraction parity (S7-O) and then RoFiRe, Swallow, WFCAT. Prioritize source-level tests and full UID-aligned output before compute-intensive runs.

## References

- Deep Fingerprinting authors: https://github.com/deep-fingerprinting/df
- VarCNN authors: https://github.com/sanjit-bhat/Var-CNN
- Tik-Tok authors: https://github.com/msrocean/Tik_Tok
- NetCLR authors: https://github.com/SPIN-UMass/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces
- CUMUL third-party implementation: https://github.com/lsvih/CUMUL
- Historical CUMUL evaluation code: https://www.cs.sfu.ca/~taowang/wf/attacks/
