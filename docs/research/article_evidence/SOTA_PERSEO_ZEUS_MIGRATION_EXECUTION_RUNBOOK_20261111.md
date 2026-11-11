# LTD-Attack: reproducible author-source SOTA battery on Perseo or Zeus

Branch: feature/macro-v2-historical-only. Date: 2026-11-11.

## Scientific scope

Seven author-code-based implementations are prepared: DF (original author Keras), VarCNN **direction branch only** (NOT complete original multimodal ensemble), Tik-Tok (original authors' ConvNet + DT), NetCLR (authors' DFNet/DFsimCLR + NetAugment, adapted 100 epochs/batch64/CDF), RoFiRe (original DFVarNet + DyWin, adapted multiplier), Swallow (authors' CIF/ResNet18/BYOL, explicit in-memory infinite-loop compatibility fix), WFCAT (authors' IAT/attention).

CUMUL remains separate: lsvih/CUMUL is an incomplete third-party public reproduction and cannot be labeled the original authors' benchmark. Proteus is only pinned as related work because it uses future/unlabeled adaptation.

B: pretraining if relevant on DEV_EARLY 19,098 + DEV_MIDDLE 20,160; supervised fit on 39,258, select epoch exclusively on DEV_LATE 18,658. C: re-fit on all 57,916 Historical DEV samples using frozen B epoch. External 09B (14,687) and Future-B (18,543): never tune. Dataset C is not used.

## A. Validate Perseo, no GPU

Update Git branch, then execute in this order:

    bash scripts/sota/run_original_sota_battery_v1.sh cpu-preflight
    bash scripts/sota/run_original_sota_battery_v1.sh cpu-integration
    bash scripts/sota/run_original_sota_battery_v1.sh cpu-prepare
    bash scripts/sota/run_original_sota_battery_v1.sh host-check
    python scripts/sota/sota_environment_inventory_v1.py --stage capture
    python scripts/sota/original_sota_orchestrator_v2.py --stage all --dry-run

The second stage uses original models for CPU optimizer mini-updates and save/load tests. The third prepares all historical representation caches, including S8 Tik-Tok DT, S11 Swallow CIF and S12 WFCAT IAT (several GiB possible). Each successful phase records its hostname + Git commit into an operational readiness certificate.

If a phase fails, do not train and do not delete previous frozen outputs. Fix only the error and rerun CPU gates. The GPU was occupied by PID 3223 on Perseo in the most recent check.

## B. Zeus migration

Clone the exact branch/commit on Zeus, then retrieve all author-source repositories in clean detached pinned checkouts using:

    python3 scripts/sota/s3_s9_original_sources_gate_v1.py --stage fetch
    python3 scripts/sota/s10_s13_upstream_source_gate_v1.py --stage fetch

Transfer with rsync/SSH (not Git): Historical DEV source CSV, frozen S1 manifest and its UID-index CSV and label-order JSON, S5 manifest with canonical 09B/Future-B NPZ references, Future-B source CSV, optionally fully manifest-verified historical cached NumPy files. Do not transfer partial .npy files. No Dataset C.

The original S1/S5 JSON may embed Perseo absolute paths. Do not edit or replace originals. Use scripts/sota/sota_zeus_evidence_bootstrap_v1.py --stage relocate with the actual old S1/S5 manifest paths, actual new location of index/order/source CSV and 09B/Future-B reference NPZ, and a new output directory. The relocator checks original SHA256 checksums before emitting host-local S1/S5 copies and host_environment.sh. Source that generated script before training.

Optional environment configuration if paths differ:
- LTD_SOTA_OPS_ROOT: location of both original author checkout trees
- LTD_SOTA_ARTIFACT_ROOT: persistent output/caches root (default /data/ltd-attack/sota)
- LTD_SOTA_STORAGE_ROOT: parent directory of historical/ and future/ vectors
- LTD_SOTA_S1_MANIFEST / LTD_SOTA_S5_MANIFEST: new host-local metadata copies
- LTD_SOTA_FUTURE_MANIFEST: repo's frozen Future-B metadata
- LTD_SOTA_PYTORCH_PY and LTD_SOTA_KERAS_PY: interpreter absolute paths

Existing environments may be reused: never reinstall PyTorch/TensorFlow globally or activate ~/ltd-venv when it lacks ML dependencies. Compare critical packages to the Perseo inventory:

    python scripts/sota/sota_environment_inventory_v1.py --stage compare --reference /path/to/perseo_sota_environments_v1.json

On Zeus, repeat all four CPU stages and CUDA smoke after confirming no competing process with nvidia-smi. Perseo's readiness certificate is deliberately invalid on Zeus.

## C. Training and final test

Only after Zeus/Perseo has current-commit CPU certificates, hash-verified S1/S5/Future-B inputs, isolated interpreters with CUDA visibility and GPU idle:

    python scripts/sota/original_sota_orchestrator_v2.py --stage all --dry-run
    bash scripts/sota/run_original_sota_battery_v1.sh gpu-all

V2 runs NetCLR, Swallow, WFCAT, DF, VarCNN direction, Tik-Tok, RoFiRe sequentially; skips only hash-verified complete stages; refuses partial output directories instead of overwriting them. Each model produces 65-way UID-aligned probabilities on 09B/Future-B. Independent final auditor recomputes accuracy, macro-F1, top5, MRR and checks original checkpoint/score SHA and canonical UID/label/date alignment.

## D. Statistical comparison to LTD-Hybrid

Provide exact frozen LTD-Hybrid 09B and Future-B NPZs, same UID/labels/date order and 65-way probability columns:

    export LTD_SOTA_LTD_REFERENCE_09B=/path/to/ltd_09b_frozen.npz
    export LTD_SOTA_LTD_REFERENCE_FUTURE_B=/path/to/ltd_futureb_frozen.npz
    bash scripts/sota/run_original_sota_battery_v1.sh statistics

Results: 14 paired comparisons, four metrics, LTD minus SOTA differences, 95% date-cluster paired bootstrap percentile CI with 2500 samples and two-sided date-block paired randomization p-values with 2500 permutations. The statistical method and assumptions are written into each result JSON. No external tuning.

## Remaining certification requirements

Seven CPU construction smokes passed on Perseo in the preceding logs. The new CPU optimizer/serialization integration stages, complete S11/S12 historical caches, Zeus host input relocation/verification and all GPU training paths **have not yet been successfully run**. A true no-error guarantee requires the actual host's CPU certificates, CUDA test, successful full run and final independent results audit. In particular, VarCNN remains direction-only and original CUMUL is unavailable. Do not report new SOTA accuracies yet.
