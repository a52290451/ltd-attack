# LTD-Attack — next CPU-only action after author-original quick PASS (2026-11-11)

The researcher ran `run_sota_original_cpu_only_v1.sh quick` on Perseo and confirmed:
- S3 DF, S4 VarCNN, S7 public CUMUL, S8 TikTok, S9 NetCLR: original/public source pin, clean tracked checkout and file SHA audit PASS.
- S9 original authors' `DFNet` / `DFsimCLR` forward tests PASS with output (2,128) and 65-class (2,65).
- Original NetAugment class on 12 **synthetic** traces PASS; this does NOT establish fidelity on real Historical DEV.
- CUDA / GPU training NOT performed.

### New preparation gate

`scripts/sota/s9_netclr_authors_augment_historical_audit_v1.py` checks authors' original NetAugment on Historical DEV_EARLY+MIDDLE only:

1. Verifies the S9 authors' pinned notebook and 57,916-row local Historical DEV cache SHA.
2. Derives original authors' burst-size sampling and histogram from 1,000 training-only traces (seed 42); no DEV_LATE contribution.
3. Quantifies the CDF indexing issue without silently changing the original source.
4. Runs 128 source-faithful CPU-only augmentation smoke trials and records failure types, if any.
5. Saves an independent diagnostic manifest (no model output, no training, no external cohort access).

### Execute on Perseo

```bash
cd ~/ltd-attack
git fetch origin feature/macro-v2-historical-only
git merge --ff-only FETCH_HEAD
bash scripts/sota/run_sota_original_cpu_only_v1.sh prepare-dev
bash scripts/sota/run_sota_original_cpu_only_v1.sh audit-augment
```

The first command prepares/validates a ~290MB array of signed directions plus label/split arrays in the existing `s9_netclr_wflib_adapted_seed42/supervised_dev_cache_v1` cache. **This cache is shared for historical directions only; its provenance remains independent from the authors' model.** The second command writes:

`/data/ltd-attack/sota/s9_netclr_author_original_v1/diagnostics/authors_augment_audit_v1.json`

If the report already exists, the script refuses to overwrite it. Never clear original frozen predictions/checkpoints or Dataset C.

### Next steps after results

- Inspect original NetAugment's `outgoing_burst_sizes`, empirical CDF and any observed outputs/failures. Decide whether using the author's literal notebook behavior is mathematically valid or requires a **documented compatibility patch**, independently tested for parity.
- Only then implement authors' original 1000-sample distribution-backed contrastive Historical DEV pretraining with source/model hashes, CPU preflight and explicit output directory, followed by chronological B/C.
- DF/VarCNN/TikTok author-source adapters and CUMUL feature parity are further CPU-preparable work; no new training metrics exist.
