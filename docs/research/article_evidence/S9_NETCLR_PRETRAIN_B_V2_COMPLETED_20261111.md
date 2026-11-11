# S9 NetCLR contrastive pretraining B — completed and independently hash-checked (11 November 2026)

## Confirmed by the researcher on Perseo

- Stage: `pretrain-B` — **100/100 epochs completed**. Contrastive loss epoch 1 = **3.0771166620985717**, epoch 100 = **2.9094990524926736**.
- Provenance: Historical DEV_EARLY (19,098) + DEV_MIDDLE (20,160) = **39,258** traces, 65 labels. No DEV_LATE in contrastive pretraining. Checkpoint manifest declares `external_data_opened=false` and `dataset_c_opened=false`; these are manifest assertions, not independently observed trace accesses.
- WFlib-adapted NetCLR/DFsimCLR, 5,000 packet directions, 128-d embeddings, InfoNCE/NT-Xent, 2 augmented views, Adam, lr 0.0003, batch 64, temperature 0.5, seed 42; not original-paper exact.
- Checkpoint path: `/data/ltd-attack/sota/s9_netclr_wflib_adapted_seed42/pretrain_B_v2/checkpoint_epoch_100.pt` (111,772,787 bytes).
- Ten checkpoints epoch 10,20,30,...,100 and `manifest.json` retained in the same run directory.
- SHA-256 independently recomputed on Perseo and matched manifest:
  `121851a09f2046ea550b8655d31e2af2b8d1d66324b19e57b233d8263d362523`
- Console marker `S9_CHECKPOINT_INTEGRITY_PASS` (11 November 2026).

**Only unsupervised pretraining is completed**. No evidence exists here of supervised fine-tuning B, phase-C historical refit, or 09B/Future-B classification scores. Do not report any classification accuracy for NetCLR.

## Dataset C parallel operations — latest evidence

- Half-hourly QA schedules were still installed, but all four direct Tor containers were stopped by 11 November; the three active instance containers ended with exit 137. Root cause not proven; inspect OOMKilled and Docker/kernel logs.
- Last successful reported QA half-lot: 37/60 PCAP-valid in a second split (dated 8 November), `SCIENTIFIC_CAPTURE=NOT_AUTHORIZED`. A subsequent first split reported `HALFSPLIT_PCAP_VALID 0/60`, `NAVIGATION_ERROR: 60`; scheduler then blocked with `ltd-c-tor-inst1 not running`.
- Host `NTPSynchronized=no`; data root `/data/ltd-c-qa-bsierra/C_DIRECT_V1` ~37 GB, /data 311 GB free, 89% used.
- GPU process PID 3223 belongs to OS user `perseo`, executes `python3 -u ./model/main.py --precomputed_root ...`, consumes ~6.8 GiB RTX5080 VRAM; not identified as S9 or Dataset C. Do not kill.
- No scientific Dataset C results, matching 65-site class crosswalk, or stable capture clock inferred from QA evidence.

## Immediate next scientific gate
Build WFlib-source-faithful supervised NetCLR evaluation using only Historical DEV:
1. Load and verify frozen pretraining checkpoint / WFlib source pin.
2. Fine-tune on 39,258 Early+Middle labels; select epoch by 18,658 DEV_LATE macro-F1, with auditable batch/epoch/elapsed logs.
3. Phase-C fit full 57,916 Historical DEV using frozen recipe and **only permitted Historical DEV** pretraining. Important: phase-C must not reuse a pretrained representation fitted on future evaluation cohorts; it may use full Historical DEV (re-pretraining there would require an explicit new frozen stage and output directory).
4. Evaluate 09B and Future-B from frozen C; export complete UID-aligned 65-class vectors, top-1, macro-F1, top-5, MRR, SHA-256. Preserve originals; do not overwrite or invent S9 numbers.
5. New SOTA candidates RoFiRe / Swallow / WFCAT after closing S9, with Proteus adaptation requirements explicitly separated.

This evidence note records confirmed state; the GitHub connector cannot access Perseo files directly.
