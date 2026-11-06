# S8 Tik-Tok — Historical DEV B selected checkpoint (2026-11-06)

Source: researcher-provided Perseo console, script scripts/sota/s8_tiktok_train_b_v1.py, commit 3055138. This records the console, not independent SHA verification of server files.

- Historical DEV feature cache: 57,916 vectors; 195,691 decreasing consecutive timestamps; no transformations were made to repair or reorder time.
- Hardware: NVIDIA GeForce RTX 5080, PyTorch CUDA SOTA virtualenv /data/ltd-sota-bsierra/venv-cu128.
- Train: DEV_EARLY+DEV_MIDDLE, n=39,258. Validation: DEV_LATE, n=18,658. 65 closed-world sites.
- Model: WFlib TikTok pinned to commit 537ec8e03c805540b04b95b144fa158ce50aa255, blob 8736ef4d7d4c00bb1559def532f511cb8e142625; directional timing 5000, Adamax lr 0.002, batch 128, 30 epochs.
- Validation best macro-F1: epoch 23, accuracy 0.9265194554614643, macro-F1 0.9258110264128552, training loss 0.13439913723447355.
- Epoch 30: accuracy 0.9264658591488906, macro-F1 0.925140539421719; epoch 23 chosen under frozen macro-F1 selection.
- Checkpoint SHA256: 8f130d14725b3f25016ac1e142225c00047179c4518ae7eae8a10f5282cbba68.
- Console: S8_TIKTOK_B_PASS 23.
- This is an internal historical DEV_LATE validation result; DO NOT compare directly with external 09B or Future-B accuracy as a final benchmark.
- Follow-up: scripts/sota/s8_tiktok_frozen_c_eval_v1.py published to reproduce selected-epoch refit C and external inference under frozen SHA/UID/class constraints. Its successful execution is NOT yet established.
- Must preserve S8 B, S7, S6, Dataset C without edits/overwrites. NetCLR pretraining remains outstanding.
