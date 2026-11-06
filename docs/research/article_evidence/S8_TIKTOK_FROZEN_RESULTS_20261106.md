# LTD-Attack — S8 Tik-Tok external frozen results (2026-11-06)

Source: researcher-provided Perseo console output, script `scripts/sota/s8_tiktok_frozen_c_eval_v1.py`. This document records the observed experimental output; external independent audit of on-server checkpoint and NPZ remains pending.

## Protocol
- B fit 39,258 (DEV_EARLY + DEV_MIDDLE), validation 18,658 (DEV_LATE).
- B selection: epoch 23 of 30 by Macro-F1 = 0.9258110264128552, Accuracy = 0.9265194554614643, B checkpoint SHA256 `8f130d14725b3f25016ac1e142225c00047179c4518ae7eae8a10f5282cbba68`.
- C refit: exactly 23 epochs on 57,916 Historical DEV examples. C training complete, `S8_TIKTOK_C_PASS`.
- C SHA256: `ed265af5fbfd1c82637fa6c6c049b23724006776ec510c1a8e38de74f96264e6`.

## Frozen external results (printed by Perseo)
| Cohort | N | Accuracy | Macro-F1 | Top-5 | MRR |
|---|---:|---:|---:|---:|---:|
| 09B | 14687 | 0.9411043780213795 | 0.9385823041866972 | 0.9913528971199019 | 0.9632330629539676 |
| Future-B DEV_FROZEN | 18543 | 0.20104621690125654 | 0.1855718749835367 | 0.5221377339157633 | 0.35346309767233847 |

All commands ended `S8_FULL_EVALUATION_PASS`. No external optimization in the S8 script; do not claim prospective blinded evaluation.

## Comparison and limitations
- LTD-Hybrid 09B Accuracy 97.440%, Future-B 49.453% (from frozen S6).
- Hybrid minus Tik-Tok Accuracy = approx +3.3296 pp (09B), +29.3484 pp (Future-B).
- This is DESCRIPTIVE only. Paired UID/day statistics and multiple-testing family for Tik-Tok remain to be generated.
- Temporal drift is not independently identified as the sole cause of error; capture confounders and source alignment require inspection.
- Methods are WFlib implementation variants; avoid asserting exact paper reproduction.

## Next experimental step
S9 NetCLR WFlib-adapted contrastive pretraining, trained only on Historical DEV_EARLY + DEV_MIDDLE and supervised validation on DEV_LATE, followed by frozen C and external inference. Archive source SHA, setup, hyperparameter deviations, GPU limits, and contrastive losses. Distinguish NetAugment variant from original paper claims.
