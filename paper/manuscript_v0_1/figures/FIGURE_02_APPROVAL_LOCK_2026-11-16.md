# Figure 2 — LTD-Macro (APPROVED / LOCKED)

Approval date: 2026-11-16
Status: approved after scientific and visual validation. Preserve exact composition until explicitly unlocked by researcher.

## Approved file and provenance

- Binary PNG in this repository: `figure_02_ltd_macro_approved.png` (2745×1699, PNG), exported directly from the **editable Figma frame**.
- Figma file: https://www.figma.com/design/gVoDKr5ImuCkvD99nL8NzZ/LTD-Attack--Figure-1--Frozen-Architecture--Editable-?node-id=61-2
- Figma page: `25:2`, editable frame: `61:2`.
- Source approval image uploaded in ChatGPT: `FIGURE 2 · LTD-Macro · native editable · v2 approved visual reference (3).png` (2048×1267); its SHA-256 is `9f6fac316df50b335e49fe38d5f771e74346296815ab87b4ec6495b72da3f3df`.
- **Provenance distinction:** the binary checked into Git is a native Figma render of the approved editable diagram, not the byte-identical ChatGPT-uploaded screenshot. Both depict the researcher-approved final diagram. Use Figma as canonical editable source.

## Final architecture depicted

- Tor Query Traffic -> BASE-128 features (128-d).
- XGBoost branch: median-imputed features -> 65-site `predict_proba`.
- Historical Tor Captures (offline) -> site-day median BASE-128 profiles -> frozen normalization -> frozen historical profile bank.
- Date-conditioned lookup: up to five observed prior site-days **per candidate**, strictly before the query date; no updating the bank with inference observations.
- LTDPairScorer: separate Query Encoder (128 -> 64) and Temporal Encoder (positional encoding, Transformer x2 with 4 attention heads, masked pooling), pair representation (256-d), Pair MLP 256->128->1, 65 logits and softmax.
- One *representative* architecture shared by **three separately trained** LTDPairScorer models; checkpoints with seeds 11, 42 and 73 (different trained weights), not three different outputs from one trained checkpoint.
- Mean of the three model distributions.
- Geometric fusion XGB 0.625 / LTD 0.375 -> 65 Macro probabilities.
- Note: historical profiles and model parameters remain frozen during external evaluation.

## Do not misrepresent

- Figure 2 is an *inference architecture diagram*, not a training timeline.
- `<=5 prior site-days` belongs to the **per-query context selection**, not the stored-bank capacity. The source bank is built offline.
- The model selection/fitting protocol requires a separate Figure 3.

## Next planned figure

**Figure 3 — Chronological Training and Evaluation Protocol.** Show Historical DEV (DEV_EARLY/DEV_MIDDLE/DEV_LATE), phase B development, phase C refit and freezing, separate 09B and Future-B evaluations, and separate DEV_FROZEN versus HISTORICAL_FINAL tracks. Avoid implying that historical refit containing 09B provides independent 09B evaluation.

Research code references: `src/features/macro_v2/07A_candidate_conditioned_temporal_encoder.py`, `src/models/final/09A_refit_final_dev_models.py`, `src/models/final/10A_refit_full_historical.py`, `src/models/final/10B_evaluate_future_concept_drift.py`.
