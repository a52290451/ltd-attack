# Figure 1 — LTD-Hybrid inference architecture (APPROVED / LOCKED)

Approval date: 2026-11-16
Status: visually frozen by researcher; do not modify figure 1 without explicit request.

## Exact approved visual reference
- Uploaded source file: `Editable(3).png`
- Dimensions: 2048 × 1852 px, RGBA PNG
- SHA-256 (exact uploaded PNG): `0ce91066ceebb08a81f81867870e2d7a824e864b25a503156f33b7633ca111a9`
- Figma project associated with figure design: https://www.figma.com/design/gVoDKr5ImuCkvD99nL8NzZ/LTD-Attack-%E2%80%94-Figure-1-%C2%B7-Frozen-Architecture--Editable-?node-id=25-2

NOTE: This file is an immutable APPROVAL RECORD and checksum for the uploaded approved PNG, **not the PNG binary itself**. Do not claim the PNG is checked into Git until the binary is explicitly added. Preserve the exact original image under this checksum.

## Reviewed model depiction
- Diagram depicts frozen-model INFERENCE, not end-to-end training.
- Tor Query Traffic -> Micro packet direction/size encoder, Transformer ×4 (8 heads), mean/max pooling, classifier/softmax.
- The same query produces Macro BASE-128 features for XGBoost and candidate-conditioned LTDPairScorer.
- Historical Tor Captures (offline) -> site-day BASE-128 profiles -> normalization -> Frozen Historical Profile Bank.
- For each query date, up to five earlier observed site-days per candidate feed the Temporal Encoder (Transformer ×2, 4 heads); the historical bank is NOT updated by future queries. The five days designate the selected context, not total bank size.
- Three independently trained LTDPairScorer models each emit a softmax distribution; arithmetic mean supplies LTD probabilities.
- Macro geometric fusion: XGB 0.625 / LTD 0.375.
- Hybrid geometric fusion: Micro 0.45 / Macro 0.55; output ranks 65 candidate sites.
- Model weights and historical profiles remain frozen during external evaluation.

## Figure 2 design constraints
- Same visual style: clean white background, light blue Micro, pastel green Macro, soft purple fusion accents, rounded boxes, dashed architectural group outlines, black directed connectors, restrained typography.
- Figure 2 must explain OFFLINE historical memory + ONLINE candidate-conditioned longitudinal scoring, not duplicate the full Figure 1 hybrid architecture.
- Keep left-to-right layout, explicit data provenance and separate Query and candidate-history paths; retain up to five prior site-days, 65 candidates, independent three-scorer ensemble, mean probabilities. XGBoost/Hybrid fusion can be referenced as Figure 1, not re-expanded.
- Figure 2 remains a PROPOSAL until explicitly approved.

## Technical audit provenance
- Branch: `feature/macro-v2-historical-only` at review time; relevant final scripts: `src/models/final/09A_refit_final_dev_models.py`, `src/models/final/10B_evaluate_future_concept_drift.py` and `src/features/macro_v2/07A_candidate_conditioned_temporal_encoder.py`.
