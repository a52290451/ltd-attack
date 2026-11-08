# Figure 1 v5 — architecture validated against frozen LTD model source (2026-11-08)

**Scope:** source/config/manifest review for the publication illustration; NOT independent binary checkpoint re-hashing on Perseo. Original editable v0.1 and earlier versions remain unchanged.

## Verified source mapping

- Micro: `configs/models/MICRO-FINAL.yaml`; `src/models/micro/08B_clean_temporal_micro_baseline.py::MultimodalTransformer`; `src/models/final/09A_refit_final_dev_models.py`.
- Macro: `configs/models/MACRO-LTD-FINAL.yaml`; `configs/models/MACRO-LTD-V2.yaml`; `src/features/macro_v2/07A_candidate_conditioned_temporal_encoder.py::LTDPairScorer`.
- Hybrid: `configs/models/LTD-HYBRID-FINAL.yaml`; `src/models/final/09B_evaluate_internal_test.py::geometric_fusion`; `src/models/final/10B_evaluate_future_concept_drift.py::evaluate_track`.
- Executed model configuration corroborated by `docs/evidence/FINAL/09A_manifest_final_dev_refit.json`: Micro 30 epochs, Macro context=5, seeds 11/42/73, Macro LTD weight 0.375, Hybrid weights Micro 0.45 and Macro 0.55.

## Correct frozen computational architecture

1. Single query Tor traffic trace is preprocessed as two Micro vectors of length **3000** (direction and packet size), plus **128** engineered Macro features. SOTA-specific 5000-length adapters are NOT this model's input.
2. **Micro:** direction Embedding(3,256) and size Linear(1,256) → concatenate/project Linear(512,256) → Conv1D(kernel=7,stride=2) → sinusoidal position → TransformerEncoder(4 layers,8 heads) → mean/max pooling → LayerNorm → MLP(512→512→65) → softmax probability vector.
3. **Macro/XGBoost:** 128 features, 120 estimators and 65-class probability output.
4. **Macro/LTDPairScorer:** same query 128 features projected to 64; candidate-specific strictly earlier daily history [65 candidates, up to 5 prior observed site-days, 128 features] with left-padding and masks; TemporalEncoder(128→64, 2 Transformer layers, 4 heads), valid-day mean pooling; pairwise vector [q,c,|q-c|,q*c] and MLP(256→128→1), yielding 65 logits. Three seeds 11/42/73 are combined as the **mean of softmax probabilities**.
5. **Macro probability:** normalized weighted geometric fusion p_M = softmax(0.625 log p_XGB + 0.375 log p_LTD), with numerical clipping EPS=1e-12.
6. **Hybrid probability:** p_H = softmax(0.45 log p_Micro + 0.55 log p_Macro), same epsilon handling. There is **NO separately trained Dense+Softmax hybrid head**, and **NO concatenation into a 130-feature vector**.
7. The final 65-component probability vector is ranked to select a monitored site; model outputs appear outside their dashed boundaries. Charts must not invent sample prediction probabilities.

## Required figure corrections applied to downloadable v5
- Macro shows TWO parallel internal paths (XGBoost + candidate-conditioned longitudinal transformer), not XGBoost alone.
- The 5-day historical reference bank and query feature path are separate and meet at the pair scorer.
- Internal Macro geometric mix alpha=0.375 and final Hybrid geometric mix 0.45/0.55 are explicit.
- Micro early fusion, CNN, positional encoding, Transformer, mean+max pool, MLP are explicit with source-verified dimensions.
- Ranked probability panels are outside Micro and Macro; arbitrary numerical output values removed.
- Chronological cohort diagram removed from architecture Figure 1; no extra trainable Hybrid classifier is depicted.

## Remaining gates
- Verify on-server checkpoint SHA and output vectors independently in a read-only reproducibility audit; repository manifest/code reconciliation is completed but not binary rehash.
- Confirm realistic availability/refresh cost of candidate-conditioned labeled site histories in the threat model.
- Perform final figure two-column typography/legibility gate once conference template/page width is chosen; use a full-width figure* if necessary.

The editable SVG, PDF and PNG are delivered separately as user-facing files with the figure audit; preserve their source master when updating the manuscript.
