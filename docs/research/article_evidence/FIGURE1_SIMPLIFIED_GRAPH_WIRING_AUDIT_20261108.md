# Figure 1 — scientific wiring audit of simplified architecture (2026-11-08)

**Verdict: NOT APPROVED AS CODE-FAITHFUL.** Composition is serviceable; critical connection/feature provenance errors must be fixed. Repository branch: `feature/macro-v2-historical-only`. Review uses read-only frozen source/config, NOT independent execution or checkpoint byte verification.

## S9 status (researcher-provided Perseo output)

S9 NetCLR `pretrain_B_v2` printed `S9_NETCLR_PRETRAIN_B_PASS 121851a09f2046ea550b8655d31e2af2b8d1d66324b19e57b233d8263d362523` at epoch 100; contrastive loss first epoch 3.0771166620985717, epoch 100 2.9094990524926736. This completes PRETRAIN-B only; supervised training and external evaluation have not been reported complete. No direct server hash re-computation was performed here.

## Micro: source-verified signal flow

`configs/models/MICRO-FINAL.yaml`; `src/models/micro/08B_clean_temporal_micro_baseline.py::MultimodalTransformer`:
- Packet direction (3000) → Embedding(3,256); packet sizes (3000) → Linear(1,256) **IN PARALLEL**.
- Concatenate 256+256 channels and project via Linear(512,256), **NOT ADDITION** and NOT direction→size sequence.
- Conv1D 256→256 with kernel=7, stride=2, padding=3.
- Sinusoidal positional encoding, 4 Transformer encoder layers, 8 attention heads (d_model 256).
- Mean and max pool over sequence, concatenate 256+256, LayerNorm(512), MLP 512→512→65 (GELU+Dropout between). 65 logits → softmax for classification probabilities.
- Direction vectors are truncated/right-padded to 3000; packet sizes use frozen historical min/max normalization separately. Generic 'Normalize' on a single common preprocessing block can misrepresent this.

## Macro: source-verified signal flow

`configs/models/MACRO-LTD-FINAL.yaml`, `configs/models/MACRO-LTD-V2.yaml`, `src/features/macro_v2/07A_candidate_conditioned_temporal_encoder.py::{build_context_for_date,QueryEncoder,CandidateTemporalEncoder,LTDPairScorer}`, `src/models/final/{09A_refit_final_dev_models,09B_evaluate_internal_test,10B_evaluate_future_concept_drift}.py`:
- Per-query selected **BASE-128 traffic features**, not 128 intrinsically longitudinal features. 128-feature vector branches into **two** paths.
- XGB classifier on median-imputed 128 features yields a 65-class probability vector; uses XGBoost 120 estimators with hyperparameters from final configuration.
- LTD pair branch: the same query's BASE-128 features are scaled and mapped by QueryEncoder 128→64.
- Separately and INDEPENDENTLY, a `labeled historical reference bank` contains previous DAILY aggregate feature vectors, grouped by each of 65 candidate site labels and filtered strictly to dates before query cutoff. Select up to 5 **previous observed site-days**, not necessarily 5 consecutive calendar days. Build tensor [65,5,128], left padding with attention mask.
- CandidateTemporalEncoder projects 128→64, adds sinusoidal temporal positions, runs 2-layer TransformerEncoder (4 heads), masked temporal mean pooling; candidate encoding dimensionality 64.
- PairScorer combines [q,c,abs(q-c),q*c] => 256-d feature, applies MLP(256→128→1) per query/candidate, obtains 65 logits; softmax yields probabilities per seed. Three model seeds 11/42/73; mean their probability vectors.
- Macro final `geometric_fusion(XGB_probs,LTD_probs,second_weight=0.375)` returns a normalized 65-class probability vector. In figure show a **weighted geometric fusion** symbol, never a plus/addition symbol. Weight XGB 0.625 / LTD 0.375.
- Figure must NOT imply candidate-history bank is generated from the single unclassified query. It is a second, separately maintained and *class-labeled historical* reference input, anchored to a time cutoff. Threat model must disclose this condition.

## Hybrid: source-verified signal flow

`configs/models/LTD-HYBRID-FINAL.yaml`; `src/models/final/09B_evaluate_internal_test.py::geometric_fusion`; `src/models/final/10B_evaluate_future_concept_drift.py::evaluate_track`:
- Only two probability vectors enter: Micro P(65) and final Macro P(65).
- `Hybrid = softmax(0.45*log(clip(Micro)) + 0.55*log(clip(Macro)))` with epsilon 1e-12. A separately trained classifier, FC layer, 130-d concatenation, or attention gate is **NOT** part of the frozen Hybrid architecture.
- Diagram should show one compact node `Weighted geometric fusion (Micro 0.45 / Macro 0.55)`, output `Hybrid probabilities (65)` → `Rank 1...65` by sorting probabilities.

## Editorial and scientific corrections (priority)

1. P0: Micro embedding and size projection side-by-side with arrows into CONCAT + Linear; replace `+` icon with concat.
2. P0: Replace Macro input label `Longitudinal feature vector (128)` with `Traffic features (128)`. Show a **separate** historical candidate bank feeding temporal encoder; never derive it from query feature vector.
3. P0: Show query feature vector fork into XGB and QueryEncoder; temporal candidate bank fork feeds CandidateTemporalEncoder; these combine at PairScorer, then ensemble mean.
4. P0: Show Macro weighted geometric fusion XGB/LTD, not addition; optional concise weights 0.625/0.375.
5. P0: Hybrid weighted geometric fusion 0.45/0.55, no FC/Dense, no concatenation. The simplified picture's single geometric-fusion stage is structurally on the right path, but specify weights and make its inputs explicit.
6. P1: Distinguish mean+max pooling and 65-class Micro softmax; keep all probability/ranking panels OUTSIDE model dashed boundaries.
7. P1: Avoid implying all normalization/padding applies to both branches equally; leave common block `Preprocessing` and explain branch-specific operations via caption and method.
8. P1: Label `up to 5 prior observed site-days` rather than `5-day context` if it could be interpreted as five calendar days; use arrows for query vs historical reference separately.
9. P1: Output mini-bars are schematic, without real percentage labels; clarify `illustrative` in caption, or replace with actual frozen sample predictions after audit.
10. P1: Typography should remain readable in two-column `figure*`, vector master preferred. Detailed dimensions/learning parameters belong in caption/methods; keep block names short.

## Result

**Diagram not yet publishable; scientific content of corrected wiring is supported by source/config.** Checkpoint artifact byte-level independent verification and any future operational use of class-labeled histories remain separate gates. The figure should be redrawn from this wiring without inventing trainable layers.
