# Figure 1 — per-element technical and scientific audit (2026-11-08)

**Status: CONDITIONAL / NO-GO for final paper figure.** The diagram is substantially architecture-faithful but incomplete in the provenance of the historical candidate reference bank and in the presentation of the three independently trained LTDPairScorer instances. This is an audit of the currently generated **PNG figure** shown in the ChatGPT conversation, not proof that the PNG is an editable vector source.

## Authoritative sources and scope

- Repo `a52290451/ltd-attack`, branch `feature/macro-v2-historical-only`.
- `configs/models/MICRO-FINAL.yaml`, `configs/models/MACRO-LTD-FINAL.yaml`, `configs/models/LTD-HYBRID-FINAL.yaml`.
- `src/models/micro/08B_clean_temporal_micro_baseline.py`: `MultimodalTransformer.forward`, `parse_vector`, `normalize_size`.
- `src/features/macro_v2/07A_candidate_conditioned_temporal_encoder.py`: `build_context_for_date`, `QueryEncoder`, `CandidateTemporalEncoder`, `LTDPairScorer`.
- `src/models/final/09A_refit_final_dev_models.py`: frozen training and XGB/LTD data transformations.
- `src/models/final/09B_evaluate_internal_test.py`: `softmax_numpy`, `geometric_fusion`, inference and rankings.
- `src/models/final/10B_evaluate_future_concept_drift.py`: exact external `DEV_FROZEN` inference and three-seed averaging.
- `docs/evidence/FINAL/09A_manifest_final_dev_refit.json`: 65 sites, 5-history site-days, checkpoint paths and frozen weights.
- This confirms source/config/manifest consistency; it does **not** re-hash checkpoint bytes on Perseo or prove deployed attacker access to historical labels.

## Review legend
- PASS: depiction is compatible with executed model.
- QUALIFY: accurate concept but inaccurate or incomplete reading is possible.
- FIX: technically inaccurate label or missing essential dependency.

| # | Figure element | Status | Code-verified interpretation / required action |
|---|---|---|---|
| 01 | Onion icon and Network Traffic | PASS | Reasonable symbol of Tor-encrypted traffic; not a Tor component that exposes content. |
| 02 | Parenthesis `Tor / WFP` | FIX | Website fingerprinting is an inference task (usually WF), not a network/protocol. Label `Tor traffic` with `Website fingerprinting` in caption, not `Tor/WFP`. |
| 03 | One observed trace → preprocessing | PASS | Query trace produces packet vectors and engineered feature record; distinct training data files are aligned by capture UID. |
| 04 | Preprocessing box | QUALIFY | Direction padding/truncation, byte-size min–max using TRAINING scales, feature selection/imputation/scaling are branch-specific. Do not suggest one shared universal scaler. |
| 05 | Direction vector length 3000 | PASS | `MAX_LEN=3000`; values -1/0/+1, 0 used as pad; fixed right truncation/zero padding. |
| 06 | Packet size vector length 3000 | PASS | Truncated/padded; normalized with Micro training min/max at model input. |
| 07 | Packet-sequence representation to Micro | PASS | Two separate input arrays, not traffic images. |
| 08 | Direction embedding | PASS | `Embedding(3,256)` applied to `x_dir+1`. |
| 09 | Size projection | PASS | `Linear(1,256)` independently applied to each packet size. |
| 10 | Micro two input paths in parallel | PASS | Both branches independently feed CONCAT; do not draw direction→size. Ensure two visually separate arrows from d and s. |
| 11 | Concat + Linear | PASS | Concatenation 256+256→512 per token, projection `Linear(512,256)`; not elementwise addition. |
| 12 | Conv1D | PASS | `kernel=7, stride=2, padding=3`; sequence downsample roughly 3000→1500. |
| 13 | Positional encoding | PASS | Sinusoidal positional encoding occurs AFTER Conv1D, BEFORE Transformer. |
| 14 | Transformer encoder ×N | QUALIFY | N is **4** encoder layers with **8 heads** and `d_model=256`. Replace unspecified ×N with `×4` in camera-ready figure. |
| 15 | Mean/Max pooling | PASS | Parallel temporal mean and max, concatenated into 512-d vector, then LayerNorm. |
| 16 | MLP + Softmax | PASS | MLP 512→512→65 yields LOGITS; evaluation uses softmax to obtain 65 probabilistic scores. These values are not demonstrated to be calibrated. |
| 17 | Micro probabilities outside blue dashed model | PASS | 65-class probability vector; plotted bars are illustrative, not observed predictions. |
| 18 | Traffic feature vector (128) | PASS | Frozen selected 128 **per-query engineered traffic features**, NOT inherently a 128-dimensional longitudinal history; do not label them all longitudinal. |
| 19 | Feature vector to XGBoost | PASS | Median imputation from historical fitting, XGB `n_estimators=120`, 65-class `predict_proba`. |
| 20 | XGB probabilities within Macro | PASS | Internal 65-component probability vector feeds Macro fusion. Bar chart must not imply only top 3 classes pass downstream. |
| 21 | Feature vector to Query Encoder | PASS | Independent LTD preprocessing/scaler, `Linear(128,64)`/GELU/LayerNorm/Dropout. |
| 22 | Historical candidate bank | **FIX** | Separate, labeled per-site DAILY history reference bank (65 candidate sites); history must be strictly EARLIER than query date. Depict as separate historical-reference DATA INPUT, not as memory produced from the current unknown trace or solely a learned layer. |
| 23 | `up to 5 prior observed site-days` | PASS | Last 5 available site-day profiles per candidate, not necessarily 5 consecutive calendar dates; left-pad/mask. Shape `[65,5,128]` if complete/padded. |
| 24 | Temporal encoder | QUALIFY | Actual `CandidateTemporalEncoder` projects 128→64, sinusoidal positions, **2 Transformer encoder layers / 4 heads**, masked mean temporal pooling; label `Temporal Transformer ×2` for fidelity. |
| 25 | Query encoder + Temporal encoder → Pair scorer | PASS | Combine `[q,c,|q-c|,q*c]` into 256-d pair feature, MLP 256→128→1 per candidate → 65 logits. |
| 26 | 3-seed ensemble | **QUALIFY** | Three **full independently trained LTDPairScorer networks** (seeds 11,42,73), not 3 output heads attached to one pair scorer. Each logits→softmax first; ensemble = average of **probabilities**. Draw a ×3 brace over the full LTD model or label `3 models → mean P`. |
| 27 | LTD probabilities within Macro | PASS | 65-class p_LTD after softmax-per-seed + averaging; bars illustrative. |
| 28 | Macro weighted geometric fusion | PASS | `softmax(0.625*log(p_XGB) + 0.375*log(p_LTD))`, clipped EPS=1e-12. |
| 29 | Macro probabilities outside green model | PASS | Final normalized 65-class Macro scores; not merely XGBoost output. |
| 30 | Micro and Macro probabilities to Hybrid | PASS | Both vectors aligned on same 65-class order. Must remain full vectors; no top-3 truncation. |
| 31 | Hybrid geometric fusion | PASS | `softmax(0.45*log(p_Micro) + 0.55*log(p_Macro))`, with clipping. No additional trainable dense/FC/softmax head, no 130-d concatenation. |
| 32 | Hybrid probabilities panel | QUALIFY | Visualization of fusion output, not a learned intermediate layer; optionally put outside dashed Hybrid boundary to make output status unambiguous. |
| 33 | Final ranking of 65 sites | PASS | Sort 65-class scores descending; first rank is closed-world prediction. |
| 34 | Mini-bar lengths | QUALIFY | Unlabeled illustrative graphics; NEVER describe as measured model outputs. Caption must say schematic example; no invented precision or probabilities. |
| 35 | Camera-ready typography / output | QUALIFY | Current art is a rendered PNG; for final publication create true editable SVG/TikZ with vector text, use full-width double-column placement if necessary and proof at final size. |

## Non-negotiable fixes before approval

**P0 / scientific**
1. Label bank as `Labeled historical site profiles (reference input)` with an arrow entering Macro from OUTSIDE, fed by prior known collections, filtered `history date < query date`. Do not draw this bank as generated by the current unknown query.
2. Clarify three complete independent LTD scorer models and averaging of post-softmax 65-class vectors.
3. Correct `Tor / WFP` terminology; write `Tor traffic` (WF is task).

**P1 / publication-quality precision**
4. Replace Transformer `×N` by `×4` and, if readable, annotate `8 heads`.
5. Annotate Temporal Transformer `×2` / `4 heads`; masked pooling may move to caption.
6. Label branch-specific preprocessing in caption rather than inserting narrative boxes.
7. Avoid intermediate probability mini-rank panels implying top-3 pruning; retain vectors of 65 entries throughout.
8. Make all examples explicitly schematic; align probability plots with same 65-class support.
9. Convert validated master to editable SVG/TikZ and typeset inside final LaTeX page width, then inspect legibility.

## Editorial caption candidate (English)

`Figure 1. Architecture of LTD-Attack. A Tor packet trace yields 3,000-element direction and size sequences for the Micro model and 128 selected traffic features for the Macro model. Micro independently embeds directions and projects normalized sizes; their concatenation is followed by a one-dimensional convolution, positional encoding, a four-layer Transformer, mean/max pooling, and an MLP. Macro combines an XGBoost classifier with an ensemble of three candidate-conditioned LTDPairScorer models, each comparing query features with up to five earlier labeled site-day profiles per candidate. Macro and Hybrid apply frozen weighted geometric probability fusion (XGB/LTD: 0.625/0.375; Micro/Macro: 0.45/0.55). Each branch and the hybrid yield aligned 65-class probability scores, from which the final ranking is obtained. Probability-bar graphics are schematic.`

**Final verdict:** `NO-GO AS CAMERA-READY` because historical data provenance and 3-seed structural meaning are not fully clear. Otherwise most blocks are demonstrably correct. Do not change frozen code/models to make the diagram fit; revise only the artwork/caption.
