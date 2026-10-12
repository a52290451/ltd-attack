# LTD-Attack — Scientific Milestone P15 (10B→15C; 15D paired follow-up pending)

**Checkpoint date:** 2026-10-12. **Status:** consolidated through P15C; P15D paired uncertainty execution pending. Future-B frozen / opened (descriptive only). Dataset C operationally isolated.

## 1. Main result and scientific problem
- Canonical 65 websites; Historical = 72,603 captures, 52 days; DEV 57,916 / 41 days, Internal Test 14,687 / 11 days.
- Frozen 10B Hybrid: Internal Test Top-1 = 97.44%, Future-B Top-1 = 49.45%, Future-B Top-5 = 80.56% on N=18,543. This is not evidence a future candidate could attain 80.56% Top-1.
- 14A verified rank counts: 9,170 rank 1, 5,768 ranks 2–5, 3,605 outside top five. 41 of the 65 sites meet the original `recoverable_gap` *descriptive* rule; 11 high-Top1, 13 residual.
- Frozen Future-B was already opened: the 65-site table is presentation material only and is **not** available for model/weight/site/feature/threshold selection.

## 2. Evidence map and decisions

| Stage | Question / experiment | Main evidence | Decision |
|---|---|---|---|
| 10B | Frozen temporal generalization | Hybrid 97.44% Internal Test → 49.45% Future-B Top-1 | Canonical frozen benchmark |
| 10C | Which Macro branch helps? | Post-hoc Future: XGB stronger, LTD branch fragile | No Future-tuned selection |
| 11A–11L | Historical Macro robustness mechanisms | Mixed historical outcomes, 11L retained | Mechanisms retained only when historical gates met |
| 12A | Integrate robust Macro into Hybrid | Gate failed | No promotion |
| 13B | Temporal Micro experts vs equal budget | T3_S42 failed ORIGIN14 FAR gate vs U3 | No promotion |
| 13C | Fixed factorial Micro × Macro | T9xM0 best descriptive FAR but 9 vs 3 experts | Exploratory only |
| 14A/14B | Characterize future rank gap | 31.11 pp rank2–5, 19.44% beyond Top5 | Post-hoc descriptive only |
| 15A | Reproduce historical rank gap | NEAR→FAR decreases Top1 and increases beyond-Top5 | Historical mechanism supported descriptively |
| 15A-MECH | Hybrid paired rescue/harm and day/site stability | 18/18 Top1 gains; ORIGIN14 Top5 down 9/9 | Rejected assumption Macro always enhances recall |
| 15B | Swap score aggregator, frozen | FAR mean Δ Macro-F1 arithmetic -0.011285, RRF -0.032588 | Both gates failed |
| 15C | Frozen M0 vs M1 (Macro + Hybrid) | Macro FAR improves both origins; Hybrid FAR inconsistent | No M1 Hybrid promotion |
| 15D | Day-block paired M0/M1 uncertainty | **Awaiting execution** | Must not claim independent confirmation |

## 3. P15C FAR — decisive numerical results

| Origin | Model | Accuracy | Macro-F1 | Top-5 |
|---|---|---:|---:|---:|
| ORIGIN14 | U3 | 0.79858 | 0.76925 | 0.92803 |
| ORIGIN14 | M0 | 0.64166 | 0.60966 | 0.84977 |
| ORIGIN14 | M1 | 0.65089 | 0.61923 | 0.84825 |
| ORIGIN14 | U3xM0 | 0.81581 | 0.78064 | 0.91080 |
| ORIGIN14 | U3xM1 | 0.81946 | 0.78446 | 0.90715 |
| ORIGIN28 | U3 | 0.87001 | 0.85235 | 0.98088 |
| ORIGIN28 | M0 | 0.77517 | 0.75966 | 0.95590 |
| ORIGIN28 | M1 | 0.77661 | 0.76046 | 0.95490 |
| ORIGIN28 | U3xM0 | 0.91732 | 0.90594 | 0.98475 |
| ORIGIN28 | U3xM1 | 0.91588 | 0.90358 | 0.98143 |

**Interpretation:** a Macro-only gain does not imply a consistent gain in the downstream Hybrid. Reweighting or swapping an aggregate component has not solved the temporal robustness problem. ORIGIN14 and ORIGIN28 differ; this does not isolate causation.

## 4. Scientific boundaries
1. All model choices since opening Future-B rely on Historical only; post-hoc B visualization does not authorize promotion.
2. 15A/B/C slices reuse the Historical data and prior trained models; results are not independent confirmation of a new model.
3. 15D is **post-hoc uncertainty quantification** of already observed P15C point estimates. No promotion gate can be declared independent retroactively.
4. M1 is a specific robust Macro comparator, **not** proof that a new memory freshness schedule is optimal.
5. Dataset C must meet independently defined clock, Tor transport and coverage protocol checks before use as a test.

## 5. Next experimental phase after P15D milestone
**Phase 16, hypothesis before new model:** evaluate longitudinal memory freshness versus age using predeclared Historical-only splits, fixed history budgets, matched site/capture composition and no Future-B/C, with a training/inference and GPU budget. Compare XGB-only, LTD-only, M0, M1 and any truly new memory modification only after input provenance and controls are validated. If historical loss traces to capture-length truncation or acquisition shift, prioritize those instead.

## 6. Canonical documentation and reproducibility
- `docs/RESULTS_HISTORY.md`; `docs/EXPERIMENT_MAP.md`
- `docs/research/P13_B1_CLOSEOUT_20261012.md`
- `docs/research/P14B_PRESENTATION_EVIDENCE_INDEX.md`
- `docs/research/P15A_MECH_CLOSEOUT_20261012.md`
- `docs/research/P15B_INTERPRETATION_AND_P15C_DECISION_20261012.md`
- `configs/experiments/P15D_HIST_PAIRED_M0_M1_V1.yaml`
- `scripts/diagnostics/run_p15d_and_milestone.py`

**Next execution:** P15D should independently verify 4+2+3 output CSV hashes, all P15C frozen source hashes, all paired captures/dates/labels, and generate `P15_MILESTONE_AUTOGENERATED.md` + its SHA256 manifest in Perseo.
