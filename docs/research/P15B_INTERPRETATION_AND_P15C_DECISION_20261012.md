# LTD-Attack — 15B result and evidence-driven 15C decision (2026-10-12)

## Status
P15B exploratory Historical-only frozen-score ablation executed on Perseo: 24 rows, all 6 origin/window B0 reconstructions matched 13C; Bneg_temp2 preserved ranking exactly; no training, no Future-B/C reads in script. Output: `/home/bsierra/ltd-storage/results/diagnostics/P15B-FROZEN-FUSION-V1/`. **Independent verification of P15B output SHA256 remains pending** and precedes formal closeout.

## Fixed FAR metrics
| Origin | Variant | Top-1 | Macro-F1 | Top-5 | Delta Macro-F1 vs B0 |
|---|---|---:|---:|---:|---:|
| ORIGIN14 | B0 geometric | 0.815813 | 0.780642 | 0.910796 | 0 |
| ORIGIN14 | B1 arithmetic | 0.804562 | 0.777097 | 0.923771 | -0.003545 |
| ORIGIN14 | B2 RRF k60 | 0.776280 | 0.742420 | 0.913533 | -0.038222 |
| ORIGIN28 | B0 geometric | 0.917321 | 0.905937 | 0.984746 | 0 |
| ORIGIN28 | B1 arithmetic | 0.899856 | 0.886911 | 0.975461 | -0.019025 |
| ORIGIN28 | B2 RRF k60 | 0.888692 | 0.878983 | 0.990052 | -0.026953 |
Both alternatives fail preregistered FAR gate (mean macro-F1 gain >= +0.005, both-origin positive accuracy and bootstrap, Top-5 guard).

## Combined evidence
- 14A Future-B diagnostic: Top-1 49.45%, Top-5 80.56%, 65 sites, 18,543 samples. Opened once for post-hoc characterization; **must not guide selecting architectures, thresholds or candidates**.
- 15A Historical: the true class moves from Top-1 into 2–5 and beyond Top-5 as temporal distance increases; classes and capture distributions are not causal controls.
- 15A-MECH: Hybrid improves Top-1 in all 18 Historical comparisons but consistently reduces Top-5 in ORIGIN14; ORIGIN28 Top-5 gains. Per-day/site support and confidence summaries independently audited.
- 15B: swapping fixed aggregators does not resolve tradeoff. Arithmetic helps Top-5 for ORIGIN14 but harms F1/Top-1; RRF hurts F1 both origins. These alternatives remain rejected.

## Why 15C is next (not 15B hyperparameter search)
15B did not establish that a simple monotonic score transform or rank aggregation robustly rescues Top-1. Selecting a fusion weight or RRF k after examining two FAR windows risks development overfitting. Prior 08F confidence gating and 08G reranking were unsuccessful Historical controls; unmotivated repeats have low information value. New Micro training/large Transformer architectures impose high compute and confound mechanism attribution. Future-B was already opened; do not use it to choose new models. Dataset C is collecting and not a comparable blind test while clock, transport and coverage remain under assessment.

## P15C-HIST preplanned direction (protocol freeze required before execution)
**Diagnostic first**: reconstruct the *existing* frozen component scores on Historical per capture: Micro U3; Macro M0/M1; where available, XGB-only, LTD-only and Macro-FINAL. Validate capture UID, date and 65-class label order. Quantify per capture: rank and confidence by component, error cross-tabs, rescue/harm transitions, top-5 retention, and changes by date/day/site. Produce day-block confidence intervals for component deltas. Inspect memory-age/sequence features **only if obtainable and aligned**; do not manufacture them.

**Then targeted ablation**: if temporal-memory deterioration is supported by Historical, preregister a fixed (not searched) equal-budget memory freshness comparison, e.g. current frozen memory vs historically fresh memory at fixed windows. Require two FAR origins, aligned captures, guardrails on macro-F1 and Top-5, and repeatable audit; training only if necessary and justified. Do not claim causal influence from an observational component comparison.

## Next action
Run independent P15B CSV/manifests integrity verification; inspect canonical Macro NPZ schema, then freeze 15C diagnostic YAML and execute read-only. No further training before these gates.
