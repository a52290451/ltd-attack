# P15A-HIST — Aggregate rank drift findings

Status: **COMPLETED DESCRIPTIVE**, validated on Perseo. Source: SHA-verified 13B/13C CSVs; 13B hashes 6/6, 13C 5/5. No training, Future-B reads or Dataset C reads.

## Historical Hybrid rank redistribution, NEAR → FAR (percentage points)
| Candidate | Origin | Delta Top-1 | Delta ranks 2–5 | Delta beyond Top-5 |
|---|---|---:|---:|---:|
| U1xM0 | ORIGIN14 | -10.858 | +3.833 | +7.025 |
| U3xM0 | ORIGIN14 | -10.508 | +3.676 | +6.831 |
| T3_S42xM0 | ORIGIN14 | -10.552 | +3.315 | +7.237 |
| T9xM0 | ORIGIN14 | -10.197 | +3.185 | +7.012 |
| U1xM0 | ORIGIN28 | -2.320 | +0.839 | +1.481 |
| U3xM0 | ORIGIN28 | -2.814 | +1.630 | +1.184 |
| T3_S42xM0 | ORIGIN28 | -2.516 | +1.415 | +1.100 |
| T9xM0 | ORIGIN28 | -2.590 | +1.453 | +1.137 |

**Interpretation:** both rank misordering and Top-5 signal loss rise with temporal distance, and the ORIGIN14 deterioration is larger. These are aggregate comparisons of distinct time windows, not paired capture-level causal effects. Site composition, timing, capture conditions and temporal memory must be investigated before causal claims. T9's higher model count prevents matched-budget promotion. Do not infer Future-B gains from Historical rank headroom.

## Evidence and next step
- Existing source CSVs: `13B_rank_gap_historical.csv`, `13C_rank_gap_historical.csv`, whose hashes are verified by their existing manifests.
- `scripts/diagnostics/report_p15a_hist_aggregates.py` reproduces the above.
- NEXT: read-only cache inventory for existing per-capture score arrays and metadata, including identity/label/date validation. No re-training.
- Continue experimental firewall: Historical-only development, Future-B post-hoc interpretation only, Dataset C isolated QA.
- Stage 15B vs 15C decision **not yet made**: this preliminary aggregation does not prove a ranking correction will work.
