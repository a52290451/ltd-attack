# P15A-MECH — independent closeout, 2026-10-12

Status: **independently verified** on Perseo. Original 13B/13C frozen manifests verified by derivation; 18 paired candidate/origin/window comparisons; 4 output files passed SHA256, per-day and per-site counts reconciled, closeout ended `P15A_MECH_CLOSEOUT_VALIDATED`.

## Principal findings
- All 18 fixed Micro-to-Hybrid M0 comparisons improve Top-1; all 18 day-block bootstrap Top-1 95% CIs exclude zero on positive side.
- **ORIGIN14:** all nine Top-5 point differences are negative; **8 of 9** day-block bootstrap Top-5 95% CIs lie wholly below zero. The exception is U1/NEAR (-0.00010; CI [-0.00948,+0.01086]).
- **ORIGIN28:** all nine Top-5 point differences positive; 7 of 9 day-block bootstrap Top-5 95% CIs positive; the two exceptions, NEAR U3 and T3_S42, include zero.
- ORIGIN14/FAR U3: rescued 577/harmed 407, net +170, Top-1 +1.7233pp, Top-5 -1.7233pp; 42 of 65 sites net benefited, 14 net harmed, 9 unchanged.
- ORIGIN28/FAR U3: rescued 532/harmed 104, net +428, Top-1 +4.7309pp, Top-5 +0.3869pp; 42 positive sites, 10 negative, 13 unchanged.
- ORIGIN14/FAR U3 wrong confidence >=0.8: Micro 4.440% of all captures, Hybrid 2.038%; this is not a calibration test. Among rescue captures Micro predicted-class confidence mean 0.55961, harmed captures confidence mean 0.67631.
- No evidence of a deployable gain from Future-B 14A oracle headroom. All data here exclusively from Historical.

## Decision and caution
The measured tension between Top-1 and Top-5 supports a *fixed-score fusion ablation* before deploying a trained reranker. A new 15B study must include existing geometric fusion B0, arithmetic and rank-based fixed alternatives, a rank-preserving negative control and Historical-only gate. Arithmetic mixing has been considered in 11K; include as named comparator, **not as a novel method**. Rank-RRF is an exploratory distinct aggregation mechanism, not a proven improvement. No weights or rank offset may be tuned using Future-B. The stage cannot claim a new successful model until both origins FAR and bootstrap guardrails pass.
