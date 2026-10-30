# CHECKPOINT — LTD-Attack frozen statistical evidence / pre-SOTA / pre-C direct (2026-10-30)

**Status: RESEARCH CHECKPOINT — DO NOT REOPEN OR REPLACE FROZEN 09A/09B/10B/10C.** This is an immutable-by-convention scientific record. This document establishes an explicit handoff, **not** a new evaluation. Repository branch: `feature/macro-v2-historical-only`; operator Perseo reported fast-forward to `97532a4` before executing the 10C p-value audit. Preserve raw and manifest artifacts by SHA256, independent of GitHub source storage.

## 1. What is already supported

Closed-world Tor website fingerprinting, 65 canonical website labels. Historical A = 72,603 traces/52 dates; DEV = 57,916/41; Internal 09B = 14,687/11; Future-B 10B DEV_FROZEN = 18,543/16. **The same pcap_uid must be used inside each pairwise cohort; never pair historical internal traces with future traces.**

Reference score SHA256:
- 09B `5b13d0c16bf4a2d418eddf2dbd11ffca42b5f45485d731d9b9680f5c5b69d0f3`
- 10B DEV_FROZEN `d0edcd2780454b5cd39ef962ec45375d8822dc26b4b3c0b02cf4c06b0b29c7bb`
- V1 evidence manifest `5d38ffb7d3a4ec2d6a01913019141250786e1f9af9f2a2d310e3748054f968c8`
- V2 sensitivity manifest `3224d634527e9df03d4d6d784cce56f47cf019feb7d9d307751b39ccb2ecc2a6`

| Original research question | Observed evidence / carefully limited interpretation |
|---|---|
| Hybrid vs Micro same captures? | **Yes, within evaluated 09B and 10B.** 10B ΔTop1 +6.3366 percentage points (95% paired day-bootstrap CI +5.8663 to +6.7497); paired exact conditional two-sided p 0.00003052, Holm9 p 0.00027466; +1,175 net correct of 18,543 (1,865 rescued, 690 harmed). |
| Hybrid vs Macro same captures? | **Yes, within evaluated 09B and 10B.** 10B ΔTop1 +27.1477 pp (CI +26.2219 to +28.0356); paired conditional p 0.00003052. |
| Does XGB help Macro / Hybrid? | Future-B 10C posthoc renormalized *Hybrid* ablation: ΔTop1 +4.7943 pp (CI +4.2763 to +5.2801). Macro-only independent XGB removal **not established**; Macro vs LTD direct is not an equivalent ablation. |
| Does LTD add a real improvement? | **No robust additional Top1 benefit demonstrated** in Future-B 10C: Hybrid−no-LTD +0.2912 pp, CI [−0.1569,+0.7114]; Top5 −0.7011 pp, CI [−0.9636,−0.4813]. Internal Macro−XGB +1.5864 pp but Future Macro−XGB −1.0300 pp. |

Statistical practice: paired day-block bootstrap 2,500 replicates for interval estimates; 9,999 Monte Carlo complete-day label swaps for V1 p values, Holm12 (3 pairs × 4 metrics per cohort); exact 2^11 and 2^16 whole-day sign permutations for additive Top1, Top5 and MRR in V2/V4, Holm9 (3 primary pairs × 3 additive metrics). Macro-F1 has V1 Monte Carlo p, not a V2/V4 exact exhaustive p. Data are retrospective and day-block exchangeability/serial dependence constrain inference. P-values are **not** probabilities of H0 and **do not prove causal superiority**. Count/verify harmed cases and cross-site differences (38 sites improved recall, 27 worsened Hybrid vs Micro on Future-B).

## 2. NEW 10C audit output received from Perseo (2026-10-30)

User/Perseo console: `10C_EXACT_ABLATION_COMPLETE /home/bsierra/ltd-storage/results/reports/FROZEN-10C-PVALUES-V1`. Six exact conditional p-values from 16-day sign enumeration (Holm family of 2 posthoc variants × 3 additive metrics):

| Contrast | Metric | Δ | exact two-sided p | Holm6 |
|---|---|---:|---:|---:|
| Hybrid − no-XGB-renorm | Top1 | +0.047943 | 0.000030517578125 | 0.00018310546875 |
| Hybrid − no-XGB-renorm | Top5 | +0.083158 | 0.000030517578125 | 0.00018310546875 |
| Hybrid − no-XGB-renorm | MRR | +0.057373 | 0.000030517578125 | 0.00018310546875 |
| Hybrid − no-LTD-renorm | Top1 | +0.002912 | 0.245208740234375 | 0.3675537109375 |
| Hybrid − no-LTD-renorm | Top5 | −0.007011 | 0.0001220703125 | 0.0003662109375 |
| Hybrid − no-LTD-renorm | MRR | −0.002024 | 0.18377685546875 | 0.3675537109375 |

**Status: console verified only;** before labeling the NEW 10C audit fully reproduced, retrieve and independently check its `10C_ablation_exact_day_swaps.csv` and `manifest.json` and input SHA hashes. Macro-F1 exact sign-swap p **not computed** from these additive day summaries. Do not invent it.

## 3. Publication assessment — checkpoint decision (no false novelty)

**A defensible empirical short/workshop paper is plausible now**: frozen fusion under temporal drift, paired uncertainty, rescue/harm and site heterogeneity, negative ablation evidence. **Not yet claim a full top-tier security paper is ready**: novelty against recent drift-aware literature and prospectively controlled SOTA baselines remain unproven, 16 future dates, site selection conditional on coverage, A/B acquisition contract not fully validated, new blind cohort absent. Reframe as `Assessing Frozen Hybrid Website Fingerprinting Under Temporal Drift: Paired Evaluation, Component Ablations, and Failure Analysis`; LTD's increment is NOT proved. Avoid claiming to solve concept drift or to outperform Proteus/Aion based on published unrelated numbers.

## 4. Planned workstream A — fair SOTA comparison

**No SOTA benchmark has been completed in this checkpoint.** Primary **FROZEN/no-target-adaptation** track: original CUMUL/SVM (rebuild; legacy 94 features were future-informed), original Deep Fingerprinting 5000 direction entries if raw available (legacy 3000 is an adapted variant), Var-CNN, and Tik-Tok *only when per-packet timestamp representation is available*. Optional other faithful modern single-tab closed-world nonadaptive architectures after primary literature review. Separate **ADAPTIVE** track: Proteus (NDSS 2026, publicly available code; uses unlabeled target traffic for updates), Aion (IEEE TDSC 2026, verify code and labeled-target policy); label differing target information and compute cost prominently. Other papers/implementations do not automatically merit the same benchmark because multi-tab/open-world/hidden-service tasks differ.

**Fair preregistered gate BEFORE implementation:** same 65 class mapping, UID and 09A DEV / 09B / 10B cohorts; train/preprocess/early stop on DEV chronology only; no future label or test-set selection; pin external commits and reference architectures; report access to packet direction, timestamp, length independently, representation fidelity, input truncation variants; comparable training budget or explicitly separate unequal-budget comparisons; seeds and hardware logs; output all per-UID 65-class scores with provenance SHA; Top1/Macro-F1/Top5/MRR, paired day-block CI and significance with multiplicity correction. 10B has been opened: results on 10B are **retrospective benchmarks**, never new blind confirmatory evidence. Reserve `C_DIRECT_V1` as prospective external blind test only if original A/B collection can be matched and its evaluation criteria frozen beforehand.

**Do not execute** existing `scripts/train/run_estado_del_arte.sh` as canonical: old SVM uses `new_features_invariantes_seguras.txt` (94 Future-informed features); old DF/Rimmer datasets/splits are not aligned with 09A/09B/10B.

SOTA gates: A1 dataset/lineage provenance; A2 faithful implementations and frozen configs; A3 train/validation; A4 same UID score export; A5 posthoc 09B/10B comparison; A6 optional prospective C. Exit condition: reproducible benchmark with both favorable and unfavorable comparators and complete costs.

## 5. Planned workstream B — Dataset C direct Tor

Original C obfs4 pilot is quarantined as non-equivalent to A/B; **retain all raw artifacts and logs**, 166 runs / approximately 87 GB reported 2026-10-18. Old cron/watchdog was stopped and four Tor containers `ltd-c-tor-inst1/2/3`, `ltd-c-manual-tor` were exited with restart=no by user's operation. Never re-enable old obfs4 collectors or the former clock offset `LTD_CLOCK_OFFSET_SECONDS=-2963` or `LD_PRELOAD` shim. No sudo; no broad deletion or daemon restart. Dataset C categories inst1 beauty/tourism, inst2 banking/health, inst3 sports/politics; historical smoke covered 120 positions / 118 unique URLs.

User reports Tor direct is now enabled; **NOT YET independently verified on Perseo.** Stage B0 read-only inventory (chrony/timedatectl UTC and NTP, old cron/containers, port, image, storage, config sanitization). Stage B1 **ephemeral/direct Tor bootstrap + IsTor=true** without bridges or custom clock offset; no browser/capture and no modifications to historic datasets. Stage B2 tiny QA capture in distinct `C_DIRECT_V1` root, fixed known site list, trace header/clock/route and checksum checks; only then B3 approve capture schedule. First restore exact A/B original collector/BPF/circuit/network/feature contract, else C is only a separate new cohort without strict comparability.

**No cron or regular collection until B0/B1/B2 evidence passes and scientific capture protocol is frozen.** Scientific storage on /data or ~/ltd-storage, not repository. Never access arbitrary sites beyond approved/ethical collection list.

## 6. Timeline / decisions

Immediate: preserve checkpoint; (A0) audit candidate-baseline source/representation and freeze fair protocol; (B0/B1) validate direct Tor operationally. Parallel after gates: implement primary SOTA batch and one controlled C_DIRECT pilot. Then write **first paper draft** grounded in verified findings (no need to claim the LTD increment); afterward resume focused experiments on LTD memory freshness, matched budgets and new blind evaluation. Full SOTA retraining and full C collection are **not** represented as accomplished by this plan.

## Canonical links

- `docs/research/LTD_ATTACK_FOUR_RESEARCH_QUESTIONS_FINAL_20261026.md`
- `docs/research/LTD_ATTACK_STATISTICAL_INFERENCE_CLOSEOUT_V4_20261030.md`
- `docs/research/LTD_ATTACK_SOTA_FAIR_COMPARISON_PROTOCOL_DRAFT_20261030.md`
- `docs/operations/DATASET_C_BRIDGE_PILOT_STOP_AND_DIRECT_RECAPTURE_20261013.md`
- `scripts/diagnostics/verify_10c_ablation_exact_p.py`

**END CHECKPOINT — original scores and manuscript conclusions preserved; new work must be separately versioned.**
