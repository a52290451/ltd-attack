# LTD-Attack — Presentation evidence inventory, 14B (2026-10-12)

## Preserve complete per-site table
The complete 65-site table is deterministically generated **read-only** on Perseo:
```bash
cd ~/ltd-attack
~/modelos/venv/bin/python scripts/diagnostics/report_p14b_site_domains.py
```
This prints `CANONICAL_DOMAINS_65_WITH_RANK_METRICS`, including numeric label, real domain, support, Top-1, Top-5, Top-5 minus Top-1, recoverable count, outside-Top-5 count, and fixed classification. Underlying immutable artifacts:
- `/home/bsierra/ltd-storage/results/final/FUTURE-RANK-GAP-PHASE14/14A_per_site_rank_gap.csv`
- `/home/bsierra/ltd-storage/results/final/FUTURE-RANK-GAP-PHASE14/14A_confusion_edges.csv`
- `/home/bsierra/ltd-storage/results/final/FUTURE-RANK-GAP-PHASE14/14A_manifest.json`
- `/home/bsierra/ltd-storage/datasets/future/CLEAN_final_features_sites_concept_drift.csv` for site_label/site mapping.

Do not commit private large raw datasets. This document preserves the authoritative table-generation path and headline values; the original table remains in the frozen 14A artifact and in the user's presentation copy.

## Slide-ready points
- 65 classes, 18,543 Future-B samples: Top-1 9,170 (49.45%); ranks 2–5 5,768 (31.11 pp); beyond Top-5 3,605 (19.44%).
- Of 9,373 Top-1 errors, 61.54% retained true rank 2–5: potential *oracle headroom*, not achieved rescue.
- 41 `recoverable_gap`, 11 `high_top1`, 13 residual `lost_beyond_top5` classes; **residual category does not imply Top-5 always low**.
- Dramatic class contrast: transfermarkt.es (Top-1 5.32%, Top-5 100%, n=282, recovered 267) versus segurcaixaadeslas.es (Top-1 1.56%, Top-5 5.06%, n=257, 244 outside Top-5).
- Strong confusion edges: druni.es→marvimundo.com 245, transfermarkt.es→bwin.es 237, senado.es→bwin.es 177.
- Descriptive evidence only: Future-B is OPEN/FROZEN, not for training, site selection, calibration, or hyperparameter tuning.
- Drift metrics table includes 69 raw-site rows; must be cohort-filtered to 65 before any corresponding aggregate slide.
- Original method/limitations and progression: `docs/research/P13_B1_CLOSEOUT_20261012.md`, `docs/research/P14B_OBSERVATIONAL_FINDINGS_20261012.md`, `docs/research/P14_TOP5_GAP_RESEARCH_ROADMAP_20261012.md`.

## Suggested presentation arc
1. Problem: Internal Test Hybrid 97.44% vs Future-B 49.45% (concept drift).
2. Main tension: Top-5 80.56% reveals hidden rank headroom.
3. All 65 sites table and segmentation (full data in source artifact).
4. Two opposite failure cases and confusion examples.
5. 13B/13C controls: generic ensemble and temporal mechanism not conflated.
6. Future experiments Historical-only; external Dataset C must meet time/protocol/coverage gates.

**Evidence status:** User supplied full 65-site site/domain table from Perseo on 2026-10-12; values here are a faithful subset. The complete table must be regenerated from the canonical frozen source for any publication and checked against its manifest. No new model experiments were conducted to create this index.
