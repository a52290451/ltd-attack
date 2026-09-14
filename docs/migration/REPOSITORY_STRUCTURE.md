# LTD-Attack — repository structure

The repository is organized by research responsibility while preserving historical lineage. This is a physical migration only: scripts retain their original contents and known historical behavior.

```text
src/
├── data/                       # PCAP ingestion, aggregation, and data mapping
├── features/                   # Active Macro feature engineering
├── models/
│   ├── micro/                  # MICRO-002 active model
│   ├── macro/                  # Active Macro models
│   ├── hybrid/                 # Current Hybrid models
│   ├── dml/                    # Current integrated DML models
│   └── sota/                   # SOTA comparison implementations
├── training/                   # Reserved shared training modules
├── evaluation/                 # Active evaluation and concept-drift code
├── metrics/                    # Reserved reusable metrics
└── utils/                      # Reserved shared utilities

scripts/
├── preprocess/                 # Data/feature pipeline launchers
├── train/                      # Training and comparison launchers
├── evaluate/                   # Evaluation orchestration launchers
└── diagnostics/                # Integrity, drift, and visualization launchers

configs/                        # Dataset, experiment, and environment configuration space
experiments/                    # Experiment metadata and reproduction records
data/manifests/                 # Data manifests, not data payloads
data/local/historical/          # Local datasets retained outside active source
artifacts/manifests/            # Artifact manifests, including external Zeus paths
results/                        # Reserved for reconciled results
legacy/                         # Immutable historical source organized by family
docs/catalog/                   # Research catalog and lineage identifiers
docs/migration/                 # This migration plan and local movement manifest
tests/                          # Reserved test suite
```

## Active lines

- `MICRO-002`: `src/models/micro/VP1_Transformers_dir_size.py` and its evaluator in `src/evaluation/`.
- Macro: current `features/` pipeline is now split between `src/features/`, `src/models/macro/`, and `src/evaluation/`.
- Hybrid: `vectores_features_v2/EXP2`–`EXP5` are in `src/models/hybrid/`.
- DML: `vectores_features_v2/EXP6`–`EXP7` are in `src/models/dml/`.
- SOTA: `estado_del_arte/SOTA_*` is in `src/models/sota/`.

## Historical lines

The `legacy/` tree contains the earlier `d_*`/`ds2_*` Micro models, 45F/30V/40F generations, `features_old/`, `concept_drift_old/`, the phased DML pipeline, exploratory P2/P3/P4 prototypes, and hardcoded-result diagnostics. Where files had identical names or duplicate historical copies, provenance subdirectories were retained instead of overwriting either file.

## Data and artifact policy

Large local CSV datasets are stored under `data/local/historical/`, which is ignored for future additions. Local figures are stored under `results/historical/` with provenance subdirectories. Any locally present result/checkpoint files remain outside `src/` until reconciliation with Zeus. Their status and physical paths are recorded in `docs/migration/LOCAL_MIGRATION.csv`; external paths remain additionally catalogued in `docs/catalog/ARTIFACTS.csv`.

## Pending path/import work

No broad import or path refactor is part of this migration. Relative paths embedded in historical scripts may now require the original working directory or later configuration work. Those findings are reported after the move and are deliberately left unchanged.
