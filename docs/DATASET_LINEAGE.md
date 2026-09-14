# LTD-Attack Dataset Lineage

## Historical campaign

Nominal collection design:

- approximately 65 target websites
- hourly collection
- approximately 60 days
- theoretical maximum: 65 × 24 × 60 = 93,600 captures

The theoretical maximum is not the observed dataset size. Capture
failures, inaccessible sites, invalid captures and subsequent
quality-control filtering reduce the effective number of samples.

### Processed generation

- 133,119 observations
- 118 labels

### CLEAN generation

- 77,685 observations
- 69 sites
- temporal range: 2025-11-18 09:00:02 to 2026-01-08 07:36:29

### MICRO-002 experimental subset

Filtering:

1. explicit removal of labels 5 and 49
2. retain classes with at least 50% of the maximum class count

Result:

- 65 sites
- 72,603 observations
- random stratified 80/20 historical split
- historical test set: 14,521 observations

## Future Concept Drift campaign

Nominal collection design:

- approximately 65 target websites
- hourly collection
- approximately 15 days
- theoretical maximum: 65 × 24 × 15 = 23,400 captures

### Processed generation

- 35,416 observations
- 118 labels

### CLEAN generation

- 20,363 observations
- 69 sites
- temporal range: 2026-03-23 17:00:02 to 2026-04-07 08:05:30

### MICRO-002 future subset

Restricting evaluation to the 65 trained classes:

- 18,543 observations
- 65 sites

## Temporal separation

Last historical capture:

2026-01-08 07:36:29

First future capture:

2026-03-23 17:00:02

Gap:

74 days 09:23:33

There are no shared `pcap_name` values between the historical and
future CLEAN datasets.

`pcap_uid` is unique only inside an individual processed dataset and
must not be treated as a global capture identifier.

## Methodological rule for new experiments

The future Concept Drift dataset is a sealed holdout.

It must not be used for:

- architecture selection
- feature selection
- hyperparameter tuning
- threshold selection
- early stopping
- model selection

All model development must use the historical campaign only.

Future evaluation is performed only after the experiment has been frozen.
