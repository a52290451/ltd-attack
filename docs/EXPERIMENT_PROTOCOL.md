# LTD-Attack Experimental Protocol

## Reproducibility states

- R0 UNTRACEABLE
- R1 CODE_ONLY
- R2 PARTIAL
- R3 TRACEABLE
- R4 REPRODUCED
- R5 VALIDATED

## Historical reproduction

Historical experiments may reproduce their original random split when
the purpose is exclusively to verify an earlier reported result.

Such reproductions do not define the methodology for new experiments.

## New experiments

New canonical experiments must use temporal separation inside the
historical campaign.

Conceptually:

TRAIN -> VALIDATION -> INTERNAL TEMPORAL TEST -> sealed FUTURE

The future dataset cannot participate in model development.

## Experiment identity

Any meaningful change creates a new experiment ID, including changes to:

- architecture
- dataset
- split
- features
- sequence length
- seed
- optimizer
- loss
- learning rate
- preprocessing

Historical experiment outputs are immutable.

## Required experiment metadata

Every canonical experiment should record:

- experiment ID
- Git commit
- environment
- dataset IDs
- dataset hashes
- temporal split
- seed
- model configuration
- preprocessing configuration
- checkpoint
- logs
- training curves
- Accuracy
- Macro F1
- Weighted F1
- per-class recall
- confusion matrix
- historical accuracy
- future accuracy
- absolute degradation
- accuracy retention
- runtime

## Execution model

Development:

local machine

Execution:

Zeus

Workflow:

local code/config
-> Git commit
-> Git push
-> Zeus git pull
-> execution
-> ltd-storage artifacts/results
