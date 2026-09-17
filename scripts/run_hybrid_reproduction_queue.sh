#!/usr/bin/env bash

set -u

RUNNER="$HOME/ltd-attack/scripts/run_python_experiment.sh"

run_stage () {
    EXP="$1"
    SCRIPT="$2"

    echo
    echo "============================================================"
    echo "$EXP"
    date -Is
    echo "============================================================"

    "$RUNNER" "$EXP" "$SCRIPT"
    return $?
}

for EXP in HYB-003-R HYB-004-R HYB-005-R HYB-006-R
do
    run_stage "$EXP" \
      "$HOME/ltd-attack/experiments/$EXP/train.py"

    RC=$?

    if [ "$RC" -ne 0 ]; then
        echo "TRAIN FAILED: $EXP"
        continue
    fi

    run_stage "${EXP}-EVAL" \
      "$HOME/ltd-attack/experiments/$EXP/evaluate.py"

done
