#!/usr/bin/env bash

set -u

REPO="$HOME/ltd-attack"
BASE="$HOME/ltd-storage/results/canonical"
QUEUE_LOG="$BASE/reproduction_queue.log"
QUEUE_STATUS="$BASE/queue_status.tsv"

cd "$REPO" || exit 1

source "$HOME/modelos/venv/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export LTD_ENV=zeus

printf "experiment\tstart\tend\texit_code\n" > "$QUEUE_STATUS"

run_one() {
    EXP="$1"
    SCRIPT="$2"

    START=$(date --iso-8601=seconds)

    echo
    echo "============================================================"
    echo "STARTING: $EXP"
    echo "TIME: $START"
    echo "============================================================"

    scripts/run_python_experiment.sh "$EXP" "$SCRIPT"
    RC=$?

    END=$(date --iso-8601=seconds)

    printf "%s\t%s\t%s\t%s\n" \
        "$EXP" "$START" "$END" "$RC" >> "$QUEUE_STATUS"

    echo
    echo "FINISHED: $EXP"
    echo "EXIT CODE: $RC"
    echo

    return "$RC"
}

# ----------------------------------------------------------
# MICRO future evaluations
# ----------------------------------------------------------

run_one \
  "MICRO-002-R-LEGACY-EVAL" \
  "experiments/MICRO-002-R-LEGACY-EVAL/evaluate.py"

run_one \
  "MICRO-002-R-CORRECTED-EVAL" \
  "experiments/MICRO-002-R-CORRECTED-EVAL/evaluate.py"

# ----------------------------------------------------------
# MACRO historical reproduction + future evaluation
# ----------------------------------------------------------

run_one \
  "MACRO-002-R" \
  "experiments/MACRO-002-R/train.py"

MACRO_RC=$?

if [ "$MACRO_RC" -eq 0 ]; then
    run_one \
      "MACRO-002-R-EVAL" \
      "experiments/MACRO-002-R/evaluate.py"
else
    echo "Skipping MACRO evaluation because training failed."
fi

# ----------------------------------------------------------
# State-of-the-art baselines
# ----------------------------------------------------------

run_one \
  "SOTA-001-R" \
  "experiments/SOTA-001-R/run.py"

run_one \
  "SOTA-002-R" \
  "experiments/SOTA-002-R/run.py"

run_one \
  "SOTA-003-R" \
  "experiments/SOTA-003-R/run.py"

echo
echo "============================================================"
echo "REPRODUCTION QUEUE FINISHED"
echo "$(date --iso-8601=seconds)"
echo "============================================================"
