#!/usr/bin/env bash

set -u

REPO="$HOME/ltd-attack"
EXP="MICRO-002-R"
RESULT_DIR="$HOME/ltd-storage/results/canonical/$EXP"
ARTIFACT_DIR="$HOME/ltd-storage/artifacts/micro/$EXP"
LOG="$RESULT_DIR/train.log"
STATUS="$RESULT_DIR/status.txt"

mkdir -p "$RESULT_DIR" "$ARTIFACT_DIR"

cd "$REPO" || exit 1

source "$HOME/modelos/venv/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export LTD_ENV=zeus

{
    echo "experiment=$EXP"
    echo "start=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "python=$(python3 --version 2>&1)"
    echo "status=RUNNING"
} > "$STATUS"

echo "==================================================" >> "$LOG"
echo "START $EXP" >> "$LOG"
echo "DATE: $(date --iso-8601=seconds)" >> "$LOG"
echo "COMMIT: $(git rev-parse HEAD)" >> "$LOG"
echo "PYTHONPATH=$PYTHONPATH" >> "$LOG"
echo "==================================================" >> "$LOG"

python3 experiments/MICRO-002-R/train_reproduction.py >> "$LOG" 2>&1
EXIT_CODE=$?

{
    echo "end=$(date --iso-8601=seconds)"
    echo "exit_code=$EXIT_CODE"
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "status=COMPLETED"
    else
        echo "status=FAILED"
    fi
} >> "$STATUS"

echo "==================================================" >> "$LOG"
echo "END $EXP" >> "$LOG"
echo "DATE: $(date --iso-8601=seconds)" >> "$LOG"
echo "EXIT_CODE: $EXIT_CODE" >> "$LOG"
echo "==================================================" >> "$LOG"

exit "$EXIT_CODE"
