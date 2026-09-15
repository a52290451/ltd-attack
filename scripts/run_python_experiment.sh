#!/usr/bin/env bash

set -u

EXP="$1"
SCRIPT="$2"

REPO="$HOME/ltd-attack"
RESULT="$HOME/ltd-storage/results/canonical/$EXP"

mkdir -p "$RESULT"

cd "$REPO" || exit 1

source "$HOME/modelos/venv/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export LTD_ENV=zeus

LOG="$RESULT/train.log"
STATUS="$RESULT/status.txt"

{
    echo "experiment=$EXP"
    echo "start=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "python=$(python3 --version 2>&1)"
    echo "status=RUNNING"
} > "$STATUS"

echo "============================================================" >> "$LOG"
echo "START $EXP" >> "$LOG"
echo "DATE $(date --iso-8601=seconds)" >> "$LOG"
echo "============================================================" >> "$LOG"

python3 -u "$SCRIPT" >> "$LOG" 2>&1

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

echo "============================================================" >> "$LOG"
echo "END $EXP" >> "$LOG"
echo "EXIT_CODE $EXIT_CODE" >> "$LOG"
echo "============================================================" >> "$LOG"

exit "$EXIT_CODE"
