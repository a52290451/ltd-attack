#!/usr/bin/env bash

BASE="$HOME/ltd-storage/results/canonical"

EXPS=(
  MICRO-002-R
  MICRO-002-R-LEGACY-EVAL
  MICRO-002-R-CORRECTED-EVAL
  MACRO-002-R
  MACRO-002-R-EVAL
  SOTA-001-R
  SOTA-002-R
  SOTA-003-R
)

while true; do

    clear

    echo "================================================================================"
    echo " LTD-ATTACK REPRODUCTION QUEUE"
    echo " $(date)"
    echo "================================================================================"

    printf "%-32s %-14s %-45s\n" \
        "EXPERIMENT" "STATUS" "LAST PROGRESS"

    echo "--------------------------------------------------------------------------------"

    for EXP in "${EXPS[@]}"; do

        STATUS="$BASE/$EXP/status.txt"
        LOG="$BASE/$EXP/train.log"

        STATE="PENDING"
        PROGRESS="-"

        if [ -f "$STATUS" ]; then
            STATE=$(grep '^status=' "$STATUS" | tail -1 | cut -d= -f2)
        fi

        if [ -f "$LOG" ]; then
            PROGRESS=$(
                grep -Ei \
                'Epoch|accuracy|early stopping|completad|finalizad|Test Set' \
                "$LOG" | tail -1 | cut -c1-45
            )

            [ -z "$PROGRESS" ] && PROGRESS="initializing/preprocessing"
        fi

        printf "%-32s %-14s %-45s\n" \
            "$EXP" "$STATE" "$PROGRESS"

    done

    echo
    echo "GPU:"
    nvidia-smi \
      --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader

    echo
    echo "Queue status:"
    if [ -f "$BASE/queue_status.tsv" ]; then
        tail -n 8 "$BASE/queue_status.tsv"
    else
        echo "Not started."
    fi

    echo
    echo "Ctrl+C closes ONLY this monitor."

    sleep 10
done
