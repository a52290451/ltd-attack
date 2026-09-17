#!/usr/bin/env bash

while true
do
    clear

    echo "=============================================================="
    echo "LTD-ATTACK — CLEAN HYBRID QUEUE"
    date
    echo "=============================================================="
    printf "%-20s %-14s %s\n" "EXPERIMENT" "STATUS" "LAST RESULT"
    echo "--------------------------------------------------------------"

    for EXP in \
      HYB-003-R HYB-003-R-EVAL \
      HYB-004-R HYB-004-R-EVAL \
      HYB-005-R HYB-005-R-EVAL \
      HYB-006-R HYB-006-R-EVAL
    do
        DIR="$HOME/ltd-storage/results/canonical/$EXP"

        STATUS="PENDING"

        if [ -f "$DIR/status.txt" ]; then
            STATUS=$(grep '^status=' "$DIR/status.txt" | cut -d= -f2)
        fi

        LAST=""

        if [ -f "$DIR/train.log" ]; then
            LAST=$(grep -Ehi \
              'Val Acc|MEJOR ACCURACY|ACCURACY FINAL|early stopping|nan' \
              "$DIR/train.log" 2>/dev/null | tail -1)
        fi

        printf "%-20s %-14s %s\n" "$EXP" "$STATUS" "$LAST"
    done

    echo
    nvidia-smi \
      --query-gpu=utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader 2>/dev/null

    echo
    echo "Ctrl+C closes monitor only."

    sleep 10
done
