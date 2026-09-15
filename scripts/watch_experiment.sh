#!/usr/bin/env bash

EXP="$1"
LOG="$HOME/ltd-storage/results/canonical/$EXP/train.log"
STATUS="$HOME/ltd-storage/results/canonical/$EXP/status.txt"

clear

while true; do
    clear

    echo "============================================================"
    echo " LTD-Attack experiment monitor"
    echo "============================================================"
    echo
    echo "Experiment: $EXP"
    echo "Time:       $(date)"
    echo

    echo "---------------- STATUS ----------------"
    if [ -f "$STATUS" ]; then
        cat "$STATUS"
    else
        echo "No status file yet."
    fi

    echo
    echo "---------------- GPU -------------------"
    nvidia-smi \
      --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader

    echo
    echo "---------------- PROGRESS --------------"

    if [ -f "$LOG" ]; then

        LAST_EPOCH=$(grep 'Epoch \[' "$LOG" | tail -1)

        if [ -n "$LAST_EPOCH" ]; then
            echo "$LAST_EPOCH"

            COUNT=$(grep -c 'Epoch \[' "$LOG")
            echo
            echo "Epochs completed: $COUNT"

            AVG=$(grep 'Epoch \[' "$LOG" \
                | tail -5 \
                | sed -n 's/.* - \([0-9.]*\)s$/\1/p' \
                | awk '{s+=$1;n++} END {if(n>0) printf "%.1f",s/n}')

            if [ -n "$AVG" ]; then
                echo "Avg last epochs: ${AVG}s"
                python3 - <<PY
count=$COUNT
avg=float("$AVG")
remaining=max(0,100-count)
seconds=remaining*avg
print(f"Nominal ETA to epoch 100: {seconds/60:.1f} min")
PY
            fi
        else
            echo "Preprocessing / model initialization..."
        fi

        echo
        echo "---------------- LAST LOG --------------"
        tail -n 12 "$LOG"
    else
        echo "Log not created yet."
    fi

    sleep 10
done
