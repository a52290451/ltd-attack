#!/usr/bin/env bash
# P16C: QA tres recolectores paralelos; no constituye Dataset C científico.
set -euo pipefail
umask 077
REPO="${HOME}/ltd-attack"
BASE="/data/ltd-c-qa-bsierra"
PY="$REPO/scripts/dataset_c/three_instance_qa.py"
MODE="${1:---status}"
case "$MODE" in
  --install)
    [[ -z "$(crontab -l 2>/dev/null | grep LTD_C_QA_HOURLY || true)" ]] || { echo "STOP: cron antiguo todavía activo"; exit 2; }
    FOUND=$(find "$BASE" -maxdepth 2 -type f -name VALIDATED_THREE_FULL.txt -path '*/three_hourly_*/*' | head -1)
    [[ -n "$FOUND" ]] || { echo "STOP: falta una prueba full completa de tres instancias"; exit 3; }
    python3 "$PY" status | tee "$BASE/three_last_status.txt"
    [[ $(grep -c 'IsTor True' "$BASE/three_last_status.txt") == 3 ]] || { echo "STOP: no hay tres Tor saludables"; exit 4; }
    {
      (crontab -l 2>/dev/null || true) | grep -v 'LTD_C_THREE_QA' || true
      printf '0 * * * * /bin/bash %s --once # LTD_C_THREE_QA\n' "$REPO/scripts/dataset_c/three_hourly_qa.sh"
    } | crontab -
    echo "INSTALADO: tres capturadores en paralelo, ronda cada hora (QA solamente)."
    ;;
  --once)
    mkdir -p "$BASE/three_logs"
    LOG="$BASE/three_logs/run_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
    echo "LOG=$LOG"
    python3 "$PY" full > "$LOG" 2>&1 || { tail -n 30 "$LOG"; exit 1; }
    RUN=$(sed -n 's/^FIN THREE_EXIT=0 OUT=//p' "$LOG" | tail -n 1)
    [[ "$RUN" == "$BASE"/three_hourly_* ]] || { echo "STOP: falta ruta de ejecución validada" >> "$LOG"; exit 3; }
    python3 "$REPO/scripts/dataset_c/audit_three_hourly.py" "$RUN" >> "$LOG" 2>&1 || { tail -n 35 "$LOG"; exit 4; }
    tail -n 18 "$LOG"
    ;;
  --status)
    echo "CRON:"
    (crontab -l 2>/dev/null || true) | grep 'LTD_C_THREE_QA' || true
    echo "TOR:"
    python3 "$PY" status
    echo "ÚLTIMA RONDA:"
    ls -t "$BASE"/three_logs/run_*.log 2>/dev/null | head -n 1 || true
    ;;
  *)
    echo "Uso: bash $0 --install|--once|--status" >&2; exit 2 ;;
esac
