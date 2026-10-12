#!/usr/bin/env bash
# P16C: 3 Tor + 3 parallel Chrome collectors, isolated, hourly operational QA.
# Cron installation must NOT fail because a single external IsTor probe fails.
# QA only, until host clock and external dataset protocol are approved.
set -euo pipefail
umask 077
REPO="${HOME}/ltd-attack"
BASE="/data/ltd-c-qa-bsierra"
PY="$REPO/scripts/dataset_c/three_instance_qa.py"
HEALTH="$REPO/scripts/dataset_c/tor_health_retry.py"
AUDIT="$REPO/scripts/dataset_c/audit_three_hourly.py"
SELF="$REPO/scripts/dataset_c/three_hourly_qa.sh"
MODE="${1:---status}"

check_static() {
  [[ -d "$BASE" && "$(stat -c %u "$BASE")" == "$(id -u)" ]] ||
    { echo "STOP: QA data directory missing/not owned"; return 2; }
  [[ -f "$PY" && -f "$HEALTH" && -f "$AUDIT" ]] ||
    { echo "STOP: QA scripts missing"; return 2; }
  [[ -f "$HOME/ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2/capture_dom.js" ]] ||
    { echo "STOP: browser collector missing"; return 2; }
  if (crontab -l 2>/dev/null || true) | grep -q 'LTD_C_QA_HOURLY'; then
    echo "STOP: old one-Tor Cron entry still active"
    return 2
  fi
  local n
  for i in 1 2 3; do
    n="ltd-c-tor-inst$i"
    [[ "$(docker inspect --format '{{.State.Running}}' "$n" 2>/dev/null)" == true ]] ||
      { echo "STOP: $n not running"; return 3; }
    docker exec "$n" test -x /usr/bin/obfs4proxy ||
      { echo "STOP: $n missing obfs4proxy"; return 3; }
  done
  echo "STATIC_CHECK_OK: 3 running Tor containers; transport installed"
}

install_job() {
  check_static
  [[ "$(systemctl is-active cron 2>/dev/null)" == active ]] ||
    { echo "STOP: cron service not active"; return 3; }
  local found original cron_entry count
  found=$(find "$BASE" -mindepth 2 -maxdepth 2 -type f -name VALIDATED_THREE_FULL.txt -path '*/three_hourly_*/*' -print -quit)
  [[ -n "$found" ]] || { echo "STOP: no validated 120-slot reference"; return 3; }
  original=$(crontab -l 2>/dev/null || true)
  count=$(printf '%s\n' "$original" | grep -c 'LTD_C_THREE_QA' || true)
  [[ "$count" -le 1 ]] || { echo "STOP: duplicate hourly jobs"; return 3; }
  cron_entry="0 * * * * /bin/bash $SELF --once # LTD_C_THREE_QA"
  if printf '%s\n' "$original" | grep -Fxq "$cron_entry"; then
    echo "CRON_ALREADY_INSTALLED"
  else
    local backup="$HOME/ltd-ops/cron-backups"
    mkdir -p "$backup"; chmod 700 "$HOME/ltd-ops" "$backup"
    printf '%s\n' "$original" > "$backup/before-three-hourly-install-$(date -u +%Y%m%dT%H%M%SZ).txt"
    chmod 600 "$backup"/before-three-hourly-install-*.txt
    { printf '%s\n' "$original" | grep -v 'LTD_C_THREE_QA' || true
      printf '%s\n' "$cron_entry"; } | crontab -
  fi
  echo "INSTALLED: 120 QA slots/hour, 3 isolated Tor+Chrome paths."
  echo "NOTE: Tor reachability is retried at RUN TIME, not required at installation."
}

once() {
  [[ -d "$BASE" ]] || { echo "STOP: QA directory absent"; exit 2; }
  exec 9>"$BASE/.three_hourly_runner.lock"
  if ! flock -n 9; then
    echo "ALREADY_RUNNING: another --once holds hourly runner lock"
    exit 0
  fi
  mkdir -p "$BASE/three_logs"
  local LOG="$BASE/three_logs/run_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
  echo "LOG=$LOG"
  {
    echo "START=$(date -u +%FT%TZ) HOST_CLOCK_NTP_NOT_VERIFIED"
    echo "===== TOR WAIT, up to 6 minutes ====="
    if ! python3 "$HEALTH" --max-wait 360 --interval 12; then
      echo "SKIPPED_NO_TOR: no capture folder created; will try next scheduled hour"
      exit 4
    fi
    echo "===== THREE PARALLEL COLLECTORS ====="
    if ! python3 "$PY" full; then
      echo "ROUND_INCOMPLETE: retain QA evidence, not scientific Dataset C"
      exit 5
    fi
  } > "$LOG" 2>&1
  local RUN
  RUN=$(sed -n 's/^FIN THREE_EXIT=0 OUT=//p' "$LOG" | tail -n 1)
  if [[ "$RUN" != "$BASE"/three_hourly_* || ! -f "$RUN/VALIDATED_THREE_FULL.txt" ]]; then
    echo "STOP: no validated full run path" >> "$LOG"
    tail -n 20 "$LOG"
    return 6
  fi
  if ! python3 "$AUDIT" "$RUN" >> "$LOG" 2>&1; then
    echo "QA_AUDIT_FAILED: preserve raw QA run; never treat as scientific" >> "$LOG"
    tail -n 28 "$LOG"
    return 7
  fi
  echo "FINISHED_AT=$(date -u +%FT%TZ)" >> "$LOG"
  tail -n 20 "$LOG"
}

case "$MODE" in
  --install) install_job ;;
  --enable-now)
    install_job
    mkdir -p "$BASE/three_logs"
    nohup /bin/bash "$SELF" --once \
      > "$BASE/three_logs/manual_start_$(date -u +%Y%m%dT%H%M%SZ).log" \
      2>&1 < /dev/null &
    echo "MANUAL_START_PID=$!"
    echo "CHECK_LOGS=$BASE/three_logs"
    ;;
  --once) once ;;
  --status)
    echo "CRON:"
    (crontab -l 2>/dev/null || true) | grep 'LTD_C_THREE_QA' || true
    echo "TOR CONTAINERS:"
    for i in 1 2 3; do
      docker inspect --format 'container={{.Name}} running={{.State.Running}}' "ltd-c-tor-inst$i" 2>/dev/null || true
    done
    echo "LAST HOURLY LOG:"
    ls -t "$BASE"/three_logs/run_*.log 2>/dev/null | head -n 1 || true
    ;;
  *)
    echo "Usage: bash $SELF --install|--enable-now|--once|--status" >&2
    exit 2 ;;
esac
