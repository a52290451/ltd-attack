#!/usr/bin/env bash
# P16C: programador de QA horario, NO Dataset C científico.
# Mantiene el reloj kernel y los datasets históricos SIN MODIFICACIONES.
set -euo pipefail
REPO="${REPO:-$HOME/ltd-attack}"
BASE="${BASE:-$HOME/ltd-storage/dataset-c-smoke-obfs4}"
TOR="${TOR_CONTAINER:-ltd-c-manual-tor}"
OFFSET="${LTD_PILOT_CLOCK_OFFSET_SECONDS:--2963}"
SCRIPT="$REPO/scripts/dataset_c/run_hourly_qa.sh"
MODE="${1:---status}"

tor_ok() {
  docker exec "$TOR" curl -fsS --max-time 35 \
    --socks5-hostname 127.0.0.1:9050 https://check.torproject.org/api/ip \
    | python3 -c 'import json,sys; assert json.load(sys.stdin).get("IsTor") is True' 2>/dev/null
}
ensure_tor() {
  if [[ "$(docker inspect -f '{{.State.Running}}' "$TOR" 2>/dev/null)" != true ]]; then
    docker start "$TOR" >/dev/null || { echo "STOP: contenedor Tor detenido" >&2; return 1; }
  fi
  if tor_ok; then echo "Tor OK"; return 0; fi
  if ! docker exec "$TOR" pgrep -u debian-tor -x tor >/dev/null 2>&1; then
    echo "Arrancando Tor piloto en modo desacoplado del terminal..."
    # Solo QA obfs4, con shim temporal NO válido para evidencia científica.
    docker exec -d --user debian-tor \
      -e LTD_CLOCK_OFFSET_SECONDS="$OFFSET" \
      -e LD_PRELOAD=/usr/local/lib/libltd_realtime.so \
      "$TOR" tor -f /etc/tor/torrc-obfs4
  fi
  for i in $(seq 1 12); do
    sleep 5
    if tor_ok; then echo "Tor OK"; return 0; fi
  done
  echo "STOP: no se verificó IsTor:true; se omite esta hora" >&2
  return 1
}
case "$MODE" in
--install)
  command -v crontab >/dev/null || { echo "STOP: crontab no está instalado" >&2; exit 2; }
  [[ -f "$SCRIPT" ]] || { echo "STOP: no existe $SCRIPT" >&2; exit 2; }
  old="$(crontab -l 2>/dev/null || true)"
  {
    printf '%s\n' "$old" | grep -v -F '/scripts/dataset_c/run_hourly_qa.sh' || true
    printf '0 * * * * /bin/bash %s --once # LTD_C_QA_HOURLY\n' "$SCRIPT"
  } | crontab -
  echo "INSTALADO: QA piloto de 120 posiciones cada hora en minuto 00 (hora del host)."
  echo "Consultar: crontab -l | tail -n 5"
  echo "Detener: crontab -l | grep -v LTD_C_QA_HOURLY | crontab -"
  ;;
--once)
  mkdir -p "$BASE/cron_logs"
  LOG="$BASE/cron_logs/hourly_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
  echo "LOG=$LOG"
  {
    echo "INICIO=$(date -u +%FT%TZ)"
    echo "PILOTO_OBFS4_NO_CIENTIFICO=1"
    ensure_tor || exit 3
    export MIN_DISK_FREE_GB=120
    bash "$REPO/scripts/dataset_c/run_120_pilot.sh"
  } >> "$LOG" 2>&1
  tail -n 12 "$LOG"
  ;;
--status)
  echo "CRON:"
  (crontab -l 2>/dev/null || true) | grep 'LTD_C_QA_HOURLY' || true
  echo "TOR:"
  docker ps --filter "name=$TOR" --format '{{.Names}} {{.Status}}'
  echo "ÚLTIMOS LOGS:"
  ls -t "$BASE"/cron_logs/hourly_*.log 2>/dev/null | head -n 3 || true
  echo "ÚLTIMA AUDITORÍA:"
  cat "$BASE/.audit120_latest" 2>/dev/null || true
  ;;
*) echo "Uso: bash $SCRIPT [--install|--once|--status]" >&2; exit 2 ;;
esac
