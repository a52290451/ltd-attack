#!/usr/bin/env bash
# P16C: única auditoría operativa de 120 posiciones; NO Dataset C científico.
# Tor/obfs4 debe estar ejecutándose en ltd-c-manual-tor; no inicia Cron.
set -euo pipefail
REPO="${REPO:-$HOME/ltd-attack}"
APP="${APP:-$HOME/ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2}"
BASE="${BASE:-$HOME/ltd-storage/dataset-c-smoke-obfs4}"
TOR_CONTAINER="${TOR_CONTAINER:-ltd-c-manual-tor}"
IMAGE="${IMAGE:-tor_1site_noise:final}"

[[ -f "$APP/run_category.sh" && -f "$REPO/scripts/dataset_c/analyze_120_pilot.py" ]] || {
  echo "STOP: faltan archivos del piloto/reporte; ejecuta git pull." >&2; exit 2;
}
[[ "$(docker inspect -f '{{.State.Running}}' "$TOR_CONTAINER" 2>/dev/null)" == true ]] || {
  echo "STOP: Tor no está ejecutándose." >&2; exit 2;
}
docker exec "$TOR_CONTAINER" curl -fsS --max-time 25 \
  --socks5-hostname 127.0.0.1:9050 https://check.torproject.org/api/ip \
  | python3 -c 'import json,sys; assert json.load(sys.stdin).get("IsTor") is True' || {
  echo "STOP: no hay salida Tor verificada." >&2; exit 2;
}
mkdir -p "$BASE"
exec 8>"$BASE/.audit120.lock"
flock -n 8 || { echo "STOP: ya se ejecuta una auditoría 120." >&2; exit 5; }
RUN_ID="audit120_$(date -u +%Y%m%dT%H%M%SZ)_$$"
DEST="$BASE/$RUN_ID"
mkdir -p "$DEST"
printf '%s\n' "$DEST" > "$BASE/.audit120_latest"
printf 'PILOTO_NO_CIENTIFICO obfs4 reloj_host_desincronizado %s\n' "$(date -u +%FT%TZ)" > "$DEST/PILOT_METADATA.txt"
echo "RUN_ID=$RUN_ID"
echo "OUT=$DEST"
echo "INICIO=$(date -u +%FT%TZ)"
echo "=== 120 posiciones secuenciales, MAX_URLS=0, sin Cron ==="

set +e
docker run --rm \
  --name "ltd-c-$RUN_ID" \
  --network "container:$TOR_CONTAINER" \
  --cap-add NET_RAW --cap-add NET_ADMIN \
  --security-opt no-new-privileges \
  --shm-size 1g \
  --mount "type=bind,src=$APP,dst=/opt/ltd-c-v2,readonly" \
  --mount "type=bind,src=$DEST,dst=/audit120" \
  -e APP_HOME=/opt/ltd-c-v2 \
  -e NODE_PATH=/app/node_modules \
  -e CAPTURE_ROOT=/audit120 \
  -e MAX_URLS=0 \
  -e CONSENT_POLICY=accept_all \
  -e SESSION_POLICY=fresh \
  -e CHROME_NO_SANDBOX=1 \
  -e MIN_DISK_FREE_GB="${MIN_DISK_FREE_GB:-120}" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  --entrypoint /bin/bash "$IMAGE" -lc '
    set -uo pipefail
    trap '\''chown -R "$HOST_UID:$HOST_GID" /audit120 2>/dev/null || true'\'' EXIT
    # Tor puede perder temporalmente su listener SOCKS aunque el contenedor siga vivo.
    # Verificar salida efectiva antes de cada categoría y reintentar SOLO si
    # run_category falló antes de registrar cualquier slot. No repetir visitas
    # parciales y no atribuir recuperaciones a una hora pasada.
    tor_ready() {
      curl -fsS --max-time 25 \
        --socks5-hostname 127.0.0.1:9050 \
        https://check.torproject.org/api/ip \
        | grep -Eq "\"IsTor\"[[:space:]]*:[[:space:]]*true"
    }
    slots_count() {
      local dir="/audit120/$1/$2"
      if compgen -G "$dir/attempts_*.tsv" > /dev/null; then
        awk "FNR>1 && NF>1 {n++} END {print n+0}" "$dir"/attempts_*.tsv
      else
        echo 0
      fi
    }
    rc=0
    for spec in "inst1 belleza" "inst1 turismo" "inst2 bancos" "inst2 salud" "inst3 deportes" "inst3 politica"; do
      read -r instance category <<< "$spec"
      echo "===== INICIO $instance/$category $(date -u +%FT%TZ) ====="
      code=4
      for attempt in 1 2 3; do
        if ! tor_ready; then
          echo "TOR_UNREACHABLE $instance/$category intento=$attempt/3; esperando..." >&2
          sleep 15
          continue
        fi
        if bash /opt/ltd-c-v2/run_category.sh "$instance" "$category"; then
          code=0
          break
        else
          code=$?
          counted=$(slots_count "$instance" "$category")
          echo "FALLO_CATEGORIA $instance/$category exit=$code slots=$counted intento=$attempt/3" >&2
          if (( code == 4 && counted == 0 && attempt < 3 )); then
            echo "REINTENTO_SEGURO $instance/$category sin capturas previas" >&2
            sleep 15
            continue
          fi
          break
        fi
      done
      counted=$(slots_count "$instance" "$category")
      echo "===== FIN $instance/$category $(date -u +%FT%TZ) exit=$code slots=$counted/20 ====="
      if (( code != 0 || counted != 20 )); then
        echo "AUDIT_PARTIAL $instance/$category exit=$code slots=$counted/20" >&2
        rc=1
      fi
    done
    exit "$rc"
  '
DOCKER_EXIT=$?
set -e
echo "===== INFORME ====="
python3 "$REPO/scripts/dataset_c/analyze_120_pilot.py" "$DEST" \
  | tee "$DEST/report_console.txt" || true
echo "FIN=$(date -u +%FT%TZ) DOCKER_EXIT=$DOCKER_EXIT OUT=$DEST"
echo "Este piloto obfs4 NO es Dataset C científico."
exit "$DOCKER_EXIT"
