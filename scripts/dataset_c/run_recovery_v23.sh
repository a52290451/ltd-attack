#!/usr/bin/env bash
# Recuperación dirigida P16C V2.3; Tor obfs4 ya operativo; sin Cron.
set -euo pipefail
MODE="${1:---consent}"
case "$MODE" in --consent|--other|--all) ;; *) echo "Uso: $0 [--consent|--other|--all]" >&2; exit 2;; esac
APP="${APP:-$HOME/ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2}"
BASE="${BASE:-$HOME/ltd-storage/dataset-c-smoke-obfs4}"
TOR="${TOR_CONTAINER:-ltd-c-manual-tor}"
[[ -f "$APP/capture_dom.js" ]] || { echo "STOP: APP ausente" >&2; exit 2; }
grep -q visible-label-v23 "$APP/capture_dom.js" || { echo "STOP: falta parche V2.3" >&2; exit 2; }
docker exec "$TOR" curl -fsS --max-time 25 --socks5-hostname 127.0.0.1:9050 https://check.torproject.org/api/ip \
  | python3 -c 'import json,sys; assert json.load(sys.stdin).get("IsTor") is True' || { echo "STOP: Tor no disponible" >&2; exit 3; }
mkdir -p "$BASE"
exec 8>"$BASE/.audit120.lock"
flock -n 8 || { echo "STOP: existe otra auditoría o recuperación en marcha" >&2; exit 5; }
DEST="$BASE/recovery_v23_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$DEST"
printf '%s\n' "$DEST" > "$BASE/.recovery_v23_latest"
echo "MODE=$MODE OUT=$DEST"
if [[ "$MODE" == --consent ]]; then
  SPECS=$(printf '%s\n' "inst1 turismo 8,16" "inst2 bancos 8" "inst3 deportes 1" "inst3 politica 8,17")
elif [[ "$MODE" == --other ]]; then
  SPECS=$(printf '%s\n' "inst1 belleza 4,16,17" "inst2 bancos 10" "inst2 salud 2,13,16" "inst3 deportes 5,17,18,20" "inst3 politica 14")
else
  SPECS=$(printf '%s\n' "inst1 belleza 4,16,17" "inst1 turismo 8,16" "inst2 bancos 8,10" "inst2 salud 2,13,16" "inst3 deportes 1,5,17,18,20" "inst3 politica 8,14,17")
fi
docker run --rm --name "ltd-c-recovery-v23-$$" \
  --network "container:$TOR" \
  --cap-add NET_RAW --cap-add NET_ADMIN \
  --security-opt no-new-privileges --shm-size 1g \
  --mount "type=bind,src=$APP,dst=/opt/ltd-c-v2,readonly" \
  --mount "type=bind,src=$DEST,dst=/qa_recovery" \
  -e APP_HOME=/opt/ltd-c-v2 -e NODE_PATH=/app/node_modules \
  -e CAPTURE_ROOT=/qa_recovery -e MAX_URLS=0 \
  -e CONSENT_POLICY=accept_all -e SESSION_POLICY=fresh \
  -e CHROME_NO_SANDBOX=1 -e MIN_DISK_FREE_GB=120 \
  -e CHROME_TIMEOUT=70 -e RUN_WALL_LIMIT=95 \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -e SPECS="$SPECS" \
  --entrypoint /bin/bash tor_1site_noise:final -lc '
    set -uo pipefail
    trap '\''chown -R "$HOST_UID:$HOST_GID" /qa_recovery 2>/dev/null || true'\'' EXIT
    while IFS= read -r spec; do
      [[ -n "$spec" ]] || continue
      read -r inst category slots <<< "$spec"
      echo "===== $inst/$category SLOTS=$slots ====="
      SITE_SLOT_FILTER="$slots" bash /opt/ltd-c-v2/run_category.sh "$inst" "$category" || true
    done <<< "$SPECS"
  '
python3 - "$DEST" <<'PY'
from pathlib import Path
from collections import Counter
import csv,sys
rows=[]
for f in Path(sys.argv[1]).rglob("attempts_*.tsv"):
    with f.open() as h: rows.extend(csv.DictReader(h,delimiter="\t"))
print("TOTAL",len(rows), "ESTADOS",dict(Counter(r["status"] for r in rows)))
for r in rows: print(r["status"],r["url"])
PY
echo "OUT=$DEST"
