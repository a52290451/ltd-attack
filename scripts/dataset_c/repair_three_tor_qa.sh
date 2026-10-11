#!/usr/bin/env bash
# Reparación conservadora para tres Tor del piloto P16C.
# No borra contenedores, volúmenes ni datos; Cron se mantiene desactivado.
set -euo pipefail
BASE=/data/ltd-c-qa-bsierra
[[ -d "$BASE" ]] || { echo "STOP: BASE ausente"; exit 2; }
[[ -z "$(crontab -l 2>/dev/null | grep -E 'LTD_C_(THREE_QA|QA_HOURLY)' || true)" ]] || { echo "STOP: desactivar Cron antes de reparar"; exit 2; }
echo "Preparando directorios de estado privados de pluggable transports..."
for i in 1 2 3; do
  n="ltd-c-tor-inst$i"
  docker inspect "$n" >/dev/null
  docker exec --user 100:101 "$n" sh -c '
    umask 077
    test -d /var/lib/tor-ltd &&
    mkdir -p /var/lib/tor-ltd/pt_state &&
    chmod 700 /var/lib/tor-ltd/pt_state &&
    test -w /var/lib/tor-ltd/pt_state
  '
  echo "PT_STATE_OK $n"
done
echo "Reiniciando únicamente los tres Tor nuevos..."
for i in 1 2 3; do docker restart "ltd-c-tor-inst$i" >/dev/null; done
echo "Esperando bootstrap de Tor (hasta 180 s)..."
for i in 1 2 3; do
 n="ltd-c-tor-inst$i"; good=0
 for j in $(seq 1 12); do
   if docker exec "$n" curl --silent --show-error --fail --max-time 12       --socks5-hostname 127.0.0.1:9050 https://check.torproject.org/api/ip 2>/dev/null |
      python3 -c 'import json,sys; assert json.load(sys.stdin).get("IsTor") is True' 2>/dev/null; then
     good=1; break
   fi
   sleep 15
 done
 if (( good )); then
    echo "TOR_OK $n"
 else
    echo "TOR_FAIL $n" >&2
    echo "Resumen del log Tor (no imprimir bridge/cert/endpoint):" >&2
    docker logs --tail 150 "$n" 2>&1 |
       grep -Ei 'Managed proxy|Bootstrapped|permission|Transport|PT_STATE' |
       sed -E 's/([0-9]{1,3}\.){3}[0-9]{1,3}/[IP]/g' | tail -n 10 >&2 || true
    exit 4
 fi
done
echo "TRES_TOR_SALUDABLES; empezando smoke y luego full."
cd "$HOME/ltd-attack"
python3 scripts/dataset_c/three_instance_qa.py smoke
python3 scripts/dataset_c/three_instance_qa.py full
bash scripts/dataset_c/three_hourly_qa.sh --install
echo "THREE_QA_INSTALADO"
