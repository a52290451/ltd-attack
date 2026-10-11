#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ -z "$(crontab -l 2>/dev/null | grep -E 'LTD_C_(THREE_QA|QA_HOURLY)' || true)" ]] || { echo "STOP: CRON ACTIVO"; exit 2; }
for i in 1 2 3; do
  N="ltd-c-tor-inst$i"
  echo "REPARANDO $N"
  docker exec "$N" sh -c '
    set -e
    command -v env >/dev/null
    command -v obfs4proxy >/dev/null
    CFG=/etc/tor/torrc-obfs4
    test "$(grep -Ec "^[[:space:]]*ClientTransportPlugin[[:space:]]+obfs4[[:space:]]+exec" "$CFG")" -eq 1
    sed -i -E "s|^[[:space:]]*ClientTransportPlugin[[:space:]]+obfs4[[:space:]]+exec.*$|ClientTransportPlugin obfs4 exec /usr/bin/env -u LD_PRELOAD /usr/bin/obfs4proxy|" "$CFG"
    chown 100:101 "$CFG"
    chmod 600 "$CFG"
  '
  docker exec --user debian-tor "$N" tor --verify-config -f /etc/tor/torrc-obfs4 >/dev/null
  echo "CONFIG_OK $N"
done
for i in 1 2 3; do docker restart "ltd-c-tor-inst$i" >/dev/null; done
for i in 1 2 3; do
 N="ltd-c-tor-inst$i"; GOOD=0
 for j in $(seq 1 12); do
   if docker exec "$N" curl -fsS --max-time 12 --socks5-hostname 127.0.0.1:9050 https://check.torproject.org/api/ip 2>/dev/null | python3 -c 'import json,sys; assert json.load(sys.stdin).get("IsTor") is True' 2>/dev/null; then GOOD=1; break; fi
   sleep 10
 done
 if (( GOOD )); then echo "TOR_OK $N"; else echo "TOR_FAIL $N"; exit 4; fi
done
cd "$HOME/ltd-attack"
python3 scripts/dataset_c/three_instance_qa.py smoke
python3 scripts/dataset_c/three_instance_qa.py full
bash scripts/dataset_c/three_hourly_qa.sh --install
echo "TRES_INSTANCIAS_LISTAS"
