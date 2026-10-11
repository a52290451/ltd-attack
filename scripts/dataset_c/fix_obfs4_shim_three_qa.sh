#!/usr/bin/env bash
# Tor P16C: prevent clock shim inherited by Go obfs4proxy; no secret output.
set -euo pipefail
umask 077
BASE=/data/ltd-c-qa-bsierra
[[ -d "$BASE" && -f "$BASE/three_tor_config/torrc-obfs4" ]] || { echo 'STOP: configuración ausente'; exit 2; }
[[ -z "$(crontab -l 2>/dev/null | grep -E 'LTD_C_(THREE_QA|QA_HOURLY)' || true)" ]] || { echo 'STOP: hay cron activo'; exit 2; }
for i in 1 2 3; do
  NAME="ltd-c-tor-inst$i"
  echo "PARCHEANDO $NAME (solo ClientTransportPlugin)"
  docker inspect "$NAME" >/dev/null
  # El wrapper env evita propagar LD_PRELOAD al binario Go del transporte.
  docker exec "$NAME" python3 -c '
import pathlib,re,sys
p=pathlib.Path("/etc/tor/torrc-obfs4")
s=p.read_text()
pattern=r"(?m)^\s*ClientTransportPlugin\s+obfs4\s+exec\s+[^\r\n]+$"
line="ClientTransportPlugin obfs4 exec /usr/bin/env -u LD_PRELOAD /usr/bin/obfs4proxy"
new,n=re.subn(pattern,line,s)
if n != 1: sys.exit("STOP: ClientTransportPlugin inesperado: "+str(n))
p.write_text(new)
p.chmod(0o600)
' 
  docker exec --user debian-tor "$NAME" tor --verify-config -f /etc/tor/torrc-obfs4 >/dev/null
  echo "CONFIG_OK $NAME"
done
for i in 1 2 3; do docker restart "ltd-c-tor-inst$i" >/dev/null; done
FAILED=0
for i in 1 2 3; do
 NAME="ltd-c-tor-inst$i"; PASSED=0
 for j in $(seq 1 16); do
  if docker exec "$NAME" curl -fsS --max-time 15 --socks5-hostname 127.0.0.1:9050 https://check.torproject.org/api/ip 2>/dev/null |
     python3 -c 'import json,sys; assert json.load(sys.stdin).get("IsTor") is True' 2>/dev/null; then
   PASSED=1; break
  fi
  sleep 12
 done
 if (( PASSED )); then echo "TOR_OK $NAME"; else echo "TOR_FAIL $NAME"; FAILED=1; fi
done
if (( FAILED )); then
 echo "STOP: algún transporte sigue fallando. No iniciamos capturas ni cron."
 for i in 1 2 3; do
  echo "ESTADO inst$i:"
  docker exec "ltd-c-tor-inst$i" sh -c 'test -x /usr/bin/obfs4proxy && echo obfs4proxy_present; stat -c "%U:%G %a %n" /var/lib/tor-ltd/pt_state' || true
  # Se evita extraer logs brutos: pueden revelar endpoints.
 done
 exit 4
fi
cd "$HOME/ltd-attack"
echo "===== SMOKE ====="
python3 scripts/dataset_c/three_instance_qa.py smoke
echo "===== FULL ====="
python3 scripts/dataset_c/three_instance_qa.py full
echo "===== CRON ====="
bash scripts/dataset_c/three_hourly_qa.sh --install
echo "TRES_INSTANCIAS_LISTAS"
