#!/usr/bin/env bash
# Dataset C: fail-closed QA evaluation of an HTTP CONNECT -> Tor SOCKS adapter.
# No changes to live Tor containers, host clock, canonical collector or historical artifacts.
set -euo pipefail
umask 077
MODE="${1:---probe-and-qa3}"
[[ "$MODE" == --probe-only || "$MODE" == --probe-and-qa3 ]] || { echo 'Usage: --probe-only | --probe-and-qa3' >&2; exit 2; }
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$HOME/ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2"
BASE=/data/ltd-c-qa-bsierra/C_DIRECT_V1
[[ -d "$BASE" && -f "$APP/capture_dom.js" && -f "$APP/run_category.sh" ]] || { echo 'STOP: baseline collector/destination missing' >&2; exit 3; }
OUT="$BASE/diagnostics/browser_tor_recovery_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$OUT"
exec > >(tee "$OUT/report.log") 2>&1
echo "RECOVERY_OUT=$OUT"
echo "SCIENTIFIC_DATASET_C_CAPTURE=NOT_AUTHORIZED"
echo "HOST_NTP_SYNC=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown)"
for n in ltd-c-tor-inst1 ltd-c-tor-inst2 ltd-c-tor-inst3; do
  [[ "$(docker inspect -f '{{.State.Running}}' "$n")" == true ]] || { echo "STOP: $n not running"; exit 3; }
  docker exec "$n" grep -qx 'UseBridges 0' /tmp/direct-torrc || { echo "STOP: $n is not direct Tor"; exit 3; }
done
probe() {
  local i="$1" mode="$2"
  timeout 125 docker run --rm --name "ltd-c-proxy-recovery-$i-$mode-$$" \
    --network "container:ltd-c-tor-inst$i" --security-opt no-new-privileges \
    --shm-size 1g -e NODE_PATH=/app/node_modules -e QA_HEADLESS_MODE="$mode" \
    --mount "type=bind,src=$SOURCE_DIR,dst=/qa_scripts,readonly" \
    --entrypoint /bin/bash tor_1site_noise:final -lc '
      node /qa_scripts/http_connect_to_tor_v1.js &
      p=$!
      trap '\''kill "$p" 2>/dev/null || true; wait "$p" 2>/dev/null || true'\'' EXIT
      sleep 0.6
      for attempt in 1 2; do
        if node /qa_scripts/probe_http_tor_browser_v1.js; then exit 0; fi
        echo "RETRY_BROWSER_PROBE attempt=$attempt"
        sleep 5
      done
      exit 7
    '
}
SELECTED=''
for candidate in new shell; do
  echo "=== PROBE_MODE=$candidate ==="
  for i in 1 2 3; do
    probe "$i" "$candidate" > "$OUT/probe_inst${i}_${candidate}.log" 2>&1 &
    eval "pid_$i=\$!"
  done
  good=0
  for i in 1 2 3; do
    eval "pid=\$pid_$i"
    if wait "$pid" && grep -q "BROWSER_TOR_PROBE_PASS $candidate" "$OUT/probe_inst${i}_${candidate}.log"; then
      good=$((good+1))
      echo "PROBE_PASS inst$i mode=$candidate"
    else
      echo "PROBE_FAIL inst$i mode=$candidate"
    fi
    grep -E 'BRIDGE READY|BRIDGE TOR_SOCKS_CONNECT_OK|CHROME|BROWSER_TOR_VERIFIED|BROWSER_EXAMPLE_VERIFIED|FRAME_DETACHED|BROWSER_TOR_PROBE_(PASS|FAIL)|TOP_FAIL' "$OUT/probe_inst${i}_${candidate}.log" | tail -22 || true
  done
  if (( good==3 )); then SELECTED="$candidate"; break; fi
done
if [[ -z "$SELECTED" ]]; then
  echo "RECOVERY_BLOCKED: no uniform browser mode achieved Tor-verified navigation in all 3 instances"
  echo "RECOVERY_OUT=$OUT"
  exit 5
fi
echo "RECOVERY_PROBES_ALL_PASS MODE=$SELECTED"
if [[ "$MODE" == --probe-only ]]; then exit 0; fi
# Create private local overlay; the baseline collector remains unmodified.
PATCHED="$OUT/capture_dom.qa.js"
python3 - "$APP/capture_dom.js" "$PATCHED" "$SELECTED" <<'PY'
import sys,pathlib
src=pathlib.Path(sys.argv[1]).read_text()
needle=chr(96)+'--proxy-server=socks5://127.0.0.1:'+chr(36)+'{socksPort}'+chr(96)
assert src.count(needle)==1, 'Unexpected baseline proxy configuration: fail closed'
src=src.replace(needle, "'--proxy-server=http://127.0.0.1:18888'")
if sys.argv[3]=='shell':
 needle2='headless: true, args, timeout: 20000'
 assert src.count(needle2)==1, 'Unexpected headless config: fail closed'
 src=src.replace(needle2,"headless: 'shell', args, timeout: 20000")
pathlib.Path(sys.argv[2]).write_text(src)
PY
sha256sum "$APP/capture_dom.js" "$PATCHED" | tee "$OUT/overlay_hashes.txt"
QA="$BASE/qa_only/recovery_qa3_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$QA"
cat > "$QA/qa_manifest.txt" <<EOF
QA_ONLY_NOT_FOR_SCIENTIFIC_USE
transport=DIRECT_TOR_VIA_LOCAL_HTTP_CONNECT_TO_SOCKS
headless_mode=$SELECTED
host_NTP_synced=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown)
collector_baseline=$APP
proxy_adapter=$SOURCE_DIR/http_connect_to_tor_v1.js
sites=inst1/turismo:slot1,inst2/salud:slot1,inst3/deportes:slot1
EOF
collect_one() {
  local i="$1" category="$2"
  timeout 145 docker run --rm --name "ltd-c-recovery-qa-inst$i-$$" \
    --network "container:ltd-c-tor-inst$i" --cap-add NET_RAW --cap-add NET_ADMIN \
    --security-opt no-new-privileges --shm-size 1g \
    --mount "type=bind,src=$APP,dst=/opt/ltd-c-v2,readonly" \
    --mount "type=bind,src=$PATCHED,dst=/opt/ltd-c-v2/capture_dom.js,readonly" \
    --mount "type=bind,src=$SOURCE_DIR,dst=/qa_scripts,readonly" \
    --mount "type=bind,src=$QA,dst=/audit" \
    -e APP_HOME=/opt/ltd-c-v2 -e NODE_PATH=/app/node_modules \
    -e CAPTURE_ROOT=/audit -e MAX_URLS=1 \
    -e CONSENT_POLICY=accept_all -e SESSION_POLICY=fresh \
    -e CHROME_NO_SANDBOX=1 -e MIN_DISK_FREE_GB=120 \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    --entrypoint /bin/bash tor_1site_noise:final -lc "
      node /qa_scripts/http_connect_to_tor_v1.js >/audit/bridge_inst$i.log 2>&1 &
      proxy_pid=\$!
      trap 'kill \"\$proxy_pid\" 2>/dev/null || true; wait \"\$proxy_pid\" 2>/dev/null || true; chown -R \"\$HOST_UID:\$HOST_GID\" /audit 2>/dev/null || true' EXIT
      sleep 0.6
      bash /opt/ltd-c-v2/run_category.sh inst$i $category
    "
}
echo '=== QA3 BEGIN: one existing canonical site per instance ==='
collect_one 1 turismo > "$QA/capture_inst1.log" 2>&1 & p1=$!
collect_one 2 salud > "$QA/capture_inst2.log" 2>&1 & p2=$!
collect_one 3 deportes > "$QA/capture_inst3.log" 2>&1 & p3=$!
for i in 1 2 3; do
  eval "p=\$p$i"
  if wait "$p"; then echo "QA_CONTAINER_EXIT inst$i=0"; else echo "QA_CONTAINER_EXIT inst$i=FAIL"; fi
  grep -E 'QA_FAIL|CATEGORY_|ERROR|STOP|TOR_SOCKS' "$QA/capture_inst$i.log" | tail -15 || true
done
set +e
python3 - "$QA" <<'PY'
import pathlib,csv,json,sys
root=pathlib.Path(sys.argv[1]); rows=[]
for p in root.glob('inst*/**/attempts_*.tsv'):
 with p.open(newline='',encoding='utf-8-sig') as f:
  for r in csv.DictReader(f,delimiter='\t'):
   rows.append(r)
   print('QA_ATTEMPT',p.relative_to(root),r['url'],'status='+r['status'],'node_exit='+r['node_exit'],'pcap_valid='+r['pcap_valid'],'packets='+r['packet_count'])
valid=sum(1 for r in rows if r.get('status')=='OK' and r.get('node_exit')=='0' and r.get('pcap_valid')=='true' and int(r.get('packet_count','0'))>=12)
print('RECOVERY_QA3_VALID',str(valid)+'/3','ATTEMPTS',len(rows))
(root/'recovery_qa_summary.json').write_text(json.dumps({'QA_ONLY':True,'attempts':len(rows),'valid':valid,'all_good':valid==3 and len(rows)==3},indent=2)+'\n')
sys.exit(0 if valid==3 and len(rows)==3 else 6)
PY
res=$?
set -e
echo "RECOVERY_QA3_OUTPUT=$QA"
echo "RECOVERY_QA3_EXIT=$res"
echo 'SCIENTIFIC_DATASET_C_CAPTURE=NOT_AUTHORIZED'
exit "$res"
