#!/usr/bin/env bash
# Dataset C 2026-11: test zombie reaping with Docker --init (tini) and, only if
# native Chrome->SOCKS5 Tor works repeatedly, run 12 QA attempts.
# Scientific capture remains blocked: no changes to A/B, Tor, baseline collector,
# Chrome/Puppeteer flags, native SOCKS proxy, host clock or cron.
set -euo pipefail
umask 077
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/scripts/dataset_c/chrome_lifecycle_forensics_v1.js"
APP="$HOME/ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2"
BASE="/data/ltd-c-qa-bsierra/C_DIRECT_V1"
IMAGE="tor_1site_noise:final"
MODE="${1:---probe-and-qa12}"
[[ "$MODE" == "--probe-only" || "$MODE" == "--probe-and-qa12" ]] || { echo "Usage: $0 [--probe-only|--probe-and-qa12]" >&2; exit 2; }
[[ -f "$SRC" && -f "$APP/capture_dom.js" && -f "$APP/run_category.sh" && -d "$BASE" && ! -L "$BASE" ]] || {
 echo 'STOP: expected pristine sources or QA root not available'; exit 3;
}
[[ "$(stat -c %u "$BASE")" == "$(id -u)" ]] || { echo 'STOP: QA root not owned by current user'; exit 3; }
docker image inspect "$IMAGE" >/dev/null
docker run --init --rm --network none --entrypoint /usr/bin/node "$IMAGE" -e 'console.log("DOCKER_INIT_TEST_OK",require("fs").readFileSync("/proc/1/comm","utf8").trim())' |
 grep -q 'DOCKER_INIT_TEST_OK'
for i in 1 2 3; do
 n="ltd-c-tor-inst$i"
 [[ "$(docker inspect -f '{{.State.Running}}' "$n")" == true ]] || { echo "STOP: $n is not running"; exit 3; }
 docker exec "$n" grep -qx 'UseBridges 0' /tmp/direct-torrc || { echo "STOP: $n unexpectedly uses bridges"; exit 3; }
done
if crontab -l 2>/dev/null | grep -Ev '^[[:space:]]*(#|$)' | grep -Eq 'LTD_C_QA_HOURLY|LTD_C_THREE_QA'; then
 echo 'STOP: unexpected active Dataset C cron'; exit 3
fi
OUT="$BASE/diagnostics/direct_tor_init_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$OUT"
exec > >(tee "$OUT/report.log") 2>&1
echo "INIT_RECOVERY_REPORT=$OUT/report.log"
echo 'SCIENTIFIC_CAPTURE=NOT_AUTHORIZED'
echo "HOST_NTP_SYNC=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown)"
echo "BASELINE_SHA256:"
sha256sum "$APP/capture_dom.js" "$APP/run_category.sh" "$APP/quality.js"
echo '=== NATIVE SOCKS / DOCKER INIT PROBE: 1 DIRECT + 3 TOR PER INSTANCE ==='
probe(){
 local i="$1"
 timeout 210 docker run --init --rm \
  --name "ltd-c-init-probe-inst$i-$$" \
  --network "container:ltd-c-tor-inst$i" \
  --security-opt no-new-privileges --shm-size 1g \
  -e NODE_PATH=/app/node_modules \
  --mount "type=bind,src=$SRC,dst=/qa/chrome_lifecycle_forensics_v1.js,readonly" \
  --entrypoint /usr/bin/node "$IMAGE" /qa/chrome_lifecycle_forensics_v1.js
}
good=0
for i in 1 2 3; do
 echo "=== INIT_PROBE_BEGIN inst$i ==="
 if probe "$i" > "$OUT/probe_inst$i.log" 2>&1 &&
    grep -q '^FORENSIC_RESULT 4/4$' "$OUT/probe_inst$i.log"; then
   echo "INIT_PROBE_PASS inst$i"
   good=$((good+1))
 else
   echo "INIT_PROBE_FAIL inst$i"
 fi
 grep -E '^RUN_RESULT|^FORENSIC_RESULT|^NAV_FAILURE|^BROWSER_TOR_VERIFIED|^RESOURCE_FIELD.*(memory.events|pids.current|pids.events)|^RESOURCE_PROCESSES' \
    "$OUT/probe_inst$i.log" | tail -42 || true
done
if ((good!=3)); then
 echo "INIT_RECOVERY_BLOCKED: successful instances $good/3; no PCAP attempted"
 echo "DETAILS=$OUT/probe_inst1.log,$OUT/probe_inst2.log,$OUT/probe_inst3.log"
 exit 5
fi
echo 'INIT_PROBES_PASS_3_OF_3'
if [[ "$MODE" == "--probe-only" ]]; then
 echo 'PROBE_ONLY_PASS; NO_QA_PCAP'
 exit 0
fi
QA="$BASE/qa_only/direct_init_qa12_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$QA"
cat > "$QA/manifest.txt" <<EOF
QA_ONLY_NOT_SCIENTIFIC
approach=docker_init_tini
chromium_flags=BASELINE_UNCHANGED
puppeteer_protocol=BASELINE_CDP
proxy=SOCKS5_DIRECT_TOR_127.0.0.1_9050
tor=NO_BRIDGES
clock_ntp_sync=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown)
expected_attempts=12
EOF
collect(){
 local i="$1" categories="$2"
 timeout 520 docker run --init --rm \
   --name "ltd-c-init-qa-inst$i-$$" \
   --network "container:ltd-c-tor-inst$i" \
   --cap-add NET_RAW --cap-add NET_ADMIN \
   --security-opt no-new-privileges --shm-size 1g \
   --mount "type=bind,src=$APP,dst=/opt/ltd-c-v2,readonly" \
   --mount "type=bind,src=$QA,dst=/audit" \
   -e APP_HOME=/opt/ltd-c-v2 -e NODE_PATH=/app/node_modules \
   -e CAPTURE_ROOT=/audit -e MAX_URLS=2 \
   -e CONSENT_POLICY=accept_all -e SESSION_POLICY=fresh \
   -e CHROME_NO_SANDBOX=1 -e MIN_DISK_FREE_GB=120 \
   -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
   --entrypoint /bin/bash "$IMAGE" -lc "
      trap 'chown -R \"\$HOST_UID:\$HOST_GID\" /audit 2>/dev/null || true' EXIT
      rc=0
      for category in $categories; do
        bash /opt/ltd-c-v2/run_category.sh inst$i \"\$category\" || rc=1
      done
      exit \"\$rc\"
   "
}
echo '=== QA12 BEGIN: unmodified baseline collector, Docker init, native SOCKS5 ==='
collect 1 'belleza turismo' > "$QA/inst1.log" 2>&1 & p1=$!
collect 2 'bancos salud' > "$QA/inst2.log" 2>&1 & p2=$!
collect 3 'deportes politica' > "$QA/inst3.log" 2>&1 & p3=$!
for i in 1 2 3; do
 eval "p=\$p$i"
 if wait "$p"; then echo "QA_INSTANCE_EXIT inst$i=0"; else echo "QA_INSTANCE_EXIT inst$i=NONZERO"; fi
 grep -E 'QA_FAIL|ERROR|STOP|CATEGORY_' "$QA/inst$i.log" | tail -15 || true
done
set +e
python3 - "$QA" <<'PY'
import pathlib,csv,json,collections,sys
root=pathlib.Path(sys.argv[1]).resolve()
expected={('inst1','belleza'),('inst1','turismo'),('inst2','bancos'),('inst2','salud'),('inst3','deportes'),('inst3','politica')}
rows=[];groups=collections.Counter()
for f in sorted(root.glob('inst*/**/attempts_*.tsv')):
 if len(f.relative_to(root).parts)!=3:continue
 inst,category=f.relative_to(root).parts[:2]
 if (inst,category) not in expected:continue
 with f.open(newline='',encoding='utf8') as inp:
  for r in csv.DictReader(inp,delimiter='\t'):
   groups[(inst,category)]+=1
   pc=r.get('pcap','')
   p=(root/pc[len('/audit/'):]) if pc.startswith('/audit/') else None
   if p is not None and not p.resolve().is_relative_to(root):p=None
   size=p.stat().st_size if p and p.is_file() else 0
   try:count=int(r.get('packet_count','0'))
   except ValueError:count=0
   valid=(r.get('status')=='OK' and r.get('node_exit')=='0' and r.get('pcap_valid')=='true' and size>=8192 and count>=12)
   rows.append(dict(instance=inst,category=category,status=r.get('status'),node_exit=r.get('node_exit'),pcap_bytes=size,packet_count=count,qa_valid=valid,url=r.get('url')))
   print('QA_VISIT',inst,category,'status',r.get('status'),'pcap_bytes',size,'packets',count,'valid',valid)
count_ok=sum(r['qa_valid'] for r in rows)
success=len(rows)==12 and count_ok==12 and all(groups[x]==2 for x in expected)
summary=dict(classification='QA_ONLY_NOT_SCIENTIFIC',count=len(rows),qa_valid=count_ok,groups={':'.join(k):v for k,v in groups.items()},success=success,visits=rows,science_authorized=False)
(root/'strict_qa_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('DIRECT_INIT_QA_VALID',str(count_ok)+'/12','ATTEMPTS',len(rows))
print('DIRECT_INIT_QA_STATUS','QA_PASS' if success else 'QA_FAILED')
sys.exit(0 if success else 6)
PY
status=$?
set -e
echo "QA_ROOT=$QA"
echo "DIAG_ROOT=$OUT"
echo "SCIENTIFIC_CAPTURE=NOT_AUTHORIZED"
exit "$status"
