#!/usr/bin/env bash
# Dataset C fail-closed direct-Tor browser repair and QA (NOT scientific release).
# Retains existing Chrome launch flags, native SOCKS5 and tcpdump; changes only
# Puppeteer's Chrome automation transport from CDP WebSocket to CDP pipe in QA overlay.
# No bridge, no HTTP proxy, no cron, no sudo, no host clock changes, no A/B writes.
set -euo pipefail
umask 077
MODE="${1:---probe-and-qa}"
[[ "$MODE" == --probe-only || "$MODE" == --probe-and-qa ]] || { echo "Usage: $0 --probe-only|--probe-and-qa" >&2; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/scripts/dataset_c"
APP="$HOME/ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2"
BASE=/data/ltd-c-qa-bsierra/C_DIRECT_V1
IMAGE=tor_1site_noise:final
[[ -d "$BASE" && ! -L "$BASE" && -f "$APP/capture_dom.js" && -f "$APP/run_category.sh" ]] || {
  echo 'STOP: original collector or unique QA root is absent'; exit 3;
}
[[ "$(id -u)" == "$(stat -c '%u' "$BASE")" ]] || { echo 'STOP: QA root is not owned by operator'; exit 3; }
docker image inspect "$IMAGE" >/dev/null
for i in 1 2 3; do
  n="ltd-c-tor-inst$i"
  [[ "$(docker inspect -f '{{.State.Running}}' "$n")" == true ]] || { echo "STOP: $n not running"; exit 3; }
  docker exec "$n" grep -qx 'UseBridges 0' /tmp/direct-torrc || { echo "STOP: bridge policy unexpected on $n"; exit 3; }
  docker exec "$n" sh -ec 'tr "\000" " " </proc/1/cmdline | grep -Fq "tor -f /tmp/direct-torrc"' || { echo "STOP: $n unexpected Tor process"; exit 3; }
done
if crontab -l 2>/dev/null | grep -Ev '^[[:space:]]*(#|$)' | grep -Eq 'LTD_C_QA_HOURLY|LTD_C_THREE_QA'; then
  echo 'STOP: obsolete Dataset C cron active'; exit 3
fi
OUT="$BASE/diagnostics/direct_socks_feature_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$OUT"
exec > >(tee "$OUT/report.log") 2>&1
echo "DIRECT_SOCKS_FEATURE_DIAG=$OUT"
echo "QA_ONLY=YES SCIENTIFIC_CAPTURE=NOT_AUTHORIZED"
echo "HOST_NTP_SYNC=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown)"
echo '=== BASELINE HASHES (unchanged) ==='
sha256sum "$APP/capture_dom.js" "$APP/run_category.sh" "$APP/quality.js"
echo "GIT=$(git -C "$REPO" rev-parse --short HEAD)"
echo '=== DIRECT SOCKS FEATURE CONTROL PROBES ==='
docker run --rm --network none --mount "type=bind,src=$SRC,dst=/qa_scripts,readonly" --entrypoint /usr/bin/node "$IMAGE" --check /qa_scripts/optimization_hints_probe_v1.js
probe() {
  local i="$1" candidate="$2"
  timeout 350 docker run --rm --name "ltd-c-feature-probe-$i-$candidate-$$" \
    --network "container:ltd-c-tor-inst$i" --security-opt no-new-privileges \
    --shm-size 1g --mount "type=bind,src=$SRC,dst=/qa_scripts,readonly" \
    -e NODE_PATH=/app/node_modules -e QA_CHROME_FEATURE_MODE="$candidate" \
    --entrypoint /usr/bin/node "$IMAGE" /qa_scripts/optimization_hints_probe_v1.js
}
SELECTED=''
for candidate in enable strip; do
  echo "=== FEATURE_CANDIDATE=$candidate ==="
  for i in 1 2 3; do
    probe "$i" "$candidate" > "$OUT/probe_inst${i}_${candidate}.log" 2>&1 &
    eval "pid_$i=\$!"
  done
  good=0
  for i in 1 2 3; do
    eval "pid=\$pid_$i"
    if wait "$pid" && grep -q "FEATURE_PROBE_RESULT $candidate 4/4" "$OUT/probe_inst${i}_${candidate}.log"; then
      echo "FEATURE_PROBE_PASS inst$i mode=$candidate"
      good=$((good+1))
    else
      echo "FEATURE_PROBE_FAIL inst$i mode=$candidate"
    fi
    grep -E 'FEATURE_(PROBE|NAV|TOR_VERIFIED|EXAMPLE_VERIFIED|PAGE_ERROR|CHROME_PREMATURE_EXIT|CHROME_DISCONNECTED|FRAME_DETACHED)|Error:' "$OUT/probe_inst${i}_${candidate}.log" | tail -32 || true
  done
  if ((good==3)); then SELECTED="$candidate"; break; fi
done
if [[ -z "$SELECTED" ]]; then
  echo 'FEATURE_RECOVERY_NOT_PROVEN: no mode passed 4/4 Chrome navigations in all 3 instances'
  echo 'CONTAINERS=UNCHANGED; ORIGINAL_COLLECTOR=UNCHANGED; NO_QA_PCAP_CREATED'
  echo "REPORT=$OUT/report.log"
  exit 5
fi
echo "FEATURE_RECOVERY_PROBES_ALL_PASS MODE=$SELECTED"
if [[ "$MODE" == --probe-only ]]; then echo 'PROBE_ONLY_DONE'; exit 0; fi
echo '=== PREPARE ISOLATED FEATURE QA OVERLAY ==='
OVERLAY="$OUT/capture_dom.qa.js"
python3 - "$APP/capture_dom.js" "$OVERLAY" "$SELECTED" <<'PY'
import pathlib,sys
original=pathlib.Path(sys.argv[1]).read_text()
mode=sys.argv[3]
needle='const launch = {executablePath: chromePath, headless: true, args, timeout: 20000};'
if original.count(needle)!=1: raise SystemExit('STOP: unexpected Chrome launcher')
if mode=='enable':
  insert='args.push("--enable-features=OptimizationHints");\n    '
  patched=original.replace(needle,insert+needle)
elif mode=='strip':
  patch="""const defaults = puppeteer.defaultArgs({headless: true, args: [...args]});
    const flags = defaults.filter(x => x.startsWith('--disable-features='));
    if (flags.length !== 1 || !flags[0].split('=')[1].split(',').includes('OptimizationHints')) {
      throw new Error('Unexpected Puppeteer defaults; abort safely');
    }
    const adjusted = defaults.map(x => x.startsWith('--disable-features=')
      ? '--disable-features=' + x.slice('--disable-features='.length).split(',').filter(f => f !== 'OptimizationHints').join(',')
      : x);
    const launch = {executablePath: chromePath, headless: true, ignoreDefaultArgs: true, args: adjusted, timeout: 20000};"""
  patched=original.replace(needle,patch)
else:raise SystemExit('STOP: unsupported mode')
if original.count('socks5://127.0.0.1:')!=1 or patched.count('socks5://127.0.0.1:')!=1:
 raise SystemExit('STOP: native SOCKS5 proxy must remain identical')
if 'http://127.0.0.1:18888' in patched:raise SystemExit('STOP: prohibited intermediary')
pathlib.Path(sys.argv[2]).write_text(patched)
print('FEATURE_QA_OVERLAY_MODE',mode)
PY
echo "ONLY QA CHANGE=OptimizationHints feature toggle MODE=$SELECTED"
sha256sum "$APP/capture_dom.js" "$OVERLAY"
docker run --rm --network none \
  --mount "type=bind,src=$OVERLAY,dst=/tmp/capture_dom.qa.js,readonly" \
  --entrypoint /usr/bin/node "$IMAGE" --check /tmp/capture_dom.qa.js
QA="$BASE/qa_only/direct_feature_qa12_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$QA"
cat > "$QA/qa_manifest.txt" <<EOF
QA_ONLY_NOT_FOR_SCIENTIFIC_DATASET
collector_original=$APP
browser_protocol=CDP_WEBSOCKET_UNCHANGED
feature_experiment=$SELECTED
browser_proxy=socks5://127.0.0.1:9050
transport=DIRECT_TOR_NO_BRIDGES
A_B_INPUTS=UNTOUCHED
host_ntp_sync=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown)
expected_visits=12
EOF
collect() {
  local i="$1" cats="$2"
  timeout 480 docker run --rm --name "ltd-c-pipe-qa-inst$i-$$" \
    --network "container:ltd-c-tor-inst$i" \
    --cap-add NET_RAW --cap-add NET_ADMIN \
    --security-opt no-new-privileges --shm-size 1g \
    --mount "type=bind,src=$APP,dst=/opt/ltd-c-v2,readonly" \
    --mount "type=bind,src=$OVERLAY,dst=/opt/ltd-c-v2/capture_dom.js,readonly" \
    --mount "type=bind,src=$QA,dst=/audit" \
    -e APP_HOME=/opt/ltd-c-v2 -e NODE_PATH=/app/node_modules \
    -e CAPTURE_ROOT=/audit -e MAX_URLS=2 \
    -e CONSENT_POLICY=accept_all -e SESSION_POLICY=fresh \
    -e CHROME_NO_SANDBOX=1 -e MIN_DISK_FREE_GB=120 \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    --entrypoint /bin/bash "$IMAGE" -lc "
       trap 'chown -R \"\$HOST_UID:\$HOST_GID\" /audit 2>/dev/null || true' EXIT
       rc=0
       for cat in $cats; do
         bash /opt/ltd-c-v2/run_category.sh inst$i \"\$cat\" || rc=1
       done
       exit \"\$rc\"
    "
}
echo '=== QA12: native direct SOCKS5, original Chrome, OptimizationHints-only overlay ==='
collect 1 'belleza turismo' > "$QA/inst1.log" 2>&1 & p1=$!
collect 2 'bancos salud' > "$QA/inst2.log" 2>&1 & p2=$!
collect 3 'deportes politica' > "$QA/inst3.log" 2>&1 & p3=$!
for i in 1 2 3; do
  eval "p=\$p$i"
  if wait "$p"; then echo "QA_PROCESS inst$i exit=0"; else echo "QA_PROCESS inst$i exit=NONZERO"; fi
  grep -E 'QA_FAIL|CATEGORY_|ERROR|STOP|Terminated' "$QA/inst$i.log" | tail -20 || true
done
echo '=== STRICT PER-ATTEMPT AUDIT ==='
set +e
python3 - "$QA" <<'PY'
import pathlib,csv,json,sys,collections
root=pathlib.Path(sys.argv[1])
records=[]; statuses=collections.Counter()
for f in sorted(root.glob('inst*/**/attempts_*.tsv')):
 with f.open(newline='',encoding='utf-8-sig') as fh:
  for row in csv.DictReader(fh,delimiter='\t'):
   s=row.get('status',''); statuses[s]+=1
   # Resolve only paths produced inside the read/write /audit mount.
   raw=row.get('pcap','')
   p=(root/raw[len('/audit/'):]) if raw.startswith('/audit/') else None
   if p is not None and not p.resolve().is_relative_to(root.resolve()): p=None
   size=p.stat().st_size if p and p.is_file() else 0
   try: packets=int(row.get('packet_count','0'))
   except ValueError: packets=0
   valid=(s=='OK' and row.get('node_exit')=='0' and row.get('pcap_valid')=='true'
          and size>=8192 and packets>=12)
   records.append({'inst':f.parts[-3],'category':f.parent.name,'url':row.get('url'),
     'status':s,'node_exit':row.get('node_exit'),'pcap_valid':row.get('pcap_valid'),
     'packets':packets,'bytes':size,'qa_valid':valid})
   print('VISIT',f.parts[-3],f.parent.name,'status',s,'node_exit',row.get('node_exit'),
         'pcap_bytes',size,'packets',packets,'valid',valid)
nvalid=sum(x['qa_valid'] for x in records)
result={'classification':'QA_ONLY_NOT_SCIENTIFIC','n_attempts':len(records),
  'n_valid':nvalid,'status_counts':dict(statuses),'visits':records,
  'capture_usable_for_science':False,'host_clock_unsynchronized':True}
(root/'strict_qa_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('DIRECT_FEATURE_QA_VALID',str(nvalid)+'/12','ATTEMPTS',len(records))
print('DIRECT_FEATURE_QA_STATUS', 'QA_PASS' if nvalid==12 and len(records)==12 else 'QA_FAILED')
sys.exit(0 if nvalid==12 and len(records)==12 else 6)
PY
rc=$?
set -e
echo "QA_OUTPUT=$QA"
echo "DIAG_REPORT=$OUT/report.log"
echo "QA_EXIT=$rc"
echo 'SCIENTIFIC_DATASET_C_CAPTURE=NOT_AUTHORIZED'
exit "$rc"
