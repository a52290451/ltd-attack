#!/usr/bin/env bash
# One-shot Chrome resource diagnostic; no PCAP, no A/B changes, no cron.
set -euo pipefail
umask 077
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/scripts/dataset_c/chrome_lifecycle_forensics_v1.js"
BASE=/data/ltd-c-qa-bsierra/C_DIRECT_V1
IMAGE=tor_1site_noise:final
[[ -f "$SRC" && -d "$BASE" && ! -L "$BASE" ]] || { echo "STOP: script or QA root missing"; exit 3; }
for i in 1 2 3; do
  n="ltd-c-tor-inst$i"
  [[ "$(docker inspect -f '{{.State.Running}}' "$n")" == true ]] || { echo "STOP: $n not running"; exit 3; }
  docker exec "$n" grep -qx "UseBridges 0" /tmp/direct-torrc || { echo "STOP: $n uses unexpected Tor config"; exit 3; }
done
docker image inspect "$IMAGE" >/dev/null
OUT="$BASE/diagnostics/chrome_resources_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$OUT"
exec > >(tee "$OUT/report.log") 2>&1
echo "CHROME_FORENSIC_OUTPUT=$OUT"
echo "HOST_DATE_UTC=$(date -u +%FT%TZ)"
echo "HOST_NTP_SYNC=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown)"
echo "GIT=$(git -C "$REPO" rev-parse --short HEAD)"
echo 'SCIENTIFIC_CAPTURE=NOT_AUTHORIZED'
echo '=== HOST RESOURCES BEFORE ==='
free -h || true
df -h /dev/shm "$BASE" || true
cat /proc/pressure/memory 2>/dev/null || true
grep -E 'MemAvailable|MemFree|SwapFree|SwapTotal' /proc/meminfo || true
echo '=== CONTAINER STATUS ==='
for i in 1 2 3; do
 docker inspect -f 'TOR={{.Name}} OOMKilled={{.State.OOMKilled}} MemoryLimit={{.HostConfig.Memory}} PidsLimit={{.HostConfig.PidsLimit}} Running={{.State.Running}}' "ltd-c-tor-inst$i" || true
done
docker stats --no-stream --format 'STATS {{.Name}} {{.MemUsage}} {{.PIDs}}' \
 ltd-c-tor-inst1 ltd-c-tor-inst2 ltd-c-tor-inst3 || true
echo '=== CONTROLLED SEQUENTIAL BROWSER PROBES ==='
ok=0
for i in 1 2 3; do
 echo "BEGIN_INSTANCE=$i"
 if timeout 210 docker run --rm \
   --name "ltd-c-chrome-forensic-inst$i-$$" \
   --network "container:ltd-c-tor-inst$i" \
   --security-opt no-new-privileges --shm-size 1g \
   -e NODE_PATH=/app/node_modules \
   --mount "type=bind,src=$SRC,dst=/qa/chrome_lifecycle_forensics_v1.js,readonly" \
   --entrypoint /usr/bin/node "$IMAGE" /qa/chrome_lifecycle_forensics_v1.js \
   > "$OUT/inst$i.log" 2>&1; then
   echo "FORENSIC_INSTANCE_PASS=$i"; ok=$((ok+1))
 else
   echo "FORENSIC_INSTANCE_FAIL=$i"
 fi
 echo "=== LIFECYCLE INSTANCE $i ==="
 grep -E '^FORENSIC_|^RUN_|^NAV_|^CHROME_(PROCESS|EXIT|DISCONNECTED)|^FRAME_DETACHED|^BROWSER_TOR_VERIFIED|^RESOURCE_FIELD.*(memory.events|memory.current|pids.current|pids.events)' "$OUT/inst$i.log" | tail -90 || true
 echo "=== RUNTIME ERRORS INSTANCE $i ==="
 grep -Ei 'out.of.memory|oom|killed|sig(segv|kill)|zygote|crash|error.*(launch|process|socket|system|dbus)|resource temporarily unavailable|too many open files|no space left on device|cannot allocate memory|fatal' "$OUT/inst$i.log" | tail -35 || true
 echo "END_INSTANCE=$i"
done
echo '=== HOST RESOURCES AFTER ==='
free -h || true
df -h /dev/shm "$BASE" || true
cat /proc/pressure/memory 2>/dev/null || true
echo "FORENSIC_INSTANCES_PASSED=$ok/3"
echo "REPORT=$OUT/report.log"
echo "DETAILED_LOGS=$OUT/inst1.log;$OUT/inst2.log;$OUT/inst3.log"
echo 'SCIENTIFIC_CAPTURE=NOT_AUTHORIZED'
if ((ok==3)); then
 echo 'RESOURCE_FORENSIC_CONTROL_PASS; NOT SCIENTIFIC AUTHORISATION'
 exit 0
fi
echo 'RESOURCE_FORENSIC_CONTROL_FAILED; REVIEW LOGS BEFORE FIX'
exit 5
