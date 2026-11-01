#!/usr/bin/env bash
# Dataset C: isolate browser image/dependency vs network/Tor using official pinned
# Puppeteer 24.37.4 image. QA probes ONLY; no PCAP or changes to Tor/collector/A/B.
set -euo pipefail
umask 077
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/scripts/dataset_c/pinned_chrome_direct_tor_probe_v1.js"
BASE=/data/ltd-c-qa-bsierra/C_DIRECT_V1
IMAGE=ghcr.io/puppeteer/puppeteer:24.37.4
[[ -f "$SRC" && -d "$BASE" && ! -L "$BASE" ]] || { echo 'STOP: missing pinned test or QA root'; exit 3; }
for i in 1 2 3; do
 n="ltd-c-tor-inst$i"
 [[ "$(docker inspect -f '{{.State.Running}}' "$n")" == true ]] || { echo "STOP: Tor instance $n is down"; exit 3; }
 docker exec "$n" grep -qx 'UseBridges 0' /tmp/direct-torrc || { echo "STOP: Tor $n unexpectedly uses bridges"; exit 3; }
done
OUT="$BASE/diagnostics/pinned_browser_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$OUT"
exec > >(tee "$OUT/report.log") 2>&1
echo "PINNED_BROWSER_DIAG=$OUT"
echo "HOST_NTP_SYNC=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown)"
echo "SCIENTIFIC_CAPTURE=NOT_AUTHORIZED"
echo "GIT=$(git -C "$REPO" rev-parse --short HEAD)"
echo "OFFICIAL_BROWSER_IMAGE=$IMAGE"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
 echo '=== FETCHING OFFICIAL VERSIONED BROWSER IMAGE INTO DOCKER CACHE ==='
 docker pull "$IMAGE" || { echo 'STOP: unable to fetch official image'; exit 4; }
fi
docker image inspect -f 'OFFICIAL_IMAGE_ID={{.Id}} IMAGE_DIGESTS={{json .RepoDigests}}' "$IMAGE"
echo '=== PROBE: 1 CLEAR-NET CONTROL + 4 NATIVE SOCKS TOR NAVIGATIONS / INSTANCE ==='
good=0
for i in 1 2 3; do
 echo "=== PINNED_INSTANCE=$i ==="
 if timeout 300 docker run --init --rm \
    --name "ltd-c-pinned-browser-inst$i-$$" \
    --network "container:ltd-c-tor-inst$i" \
    --cap-drop ALL --security-opt no-new-privileges --shm-size 1g \
    --mount "type=bind,src=$SRC,dst=/home/pptruser/qa_browser_probe.js,readonly" \
    --entrypoint node "$IMAGE" /home/pptruser/qa_browser_probe.js \
    > "$OUT/inst$i.log" 2>&1 \
    && grep -q '^PINNED_PROBE_SUMMARY 5/5$' "$OUT/inst$i.log"; then
    echo "PINNED_INSTANCE_PASS=$i"; good=$((good+1))
 else
    echo "PINNED_INSTANCE_FAIL=$i"
 fi
 grep -E '^(PINNED_(PROBE_VERSION|CHROME_VERSION|PROBE_RESULT|PROBE_SUMMARY|TOR_VERIFIED|CONTROL_VERIFIED|PROBE_ERROR|PROBE_FATAL|CHROME_PREMATURE_EXIT|CHROME_DISCONNECTED|FRAME_DETACHED)|Error)' \
    "$OUT/inst$i.log" | tail -40 || true
done
echo "PINNED_INSTANCES_PASSED=$good/3"
echo "REPORT=$OUT/report.log"
echo "INSTANCE_LOGS=$OUT/inst1.log;$OUT/inst2.log;$OUT/inst3.log"
echo 'SCIENTIFIC_CAPTURE=NOT_AUTHORIZED'
if ((good==3));then
 echo 'NEXT_PATH=PINNED_IMAGE_SUCCESS_CHECK_COVERAGE_THEN_BUILD_VERSIONED_COLLECTOR_QA'
 exit 0
fi
echo 'NEXT_PATH=PINNED_IMAGE_FAILED_INVESTIGATE_CHROME_TOR_NETWORK_AND_BROWSER_EXIT'
exit 5
