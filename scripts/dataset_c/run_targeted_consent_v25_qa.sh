#!/usr/bin/env bash
# Single-targeted QA V2.5 consent regression (3 sites), no sudo, no PCAP.
set -Eeuo pipefail
umask 077
APP="$HOME/ltd-ops/collector-qa-recovery-v25"
BASE="/data/ltd-c-qa-bsierra/C_DIRECT_V1"
IMAGE="ghcr.io/puppeteer/puppeteer@sha256:ffdb72533aa06e9adf99d1056fbe5b79c915869f6c2e7de8fa1f1ce8ffda0790"
CAPTURE="tor_1site_noise:final"
CHROME="/home/pptruser/.cache/puppeteer/chrome/linux-145.0.7632.76/chrome-linux64/chrome"
EXPECTED="1bc06ede565bc3dc16a79900424f32e12bca4da8f3fa7a73667616bdde786060"
[[ -d "$BASE/qa_only" && -f "$APP/capture_dom.js" && -f "$APP/quality.js" ]] ||
 { echo "BLOCKED: QA root or patch missing";exit 3; }
[[ "$(sha256sum "$APP/capture_dom.js" | awk '{print $1}')" == "$EXPECTED" ]] ||
 { echo "BLOCKED: QA V2.5 source SHA mismatch";exit 3; }
docker image inspect "$IMAGE" >/dev/null
docker image inspect "$CAPTURE" >/dev/null
for i in 1 2;do
  [[ "$(docker inspect -f '{{.State.Running}}' "ltd-c-tor-inst$i")" == true ]] ||
   { echo "BLOCKED: direct Tor inst$i is not running";exit 3; }
done
docker run --init --rm --network none --cap-drop ALL \
 --security-opt no-new-privileges \
 --mount "type=bind,src=$APP,dst=/home/pptruser/ltd-c-v2,readonly" \
 --entrypoint node "$IMAGE" --check /home/pptruser/ltd-c-v2/capture_dom.js
echo "QA25_PREFLIGHT=PASS"
RUN="$BASE/qa_only/recovery_consent_v25_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$RUN/logs" "$RUN/outputs"
restore(){
 docker run --rm --network none --user 0 \
  --mount "type=bind,src=$RUN/outputs,dst=/audit" \
  --entrypoint chown "$CAPTURE" -R "$(id -u):$(id -g)" /audit >/dev/null
}
trap restore EXIT
docker run --rm --network none --user 0 \
 --mount "type=bind,src=$RUN/outputs,dst=/audit" \
 --entrypoint chown "$CAPTURE" -R 10042:10042 /audit >/dev/null
echo "QA25_ROOT=$RUN"
echo "QA25_NETWORK=original_native_direct_tor_no_bridges"
while IFS='|' read -r INST SITE URL;do
  [[ -n "$INST" ]] || continue
  CAPTURE_UUID="$(cat /proc/sys/kernel/random/uuid)"
  echo "QA25_START=$SITE"
  if timeout -k 5s 150s docker run --init --rm \
      --network "container:ltd-c-tor-$INST" \
      --cap-drop ALL --security-opt no-new-privileges --shm-size 1g \
      --mount "type=bind,src=$APP,dst=/home/pptruser/ltd-c-v2,readonly" \
      --mount "type=bind,src=$RUN/outputs,dst=/audit" \
      -e NODE_PATH=/home/pptruser/node_modules \
      -e CHROME_NO_SANDBOX=1 -e SESSION_POLICY=fresh -e CONSENT_POLICY=accept_all \
      -e CAPTURE_UID="$CAPTURE_UUID" \
      --entrypoint node "$IMAGE" /home/pptruser/ltd-c-v2/capture_dom.js \
      "$URL" "/audit/$SITE"_"$CAPTURE_UUID.html" 9050 "$CHROME" 90 \
      > "$RUN/logs/$SITE.log" 2>&1;then
     echo "QA25_NODE_EXIT $SITE 0"
  else
     echo "QA25_NODE_EXIT $SITE $?"
  fi
done <<'SITES'
inst1|eurostars|https://www.eurostarshotels.com/
inst2|bolsamania|https://www.bolsamania.com/
inst2|saludcastillayleon|https://saludcastillayleon.es/es
SITES
restore
trap - EXIT
python3 - "$RUN" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); valid=0
for site in ('eurostars','bolsamania','saludcastillayleon'):
 files=list((root/'outputs').glob(site+'_*.json'))
 if len(files)!=1:
  print('QA25_SITE',site,'NO_UNIQUE_JSON')
  log=root/'logs'/(site+'.log')
  if log.exists():print('QA25_LOG_TAIL',log.read_text(errors='replace')[-550:])
  continue
 d=json.loads(files[0].read_text())
 consent=d.get('consent') or {}
 good=(d.get('status')=='OK' and consent.get('action')=='clicked' and
       d.get('http_status')==200 and d.get('visible_chars',0)>=100)
 valid+=bool(good)
 print('QA25_SITE',site,'STATUS',d.get('status'),'HTTP',d.get('http_status'),
  'CONSENT',consent.get('action'),'SELECTOR',consent.get('selector'),
  'LABEL',consent.get('label'),'VISIBLE_CHARS',d.get('visible_chars'),
  'CANDIDATE_VALID',good,'ERRORS',str(d.get('errors',[]))[:240],
  'DIAGNOSTIC_LABELS',consent.get('qa_control_labels'))
print('QA25_STRICT_CONSENT_CANDIDATES',str(valid)+'/3')
print('QA25_ROOT',root)
print('SCIENTIFIC_CAPTURE=NOT_AUTHORIZED')
PY
