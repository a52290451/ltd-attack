#!/usr/bin/env bash
# LTD Dataset C: end-to-end QA12 with the already verified official Puppeteer image.
# Browser: pinned Chrome for Testing + exact ORIGINAL capture_dom.js and quality.js.
# Traffic: native SOCKS5 -> existing direct Tor, tcpdump sidecar on SAME eth0 namespace.
# No mutation to A/B, historical frozen datasets, Tor containers, original collector or cron.
# Host clock is unsynchronized; results are QA ONLY, NEVER scientific input.
set -euo pipefail
umask 077
APP="$HOME/ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2"
BASE="/data/ltd-c-qa-bsierra/C_DIRECT_V1"
BROWSER="ghcr.io/puppeteer/puppeteer@sha256:ffdb72533aa06e9adf99d1056fbe5b79c915869f6c2e7de8fa1f1ce8ffda0790"
CAPTURE="tor_1site_noise:final"
RUN_ID="pinned_qa12_$(date -u +%Y%m%dT%H%M%SZ)_$$"
OUT="$BASE/diagnostics/$RUN_ID"
QA="$BASE/qa_only/$RUN_ID"
DAY="$(date -u +%F)"
UID_HOST="$(id -u)"
GID_HOST="$(id -g)"
[[ -d "$BASE" && ! -L "$BASE" && -f "$APP/capture_dom.js" && -f "$APP/quality.js" &&
   -f "$APP/check_sites.py" && -d "$APP/site_lists" ]] || { echo 'BLOCKED: pristine collector missing'; exit 3; }
[[ "$(stat -c %u "$BASE")" == "$UID_HOST" ]] || { echo 'BLOCKED: QA root ownership mismatch'; exit 3; }
docker image inspect "$BROWSER" >/dev/null
docker image inspect "$CAPTURE" >/dev/null
for i in 1 2 3; do
 n="ltd-c-tor-inst$i"
 [[ "$(docker inspect -f '{{.State.Running}}' "$n")" == true ]] || { echo "BLOCKED: $n stopped"; exit 3; }
 docker exec "$n" grep -qx 'UseBridges 0' /tmp/direct-torrc || { echo "BLOCKED: $n bridges enabled"; exit 3; }
done
# Retain exact baseline content and frozen category inventories.
printf '%s  %s\n' \
 '137c163bfed44d17d702dffb8e4c15541469ea1d5c41f3c1e96c80cc8c9c6d4c' "$APP/capture_dom.js" \
 '2e95a90ea77115320d37aaff0ae14157727f7b47deb28a9b18de7237e84f59e1' "$APP/quality.js" \
 | sha256sum -c --status || { echo 'BLOCKED: original JS collector hash changed'; exit 3; }
python3 "$APP/check_sites.py" || { echo 'BLOCKED: canonical site lists mismatch historical reference'; exit 3; }
mkdir -p "$OUT" "$QA"
exec > >(tee "$OUT/report.log") 2>&1
echo "QA12_REPORT=$OUT/report.log"
echo "QA12_ROOT=$QA"
echo 'SCIENTIFIC_CAPTURE=NOT_AUTHORIZED'
echo "HOST_NTP_SYNC=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown)"
echo "BROWSER_IMAGE=$BROWSER"
echo 'CAPTURE_PROTOCOL=ORIGINAL_JS_NATIVE_SOCKS5_TOR_TCPDUMP_ETH0'
# Ensure the exact pinned image has the Chrome build that already passed 15/15.
CHROME="$(docker run --init --rm --network none --entrypoint node "$BROWSER" -p 'require("puppeteer").executablePath()')"
docker run --init --rm --network none --entrypoint node "$BROWSER" -e '
 const p=require("puppeteer"); const v=require("puppeteer/package.json").version;
 if(v!=="24.37.4"||!p.executablePath().includes("145.0.7632.76"))process.exit(9);
 console.log("PINNED_IMAGE_CONFIRMED",v,p.executablePath());
'
echo "PINNED_CHROME_PATH=$CHROME"
echo '=== PRECHECK ORIGINAL COLLECTOR WITH PINNED NODE/PUPPETEER ==='
docker run --init --rm --network none --cap-drop ALL --security-opt no-new-privileges \
 --mount "type=bind,src=$APP,dst=/home/pptruser/ltd-c-v2,readonly" \
 -e NODE_PATH=/home/pptruser/node_modules \
 --entrypoint node "$BROWSER" --check /home/pptruser/ltd-c-v2/capture_dom.js
docker run --init --rm --network none --cap-drop ALL --security-opt no-new-privileges \
 --mount "type=bind,src=$APP,dst=/home/pptruser/ltd-c-v2,readonly" \
 -e NODE_PATH=/home/pptruser/node_modules \
 --entrypoint node "$BROWSER" -e 'require("/home/pptruser/ltd-c-v2/quality.js");require("puppeteer");console.log("COLLECTOR_DEPENDENCIES_OK")'
# Validate six canonical category lists (20 positions each, 118 unique URLs).
python3 - "$APP/site_lists" "$OUT/visits.tsv" "$DAY" <<'PY'
import pathlib,sys,uuid,urllib.parse
base=pathlib.Path(sys.argv[1])
pairs=[('inst1','belleza'),('inst1','turismo'),('inst2','bancos'),('inst2','salud'),('inst3','deportes'),('inst3','politica')]
all_urls=[]; rows=[]
for inst,cat in pairs:
 f=base/inst/(cat+'.txt')
 lines=[x.strip() for x in f.read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')]
 if len(lines)!=20:raise SystemExit(f'BLOCKED: {f} expected 20 canonical slots, found {len(lines)}')
 for idx,url in enumerate(lines,1):
  parsed=urllib.parse.urlsplit(url)
  if parsed.scheme not in ('https','http') or not parsed.hostname:raise SystemExit(f'BLOCKED: invalid URL: {url}')
  all_urls.append(url)
  if idx<=2:
   rows.append((inst,cat,str(idx),url,parsed.hostname,str(uuid.uuid4()),sys.argv[3]))
if len(all_urls)!=120 or len(set(all_urls))!=118:raise SystemExit('BLOCKED: canonical 120 slots/118 unique URLs invariant failed')
with pathlib.Path(sys.argv[2]).open('w') as g:
 g.write('instance\tcategory\tslot\turl\thost\tuid\tday\n')
 for row in rows:g.write('\t'.join(row)+'\n')
print('CANONICAL_INVENTORY_OK','120_slots','118_unique_urls','12_QA_visits')
PY
# Precreate exactly the planned QA directories, then grant the official image's
# non-root pptruser (UID 10042) write access. Restore host ownership at end.
while IFS=$'\t' read -r inst cat slot url host uid day; do
 [[ "$inst" == instance ]] && continue
 mkdir -p "$QA/$inst/$cat/$host/$day"
done < "$OUT/visits.tsv"
owner_restored=0
restore_owner(){
 if ((owner_restored==0));then
  docker run --rm --network none --user 0 \
   --mount "type=bind,src=$QA,dst=/audit" \
   --entrypoint chown "$CAPTURE" -R "$UID_HOST:$GID_HOST" /audit >/dev/null 2>&1 || true
  owner_restored=1
 fi
}
cleanup(){
 for i in 1 2 3; do
  docker rm -f "ltd-c-pcap-qa-inst$i-$$" >/dev/null 2>&1 || true
 done
 restore_owner
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
docker run --rm --network none --user 0 \
 --mount "type=bind,src=$QA,dst=/audit" \
 --entrypoint chown "$CAPTURE" -R 10042:10042 /audit
echo 'QA_OUTPUT_OWNER=10042 (official Puppeteer non-root runtime)'
echo '=== ACTUAL QA12 CAPTURES: 3 workers, 4 sites per worker ==='
visit(){
 local inst="$1" cat="$2" slot="$3" url="$4" host="$5" uuid="$6" day="$7"
 local i="${inst#inst}" stem="$inst/$cat/$host/$day/$uuid"
 local cap_name="ltd-c-pcap-qa-inst$i-$$" rc=0 started="$(date -u +%FT%TZ)"
 echo "VISIT_BEGIN $inst/$cat/$slot $url"
 docker run -d --name "$cap_name" --user 0 \
  --network "container:ltd-c-tor-inst$i" \
  --cap-add NET_RAW --cap-add NET_ADMIN --security-opt no-new-privileges \
  --mount "type=bind,src=$QA,dst=/audit" \
  --entrypoint tcpdump "$CAPTURE" -i eth0 -n -s 0 -U -w "/audit/$stem.pcap" tcp \
  > "$OUT/capture_id_$i.txt" || rc=98
 if ((rc==0));then
  sleep 0.65
  if [[ "$(docker inspect -f '{{.State.Running}}' "$cap_name" 2>/dev/null || true)" != true ]];then
    echo "CAPTURE_FAILED_START $inst/$cat/$slot";rc=99
  fi
 fi
 if ((rc==0));then
  # Only browser runs in the exact official image that passed 15/15 control probes.
  timeout -k 5s 95s docker run --init --rm \
   --name "ltd-c-pinned-nav-inst$i-$$" --network "container:ltd-c-tor-inst$i" \
   --cap-drop ALL --security-opt no-new-privileges --shm-size 1g \
   --mount "type=bind,src=$APP,dst=/home/pptruser/ltd-c-v2,readonly" \
   --mount "type=bind,src=$QA,dst=/audit" \
   -e NODE_PATH=/home/pptruser/node_modules \
   -e CHROME_NO_SANDBOX=1 -e SESSION_POLICY=fresh -e CONSENT_POLICY=accept_all \
   -e CAPTURE_UID="$uuid" -e SITE_SLOT_INDEX="$slot" -e SITE_INSTANCE="$inst" -e SITE_CATEGORY="$cat" \
   --entrypoint node "$BROWSER" /home/pptruser/ltd-c-v2/capture_dom.js \
   "$url" "/audit/$stem.html" 9050 "$CHROME" 60 \
   > "$OUT/browser_$uuid.log" 2>&1 || rc=$?
 fi
 docker kill --signal=SIGINT "$cap_name" >/dev/null 2>&1 || true
 docker wait "$cap_name" >/dev/null 2>&1 || true
 docker logs "$cap_name" > "$OUT/tcpdump_$uuid.log" 2>&1 || true
 docker rm -f "$cap_name" >/dev/null 2>&1 || true
 echo -e "$inst\t$cat\t$slot\t$url\t$uuid\t$rc\t$stem" >> "$OUT/exits_$inst.tsv"
 echo "VISIT_END $inst/$cat/$slot NODE_EXIT=$rc"
}
for i in 1 2 3; do
 (
   while IFS=$'\t' read -r inst cat slot url host uid day; do
    [[ "$inst" == "inst$i" ]] || continue
    visit "$inst" "$cat" "$slot" "$url" "$host" "$uid" "$day"
   done < "$OUT/visits.tsv"
 ) > "$OUT/worker_inst$i.log" 2>&1 &
 eval "worker_$i=\$!"
done
workers_failed=0
for i in 1 2 3; do
 eval "p=\$worker_$i"
 if wait "$p"; then echo "QA_WORKER_COMPLETE inst$i"; else echo "QA_WORKER_FAILED inst$i";workers_failed=1; fi
 grep -E 'VISIT_(BEGIN|END)|CAPTURE_FAILED_START|Error:|error' "$OUT/worker_inst$i.log" | tail -16 || true
done
restore_owner
echo '=== STRICT QA AUDIT ==='
set +e
python3 - "$OUT" "$QA" <<'PY'
import sys,csv,json,struct,pathlib,collections
out,root=map(lambda x:pathlib.Path(x).resolve(),sys.argv[1:3])
def count_pcap(p):
 try:
  with p.open('rb') as f:
   raw=f.read()
  if len(raw)<24:return 0,len(raw),False
  magic=raw[:4]
  if magic in (bytes.fromhex('d4c3b2a1'),bytes.fromhex('4d3cb2a1')): e='<'
  elif magic in (bytes.fromhex('a1b2c3d4'),bytes.fromhex('a1b23c4d')):e='>'
  else:return 0,len(raw),False
  _,_,_,_,snap,network=struct.unpack(e+'HHIIII',raw[4:24])
  if snap<64 or network!=1:return 0,len(raw),False
  pos=24;n=0
  while pos+16<=len(raw):
   sec,usec,incl,orig=struct.unpack(e+'IIII',raw[pos:pos+16])
   if incl>snap or incl>orig or pos+16+incl>len(raw):return n,len(raw),False
   pos+=16+incl;n+=1
  return n,len(raw),pos==len(raw)
 except OSError:return 0,0,False
plans=list(csv.DictReader((out/'visits.tsv').open(),delimiter='\t'))
exits={}
for p in out.glob('exits_inst*.tsv'):
 with p.open() as f:
  for line in f:
   cells=line.rstrip('\n').split('\t')
   if len(cells)==7:exits[cells[4]]=cells[5]
records=[];good=0
for r in plans:
 rel=pathlib.Path(r['instance'])/r['category']/r['host']/r['day']/r['uid']
 meta= root/(str(rel)+'.json')
 html= root/(str(rel)+'.html')
 pcap= root/(str(rel)+'.pcap')
 packets,size,pcap_parsed=count_pcap(pcap)
 try:data=json.loads(meta.read_text())
 except (OSError,ValueError):data={}
 status=data.get('status','NO_METADATA')
 ok=(exits.get(r['uid'])=='0' and status=='OK' and data.get('requested_url')==r['url']
    and data.get('final_url') and data.get('http_status') in range(200,400)
    and data.get('visible_chars',0)>0 and html.is_file() and html.stat().st_size>512
    and pcap_parsed and size>=8192 and packets>=12)
 good+=bool(ok)
 records.append({'instance':r['instance'],'category':r['category'],'slot':r['slot'],
   'url':r['url'],'status':status,'node_exit':exits.get(r['uid']),'html_bytes':html.stat().st_size if html.exists() else 0,
   'pcap_bytes':size,'packet_count':packets,'pcap_parsed':pcap_parsed,'qa_valid':bool(ok),
   'errors':data.get('errors',[])})
 print('QA_VISIT',r['instance'],r['category'],r['slot'],status,'exit',exits.get(r['uid']),
       'pcap_bytes',size,'packets',packets,'valid',bool(ok))
passed=(len(plans)==12 and len(exits)==12 and good==12)
result={'classification':'QA_ONLY_NOT_SCIENTIFIC','attempts':len(plans),'valid':good,'passed':passed,
        'browser_image':'ghcr.io/puppeteer/puppeteer:24.37.4',
        'native_proxy':'SOCKS5_DIRECT_TOR_9050','scientific_authorized':False,'visits':records}
(out/'qa12_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print('PINNED_COLLECTOR_QA_VALID',str(good)+'/12','ATTEMPTS',len(plans))
print('PINNED_COLLECTOR_QA_STATUS','QA_PASS' if passed else 'QA_FAILED')
sys.exit(0 if passed else 6)
PY
status=$?
set -e
echo "QA12_SUMMARY=$OUT/qa12_summary.json"
echo "QA12_REPORT=$OUT/report.log"
echo "QA12_ROOT=$QA"
echo "QA12_EXIT=$status"
echo 'SCIENTIFIC_CAPTURE=NOT_AUTHORIZED'
exit "$status"
