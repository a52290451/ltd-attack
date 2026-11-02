#!/usr/bin/env bash
# LTD Dataset C: one controlled PCAP QA sweep over ONLY 20 missing frozen65 classes.
# Browser: pinned Chrome for Testing + exact ORIGINAL capture_dom.js and quality.js.
# Traffic: native SOCKS5 -> existing direct Tor, tcpdump sidecar on SAME eth0 namespace.
# No mutation to A/B, historical frozen datasets, Tor containers, original collector or cron.
# Host clock is unsynchronized; results are QA ONLY, NEVER scientific input.
set -euo pipefail
umask 077
APP="$HOME/ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2"
PATCH="$HOME/ltd-ops/collector-qa-recovery-v25"
PATCH_SHA="0150884d2b0faf568289660ffce31d0536b737b3934f7bc37118c0f34fbec774"
[[ -f "$PATCH/capture_dom.js" && -f "$PATCH/quality.js" ]] || { echo 'BLOCKED: V2.5 QA consent patch missing'; exit 3; }
[[ "$(sha256sum "$PATCH/capture_dom.js" | awk '{print $1}')" == "$PATCH_SHA" ]] || { echo 'BLOCKED: V2.5 QA consent source SHA mismatch'; exit 3; }
[[ "$(sha256sum "$PATCH/quality.js" | cut -d' ' -f1)" == "2e95a90ea77115320d37aaff0ae14157727f7b47deb28a9b18de7237e84f59e1" ]] || { echo 'BLOCKED: QA quality.js modified'; exit 3; }
BASE="/data/ltd-c-qa-bsierra/C_DIRECT_V1"
BROWSER="ghcr.io/puppeteer/puppeteer@sha256:ffdb72533aa06e9adf99d1056fbe5b79c915869f6c2e7de8fa1f1ce8ffda0790"
CAPTURE="tor_1site_noise:final"
RUN_ID="qa_remaining20_pcap_$(date -u +%Y%m%dT%H%M%SZ)_$$"
if find "$BASE/diagnostics" -mindepth 2 -maxdepth 2 -path '*/qa_remaining20_pcap_*/remaining20_pcap_summary.json' -type f 2>/dev/null | grep -q .; then
 echo 'BLOCKED: completed QA20 exists; inspect its outcomes instead of repeating';exit 3
fi
exec 9>"$BASE/diagnostics/.remaining20_pcap.lock"
flock -n 9 || { echo 'BLOCKED: targeted recovery QA already running'; exit 3; }
if docker ps --format '{{.Names}}' | grep -Eq '^ltd-c-(pcap-qa|pinned-nav|qa4-pcap)-'; then
 echo 'BLOCKED: another Dataset C capture currently running';exit 3
fi
free_bytes=$(df -PB1 "$BASE" | awk 'NR==2{print $4}')
[[ "$free_bytes" =~ ^[0-9]+$ ]] && ((free_bytes>=120000000000)) ||
 { echo 'BLOCKED: less than 120GB free or unknown disk space';exit 3; }
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
echo "REMAINING20_REPORT=$OUT/report.log"
echo "REMAINING20_ROOT=$QA"
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
 --mount "type=bind,src=$PATCH/capture_dom.js,dst=/home/pptruser/ltd-c-v2/capture_dom.js,readonly" \
 -e NODE_PATH=/home/pptruser/node_modules \
 --entrypoint node "$BROWSER" --check /home/pptruser/ltd-c-v2/capture_dom.js
docker run --init --rm --network none --cap-drop ALL --security-opt no-new-privileges \
 --mount "type=bind,src=$APP,dst=/home/pptruser/ltd-c-v2,readonly" \
 -e NODE_PATH=/home/pptruser/node_modules \
 --entrypoint node "$BROWSER" -e 'require("/home/pptruser/ltd-c-v2/quality.js");require("puppeteer");console.log("COLLECTOR_DEPENDENCIES_OK")'
# Validate six canonical category lists (20 positions each, 118 unique URLs).
python3 - "$APP/site_lists" "$OUT/visits.tsv" "$DAY" "$BASE" <<'PY'
import collections,json,pathlib,sys,urllib.parse,uuid
site_dir=pathlib.Path(sys.argv[1])
out=pathlib.Path(sys.argv[2])
day=sys.argv[3]
base=pathlib.Path(sys.argv[4])
triage=base/'diagnostics/pinned_qa108_20261101T211133Z_465267/frozen65_access_triage.json'
prior=base/'diagnostics/qa_recovered4_pcap_20261102T100945Z_2722588/remaining20_pcap_summary.json'
def read(p):
 if not p.is_file() or p.is_symlink():raise SystemExit('BLOCKED: missing/symlink evidence '+str(p))
 return json.loads(p.read_text())
a=read(triage)
b=read(prior)
missing=a['frozen_65_missing_records']
if len(missing)!=24 or len(set(x['frozen_site'] for x in missing))!=24:
 raise SystemExit('BLOCKED: frozen baseline 24-class triage mismatch')
if b.get('attempts')!=4 or b.get('valid')!=4 or len(b.get('visits',[]))!=4 or any(not v.get('qa_valid') for v in b['visits']):
 raise SystemExit('BLOCKED: preceding 4 PCAP QA proofs incomplete')
def key(r):return (str(r['instance']),str(r['category']),int(r['slot']))
successful={key(r) for r in b['visits']}
expected={('inst1','turismo',8),('inst2','bancos',8),('inst2','salud',6),('inst3','politica',15)}
if successful!=expected:raise SystemExit('BLOCKED: recovered4 slot provenance mismatch')
if not successful.issubset({key(r) for r in missing}):raise SystemExit('BLOCKED: recovered4 not part of baseline24')
pending=[r for r in missing if key(r) not in successful]
if len(pending)!=20 or len({key(r) for r in pending})!=20 or len({r['frozen_site'] for r in pending})!=20:
 raise SystemExit('BLOCKED: twenty unique remaining frozen labels not found')
kinds=collections.Counter(r['status'] for r in pending)
if kinds!={'BLOCKED':17,'HTTP_ERROR':2,'NAVIGATION_ERROR':1}:
 raise SystemExit('BLOCKED: unexpected pending status composition '+str(kinds))
targets={key(r):r for r in pending}
pairs=[('inst1','belleza'),('inst1','turismo'),('inst2','bancos'),('inst2','salud'),('inst3','deportes'),('inst3','politica')]
all_urls=[];rows=[]
for inst,cat in pairs:
 f=site_dir/inst/(cat+'.txt')
 lines=[x.strip() for x in f.read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')]
 if len(lines)!=20:raise SystemExit('BLOCKED: 20 canonical positions expected '+str(f))
 for idx,url in enumerate(lines,1):
  parsed=urllib.parse.urlsplit(url)
  if parsed.scheme not in ('https','http') or not parsed.hostname:raise SystemExit('BLOCKED: invalid site URL')
  all_urls.append(url)
  k=(inst,cat,idx)
  if k in targets:
   if url!=targets[k]['url']:raise SystemExit('BLOCKED: original URL mismatch '+str(k))
   rows.append((inst,cat,str(idx),url,parsed.hostname,str(uuid.uuid4()),day))
if len(all_urls)!=120 or len(set(all_urls))!=118 or len(rows)!=20:
 raise SystemExit('BLOCKED: 120/118 catalog or targeted 20 mismatch')
with out.open('w') as g:
 g.write('instance\tcategory\tslot\turl\thost\tuid\tday\n')
 for row in rows:g.write('\t'.join(row)+'\n')
print('QA20_PLAN',len(rows),'FROZEN65_PENDING20_AFTER_RECOVERED4')
print('QA20_PRIOR_STATUS',dict(kinds))
print('QA20_NO_FROZEN_MODEL_AB_MUTATION=YES')
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
echo '=== TWENTY REMAINING FROZEN65 TARGETED PCAP QA ATTEMPTS (NOT SCIENTIFIC DATA) ==='
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
  timeout -k 5s 135s docker run --init --rm \
   --name "ltd-c-pinned-nav-inst$i-$$" --network "container:ltd-c-tor-inst$i" \
   --cap-drop ALL --security-opt no-new-privileges --shm-size 1g \
   --mount "type=bind,src=$APP,dst=/home/pptruser/ltd-c-v2,readonly" \
   --mount "type=bind,src=$QA,dst=/audit" \
   --mount "type=bind,src=$PATCH/capture_dom.js,dst=/home/pptruser/ltd-c-v2/capture_dom.js,readonly" \
   -e NODE_PATH=/home/pptruser/node_modules \
   -e CHROME_NO_SANDBOX=1 -e SESSION_POLICY=fresh -e CONSENT_POLICY=accept_all \
   -e CAPTURE_UID="$uuid" -e SITE_SLOT_INDEX="$slot" -e SITE_INSTANCE="$inst" -e SITE_CATEGORY="$cat" \
   --entrypoint node "$BROWSER" /home/pptruser/ltd-c-v2/capture_dom.js \
   "$url" "/audit/$stem.html" 9050 "$CHROME" 90 \
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
    and pcap_parsed and size>=8192 and packets>=12
    and (not data.get('consent_visible')
       or (isinstance(data.get('consent'),dict)
           and data['consent'].get('action')=='clicked')))
 good+=bool(ok)
 records.append({'instance':r['instance'],'category':r['category'],'slot':r['slot'],
   'url':r['url'],'status':status,'node_exit':exits.get(r['uid']),'html_bytes':html.stat().st_size if html.exists() else 0,
   'pcap_bytes':size,'packet_count':packets,'pcap_parsed':pcap_parsed,'qa_valid':bool(ok),
   'errors':data.get('errors',[])})
 print('QA_VISIT',r['instance'],r['category'],r['slot'],status,'exit',exits.get(r['uid']),
       'pcap_bytes',size,'packets',packets,'valid',bool(ok))
passed=(len(plans)==20 and len(exits)==20 and good==20)
result={'classification':'QA_ONLY_NOT_SCIENTIFIC','attempts':len(plans),'valid':good,'passed':passed,
        'browser_image':'ghcr.io/puppeteer/puppeteer:24.37.4',
        'native_proxy':'SOCKS5_DIRECT_TOR_9050','scientific_authorized':False,'visits':records}
(out/'remaining20_pcap_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print('REMAINING20_PCAP_VALID',str(good)+'/20','ATTEMPTS',len(plans))
print('FROZEN65_TOTAL_QA_VALID_AFTER_20',str(45+good)+'/65')
print('REMAINING20_QA_STATUS_COUNTS',dict(collections.Counter(r['status'] for r in records)))
for r in records:
 print('REMAINING20_SITE',r['instance'],r['category'],r['slot'],
       r['url'],r['status'],'VALID',r['qa_valid'],'ERRORS',str(r['errors'])[:160])
print('REMAINING20_PCAP_REQUIREMENTS','http_2xx_3xx','content_html','pcap_parse_and_size','consent_policy_v25')
print('REMAINING20_PCAP_QA_STATUS','QA_PASS' if passed else 'SITES_STILL_UNAVAILABLE_OR_QA_UNRESOLVED')
sys.exit(0 if passed else 6)
PY
status=$?
set -e
echo "REMAINING20_PCAP_QA_SUMMARY=$OUT/remaining20_pcap_summary.json"
echo "REMAINING20_REPORT=$OUT/report.log"
echo "REMAINING20_ROOT=$QA"
echo "REMAINING20_PCAP_QA_EXIT=$status"
echo 'SCIENTIFIC_CAPTURE=NOT_AUTHORIZED'
exit "$status"
