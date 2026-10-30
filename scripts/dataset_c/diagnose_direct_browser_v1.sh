#!/usr/bin/env bash
# Dataset C: one-shot diagnostic, read-only existing PCAP, plus disposable browser probes.
# Does NOT run category collector or capture packets. No cron, no production changes.
set -uo pipefail
umask 077
APP="$HOME/ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2"
RUN="/data/ltd-c-qa-bsierra/C_DIRECT_V1/qa_only/direct_smoke_20261030T115300Z_2361887"
OUT="/data/ltd-c-qa-bsierra/C_DIRECT_V1/diagnostics/browser_diag_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$OUT"
chmod 700 "$OUT"
exec > >(tee "$OUT/report.log") 2>&1
echo "DIRECT_C_BROWSER_DIAGNOSTIC_START"
echo "REPORT=$OUT/report.log"
echo "HOST=$(hostname) HOST_TIME_UNVERIFIED=$(date -u +%FT%TZ)"
echo "NTP=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown)"
echo "GIT=$(git -C "$HOME/ltd-attack" rev-parse --short HEAD 2>/dev/null || true)"
echo "=== SCIENTIFIC GATE: QA ONLY. NEVER PROMOTE EXISTING PCAP ==="
[[ -d "$RUN" && -d "$APP" ]] || { echo "STOP existing run or collector absent"; exit 2; }
echo "=== SOURCE INFORMATION ==="
sha256sum "$APP/capture_dom.js" "$APP/run_category.sh" "$APP/quality.js" 2>/dev/null || true
echo "=== NODE CONTROL AND FRAME EVENTS ==="
nl -ba "$APP/capture_dom.js" | sed -n '1,105p;170,290p'
echo "=== CATEGORY ENTRYPOINT & CONFIG ==="
nl -ba "$APP/run_category.sh" | sed -n '1,135p'
echo "=== ERRORS AND RECORDS FROM EXISTING PILOT ==="
python3 - "$RUN" <<'PY'
import sys,pathlib,json,csv,collections
root=pathlib.Path(sys.argv[1]);statuses=collections.Counter(); errors=collections.Counter()
print("RUN",root)
for p in sorted(root.glob("inst*/**/attempts_*.tsv")):
 with p.open(newline="",encoding="utf-8-sig") as f:
  rs=list(csv.DictReader(f,delimiter="\t"))
 print("TSV",p.relative_to(root),"ROWS",len(rs),"FIELDS", list(rs[0]) if rs else [])
 for r in rs:
  statuses[r.get("status","")]+=1
  print(" ATTEMPT",r.get("url"),"STATUS",r.get("status"),"EXIT",r.get("exit_code"),"FLAG",r.get("pcap_valid"),"BYTES",r.get("pcap_bytes"))
for p in sorted(root.glob("inst*/**/*.json")):
 if p.name=="metadata.json": continue
 try:
  d=json.loads(p.read_text())
 except Exception as e:
  print("JSON_INVALID",p.relative_to(root),str(e));continue
 er=d.get("errors",[]) or []
 for e in er:errors[str(e)]+=1
 print("DETAIL",p.relative_to(root),"status",d.get("status"),"HTTP",d.get("http_status"),"duration_ms",d.get("duration_monotonic_ms"),"errors",repr(er),"url",d.get("requested_url"))
print("ALL_STATUSES",dict(statuses))
print("ALL_ERRORS",dict(errors))
for p in sorted(root.glob("inst*/**/*.pcap")):
 print("PCAP",p.relative_to(root),"BYTES",p.stat().st_size)
PY
echo "=== BROWSER LOGS / TCPDUMP ERRORS ==="
find "$RUN" -type f \( -name '*.browser.log' -o -name '*.tcpdump.log' \) -print0 |
while IFS= read -r -d '' p; do
 echo "FILE=$p"; tail -n 10 "$p" || true
done
echo "=== CONTAINER/ROUTE CONFIG ==="
for i in 1 2 3; do
 n="ltd-c-tor-inst$i"
 docker inspect -f 'container={{.Name}} state={{.State.Status}} network={{.HostConfig.NetworkMode}}' "$n"
 docker exec "$n" sh -c 'cat /tmp/direct-torrc; tr "\000" " " </proc/1/cmdline; echo' || true
done
echo "=== ONE DISPOSABLE BROWSER-IMAGE DIAGNOSTIC PER INSTANCE ==="
for i in 1 2 3; do
 n="ltd-c-tor-inst$i"
 echo "=== PROBE INSTANCE $i ==="
 # Reuses the browser image and network namespace, NOT the category capture pipeline.
 # Bounded check of SOCKS and headless Chrome on a neutral URL.
 timeout 110 docker run --rm \
   --name "ltd-c-direct-diag-inst$i-$$" \
   --network "container:$n" \
   --security-opt no-new-privileges --shm-size 512m \
   --mount "type=bind,src=$APP,dst=/opt/ltd-c-v2,readonly" \
   -e NODE_PATH=/app/node_modules -e CHROME_NO_SANDBOX=1 \
   --entrypoint /bin/bash tor_1site_noise:final -lc '
     echo "UTC_INSIDE=$(date -u +%FT%TZ)"
     echo "NODE=$(node --version)"
     echo "CHROME_BIN=$(command -v chromium || command -v chromium-browser || command -v google-chrome || true)"
     curl -fsS --connect-timeout 8 --max-time 25 --socks5-hostname 127.0.0.1:9050 https://check.torproject.org/api/ip || true
     node - <<'\''JS'\''
const puppeteer = require("puppeteer");
(async () => {
 let browser, page;
 try {
  const executablePath = process.env.CHROME_BIN || "/usr/bin/chromium";
  browser = await puppeteer.launch({
    executablePath, headless:true, timeout:20000,
    args:["--no-sandbox","--disable-setuid-sandbox","--disable-quic","--proxy-server=socks5://127.0.0.1:9050"]
  });
  console.log("BROWSER_VERSION", await browser.version());
  page = await browser.newPage();
  page.on("error", e=>console.log("PAGE_ERROR",String(e.message||e).slice(0,300)));
  page.on("framenavigated", f=>console.log("FRAME_NAVIGATED",f.url().slice(0,200)));
  page.on("framedetached", f=>console.log("FRAME_DETACHED",f.url().slice(0,200)));
  page.on("requestfailed",r=>console.log("REQUEST_FAILED",r.url().slice(0,120),String(r.failure()?.errorText||"")));
  for (const url of ["https://check.torproject.org/","https://example.org/"]) {
   try {
    console.log("NAV_BEGIN",url);
    const response=await page.goto(url,{waitUntil:"domcontentloaded",timeout:30000});
    console.log("NAV_RESULT",url,response?.status()??null,page.url());
   } catch(e) {console.log("NAV_ERROR",url,String(e.stack||e).slice(0,700));}
  }
 }catch(e){console.log("BROWSER_PROBE_ERROR",String(e.stack||e).slice(0,900));}
 finally {if(browser)try{await browser.close()}catch(_){}}
})()
JS
   ' || echo "DISPOSABLE_PROBE_FAILED_INST$i (inspect logs)"
done
echo "=== END: NO CAPTURE, NO CRON, NO DELETE ==="
echo "REPORT=$OUT/report.log"
