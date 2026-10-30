#!/usr/bin/env bash
# Dataset C: browser/Tor navigation matrix; disposable; no PCAP, no cron.
set -euo pipefail
umask 077
OUT="/data/ltd-c-qa-bsierra/C_DIRECT_V1/diagnostics/frame_matrix_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$OUT"
exec > >(tee "$OUT/report.log") 2>&1
echo "MATRIX_OUTPUT=$OUT"
echo "HOST_NTP=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || true)"
for i in 1 2 3; do
 echo "=== INSTANCE=$i ==="
 timeout 180 docker run --rm --name "ltd-c-frame-matrix-inst$i-$$" \
  --network "container:ltd-c-tor-inst$i" --security-opt no-new-privileges \
  --shm-size 1g -e NODE_PATH=/app/node_modules \
  --entrypoint /bin/bash tor_1site_noise:final -lc '
    node - <<'\''JS'\''
const puppeteer = require("puppeteer");
(async()=>{
try{
 console.log("PUPPETEER_VERSION",require("puppeteer/package.json").version);
 console.log("PUPPETEER_CORE_VERSION",require("puppeteer-core/package.json").version);
}catch(e){console.log("VERSION_ERROR",e.message)}
const tests=[
 ["unproxied-new",[],true],
 ["proxied-new",["--proxy-server=socks5://127.0.0.1:9050"],true],
 ["proxied-old",["--proxy-server=socks5://127.0.0.1:9050"],"shell"],
];
for(const [name,extra,headless] of tests){
 console.log("CASE_BEGIN",name);
 let browser, page;
 try{
  browser=await puppeteer.launch({executablePath:"/usr/bin/google-chrome",headless,timeout:20000,args:["--no-sandbox","--disable-setuid-sandbox","--disable-quic",...extra]});
  console.log("CHROME",await browser.version());
  page=await browser.newPage();
  let detach=0, reqfails=0;
  page.on("framedetached",f=>{detach++;console.log("FRAME_DETACHED",name,f.url().slice(0,100));});
  page.on("framenavigated",f=>console.log("FRAME_NAVIGATED",name,f.url().slice(0,100)));
  page.on("requestfailed",r=>{reqfails++;if(reqfails<=4)console.log("REQUEST_FAILED",name,r.failure()?.errorText,r.url().slice(0,100));});
  for(const url of ["https://example.org/","https://check.torproject.org/"]){
   try{
    const res=await page.goto(url,{waitUntil:"domcontentloaded",timeout:18000});
    console.log("NAV_OK",name,res?.status(),page.url());
   }catch(e){
    console.log("NAV_FAIL",name,String(e.message||e).slice(0,250));
    if(page.isClosed())break;
   }
  }
  console.log("CASE_SUMMARY",name,"DETACH",detach,"REQUEST_FAILURES",reqfails);
 }catch(e){console.log("CASE_EXCEPTION",name,String(e.message||e).slice(0,350));}
 finally{if(browser)try{await browser.close()}catch(_){}}
}
})().catch(e=>console.log("MATRIX_TOP_ERROR",e.message));
JS
  ' || echo "MATRIX_TIMEOUT_OR_CONTAINER_FAILURE instance=$i"
done
echo "MATRIX_COMPLETE"
echo "MATRIX_OUTPUT=$OUT"
