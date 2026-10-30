#!/usr/bin/env node
'use strict';
// Dataset C QA: original Chrome binary/launch args/SOCKS5, alternate control
// protocol WebDriver BiDi. No HTTP proxy adapter, no host clock modification.
const puppeteer=require('puppeteer');
const cases=[
 {url:'https://check.torproject.org/api/ip',name:'tor_ip'},
 {url:'https://example.org/',name:'example'}
];
async function one(entry) {
 let browser;
 try {
  browser=await puppeteer.launch({
   executablePath:'/usr/bin/google-chrome',
   protocol:'webDriverBiDi',
   headless:true,
   timeout:20000,
   args:['--proxy-server=socks5://127.0.0.1:9050',
     '--disable-quic','--force-webrtc-ip-handling-policy=disable_non_proxied_udp',
     '--no-sandbox','--disable-setuid-sandbox']
  });
  const page=await browser.newPage();
  page.setDefaultNavigationTimeout(35000);
  page.on('framedetached', f=>console.log('BIDI_FRAME_DETACHED',entry.name,f.url().slice(0,120)));
  page.on('framenavigated',f=>console.log('BIDI_FRAME_NAVIGATED',entry.name,f.url().slice(0,120)));
  const response=await page.goto(entry.url,{waitUntil:'domcontentloaded',timeout:35000});
  const status=response?.status() ?? null;
  if(status!==200)throw Error('HTTP_STATUS_'+status);
  if(entry.name==='tor_ip'){
    const raw=await page.evaluate(()=>document.body?.innerText || '');
    const obj=JSON.parse(raw.trim());
    if(obj.IsTor!==true)throw Error('NOT_TOR_'+raw.slice(0,100));
    console.log('BIDI_DIRECT_TOR_VERIFIED',JSON.stringify({IsTor:obj.IsTor,IP:obj.IP}));
  } else {
    const title=await page.title();
    if(!title)throw Error('EMPTY_TITLE');
    console.log('BIDI_EXAMPLE_OK',status,title.slice(0,100));
  }
  console.log('BIDI_NAV_OK',entry.name);
  return true;
 }catch(e){
  console.log('BIDI_NAV_FAIL',entry.name,String(e.stack||e).slice(0,950));
  return false;
 }finally{if(browser)try{await browser.close()}catch(e){console.log('BIDI_CLOSE_WARN',String(e.message||e))}}
}
(async()=>{
 console.log('BIDI_PROBE_START','node',process.version,'puppeteer',require('puppeteer/package.json').version);
 let passed=0;
 for(const entry of cases) {
   let ok=false;
   for(let attempt=1;attempt<=2&&!ok;attempt++){
      console.log('BIDI_CASE',entry.name,'ATTEMPT',attempt);
      ok=await one(entry);
      if(!ok && attempt<2) await new Promise(r=>setTimeout(r,4500));
   }
   if(ok)passed++;
 }
 console.log('BIDI_PROBE_RESULT',passed+'/2');
 process.exitCode=passed===2?0:6;
})().catch(e=>{console.error('BIDI_PROBE_TOP_ERROR',String(e.stack||e).slice(0,950));process.exitCode=6});
