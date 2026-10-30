#!/usr/bin/env node
'use strict';
// QA-only: Chromium must demonstrably browse through the HTTP CONNECT -> Tor SOCKS bridge.
const puppeteer=require('puppeteer');
(async()=>{
 const mode=process.env.QA_HEADLESS_MODE||'new';
 if(!['new','shell'].includes(mode))throw Error('bad QA_HEADLESS_MODE');
 let browser;
 try {
  browser=await puppeteer.launch({executablePath:'/usr/bin/google-chrome',headless:mode==='shell'?'shell':true,timeout:20000,args:[
   '--proxy-server=http://127.0.0.1:18888','--disable-quic',
   '--force-webrtc-ip-handling-policy=disable_non_proxied_udp',
   '--no-sandbox','--disable-setuid-sandbox']});
  console.log('CHROME',await browser.version(),'MODE',mode);
  const page=await browser.newPage();
  page.on('framedetached',f=>console.log('FRAME_DETACHED',f.url().slice(0,160)));
  page.on('requestfailed',r=>console.log('REQUEST_FAILED',String(r.failure()?.errorText||''),r.url().slice(0,160)));
  console.log('PROBE_TOR_BEGIN');
  const tor=await page.goto('https://check.torproject.org/api/ip',{waitUntil:'domcontentloaded',timeout:35000});
  const raw=await tor.text();
  const parsed=JSON.parse(raw);
  if(tor.status()!==200||parsed.IsTor!==true)throw Error('Tor browser IP not verified: '+raw.slice(0,180));
  console.log('BROWSER_TOR_VERIFIED',JSON.stringify({IsTor:parsed.IsTor,IP:parsed.IP}));
  console.log('PROBE_EXAMPLE_BEGIN');
  const res=await page.goto('https://example.org/',{waitUntil:'domcontentloaded',timeout:35000});
  if(res.status()!==200)throw Error('Example unexpected HTTP '+res.status());
  const title=await page.title();
  console.log('BROWSER_EXAMPLE_VERIFIED',res.status(),title.slice(0,100));
  console.log('BROWSER_TOR_PROBE_PASS',mode);
 }catch(err){console.error('BROWSER_TOR_PROBE_FAIL',mode,String(err.stack||err).slice(0,1400));process.exitCode=7;}
 finally{if(browser)try{await browser.close()}catch(err){console.error('CLOSE_ERROR',String(err.message||err))}}
})().catch(err=>{console.error('TOP_FAIL',String(err.stack||err));process.exitCode=7});
