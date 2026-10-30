#!/usr/bin/env node
'use strict';
// Direct SOCKS5 Chrome QA with Puppeteer CDP over pipe, NOT a proxy/timing change.
// Chrome flags and launch binary are kept identical to baseline collector.
// Each navigation gets a fresh Chrome process, as in the real collector.
const puppeteer=require('puppeteer');
const cases=[
 ['tor','https://check.torproject.org/api/ip'],
 ['example','https://example.org/'],
];
async function nav(kind,url,round){
 let browser=null,done=false,closed=false;
 try{
  const flags=['--proxy-server=socks5://127.0.0.1:9050',
    '--disable-quic','--force-webrtc-ip-handling-policy=disable_non_proxied_udp',
    '--no-sandbox','--disable-setuid-sandbox'];
  browser=await puppeteer.launch({
   executablePath:'/usr/bin/google-chrome',headless:true,
   args:flags,timeout:20000,pipe:true,dumpio:true
  });
  console.log('CDP_PIPE_BROWSER',round,kind,await browser.version());
  const proc=browser.process();
  if(proc)proc.on('exit',(code,signal)=>console.log('CDP_PIPE_CHROME_EXIT',round,kind,'code',code,'signal',signal,'BEFORE_CLEANUP',!closed));
  browser.on('disconnected',()=>console.log('CDP_PIPE_DISCONNECTED',round,kind,'BEFORE_CLEANUP',!closed));
  const page=await browser.newPage();
  page.setDefaultNavigationTimeout(35000);
  page.on('error',e=>console.log('CDP_PIPE_PAGE_ERROR',round,kind,String(e.message||e).slice(0,350)));
  page.on('close',()=>console.log('CDP_PIPE_PAGE_CLOSED',round,kind,'BEFORE_CLEANUP',!closed));
  page.on('framedetached',f=>{if(!closed)console.log('CDP_PIPE_FRAME_DETACHED_DURING_NAV',round,kind,f.url().slice(0,120));});
  console.log('CDP_PIPE_NAV_BEGIN',round,kind,url);
  const resp=await page.goto(url,{waitUntil:'domcontentloaded',timeout:35000});
  if(!resp||resp.status()!==200)throw Error('Unexpected HTTP '+String(resp?.status()));
  if(kind==='tor'){
   const raw=await page.evaluate(()=>document.body?.innerText||'');
   const obj=JSON.parse(raw.trim());
   if(obj.IsTor!==true)throw Error('BROWSER_NOT_TOR');
   console.log('CDP_PIPE_TOR_VERIFIED',round,JSON.stringify({IsTor:obj.IsTor,IP:obj.IP}));
  }else{
   const title=await page.title();
   if(!title)throw Error('EMPTY_TITLE');
   console.log('CDP_PIPE_EXAMPLE_VERIFIED',round,title.slice(0,90));
  }
  done=true;
  console.log('CDP_PIPE_NAV_OK',round,kind);
 }catch(e){
  console.error('CDP_PIPE_NAV_FAIL',round,kind,String(e.stack||e).slice(0,950));
 }finally{
  closed=true;
  if(browser)try{await browser.close();}catch(e){console.log('CDP_PIPE_CLOSE_WARN',round,kind,String(e.message||e))}
 }
 return done;
}
(async()=>{
 console.log('CDP_PIPE_PROBE_START','puppeteer',require('puppeteer/package.json').version);
 const expected=4;
 let success=0;
 for(let round=1;round<=2;round++){
  for(const [kind,url] of cases){
   if(await nav(kind,url,round)) success++;
  }
 }
 console.log('CDP_PIPE_PROBE_RESULT',success+'/'+expected);
 process.exitCode=success===expected?0:6;
})().catch(e=>{console.error('CDP_PIPE_UNCAUGHT',String(e.stack||e));process.exitCode=6});
