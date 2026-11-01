#!/usr/bin/env node
'use strict';
// Controlled diagnostic for Chrome renderer exit with native direct-Tor SOCKS.
// OptimizationHints is a Puppeteer default disabled feature with known renderer
// regressions. No HTTP adapter, no third-party proxy, no science capture.
const puppeteer=require('puppeteer');
const mode=process.env.QA_CHROME_FEATURE_MODE||'enable';
if(!['enable','strip'].includes(mode)){console.error('BAD_MODE');process.exit(2)}
const base=[
 '--proxy-server=socks5://127.0.0.1:9050',
 '--disable-quic',
 '--force-webrtc-ip-handling-policy=disable_non_proxied_udp',
 '--no-sandbox','--disable-setuid-sandbox'
];
function launchConfig(){
  if(mode==='enable')return {executablePath:'/usr/bin/google-chrome',headless:true,timeout:20000,args:[...base,'--enable-features=OptimizationHints']};
  const defaults=puppeteer.defaultArgs({headless:true,args:[...base]});
  let found=0;
  const adjusted=defaults.map(a=>{
    if(!a.startsWith('--disable-features='))return a;
    const parts=a.slice('--disable-features='.length).split(',');
    if(!parts.includes('OptimizationHints'))return a;
    found++;
    return '--disable-features='+parts.filter(x=>x!=='OptimizationHints').join(',');
  });
  if(found!==1)throw Error('Unexpected Puppeteer feature flag; stop safely');
  return {executablePath:'/usr/bin/google-chrome',headless:true,timeout:20000,ignoreDefaultArgs:true,args:adjusted};
}
async function one(caseName,url,round){
  let browser=null,cleaning=false,ok=false;
  try{
    const cfg=launchConfig();
    console.log('FEATURE_LAUNCH_FLAGS',round,caseName,JSON.stringify(cfg.args.filter(x=>x.includes('features='))));
    browser=await puppeteer.launch(cfg);
    const proc=browser.process();
    if(proc)proc.on('exit',(code,signal)=>{if(!cleaning)console.log('FEATURE_CHROME_PREMATURE_EXIT',round,caseName,code,signal)});
    browser.on('disconnected',()=>{if(!cleaning)console.log('FEATURE_CHROME_DISCONNECTED',round,caseName)});
    const page=await browser.newPage();
    page.on('framedetached',f=>{if(!cleaning)console.log('FEATURE_FRAME_DETACHED',round,caseName,f.url().slice(0,120))});
    page.on('error',e=>console.log('FEATURE_PAGE_ERROR',round,caseName,String(e.message||e).slice(0,200)));
    const response=await page.goto(url,{waitUntil:'domcontentloaded',timeout:45000});
    if(!response||response.status()!==200)throw Error('UNEXPECTED_HTTP_'+String(response?.status()));
    if(caseName==='tor'){
      const obj=JSON.parse((await page.evaluate(()=>document.body?.innerText||'')).trim());
      if(obj.IsTor!==true)throw Error('TOR_NOT_VERIFIED');
      console.log('FEATURE_TOR_VERIFIED',round,JSON.stringify({IsTor:obj.IsTor,IP:obj.IP}));
    }else{
      console.log('FEATURE_EXAMPLE_VERIFIED',round,(await page.title()).slice(0,70));
    }
    ok=true;
    console.log('FEATURE_NAV_OK',round,caseName);
  }catch(e){console.error('FEATURE_NAV_FAIL',round,caseName,String(e.stack||e).slice(0,1000))}
  finally{cleaning=true;if(browser)try{await browser.close()}catch(_){}}
  return ok;
}
(async()=>{
 console.log('FEATURE_PROBE_START','mode',mode,'puppeteer',require('puppeteer/package.json').version);
 let pass=0;
 for(let round=1;round<=2;round++){
  // Each attempt uses its own Chrome profile/process just like real collector.
  if(await one('tor','https://check.torproject.org/api/ip',round))pass++;
  if(await one('example','https://example.org/',round))pass++;
 }
 console.log('FEATURE_PROBE_RESULT',mode,pass+'/4');
 process.exitCode=pass===4?0:6;
})().catch(e=>{console.error('FEATURE_TOP_FAIL',String(e.stack||e).slice(0,1000));process.exitCode=6});
