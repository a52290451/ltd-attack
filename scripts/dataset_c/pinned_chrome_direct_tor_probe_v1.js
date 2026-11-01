#!/usr/bin/env node
'use strict';
// Scientific Dataset C - QA browser compatibility isolation.
// Official image puppeteer:24.37.4 / Chrome for Testing 145.0.7632.76.
// Direct native SOCKS5 to existing Tor; NO intermediary, NO PCAP, NO dataset writes.
const puppeteer=require('puppeteer');
const REQUIRED_PPTR='24.37.4';
const REQUIRED_CHROME='145.0.7632.76';
const flags=[
  '--disable-quic',
  '--force-webrtc-ip-handling-policy=disable_non_proxied_udp',
  '--no-sandbox','--disable-setuid-sandbox'
];
const probes=[
  {name:'control',url:'https://example.org/',tor:false},
  ...Array.from({length:4},(_,i)=>({name:'tor_'+(i+1),url:'https://check.torproject.org/api/ip',tor:true}))
];
async function run(item){
  let browser,cleaning=false,passed=false;
  const args=[...(item.tor?['--proxy-server=socks5://127.0.0.1:9050']:[]),...flags];
  console.log('PINNED_PROBE_BEGIN',item.name,JSON.stringify({proxy:item.tor?'SOCKS5_DIRECT_TOR':'NONE',args}));
  try{
    browser=await puppeteer.launch({headless:true,args,timeout:25000,dumpio:true});
    const version=await browser.version();
    console.log('PINNED_CHROME_VERSION',item.name,version);
    if(!version.includes(REQUIRED_CHROME))throw Error('CHROME_VERSION_MISMATCH_EXPECTED_'+REQUIRED_CHROME);
    const proc=browser.process();
    if(proc)proc.on('exit',(code,signal)=>{if(!cleaning)console.log('PINNED_CHROME_PREMATURE_EXIT',item.name,code,signal)});
    browser.on('disconnected',()=>{if(!cleaning)console.log('PINNED_CHROME_DISCONNECTED',item.name)});
    const page=await browser.newPage();
    page.on('error',err=>console.log('PINNED_PAGE_ERROR',item.name,String(err.message||err)));
    page.on('framedetached',frame=>{if(!cleaning)console.log('PINNED_FRAME_DETACHED',item.name,frame.url().slice(0,150))});
    const resp=await page.goto(item.url,{waitUntil:'domcontentloaded',timeout:40000});
    if(resp?.status()!==200)throw Error('UNEXPECTED_HTTP_'+(resp?.status()??'null'));
    if(item.tor){
      const data=JSON.parse((await page.evaluate(()=>document.body?.innerText||'')).trim());
      if(data.IsTor!==true)throw Error('TOR_ROUTE_NOT_VERIFIED');
      console.log('PINNED_TOR_VERIFIED',item.name,'IsTor=true','IP='+data.IP);
    }else{
      const title=await page.title();
      if(!title)throw Error('EMPTY_CONTROL_TITLE');
      console.log('PINNED_CONTROL_VERIFIED',item.name,'HTTP=200',title.slice(0,80));
    }
    passed=true;
  }catch(err){
    console.log('PINNED_PROBE_ERROR',item.name,String(err.stack||err).slice(0,1300));
  }finally{
    cleaning=true;
    if(browser)try{await browser.close()}catch(err){console.log('PINNED_CLOSE_WARNING',item.name,String(err.message||err))}
  }
  console.log('PINNED_PROBE_RESULT',item.name,passed?'PASS':'FAIL');
  return passed;
}
(async()=>{
 const ver=require('puppeteer/package.json').version;
 console.log('PINNED_PROBE_VERSION','node='+process.version,'puppeteer='+ver,'expected='+REQUIRED_PPTR,'chrome_for_testing_expected='+REQUIRED_CHROME);
 if(ver!==REQUIRED_PPTR)throw Error('PUPPETEER_VERSION_MISMATCH');
 let passed=0;
 for(const item of probes)if(await run(item))passed++;
 console.log('PINNED_PROBE_SUMMARY',passed+'/'+probes.length);
 process.exitCode=passed===probes.length?0:6;
})().catch(err=>{console.log('PINNED_PROBE_FATAL',String(err.stack||err).slice(0,1400));process.exitCode=6});
