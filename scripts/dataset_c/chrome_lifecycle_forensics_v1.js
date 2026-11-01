#!/usr/bin/env node
'use strict';
// Diagnostic only. One browser at a time. No PCAP, no collector writes.
// Records Chrome exit status, stderr, cgroup/resource pressure, SOCKS vs direct.
const fs=require('fs');
const {execFileSync}=require('child_process');
const puppeteer=require('puppeteer');
function read(p){try{return fs.readFileSync(p,'utf8').trim().slice(0,5000)}catch(e){return 'NOT_READABLE:'+e.code}}
function resources(label) {
 console.log('RESOURCE',label,'DATE',new Date().toISOString());
 for(const p of ['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current','/sys/fs/cgroup/memory.events','/sys/fs/cgroup/memory/memory.oom_control','/sys/fs/cgroup/pids.max','/sys/fs/cgroup/pids.current','/sys/fs/cgroup/pids.events','/proc/pressure/memory']){
  console.log('RESOURCE_FIELD',label,p,JSON.stringify(read(p)));
 }
 const limits=read('/proc/self/limits').split('\n').filter(s=>/processes|open files|stack size/i.test(s)); 
 console.log('RESOURCE_LIMITS',label,JSON.stringify(limits));
 try{
  const out=execFileSync('/bin/ps',['-eo','pid,ppid,stat,comm','--no-headers'],{encoding:'utf8',timeout:2000});
  const lines=out.split('\n').filter(x=>/chrome|chromium|zygote|node|tor/.test(x));
  console.log('RESOURCE_PROCESSES',label,'count',lines.length,'sample',JSON.stringify(lines.slice(0,35)));
 }catch(e){console.log('RESOURCE_PROCESSES_UNAVAILABLE',String(e.message||e).slice(0,250))}
}
async function once(kind,index){
 const proxy=(kind==='tor');
 const flags=[
  ...(proxy?['--proxy-server=socks5://127.0.0.1:9050']:[]),
  '--disable-quic','--force-webrtc-ip-handling-policy=disable_non_proxied_udp',
  '--no-sandbox','--disable-setuid-sandbox'
 ];
 const cfg={executablePath:'/usr/bin/google-chrome',headless:true,args:flags,timeout:20000,dumpio:true};
 console.log('RUN_START',kind,index,'flags',JSON.stringify(flags));
 let browser,clean=false,passed=false;
 try{
   browser=await puppeteer.launch(cfg);
   const proc=browser.process();
   console.log('CHROME_PROCESS',kind,index,proc?.pid??null,'version',await browser.version());
   if(proc){
     proc.on('exit',(code,signal)=>console.log('CHROME_EXIT',kind,index,code,signal,'CLEANUP',clean));
     proc.on('error',e=>console.log('CHROME_PROCESS_ERROR',kind,index,String(e.message||e)));
   }
   browser.on('disconnected',()=>console.log('CHROME_DISCONNECTED',kind,index,'CLEANUP',clean));
   const page=await browser.newPage();
   page.on('error',e=>console.log('PAGE_ERROR',kind,index,String(e.message||e).slice(0,250)));
   page.on('close',()=>console.log('PAGE_CLOSED',kind,index,'CLEANUP',clean));
   page.on('framedetached',f=>console.log('FRAME_DETACHED',kind,index,JSON.stringify(f.url().slice(0,160)),'CLEANUP',clean));
   page.on('requestfailed',r=>console.log('REQUEST_FAILED',kind,index,String(r.failure()?.errorText||''),r.url().slice(0,130)));
   const url=proxy?'https://check.torproject.org/api/ip':'https://example.org/';
   const response=await page.goto(url,{waitUntil:'domcontentloaded',timeout:35000});
   console.log('NAV_HTTP',kind,index,response?.status(),page.url());
   if(response?.status()!==200)throw Error('HTTP not 200');
   if(proxy){
     const body=await page.evaluate(()=>document.body?.innerText||'');
     const obj=JSON.parse(body.trim());
     if(obj.IsTor!==true)throw Error('BROWSER_NOT_TOR');
     console.log('BROWSER_TOR_VERIFIED',kind,index,JSON.stringify({IsTor:true,IP:obj.IP}));
   }else{
     console.log('DIRECT_EXAMPLE_VERIFIED',kind,index,(await page.title()).slice(0,70));
   }
   passed=true;
 }catch(e){
   console.log('NAV_FAILURE',kind,index,String(e.stack||e).slice(0,1500));
 }finally{
   clean=true;
   if(browser)try{await browser.close()}catch(e){console.log('CLOSE_WARNING',kind,index,String(e.message||e).slice(0,300))}
 }
 console.log('RUN_RESULT',kind,index,passed?'PASS':'FAIL');
 return passed;
}
(async()=>{
 console.log('FORENSIC_START','node',process.version,'puppeteer',require('puppeteer/package.json').version,'chrome',execFileSync('/usr/bin/google-chrome',['--version'],{encoding:'utf8'}).trim());
 resources('BEFORE');
 let good=0;
 if(await once('direct',1))good++;
 resources('AFTER_DIRECT');
 for(let i=1;i<=3;i++){
   if(await once('tor',i))good++;
   resources('AFTER_TOR_'+i);
 }
 console.log('FORENSIC_RESULT',good+'/4');
 process.exitCode=good===4?0:6;
})().catch(e=>{console.error('FORENSIC_FATAL',String(e.stack||e).slice(0,1800));process.exitCode=6});
