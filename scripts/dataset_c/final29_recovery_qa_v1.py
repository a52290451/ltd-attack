#!/usr/bin/env python3
"""One bounded last QA run for 29 non-frozen canonical slots not valid in QA120.

Phase A: one original Tor-instance visit per slot; Phase B: one visit through
another already-existing direct Tor instance ONLY for failures. Each visit has
PCAP, HTML, JSON, and provenance. Workers are serialized per Tor namespace.
No sudo, circuit forcing, challenge bypass, training, cron or scientific capture.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import uuid

BASE=Path('/data/ltd-c-qa-bsierra/C_DIRECT_V1')
SOURCE=BASE/'diagnostics/pinned_qa108_20261101T211133Z_465267/qa120_coverage_summary.json'
APP=Path.home()/'ltd-ops/collector-qa-recovery-v25'
BROWSER='ghcr.io/puppeteer/puppeteer@sha256:ffdb72533aa06e9adf99d1056fbe5b79c915869f6c2e7de8fa1f1ce8ffda0790'
CAPTURE='tor_1site_noise:final'
CHROME='/home/pptruser/.cache/puppeteer/chrome/linux-145.0.7632.76/chrome-linux64/chrome'
PATCH_SHA='0150884d2b0faf568289660ffce31d0536b737b3934f7bc37118c0f34fbec774'
QUALITY_SHA='2e95a90ea77115320d37aaff0ae14157727f7b47deb28a9b18de7237e84f59e1'
ALT={'inst1':'inst2','inst2':'inst3','inst3':'inst1'}
RESERVE_BYTES=120_000_000_000

def call(argv,timeout=30):
    return subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)

def require(success,message):
    if not success:raise RuntimeError('BLOCKED: '+message)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def canonical_plan():
    require(SOURCE.is_file() and not SOURCE.is_symlink(),'canonical QA120 summary missing')
    source=json.loads(SOURCE.read_text())
    rows=source.get('records')
    require(isinstance(rows,list) and len(rows)==120,'canonical QA120 must contain 120 slot records')
    require(source.get('inventory_distinct_urls')==118,'118 unique historical URL invariant')
    def key(r):return (str(r['instance']),str(r['category']),int(r['slot']))
    require(len(set(key(r) for r in rows))==120,'duplicate or absent canonical slot')
    require(sum(bool(r.get('frozen_65')) for r in rows)==65,'frozen 65 label invariant')
    # qa120_coverage_summary stores the 120 canonical slot identities. Its per-slot
    # QA flag need not already include all cumulative QA12/retry5 successes, so the
    # frozen65's later 41/65 number must NOT be asserted against this source.
    # We select only the 29 nonfrozen missing slots and keep the exact URL/slot.
    nonfrozen=[r for r in rows if not r.get('frozen_65')]
    require(len(nonfrozen)==55,'expected 55 nonfrozen original positions')
    candidates=(
        ('qa_valid',lambda r:r.get('qa_valid') is True),
        ('valid',lambda r:r.get('valid') is True),
        ('status_OK',lambda r:r.get('status')=='OK'),
    )
    diagnostics={}
    selected=None
    for name,checker in candidates:
        valid=sum(bool(checker(r)) for r in nonfrozen)
        diagnostics[name]=valid
        if valid==26 and selected is None:
            selected=(name,checker)
    print('FINAL29_BASELINE_VALIDATION',diagnostics,flush=True)
    require(selected is not None,
            'No supported QA120 success field yields the expected 26/55 nonfrozen baseline; inspect source without guessing')
    source_field,valid_func=selected
    pending=[r for r in nonfrozen if not valid_func(r)]
    require(len(pending)==29,'expected exactly 29 missing nonfrozen positions')
    print('FINAL29_BASELINE_SOURCE_FIELD',source_field,flush=True)
    out=[]
    for r in pending:
        from urllib.parse import urlsplit
        url=r['url']; parsed=urlsplit(url)
        require(parsed.scheme in ('http','https') and parsed.hostname is not None,
                'invalid canonical URL')
        out.append({'instance':str(r['instance']),'category':str(r['category']),
                    'slot':int(r['slot']),'url':url,'domain':parsed.hostname,
                    'initial_status':r.get('status'),'original_qa_valid':False})
    return out

def free_bytes():
    v=os.statvfs(BASE)
    return v.f_bavail*v.f_frsize

def preflight():
    require(digest(APP/'capture_dom.js')==PATCH_SHA,'V2.5 collector SHA mismatch')
    require(digest(APP/'quality.js')==QUALITY_SHA,'original quality.js SHA mismatch')
    active=call(['docker','ps','--format','{{.Names}}'],15)
    require(active.returncode==0,'cannot inspect active containers')
    require(not any(n.startswith(('ltd-c-pcap-qa-','ltd-c-pinned-nav-',
                                  'ltd-c-alt19-nav-','ltd-c-final29-'))
                    for n in active.stdout.splitlines()),'overlapping Dataset C QA')
    for img in (BROWSER,CAPTURE):
        require(call(['docker','image','inspect',img],20).returncode==0,'Docker image absent: '+img)
    for i in (1,2,3):
        n=f'ltd-c-tor-inst{i}'
        require(call(['docker','inspect','-f','{{.State.Running}}',n],15).stdout.strip()=='true',
                n+' not running')
        require(call(['docker','exec',n,'grep','-qx','UseBridges 0','/tmp/direct-torrc'],15).returncode==0,
                n+' is not configured as direct Tor')
    require(free_bytes()>=RESERVE_BYTES,'less than 120GB free on QA filesystem')
    check=call(['docker','run','--rm','--network','none','--cap-drop','ALL',
                '--security-opt','no-new-privileges',
                '--mount',f'type=bind,src={APP},dst=/home/pptruser/ltd-c-v2,readonly',
                '--entrypoint','node',BROWSER,'--check',
                '/home/pptruser/ltd-c-v2/capture_dom.js'],35)
    require(check.returncode==0,'V2.5 collector cannot run with pinned image')

def chown_dir(directory,uid,gid):
    x=call(['docker','run','--rm','--network','none','--user','0',
            '--mount',f'type=bind,src={directory},dst=/audit','--entrypoint','chown',
            CAPTURE,'-R',f'{uid}:{gid}','/audit'],45)
    require(x.returncode==0,'QA output ownership: '+x.stderr[-240:])

def packets_valid(path):
    if not path.is_file():return (False,0,0)
    size=path.stat().st_size
    if size<8192:return (False,0,size)
    try:
        with path.open('rb') as f:
            endian={b'\xd4\xc3\xb2\xa1':'<',b'\xa1\xb2\xc3\xd4':'>'}.get(f.read(4))
            if not endian:return (False,0,size)
            hdr=f.read(20)
            if len(hdr)!=20:return (False,0,size)
            major,minor,zone,sig,snap,network=struct.unpack(endian+'HHiIII',hdr)
            if snap<64 or network!=1:return (False,0,size)
            count=0
            while True:
                rec=f.read(16)
                if not rec:break
                if len(rec)!=16:return (False,count,size)
                _,_,inc,original=struct.unpack(endian+'IIII',rec)
                if inc>snap or inc>original:return (False,count,size)
                if len(f.read(inc))!=inc:return (False,count,size)
                count+=1
            return (count>=12,count,size)
    except (OSError,ValueError,struct.error):return (False,0,size)

def evaluate(result,out):
    stem=result['stem']
    meta=out/(stem+'.json');html=out/(stem+'.html');pcap=out/(stem+'.pcap')
    try:d=json.loads(meta.read_text())
    except (OSError,ValueError):d={}
    parsed,n,size=packets_valid(pcap)
    consent=d.get('consent') or {}
    ok=(result['node_exit']==0 and d.get('status')=='OK'
        and d.get('requested_url')==result['url']
        and d.get('final_url')
        and d.get('http_status') in range(200,400)
        and (d.get('visible_chars') or 0)>0
        and html.is_file() and html.stat().st_size>512
        and parsed and size>=8192 and n>=12
        and (not d.get('consent_visible') or consent.get('action')=='clicked'))
    return {**result,'status':d.get('status','NO_METADATA'),
            'http_status':d.get('http_status'),'consent_action':consent.get('action'),
            'errors':d.get('errors',[]),'pcap_bytes':size,'pcap_packets':n,
            'pcap_parsed':parsed,'html_bytes':html.stat().st_size if html.exists() else 0,
            'qa_valid':bool(ok)}

def visit(r,tor,phase,diag,out):
    require(free_bytes()>=RESERVE_BYTES,'disk reserve during QA')
    uid=str(uuid.uuid4());stem='_'.join([phase,r['instance'],r['category'],str(r['slot']),uid])
    capname='ltd-c-final29-cap-'+uid[:12]
    navname='ltd-c-final29-nav-'+uid[:12]
    result={**r,'phase':phase,'tor_instance':tor,'uuid':uid,'stem':stem,
            'node_exit':None,'capture_exit':None}
    capture_argv=['docker','run','-d','--name',capname,'--user','0',
        '--network','container:ltd-c-tor-'+tor,'--cap-add','NET_RAW',
        '--cap-add','NET_ADMIN','--security-opt','no-new-privileges',
        '--mount',f'type=bind,src={out},dst=/audit',
        '--entrypoint','tcpdump',CAPTURE,'-i','eth0','-n','-s','0','-U',
        '-w',f'/audit/{stem}.pcap','tcp']
    start=call(capture_argv,35)
    result['capture_exit']=start.returncode
    if start.returncode!=0:
        (diag/f'{stem}.error.log').write_text('PCAP START '+start.stderr[-1200:])
        result['node_exit']=98
        return evaluate(result,out)
    try:
        time.sleep(.8)
        running=call(['docker','inspect','-f','{{.State.Running}}',capname],15)
        if running.stdout.strip()!='true':
            result['node_exit']=99
            return evaluate(result,out)
        browser=['docker','run','--init','--rm','--name',navname,
            '--network','container:ltd-c-tor-'+tor,
            '--cap-drop','ALL','--security-opt','no-new-privileges','--shm-size','1g',
            '--mount',f'type=bind,src={APP},dst=/home/pptruser/ltd-c-v2,readonly',
            '--mount',f'type=bind,src={out},dst=/audit',
            '-e','NODE_PATH=/home/pptruser/node_modules',
            '-e','CHROME_NO_SANDBOX=1','-e','SESSION_POLICY=fresh',
            '-e','CONSENT_POLICY=accept_all',
            '-e','CAPTURE_UID='+uid,'-e','SITE_INSTANCE='+r['instance'],
            '-e','SITE_CATEGORY='+r['category'],'-e','SITE_SLOT_INDEX='+str(r['slot']),
            '--entrypoint','node',BROWSER,'/home/pptruser/ltd-c-v2/capture_dom.js',
            r['url'],f'/audit/{stem}.html','9050',CHROME,'90']
        try:
            b=call(browser,115)
            result['node_exit']=b.returncode
            (diag/f'{stem}.browser.log').write_text((b.stdout+'\n'+b.stderr)[-18000:])
        except subprocess.TimeoutExpired:
            result['node_exit']=124
            (diag/f'{stem}.browser.log').write_text('BROWSER_TIMEOUT_115s\n')
            call(['docker','rm','-f',navname],15)
    finally:
        call(['docker','kill','--signal=SIGINT',capname],20)
        call(['docker','wait',capname],20)
        logs=call(['docker','logs',capname],20)
        (diag/f'{stem}.tcpdump.log').write_text((logs.stdout+'\n'+logs.stderr)[-5000:])
        call(['docker','rm','-f',capname],20)
    return evaluate(result,out)

def phase_worker(inst,rows,phase,diag,out):
    results=[]
    for r in rows:
        try:
            result=visit(r,inst,phase,diag,out)
        except Exception as exc:
            result={**r,'phase':phase,'tor_instance':inst,'qa_valid':False,
                    'status':'QA_INFRA_ERROR','errors':[type(exc).__name__+': '+str(exc)[:300]]}
        results.append(result)
        print('FINAL29_SITE',phase,r['domain'],inst,result['status'],
              result.get('http_status'),'PCAP_VALID',result['qa_valid'],flush=True)
    return results

def run_phase(pending,phase,diag,out):
    groups=defaultdict(list)
    for r in pending:
        tor=r['instance'] if phase=='original' else ALT[r['instance']]
        groups[tor].append(r)
    completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(phase_worker,tor,items,phase,diag,out) for tor,items in groups.items()]
        for future in concurrent.futures.as_completed(futures):
            completed.extend(future.result())
            print('FINAL29_PROGRESS',phase,len(completed),'/',len(pending),flush=True)
    return completed

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan-only',action='store_true')
    args=ap.parse_args()
    plan=canonical_plan()
    print('FINAL29_PLAN',len(plan),'nonfrozen_failed_canonical_slots',flush=True)
    print('FINAL29_BASELINE=67/120; frozen_valid=41/65; nonfrozen_valid=26/55',flush=True)
    if args.plan_only:
        print('FINAL29_PLAN_ONLY=PASS; NO_BROWSER_NO_PCAP',flush=True)
        return 0
    lock=BASE/'diagnostics/.final29_recovery.lock'
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if list((BASE/'diagnostics').glob('qa_final29_*/final29_summary.json')):
            raise RuntimeError('Final29 QA already completed: inspect existing report, do not rerun')
        preflight()
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run_id=f'qa_final29_{stamp}_{os.getpid()}'
        diag=BASE/'diagnostics'/run_id
        qa=BASE/'qa_only'/run_id
        diag.mkdir(parents=True)
        out=qa/'outputs';out.mkdir(parents=True)
        print('FINAL29_DIAGNOSTICS',diag,flush=True)
        print('FINAL29_QA_FILES',out,flush=True)
        print('SCIENTIFIC_CAPTURE=NOT_AUTHORIZED',flush=True)
        chown_dir(out,10042,10042)
        results=[]
        try:
            first=run_phase(plan,'original',diag,out)
            print('FINAL29_PHASE1_COMPLETE',sum(x['qa_valid'] for x in first),'/',len(first),flush=True)
            unsuccessful=[r for r in plan if not any(x['qa_valid'] and x['url']==r['url'] and x['instance']==r['instance'] and x['slot']==r['slot'] for x in first)]
            second=run_phase(unsuccessful,'alternate',diag,out) if unsuccessful else []
            results=first+second
        finally:
            chown_dir(out,os.getuid(),os.getgid())
        success={}
        for x in results:
            key=(x['instance'],x['category'],x['slot'])
            if x.get('qa_valid') and key not in success:
                success[key]=x
        counts=Counter(x['status'] for x in results)
        report={'classification':'QA_ONLY_NOT_SCIENTIFIC','baseline_qa120':str(SOURCE),
                'attempts':len(results),'target_slots':len(plan),'newly_valid':len(success),
                'nonfrozen_after_qa':26+len(success),'frozen_after_qa':47,
                'total_after_qa':73+len(success),
                'remaining_original_120_positions':47-len(success),
                'scientific_authorized':False,
                'primary_vantage':'original direct Tor container',
                'fallback_vantage':'other existing direct Tor container, one attempt maximum',
                'status_counts':dict(counts),'successful_slots':[list(k) for k in success],
                'records':results}
        path=diag/'final29_summary.json'
        path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print('FINAL29_NEW_VALID',len(success),'/29',flush=True)
        print('FINAL29_TOTAL_QA_VALID',str(73+len(success))+'/120',flush=True)
        print('FINAL29_FROZEN65_QA_VALID=47/65',flush=True)
        print('FINAL29_STATUS_COUNTS',dict(counts),flush=True)
        print('FINAL29_SUMMARY',path,flush=True)
        print('SCIENTIFIC_CAPTURE=NOT_AUTHORIZED',flush=True)
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        print('FINAL29_ABORTED',type(exc).__name__,str(exc)[:700],file=sys.stderr,flush=True)
        sys.exit(4)
