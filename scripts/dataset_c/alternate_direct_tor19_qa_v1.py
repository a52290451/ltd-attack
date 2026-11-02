#!/usr/bin/env python3
"""One bounded QA-only access comparison for 19 failed frozen classes.

Each site is visited ONCE through another already-running direct Tor instance.
Identical pinned browser, collector V2.5, URL, and consent policy.
No PCAP, no NEWNYM, no CAPTCHA/auth bypass, no scientific data, no sudo.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from collections import Counter
from urllib.parse import urlsplit

BASE=Path("/data/ltd-c-qa-bsierra/C_DIRECT_V1")
SOURCE=BASE/"diagnostics/qa_remaining20_pcap_20261102T102017Z_2807883/remaining20_pcap_summary.json"
CANONICAL=BASE/"diagnostics/pinned_qa108_20261101T211133Z_465267/qa120_coverage_summary.json"
APP=Path.home()/"ltd-ops/collector-qa-recovery-v25"
IMAGE="ghcr.io/puppeteer/puppeteer@sha256:ffdb72533aa06e9adf99d1056fbe5b79c915869f6c2e7de8fa1f1ce8ffda0790"
HELPER_IMAGE="tor_1site_noise:final"
CHROME="/home/pptruser/.cache/puppeteer/chrome/linux-145.0.7632.76/chrome-linux64/chrome"
EXPECTED="0150884d2b0faf568289660ffce31d0536b737b3934f7bc37118c0f34fbec774"
ORIGINAL_QUALITY="2e95a90ea77115320d37aaff0ae14157727f7b47deb28a9b18de7237e84f59e1"
ALTERNATE={"inst1":"inst2","inst2":"inst3","inst3":"inst1"}
PENDING_STATES=Counter({"BLOCKED":16,"HTTP_ERROR":2,"NAVIGATION_ERROR":1})

def run(args,timeout=30,check=False):
    return subprocess.run(args,capture_output=True,text=True,timeout=timeout,check=check)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p):
    if not p.is_file() or p.is_symlink():
        raise RuntimeError("Missing or symlink source: "+str(p))
    return json.loads(p.read_text())

def key(r):
    return (str(r["instance"]),str(r["category"]),int(r["slot"]))

def build_plan():
    data=load(SOURCE);full=load(CANONICAL)
    rows=data.get("visits",[])
    if (data.get("attempts"),data.get("valid"),len(rows))!=(20,1,20):
        raise RuntimeError("QA20 frozen result mismatch")
    original=full.get("records",[])
    if len(original)!=120 or full.get("inventory_distinct_urls")!=118:
        raise RuntimeError("QA120 inventory incomplete")
    indexed={key(r):r for r in original}
    if len(indexed)!=120:
        raise RuntimeError("Duplicate canonical inventory keys")
    failed=[r for r in rows if not r.get("qa_valid")]
    if len(failed)!=19 or Counter(r["status"] for r in failed)!=PENDING_STATES:
        raise RuntimeError("Expected 16 blocked + 2 HTTP error + 1 timeout")
    if len(set(key(r) for r in failed))!=19:
        raise RuntimeError("Duplicate missing slot keys")
    plan=[]
    for r in failed:
        k=key(r)
        url=r["url"]
        c=indexed.get(k)
        if not c or not c.get("frozen_65") or c["url"]!=url:
            raise RuntimeError("Frozen-class URL mismatch: "+str(k))
        host=urlsplit(url).hostname
        if not host:
            raise RuntimeError("Malformed original URL")
        plan.append({
            "instance":k[0],"category":k[1],"slot":k[2],
            "url":url,"domain":host,"initial_status":r["status"],
            "initial_http":r.get("http_status"),
            "alternate_instance":ALTERNATE[k[0]]
        })
    if len(plan)!=19 or len({r["domain"] for r in plan})!=19:
        raise RuntimeError("Expected exactly 19 unique missing sites")
    return plan

def prerequisites():
    if (digest(APP/"capture_dom.js")!=EXPECTED or
        digest(APP/"quality.js")!=ORIGINAL_QUALITY):
        raise RuntimeError("Verified QA V2.5 collector hash mismatch")
    for image in (IMAGE,HELPER_IMAGE):
        run(["docker","image","inspect",image],timeout=20,check=True)
    active=run(["docker","ps","--format","{{.Names}}"],timeout=15,check=True).stdout.splitlines()
    if any(name.startswith(("ltd-c-pcap-qa-","ltd-c-pinned-nav-","ltd-c-alt19-nav-")) for name in active):
        raise RuntimeError("Another C capture or alternate QA is running")
    for i in (1,2,3):
        name="ltd-c-tor-inst"+str(i)
        if run(["docker","inspect","-f","{{.State.Running}}",name],check=True).stdout.strip()!="true":
            raise RuntimeError(name+" is stopped")
        if run(["docker","exec",name,"grep","-qx","UseBridges 0","/tmp/direct-torrc"]).returncode:
            raise RuntimeError(name+" is no longer direct Tor")
    check=run(["docker","run","--rm","--network","none","--cap-drop","ALL",
        "--security-opt","no-new-privileges",
        "--mount",f"type=bind,src={APP},dst=/home/pptruser/ltd-c-v2,readonly",
        "--entrypoint","node",IMAGE,"--check",
        "/home/pptruser/ltd-c-v2/capture_dom.js"],timeout=35)
    if check.returncode:
        raise RuntimeError("Pinned browser cannot read QA V2.5 code: "+check.stderr[-400:])
    if os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize<120_000_000_000:
        raise RuntimeError("Less than 120 GB free on C filesystem")

def chown_output(path,uid,gid):
    r=run(["docker","run","--rm","--network","none","--user","0",
        "--mount",f"type=bind,src={path},dst=/audit",
        "--entrypoint","chown",HELPER_IMAGE,"-R",f"{uid}:{gid}","/audit"],timeout=40)
    if r.returncode:
        raise RuntimeError("QA-only Docker ownership operation failed: "+r.stderr[-250:])

def visit(record,diag,output):
    rid=str(uuid.uuid4())
    stem="_".join((record["instance"],record["category"],str(record["slot"]),rid))
    tor=record["alternate_instance"]
    container="ltd-c-alt19-nav-"+str(os.getpid())+"-"+rid[:8]
    command=[
        "docker","run","--init","--rm","--name",container,
        "--network","container:ltd-c-tor-"+tor,
        "--cap-drop","ALL","--security-opt","no-new-privileges","--shm-size","1g",
        "--mount",f"type=bind,src={APP},dst=/home/pptruser/ltd-c-v2,readonly",
        "--mount",f"type=bind,src={output},dst=/audit",
        "-e","NODE_PATH=/home/pptruser/node_modules",
        "-e","CHROME_NO_SANDBOX=1","-e","SESSION_POLICY=fresh",
        "-e","CONSENT_POLICY=accept_all",
        "-e","CAPTURE_UID="+rid,
        "-e","SITE_INSTANCE="+record["instance"],
        "-e","SITE_CATEGORY="+record["category"],
        "-e","SITE_SLOT_INDEX="+str(record["slot"]),
        "--entrypoint","node",IMAGE,
        "/home/pptruser/ltd-c-v2/capture_dom.js",
        record["url"],"/audit/"+stem+".html","9050",CHROME,"90"
    ]
    code=None
    try:
        result=run(command,timeout=115)
        code=result.returncode
        log=(result.stdout or "")+"\n"+(result.stderr or "")
    except subprocess.TimeoutExpired:
        code=124
        log="QA browser Docker invocation timed out after 115 seconds"
        run(["docker","rm","-f",container],timeout=15)
    (diag/(stem+".log")).write_text(log[-12000:])
    return {**record,"capture_id":rid,"stem":stem,"node_exit":code}

def summarize(rows,diag,output):
    results=[]
    for r in rows:
        stem=r["stem"]
        metadata=output/(stem+".json")
        html=output/(stem+".html")
        try:
            d=json.loads(metadata.read_text())
        except (OSError,ValueError):
            d={}
        consent=d.get("consent") or {}
        valid=(
            r["node_exit"]==0 and d.get("status")=="OK"
            and d.get("requested_url")==r["url"]
            and d.get("http_status") in range(200,400)
            and int(d.get("visible_chars") or 0)>150
            and html.is_file() and html.stat().st_size>512
            and (not d.get("consent_visible") or consent.get("action")=="clicked")
        )
        result={**r,"status":d.get("status","NO_METADATA"),
                "http_status":d.get("http_status"),"errors":d.get("errors",[]),
                "consent_action":consent.get("action"),"qa_navigation_candidate":bool(valid),
                "html_bytes":html.stat().st_size if html.exists() else 0}
        results.append(result)
        print("ALT19_SITE",r["domain"],"ORIG",r["initial_status"],r["initial_http"],
              "ALT",r["alternate_instance"],"NEW",result["status"],result["http_status"],
              "NAV_CANDIDATE",bool(valid),"ERRORS",str(result["errors"])[:130],flush=True)
    count=sum(r["qa_navigation_candidate"] for r in results)
    report={
        "classification":"QA_ONLY_ALTERNATE_EXISTING_DIRECT_TOR_NO_PCAP",
        "attempts":len(results),"navigable_candidates":count,
        "not_recovered":len(results)-count,
        "status_counts":dict(Counter(r["status"] for r in results)),
        "source_qa20":str(SOURCE),"frozen_ab_unmodified":True,
        "scientific_capture_authorized":False,"pcap_collected":False,
        "methodology":"One alternate existing direct Tor instance per site; same pinned Chrome/V2.5, fresh session, original URL",
        "records":results
    }
    p=diag/"alt19_summary.json"
    p.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("ALTTOR19_NAVIGABLE",str(count)+"/19")
    print("ALTTOR19_STATUS_COUNTS",report["status_counts"])
    print("ALTTOR19_SUMMARY",p)
    print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan-only",action="store_true")
    args=parser.parse_args()
    plan=build_plan()
    print("ALTTOR19_PLAN",len(plan),"sites; 16 blocked, 2 HTTP errors, 1 timeout")
    print("ALTTOR19_ALTERNATE_INSTANCES inst1->inst2, inst2->inst3, inst3->inst1")
    print("ALTTOR19_ORIGINAL_URLS_AND_FROZEN_LABELS_VERIFIED=YES")
    if args.plan_only:
        print("ALTTOR19_PLAN_ONLY=PASS; NO_NAVIGATION_NO_PCAP")
        return 0
    lock_path=BASE/"diagnostics/.alt_tor19.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another alternate Tor QA run holds the lock")
        if list((BASE/"diagnostics").glob("qa_alt_tor19_*/alt19_summary.json")):
            raise RuntimeError("An earlier completed alternate Tor QA19 exists; do not repeat")
        prerequisites()
        name="qa_alt_tor19_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+str(os.getpid())
        diag=BASE/"diagnostics"/name
        qa=BASE/"qa_only"/name
        diag.mkdir(parents=True)
        (qa/"outputs").mkdir(parents=True)
        output=qa/"outputs"
        print("ALTTOR19_DIAGNOSTICS",diag)
        print("ALTTOR19_BROWSER_OUTPUT",qa)
        print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED")
        chown_output(output,10042,10042)
        try:
            buckets={inst:[] for inst in ALTERNATE}
            for r in plan:
                buckets[r["alternate_instance"]].append(r)
            def worker(rows):
                return [visit(r,diag,output) for r in rows]
            results=[]
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures=[executor.submit(worker,rows) for rows in buckets.values()]
                for f in concurrent.futures.as_completed(futures):
                    part=f.result()
                    results.extend(part)
                    print("ALTTOR19_PROGRESS",len(results),"/19",flush=True)
        finally:
            chown_output(output,os.getuid(),os.getgid())
        if len(results)!=19:raise RuntimeError("Missing QA results")
        summarize(results,diag,output)
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:
        print("ALTTOR19_INCONCLUSIVE",type(exc).__name__,str(exc)[:500],file=sys.stderr)
        sys.exit(4)
