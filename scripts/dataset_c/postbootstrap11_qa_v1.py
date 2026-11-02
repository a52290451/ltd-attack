#!/usr/bin/env python3
"""Dataset C: bounded post-bootstrap retry, ONLY 11 sites tested before Tor was ready.

Uses same pinned Puppeteer / collector V2.5 / canonical URLs via existing
alternate_direct_tor19_qa_v1; no PCAP, sudo, changes to frozen data, or Tor resets.
"""
import argparse
import concurrent.futures
from collections import Counter
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import alternate_direct_tor19_qa_v1 as base

ALT19=base.BASE/"diagnostics/qa_alt_tor19_20261102T103354Z_3873/alt19_summary.json"
EXPECTED_ALT=Counter({"NAVIGATION_ERROR":7,"BLOCKED":4})

def canonical_11():
    plan=base.build_plan()  # original canonical 19 and QA120 proven URLs
    previous=base.load(ALT19)
    rows=previous.get("records",[])
    if previous.get("attempts")!=19 or previous.get("navigable_candidates")!=1 or len(rows)!=19:
        raise RuntimeError("Alternate19 evidence mismatch")
    prev={base.key(r):r for r in rows}
    if len(prev)!=19:
        raise RuntimeError("Alternate19 repeated scientific slot")
    returned=[]
    for r in plan:
        key=base.key(r)
        p=prev.get(key)
        if not p or r["url"]!=p["url"] or r["alternate_instance"]!=p["alternate_instance"]:
            raise RuntimeError("Original URL / alternate Tor provenance mismatch for "+str(key))
        if r["alternate_instance"] in ("inst1","inst3"):
            if p["status"] not in ("NAVIGATION_ERROR","BLOCKED") or p["qa_navigation_candidate"]:
                raise RuntimeError("Unexpected prior result "+str(key))
            returned.append(r)
    if len(returned)!=11 or Counter(prev[base.key(r)]["status"] for r in returned)!=EXPECTED_ALT:
        raise RuntimeError("Expected exactly 7 timeouts and 4 blocks from Tor that had not fully bootstrapped")
    return returned

def verify_bootstrap_after_restart():
    for i in (1,3):
        name=f"ltd-c-tor-inst{i}"
        started=base.run(["docker","inspect","-f","{{.State.StartedAt}}",name],check=True).stdout.strip()
        logs=base.run(["docker","logs","--since",started,name],timeout=20)
        combined=logs.stdout+"\n"+logs.stderr
        if "Bootstrapped 100% (done)" not in combined:
            raise RuntimeError(f"Tor {name} has not reached bootstrap 100% since latest start")
        print("POSTBOOT11_TOR_READY",name,"100%",flush=True)

def diagnostic_results(attempts,diag,qa):
    results=[]
    for r in attempts:
        stem=r["stem"]
        j=qa/(stem+".json")
        h=qa/(stem+".html")
        try: d=json.loads(j.read_text())
        except (OSError,ValueError):d={}
        consent=d.get("consent") or {}
        good=(r["node_exit"]==0 and d.get("status")=="OK"
              and d.get("requested_url")==r["url"]
              and d.get("http_status") in range(200,400)
              and int(d.get("visible_chars") or 0)>150
              and h.is_file() and h.stat().st_size>512
              and (not d.get("consent_visible") or consent.get("action")=="clicked"))
        item={**r,"status":d.get("status","NO_METADATA"),"http_status":d.get("http_status"),
              "consent_action":consent.get("action"),"errors":d.get("errors",[]),
              "qa_navigation_candidate":bool(good),"html_bytes":h.stat().st_size if h.exists() else 0}
        results.append(item)
        print("POSTBOOT11_SITE",r["domain"],"ALT",r["alternate_instance"],
              "STATUS",item["status"],"HTTP",item["http_status"],
              "NAV_CANDIDATE",bool(good),"ERRORS",str(item["errors"])[:140],flush=True)
    candidates=sum(x["qa_navigation_candidate"] for x in results)
    report={
        "classification":"QA_ONLY_AFTER_TOR_BOOTSTRAP_100_NO_PCAP",
        "original_ab_frozen_untouched":True,"scientific_capture_authorized":False,
        "pcap_collected":False,"attempts":len(results),
        "navigation_candidates":candidates,
        "status_counts":dict(Counter(x["status"] for x in results)),
        "original_alt19_summary":str(ALT19),"records":results
    }
    target=diag/"postbootstrap11_summary.json"
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("POSTBOOT11_NAVIGABLE",str(candidates)+"/11",flush=True)
    print("POSTBOOT11_STATUS_COUNTS",report["status_counts"],flush=True)
    print("POSTBOOT11_SUMMARY",target,flush=True)
    print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED",flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan-only",action="store_true")
    args=parser.parse_args()
    plan=canonical_11()
    print("POSTBOOT11_PLAN",len(plan),"7_previous_timeouts","4_previous_blocks",flush=True)
    if args.plan_only:
        print("POSTBOOT11_PLAN_ONLY=PASS; NO_BROWSER_NO_PCAP")
        return 0
    lockfile=base.BASE/"diagnostics/.postbootstrap11.lock"
    with lockfile.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if list((base.BASE/"diagnostics").glob("qa_postbootstrap11_*/postbootstrap11_summary.json")):
            raise RuntimeError("Completed postbootstrap11 exists; no repeat")
        base.prerequisites()
        verify_bootstrap_after_restart()
        name="qa_postbootstrap11_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+str(os.getpid())
        diag=base.BASE/"diagnostics"/name
        out=base.BASE/"qa_only"/name/"outputs"
        diag.mkdir(parents=True)
        out.mkdir(parents=True)
        base.chown_output(out,10042,10042)
        print("POSTBOOT11_ROOT",diag,flush=True)
        print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED",flush=True)
        completed=[]
        try:
            groups={i:[r for r in plan if r["alternate_instance"]==i] for i in ("inst1","inst3")}
            def worker(items):
                res=[]
                for r in items:
                    res.append(base.visit(r,diag,out))
                    print("POSTBOOT11_PROGRESS",len(res),"/",len(items),
                          "worker="+r["alternate_instance"],flush=True)
                return res
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(worker,items) for items in groups.values()]
                for future in concurrent.futures.as_completed(futures):
                    completed.extend(future.result())
        finally:
            base.chown_output(out,os.getuid(),os.getgid())
        if len(completed)!=11:raise RuntimeError("Incomplete 11-site experiment")
        diagnostic_results(completed,diag,out)
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:
        print("POSTBOOT11_INCONCLUSIVE",type(exc).__name__,str(exc)[:500],file=sys.stderr)
        sys.exit(4)
