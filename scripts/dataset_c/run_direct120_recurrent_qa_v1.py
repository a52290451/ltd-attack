#!/usr/bin/env python3
"""Dataset C direct Tor recurring QA: 120 immutable slots, three parallel workers.

Preserves all A/B, site-list, Docker Tor configuration and scientific artifacts.
Periodic runs are QA_ONLY until NTP, original A/B parity, isolated PCAP vantage,
storage plan and formal release gates have independently passed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

import final29_recovery_qa_v1 as capture

BASE=capture.BASE
APP=capture.APP
CANONICAL=capture.SOURCE
REPORTS=BASE/"diagnostics"
CAPTURES=BASE/"qa_only"
OPERATION=Path.home()/"ltd-ops"/"dataset-c-direct120"
MARKER="LTD_C_DIRECT120_QA"
EXPECTED_PAIRS={
 "inst1":("belleza","turismo"),
 "inst2":("bancos","salud"),
 "inst3":("deportes","politica"),
}
LOCK=REPORTS/".direct120_recurrent_qa.lock"
MIN_FREE_BYTES=120_000_000_000

def require(ok,why):
    if not ok:raise RuntimeError("BLOCKED: "+why)

def key(row):
    return row["instance"],row["category"],int(row["slot"])

def inventory():
    source=json.loads(CANONICAL.read_text())
    records=source.get("records",[])
    require(len(records)==120 and source.get("inventory_distinct_urls")==118,
            "canonical120 inventory not 120 positions and 118 unique URLs")
    require(len({key(x) for x in records})==120,"duplicate slot identity")
    require(sum(bool(r.get("frozen_65")) for r in records)==65,"not 65 frozen-class slot labels")
    bykey={key(x):x for x in records}
    plan=[]
    all_urls=[]
    for inst,categories in EXPECTED_PAIRS.items():
        for cat in categories:
            f=Path.home()/"ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2/site_lists"/inst/(cat+".txt")
            require(f.is_file() and not f.is_symlink(),"canonical URL list absent")
            urls=[s.strip() for s in f.read_text().splitlines()
                  if s.strip() and not s.lstrip().startswith("#")]
            require(len(urls)==20,"canonical category not exactly 20 URLs: "+str(f))
            for position,url in enumerate(urls,1):
                sample=bykey[(inst,cat,position)]
                require(sample["url"]==url,"URL changed from original 120-source inventory")
                parsed=urlsplit(url)
                require(parsed.scheme in ("http","https") and bool(parsed.hostname),
                        "invalid original URL")
                all_urls.append(url)
                plan.append({
                    "instance":inst,"category":cat,"slot":position,"url":url,
                    "domain":parsed.hostname,"frozen_65":bool(sample.get("frozen_65")),
                    "original_slot_identity":f"{inst}/{cat}/{position}"
                })
    require(len(plan)==120 and len(set(all_urls))==118,
            "canonical 120 slot / 118 URL invariants changed")
    return plan

def clock_info():
    p=subprocess.run(["timedatectl","show","-p","NTPSynchronized","--value"],
                     capture_output=True,text=True,timeout=8)
    return p.stdout.strip() if p.returncode==0 else "unknown"

def active_old_cron(lines):
    excludes=(MARKER,"#")
    legacy=("LTD_C_THREE_QA","LTD_C_QA_HOURLY",
            "three_hourly_qa.sh --once","run_hourly_qa.sh --once")
    return [s for s in lines if s.strip() and not s.lstrip().startswith("#")
            and any(mark in s for mark in legacy)]

def cron_lines():
    p=subprocess.run(["crontab","-l"],capture_output=True,text=True,timeout=10)
    if p.returncode not in (0,1):raise RuntimeError("cannot read user crontab")
    return p.stdout.splitlines()

def cronline():
    python=sys.executable
    script=Path(__file__).resolve()
    # Every hour. fcntl keeps one batch at a time; late rounds simply skip.
    return f"0 * * * * {python} {script} --once >> {OPERATION}/scheduler.log 2>&1 # {MARKER}"

def install():
    OPERATION.mkdir(parents=True,exist_ok=True)
    lines=cron_lines()
    require(not active_old_cron(lines),
            "legacy Dataset C cron exists; inspect before installing the new runner")
    require(shutil.which("crontab") is not None,"crontab unavailable")
    service=subprocess.run(["systemctl","is-active","cron"],
                           capture_output=True,text=True,timeout=10)
    require(service.stdout.strip()=="active","cron service not active")
    existing=[s for s in lines if MARKER in s and not s.lstrip().startswith("#")]
    require(len(existing)<=1,"duplicate recurring C entries")
    if existing:
        require(existing[0]==cronline(),"different recurring C job already installed")
        print("DIRECT120_CRON_ALREADY_INSTALLED=YES",flush=True)
        return
    stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup=OPERATION/("crontab_before_direct120_"+stamp+".txt")
    backup.write_text("\n".join(lines)+("\n" if lines else ""))
    backup.chmod(0o600)
    new="\n".join(lines+[cronline()])+"\n"
    p=subprocess.run(["crontab","-"],input=new,text=True,
                     capture_output=True,timeout=15)
    require(p.returncode==0,"could not install recurring QA cron: "+p.stderr[-200:])
    print("DIRECT120_CRON_HOURLY=INSTALLED")
    print("DIRECT120_CRON_BACKUP",backup)
    print("DIRECT120_CRON_ENTRY",cronline())

def verify_tor_started():
    for i in (1,2,3):
        name=f"ltd-c-tor-inst{i}"
        p=subprocess.run(["docker","inspect","-f","{{.State.StartedAt}}",name],
                         capture_output=True,text=True,timeout=12)
        require(p.returncode==0,name+" not found")
        logs=subprocess.run(["docker","logs","--since",p.stdout.strip(),name],
                            capture_output=True,text=True,timeout=25)
        require("Bootstrapped 100% (done)" in logs.stdout+logs.stderr,
                name+" has not bootstrapped to 100% since latest restart")

def visit_worker(inst,items,run,output):
    results=[]
    for r in items:
        try:
            item=capture.visit(r,inst,"direct120",run,output)
        except Exception as e:
            item={**r,"phase":"direct120","tor_instance":inst,
                  "status":"QA_INFRA_ERROR","qa_valid":False,
                  "node_exit":None,"error":str(e)[:400]}
        results.append(item)
        print("DIRECT120_VISIT",inst,r["category"],r["slot"],
              "NODE_EXIT",item.get("node_exit"),flush=True)
    return results

def one_round():
    plan=inventory()
    REPORTS.mkdir(parents=True,exist_ok=True)
    CAPTURES.mkdir(parents=True,exist_ok=True)
    with LOCK.open("a+") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print("DIRECT120_ALREADY_RUNNING_SKIP=YES")
            return 0
        capture.preflight()
        verify_tor_started()
        require(capture.free_bytes()>=MIN_FREE_BYTES,"disk reserve missing")
        clock=clock_info()
        stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name=f"direct120_qa_{stamp}_{os.getpid()}"
        diagnostic=REPORTS/name
        output=CAPTURES/name/"outputs"
        diagnostic.mkdir(parents=True)
        output.mkdir(parents=True)
        print("DIRECT120_RUN",name,flush=True)
        print("DIRECT120_TOR=3_EXISTING_DIRECT_INSTANCES")
        print("DIRECT120_NTP",clock)
        print("DIRECT120_CLOCK_VALID_FOR_SCIENCE=NO")
        print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED")
        capture.chown_dir(output,10042,10042)
        allresults=[]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                future={
                    pool.submit(visit_worker,inst,[r for r in plan if r["instance"]==inst],diagnostic,output):inst
                    for inst in EXPECTED_PAIRS
                }
                for completed in concurrent.futures.as_completed(future):
                    finished=completed.result()
                    allresults.extend(finished)
                    print("DIRECT120_WORKER_DONE",future[completed],len(finished),flush=True)
        finally:
            # Browser writes as 10042. Reconcile only after returning ownership;
            # earlier final29 mistakenly read all 58 metadata files too soon.
            capture.chown_dir(output,os.getuid(),os.getgid())
        verified=[]
        for r in allresults:
            if r.get("stem"):
                item=capture.evaluate(r,output)
            else:
                item=r
            verified.append(item)
        grouped={key(r):r for r in verified}
        require(len(grouped)==120 and len(verified)==120,
                "incomplete / duplicate 120-slot round")
        qa_good=sum(bool(r.get("qa_valid")) for r in verified)
        frozen_good=sum(bool(r.get("qa_valid")) and bool(r.get("frozen_65"))
                        for r in verified)
        statuses=Counter(r.get("status","UNKNOWN") for r in verified)
        report={
            "classification":"C_DIRECT_120_RECURRENT_QA_ONLY",
            "run_id":name,"utc_unverified":stamp,
            "host_ntp_synchronized":clock,
            "scientific_capture_authorized":False,
            "future_b_or_ab_accessed":False,
            "canonical_positions":120,"unique_urls":118,
            "frozen_class_labels":65,
            "attempts":len(verified),"pcap_qa_valid":qa_good,
            "frozen65_qa_valid":frozen_good,
            "status_counts":dict(statuses),
            "original_qa_files":str(output),
            "collector_variant":"V2.5_QA_ONLY",
            "image":capture.BROWSER,
            "tor_mode":"preexisting_direct_no_bridges",
            "capture_method":"tcpdump on existing Tor namespace; includes possible background Tor traffic",
            "protocol_parity_ab_approved":False,
            "records":verified
        }
        outfile=diagnostic/"direct120_summary.json"
        outfile.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        print("DIRECT120_ATTEMPTS",len(verified),"/120")
        print("DIRECT120_QA_VALID",qa_good,"/120")
        print("DIRECT120_FROZEN65_QA_VALID",frozen_good,"/65")
        print("DIRECT120_STATUS_COUNTS",dict(statuses))
        print("DIRECT120_SUMMARY",outfile)
        print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED")
        return 0

def launch_now():
    plan=inventory()
    require(len(plan)==120,"canonical 120 plan is invalid")
    OPERATION.mkdir(parents=True,exist_ok=True)
    logfile=OPERATION/("manual_start_"+dt.datetime.now(dt.timezone.utc).
                      strftime("%Y%m%dT%H%M%SZ")+".log")
    f=logfile.open("w")
    p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--once"],
                       cwd=Path(__file__).resolve().parents[2],
                       stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                       start_new_session=True,close_fds=True)
    f.close()
    print("DIRECT120_MANUAL_START_PID",p.pid)
    print("DIRECT120_MANUAL_LOG",logfile)
    print("DIRECT120_STATUS_COMMAND",sys.executable,Path(__file__).resolve(),"--status")

def status():
    matches=sorted(REPORTS.glob("direct120_qa_*/direct120_summary.json"),
                   key=lambda p:p.stat().st_mtime,reverse=True)
    lines=cron_lines()
    print("DIRECT120_CRON_INSTALLED",
          any(MARKER in x and not x.lstrip().startswith("#") for x in lines))
    print("DIRECT120_MOST_RECENT_ROUND",matches[0] if matches else "NONE_YET")
    if matches:
        data=json.loads(matches[0].read_text())
        print("DIRECT120_LATEST_VALID",data.get("pcap_qa_valid"),"/120")
        print("DIRECT120_LATEST_STATUS_COUNTS",data.get("status_counts"))
    logs=sorted(OPERATION.glob("manual_start_*.log"),
                key=lambda p:p.stat().st_mtime,reverse=True)
    if logs:
        print("DIRECT120_START_LOG",logs[0])
        tail=logs[0].read_text(errors="replace").splitlines()[-10:]
        print("\n".join(tail))
    print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED")

def main():
    ap=argparse.ArgumentParser()
    modes=ap.add_mutually_exclusive_group(required=True)
    for m in ("plan-only","once","install","enable-now","start-now","status"):
        modes.add_argument("--"+m,action="store_true")
    args=ap.parse_args()
    if args.status:
        status();return 0
    plan=inventory()
    print("DIRECT120_INVENTORY",len(plan),"slots",len({r["url"] for r in plan}),
          "URLs",sum(bool(r["frozen_65"]) for r in plan),"frozen")
    if args.plan_only:
        print("DIRECT120_PLAN_ONLY=PASS_NO_CAPTURE")
        return 0
    if args.install or args.enable_now:
        install()
    if args.start_now or args.enable_now:
        launch_now()
    elif args.once:
        return one_round()
    print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED")
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as error:
        print("DIRECT120_BLOCKED",type(error).__name__,str(error)[:600],
              file=sys.stderr,flush=True)
        sys.exit(4)
