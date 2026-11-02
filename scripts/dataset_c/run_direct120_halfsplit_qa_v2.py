#!/usr/bin/env python3
"""Dataset C: 3 Tor instances x (20 sites at :00 + 20 at :30), QA only.

Existing direct Tor containers stay running. Cron is managed on the HOST.
At minute 00: inst1/belleza, inst2/bancos, inst3/deportes (60 visits).
At minute 30: inst1/turismo, inst2/salud, inst3/politica (60 visits).
Each worker visits its assigned 20 sequentially. The SAME lock as the old
120-once runner skips overlapping jobs (never contaminates PCAP concurrently).
No A/B or scientific release, no Docker/Tor configuration changes.
"""
from __future__ import annotations
import argparse
from collections import Counter
import concurrent.futures
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import final29_recovery_qa_v1 as capture
import run_direct120_recurrent_qa_v1 as legacy

MARKER="LTD_C_DIRECT120_HALFSPLIT_QA"
OLD_MARKER=legacy.MARKER
OPERATIONS=legacy.OPERATION
BATCHES={
 "first":{"inst1":"belleza","inst2":"bancos","inst3":"deportes"},
 "second":{"inst1":"turismo","inst2":"salud","inst3":"politica"}
}

def require(ok,reason):
    if not ok:raise RuntimeError("BLOCKED: "+reason)

def batch_plan(batch):
    allrows=legacy.inventory()
    selection=BATCHES[batch]
    rows=[r for r in allrows if selection.get(r["instance"])==r["category"]]
    count=Counter((r["instance"],r["category"]) for r in rows)
    require(len(rows)==60 and len(count)==3 and all(c==20 for c in count.values()),
            "split batch must have precisely 3 x 20 positions")
    require(len({legacy.key(x) for x in rows})==60,"duplicate positions within split")
    return rows

def schedule_rows():
    py=sys.executable
    script=Path(__file__).resolve()
    return [
        f"0 * * * * {py} {script} --once-first >> {OPERATIONS}/scheduler_split_first.log 2>&1 # {MARKER}_FIRST",
        f"30 * * * * {py} {script} --once-second >> {OPERATIONS}/scheduler_split_second.log 2>&1 # {MARKER}_SECOND"
    ]

def migrate_cron():
    # Only modify the exact existing C job; preserve every unrelated cron entry.
    require(OPERATIONS.is_dir(),"existing Dataset C operations directory absent")
    lines=legacy.cron_lines()
    old=[l for l in lines if OLD_MARKER in l and not l.lstrip().startswith("#")]
    new=[l for l in lines if MARKER in l and not l.lstrip().startswith("#")]
    require(len(old)<=1,"multiple old C jobs, manual inspection required")
    require(all(l==legacy.cronline() for l in old),
            "old C cron differs from the expected installed entry")
    expected=schedule_rows()
    require(not new or (len(new)==2 and sorted(new)==sorted(expected)),
            "different half-hour C cron already installed")
    require(not legacy.active_old_cron(lines),
            "other preexisting Dataset C legacy cron exists")
    service=subprocess.run(["systemctl","is-active","cron"],capture_output=True,text=True,timeout=10)
    require(service.stdout.strip()=="active","cron service is not active")
    if not old and len(new)==2:
        print("HALFSPLIT_CRON_ALREADY_INSTALLED=YES")
        for l in new:print("HALFSPLIT_CRON_ENTRY",l)
        return 0
    # Existing long 120-slot run is allowed to finish: both scripts use SAME lock.
    # Validate 120 frozen positions and local source before making cron persistent.
    require(len(legacy.inventory())==120,"canonical URL inventory incomplete")
    require(capture.digest(capture.APP/"capture_dom.js")==capture.PATCH_SHA,
            "QA consent collector hash mismatch")
    require(capture.digest(capture.APP/"quality.js")==capture.QUALITY_SHA,
            "original quality module hash mismatch")
    require(capture.free_bytes()>=120_000_000_000,"disk reserve below 120GB")
    now=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup=OPERATIONS/("crontab_before_halfsplit_"+now+".txt")
    backup.write_text("\n".join(lines)+("\n" if lines else ""))
    backup.chmod(0o600)
    clean=[l for l in lines if l not in old and l not in new]
    combined="\n".join(clean+expected)+"\n"
    result=subprocess.run(["crontab","-"],input=combined,text=True,
                          capture_output=True,timeout=15)
    require(result.returncode==0,"cannot install half-hour cron: "+result.stderr[-300:])
    actual=legacy.cron_lines()
    require(all(l in actual for l in expected) and not any(
            OLD_MARKER in l and not l.lstrip().startswith("#") for l in actual),
            "post-install crontab does not contain exactly new entries")
    print("HALFSPLIT_CRON_MIGRATION=PASS")
    print("HALFSPLIT_OLD_HOURLY_REMOVED",len(old))
    print("HALFSPLIT_CRON_BACKUP",backup)
    for line in expected:print("HALFSPLIT_CRON_ENTRY",line)
    print("HOST_CLOCK_NTP",legacy.clock_info())
    print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED")
    return 0

def one_worker(instance,rows,run,outputs):
    completed=[]
    for r in rows:
        try:item=capture.visit(r,instance,"halfsplit",run,outputs)
        except Exception as exc:
            item={**r,"phase":"halfsplit","tor_instance":instance,
                  "status":"QA_INFRA_ERROR","qa_valid":False,
                  "node_exit":None,"error":type(exc).__name__+": "+str(exc)[:320]}
        completed.append(item)
        print("HALFSPLIT_VISIT",instance,r["category"],r["slot"],
              "NODE_EXIT",item.get("node_exit"),flush=True)
    return completed

def one_batch(batch):
    plan=batch_plan(batch)
    legacy.REPORTS.mkdir(parents=True,exist_ok=True)
    legacy.CAPTURES.mkdir(parents=True,exist_ok=True)
    # Identical lock to legacy v1: don't overlap current 120-slot round.
    with legacy.LOCK.open("a+") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print("HALFSPLIT_SKIPPED_OVERLAP",batch,flush=True)
            print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED",flush=True)
            return 0
        capture.preflight()
        legacy.verify_tor_started()
        now=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rid=f"direct120_half_{batch}_{now}_{os.getpid()}"
        diagnostic=legacy.REPORTS/rid
        output=legacy.CAPTURES/rid/"outputs"
        diagnostic.mkdir(parents=True)
        output.mkdir(parents=True)
        print("HALFSPLIT_RUN",rid,flush=True)
        print("HALFSPLIT_EXPECTED 3_GROUPS_X_20 60_SLOTS",flush=True)
        print("HOST_NTP",legacy.clock_info(),flush=True)
        print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED",flush=True)
        capture.chown_dir(output,10042,10042)
        allresults=[]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures={
                    pool.submit(one_worker,i,[r for r in plan if r["instance"]==i],
                                diagnostic,output):i
                    for i in BATCHES[batch]
                }
                for f in concurrent.futures.as_completed(futures):
                    results=f.result()
                    allresults.extend(results)
                    print("HALFSPLIT_WORKER_FINISHED",futures[f],len(results),flush=True)
        finally:
            # Re-evaluate only AFTER all JSONs and PCAPs are chowned back.
            capture.chown_dir(output,os.getuid(),os.getgid())
        qa=[]
        for r in allresults:
            qa.append(capture.evaluate(r,output) if r.get("stem") else r)
        require(len(qa)==60 and len({legacy.key(x) for x in qa})==60,
                "split output not precisely 60 unique positions")
        statuses=dict(Counter(r.get("status","NO_STATUS") for r in qa))
        valid=sum(bool(x.get("qa_valid")) for x in qa)
        out={
            "classification":"C_DIRECT_120_HALFHOUR_QA_ONLY",
            "batch":batch,"run_id":rid,"attempts":60,
            "inst1_category":BATCHES[batch]["inst1"],
            "inst2_category":BATCHES[batch]["inst2"],
            "inst3_category":BATCHES[batch]["inst3"],
            "20_positions_per_instance":True,
            "valid_pcap_qa":valid,"status_counts":statuses,
            "scientific_capture_authorized":False,
            "host_time_unverified":legacy.clock_info(),
            "raw_outputs":str(output),"records":qa
        }
        report=diagnostic/"half60_summary.json"
        report.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
        print("HALFSPLIT_ATTEMPTS 60/60",flush=True)
        print("HALFSPLIT_PCAP_VALID",str(valid)+"/60",flush=True)
        print("HALFSPLIT_STATUS_COUNTS",statuses,flush=True)
        print("HALFSPLIT_REPORT",report,flush=True)
        print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED",flush=True)
        return 0

def status():
    lines=legacy.cron_lines()
    expected=schedule_rows()
    print("HALFSPLIT_CRON_READY",all(x in lines for x in expected))
    print("HALFSPLIT_OLD_CRON_PRESENT",any(
        OLD_MARKER in l and not l.lstrip().startswith("#") for l in lines))
    for l in lines:
        if MARKER in l or OLD_MARKER in l: print("CRON_ENTRY",l)
    for batch in ("first","second"):
        files=sorted(legacy.REPORTS.glob(f"direct120_half_{batch}_*/half60_summary.json"),
                     key=lambda x:x.stat().st_mtime,reverse=True)
        print("HALFSPLIT_LAST_REPORT",batch,files[0] if files else "NOT_COMPLETED")
        if files:
            data=json.loads(files[0].read_text())
            print("HALFSPLIT_LAST_RESULT",batch,data["attempts"],data["valid_pcap_qa"])
        log=OPERATIONS/("scheduler_split_"+batch+".log")
        if log.is_file(): print("HALFSPLIT_LOG",batch,log)
    print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED")

def main():
    ap=argparse.ArgumentParser()
    choices=ap.add_mutually_exclusive_group(required=True)
    for mode in ("plan-only","install","once-first","once-second","status"):
        choices.add_argument("--"+mode,action="store_true")
    args=ap.parse_args()
    if args.status:return status()
    a,b=batch_plan("first"),batch_plan("second")
    require(len({legacy.key(x) for x in a+b})==120,"halfsplit cannot reconstruct 120 original positions")
    if args.plan_only:
        for name,items in (("00",a),("30",b)):
            print("HALFSPLIT_PLAN",name,[(i,cat,20) for i,cat in BATCHES["first" if name=="00" else "second"].items()])
        print("HALFSPLIT_PLAN=PASS_120_SLOTS_6_CATEGORIES")
        print("HOST_CRON_NOT_CHANGED=YES")
        return 0
    if args.install:return migrate_cron()
    return one_batch("first" if args.once_first else "second")

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:
        print("HALFSPLIT_BLOCKED",type(exc).__name__,str(exc)[:600],file=sys.stderr,flush=True)
        sys.exit(4)
