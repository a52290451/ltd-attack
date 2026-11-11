#!/usr/bin/env python3
"""Dataset C QA half-split: preserve evidence, diagnose failed Tor runs, optionally pause QA cron.

No capture, no restart, no delete. --pause removes ONLY the two expected
Dataset-C QA halfsplit cron lines, with a private timestamped backup.
Other cron entries are preserved verbatim. Do not use to approve scientific C.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

HOME=Path.home()
OPS=HOME/"ltd-ops/dataset-c-direct120"
MARKERS=("LTD_C_DIRECT120_HALFSPLIT_QA_FIRST",
         "LTD_C_DIRECT120_HALFSPLIT_QA_SECOND")
CONTAINERS=("ltd-c-tor-inst1","ltd-c-tor-inst2","ltd-c-tor-inst3")

def run(cmd,timeout=15):
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
        return {"rc":p.returncode,"stdout":p.stdout.strip(),"stderr":p.stderr.strip()[:300]}
    except Exception as e:
        return {"error":type(e).__name__+": "+str(e)[:200]}

def report():
    print("DATASET_C_QA_DIAGNOSTIC_ONLY")
    print("UTC:",datetime.now(timezone.utc).isoformat())
    print("CRON:",json.dumps(run(["systemctl","is-active","cron"])))
    print("CLOCK:",json.dumps(run(["timedatectl","show","-p","NTP",
                                   "-p","NTPSynchronized","-p","TimeUSec"])))
    for n in CONTAINERS:
        value=run(["docker","inspect","--format",
                   "{{json .State}}",n])
        if value.get("rc")==0:
            try:
                x=json.loads(value["stdout"])
                selected={k:x.get(k) for k in ("Status","ExitCode",
                                             "OOMKilled","StartedAt",
                                             "FinishedAt","Error")}
                print("TOR_CONTAINER",n,json.dumps(selected,sort_keys=True))
            except Exception:print("TOR_INSPECT_ERROR",n,str(value)[:600])
        else:print("TOR_INSPECT_ERROR",n,str(value)[:600])
    for name in ("scheduler_split_first.log","scheduler_split_second.log"):
        p=OPS/name
        if p.is_file():
            lines=p.read_text(errors="replace").splitlines()
            events=[s for s in lines if any(v in s for v in
                   ("HALFSPLIT_PCAP_VALID","HALFSPLIT_BLOCKED",
                    "HALFSPLIT_STATUS_COUNTS","SCIENTIFIC_CAPTURE="))]
            print("QA_LOG",name,"last_markers",json.dumps(events[-8:]))
        else:print("QA_LOG_MISSING",str(p))
    print("KERNEL_OOM_READ_ONLY",json.dumps(run(
        ["journalctl","-k","--since","2026-11-07","--no-pager",
         "-g","oom-kill|Out of memory|Killed process"],25))[:4000])

def pause():
    listed=run(["crontab","-l"])
    if listed.get("rc")==1 and "no crontab" in listed.get("stderr","").lower():
        print("QA_CRON_ALREADY_ABSENT");return
    if listed.get("rc")!=0:
        raise RuntimeError("Cannot read user's crontab: "+str(listed))
    orig=listed["stdout"]
    lines=orig.splitlines()
    selected=[l for l in lines if not l.lstrip().startswith("#") and
              any(m in l for m in MARKERS)]
    found={m:sum(m in l for l in selected) for m in MARKERS}
    if found=={m:0 for m in MARKERS}:
        print("QA_CRON_ALREADY_PAUSED");return
    if found!={m:1 for m in MARKERS} or len(selected)!=2:
        raise RuntimeError("Unexpected Dataset C QA cron entries: "+str(found))
    for line in selected:
        if ("/scripts/dataset_c/run_direct120_halfsplit_qa_v2.py" not in line
            or not any(x in line for x in ("--once-first","--once-second"))):
            raise RuntimeError("Refusing to modify noncanonical cron line")
    backup=OPS/("crontab_before_QA_pause_"+datetime.now(timezone.utc)
                .strftime("%Y%m%dT%H%M%SZ")+".txt")
    if backup.exists():raise RuntimeError("backup collision")
    OPS.mkdir(parents=True,exist_ok=True)
    backup.write_text(orig+"\n")
    backup.chmod(0o600)
    remaining=[l for l in lines if l not in selected]
    payload="\n".join(remaining)+("\n" if remaining else "")
    r=subprocess.run(["crontab","-"],input=payload,text=True,capture_output=True,
                     timeout=15)
    if r.returncode:raise RuntimeError("crontab update failed; backup: "+str(backup))
    print("QA_CRON_PAUSED_ONLY_HALFSPLIT","removed",len(selected),
          "backup",str(backup))
    print("NO_TOR_OR_CAPTURE_CHANGED")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["diagnose","pause"],required=True)
    args=ap.parse_args()
    if args.stage=="diagnose":report()
    else:pause()

if __name__=="__main__":
    main()
