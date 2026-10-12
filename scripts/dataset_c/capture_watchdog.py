#!/usr/bin/env python3
"""Non-destructive hourly QA watchdog.

Runs at hh:30 via a separate, user-owned cron. If there was no COMPLETE,
audited run for >=80 minutes and no active collectors, launch --once.
Never resets Tor containers/circuits, never changes original manifests,
does not delete PCAP, and uses the existing run locks.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

BASE=Path("/data/ltd-c-qa-bsierra")
LOGS=BASE/"three_logs"
SCRIPT=Path.home()/"ltd-attack/scripts/dataset_c/three_hourly_qa.sh"
MARKER="LTD_C_WATCHDOG"
CRON=f"30 * * * * /usr/bin/python3 {Path.home()}/ltd-attack/scripts/dataset_c/capture_watchdog.py --check-and-repair # {MARKER}"

def command(*args,check=False):
    return subprocess.run(args,capture_output=True,text=True,timeout=20,check=check)

def cron_check():
    p=command("crontab","-l")
    if p.returncode not in (0,1):raise RuntimeError("cannot read crontab")
    return p.stdout

def install():
    if not SCRIPT.is_file() or not BASE.is_dir():raise RuntimeError("missing collection installation")
    current=cron_check()
    if "LTD_C_THREE_QA" not in current:raise RuntimeError("hourly collection Cron not installed")
    if "LTD_C_QA_HOURLY" in current:raise RuntimeError("obsolete collector Cron still active")
    if command("systemctl","is-active","cron").stdout.strip()!="active":
        raise RuntimeError("Cron daemon not running")
    if CRON in current:
        print("WATCHDOG_ALREADY_INSTALLED")
        return
    if MARKER in current:raise RuntimeError("conflicting watchdog Cron exists")
    backup=Path.home()/"ltd-ops/cron-backups"
    backup.mkdir(parents=True,exist_ok=True)
    backup.chmod(0o700)
    snapshot=backup/("before-watchdog-"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".txt")
    snapshot.write_text(current)
    snapshot.chmod(0o600)
    proc=subprocess.run(["crontab","-"],input=current.rstrip("\n")+"\n"+CRON+"\n",
                        text=True,capture_output=True)
    if proc.returncode:raise RuntimeError("cannot install watchdog Cron: "+proc.stderr[:180])
    print("WATCHDOG_INSTALLED",CRON)
    print("CRON_BACKUP",snapshot)

def last_completed():
    completed=[]
    for p in sorted(LOGS.glob("run_*.log"),key=lambda x:x.stat().st_mtime,reverse=True)[:100]:
        if not p.is_file():continue
        try:
            contents=p.read_text(errors="replace")
        except OSError:continue
        if not ("FIN THREE_EXIT=0 " in contents and
                "INTENTOS=120/120" in contents and
                "FINISHED_AT=" in contents and
                "OK_PCAPP_VALID=" in contents):continue
        ts=contents.rsplit("FINISHED_AT=",1)[-1].splitlines()[0].strip()
        try:when=dt.datetime.fromisoformat(ts.replace("Z","+00:00"))
        except ValueError:continue
        completed.append((when,p))
    return max(completed,key=lambda x:x[0]) if completed else None

def active_collectors():
    p=command("docker","ps","--format","{{.Names}}")
    if p.returncode:raise RuntimeError("Cannot read Docker collectors")
    return [s for s in p.stdout.splitlines() if s.startswith(("ltd-c-capture-","ltd-c-recovery-"))]

def tor_containers():
    states={}
    for i in range(1,4):
        name=f"ltd-c-tor-inst{i}"
        p=command("docker","inspect","--format","{{.State.Running}}",name)
        states[name]=p.returncode==0 and p.stdout.strip()=="true"
    return states

def write_report(status,age,busy,tor,detail=None):
    LOGS.mkdir(parents=True,exist_ok=True)
    payload={"timestamp_host_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
             "status":status,"minutes_since_last_completed":age,
             "running_capturers":busy,"tor_containers":tor,
             "detail":detail,"scientific_dataset":False}
    temp=LOGS/(".watchdog_"+str(os.getpid())+".json")
    temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    temp.replace(LOGS/"watchdog_status.json")
    print("WATCHDOG",status,"AGE_MIN",age,"TOR_RUNNING",sum(tor.values()),"ACTIVE_CAPTURERS",len(busy))
    if detail:print("DETAIL",detail)

def check_and_repair():
    if not BASE.is_dir() or not SCRIPT.is_file():
        raise RuntimeError("QA storage or hourly collector script missing")
    if "LTD_C_THREE_QA" not in cron_check():
        write_report("ALERT_HOURLY_CRON_MISSING",None,[],{},detail="Manual action needed; do not silently override a deliberate Cron pause")
        return 5
    tor=tor_containers()
    if not all(tor.values()):
        write_report("ALERT_TOR_CONTAINER_DOWN",None,[],tor)
        return 5
    busy=active_collectors()
    if busy:
        write_report("RUNNING_PARALLEL_CAPTURE",None,busy,tor)
        return 0
    last=last_completed()
    now=dt.datetime.now(dt.timezone.utc)
    age=round((now-last[0]).total_seconds()/60,1) if last else None
    if age is not None and 0<=age<80:
        write_report("HEALTHY",age,busy,tor,detail=str(last[1]))
        return 0
    name=LOGS/("watchdog_relaunch_"+now.strftime("%Y%m%dT%H%M%SZ")+".log")
    LOGS.mkdir(parents=True,exist_ok=True)
    with name.open("w") as f:
        p=subprocess.Popen(["/bin/bash",str(SCRIPT),"--once"],stdin=subprocess.DEVNULL,
                           stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write_report("RECOVERY_LAUNCHED",age,busy,tor,detail=f"PID={p.pid} LOG={name}")
    return 0

def main():
    a=argparse.ArgumentParser()
    a.add_argument("action",nargs="?",choices=["--install","--check-and-repair","--status"],default="--status")
    # argparse positional arguments that begin with -- are interpreted as options;
    # use explicit named flag instead.
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--install",action="store_true")
    group.add_argument("--check-and-repair",action="store_true")
    group.add_argument("--status",action="store_true")
    v=p.parse_args()
    if v.install:install()
    elif v.check_and_repair:sys.exit(check_and_repair())
    else:
        report=LOGS/"watchdog_status.json"
        print(report.read_text() if report.is_file() else "WATCHDOG_NOT_EXECUTED_YET")
        print("WATCHDOG_CRON",MARKER in cron_check())

if __name__=="__main__":
    try:main()
    except Exception as e:
        print("WATCHDOG_ERROR",str(e),file=sys.stderr)
        sys.exit(2)
