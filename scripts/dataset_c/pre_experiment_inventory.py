#!/usr/bin/env python3
"""Pre-experiment inventory of LTD-Attack Perseo. No deletion.
--tidy moves only known LTD-C home-level logs and cron text backups to a private ops directory.
Never traverses research data, models, or bridge secrets.
"""
import argparse
import datetime as dt
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HOME=Path.home()
REPO=HOME/"ltd-attack"
BASE=Path("/data/ltd-c-qa-bsierra")
KNOWN={
    "ltd-attack":"ACTIVE research Git repository; keep",
    "ltd-collector-c-smoke-v22":"ACTIVE browser collector; keep",
    "ltd-storage":"Dataset-C QA link/backup; inspect, keep",
    "ltd-venv":"Python virtual environment; inspect, keep",
    "modelos":"Possible trained models; inspect, keep",
    "codigo":"Unclassified source directory; inspect, keep",
    "ltd-battery-logs":"Potential experiment logs; inspect, keep",
    "ltd-dataset-c-recovery":"Recovery material; inspect, keep",
    "ltd-perseo-backup-20261004":"Older backup; verify before deleting",
    "dataset-c-container-audit-20261006":"Earlier QA audit; verify before deleting",
    "snap":"Ubuntu Snap application data; keep",
    "LTD_ATTACK_P13B_P13C_P14A_ANALYSIS.tar.gz":"Research archive; verify before deleting",
    "LTD_Attack_P16C_Recolector_V2_2_SMOKE.zip":"Collector source archive; verify before deleting",
    "LTD_C_CAPTURE_SOURCE_REVIEW.tar.gz":"Capture source-review archive; verify before deleting",
}
def run(args,timeout=20):
    try:
        p=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
        return p.stdout.strip() or (f"exit={p.returncode}; "+p.stderr.strip()[:170])
    except Exception as e:
        return f"unavailable ({type(e).__name__})"
def du(p):
    result=run(["du","-sh","--",str(p)],timeout=50)
    return result.split("\\t",1)[0] if "\\t" in result else result[:50]
def sort_home():
    print("\\n===== HOME: ARCHIVOS Y CARPETAS (NO SE BORRAN) =====")
    for p in sorted(HOME.iterdir(),key=lambda x:x.name.lower()):
        if p.name.startswith("."): continue
        if p.name=="ltd-ops": continue
        typ="SYMLINK -> "+os.readlink(p) if p.is_symlink() else "DIR" if p.is_dir() else "FILE"
        note=KNOWN.get(p.name, "Known LTD-C log/back-up" if p.name.startswith("ltd-c-") else "UNKNOWN — review before deleting")
        print(f"{du(p):>9}  {typ:<24} {p.name}  :: {note}")
def tidy():
    if not REPO.is_dir() or not BASE.is_dir():
        raise RuntimeError("STOP: expected research repo or data base missing")
    ops=HOME/"ltd-ops"
    logdest=ops/"archived-logs"
    cron=ops/"cron-backups"
    for p in (ops,logdest,cron):
        p.mkdir(mode=0o700,parents=True,exist_ok=True)
        p.chmod(0o700)
    moved=[]
    for p in sorted(HOME.iterdir()):
        if not p.is_file() or p.is_symlink(): continue
        if p.name.startswith("ltd-c-") and p.suffix==".log":
            dst=logdest/p.name
        elif p.name.startswith(("ltd-c-cron-backup-","ltd-c-crontab-before-")) and p.suffix==".txt":
            dst=cron/p.name
        else: continue
        if dst.exists():
            print("SKIP_DUPLICATE",p.name)
            continue
        shutil.move(str(p),str(dst))
        moved.append(p.name)
    print("ARCHIVED_HOME_FILES",len(moved),"IN",ops)
    for name in moved: print("  MOVED",name)
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--tidy",action="store_true",help="organize named LTD-C home logs and cron backups, no deletion")
    args=a.parse_args()
    print("LTD PRE-EXPERIMENT INVENTORY")
    print("Generated (host time; may be unsynchronized):",dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    print("HOST DATE",run(["date","-Is"]))
    print("HOST DATE UTC",run(["date","-u","+%FT%TZ"]))
    print("NTP",run(["timedatectl","show","-p","NTP","-p","NTPSynchronized"]))
    print("CRON_SERVICE",run(["systemctl","is-active","cron"]))
    print("CRON_ENTRIES",run(["crontab","-l"]))
    print("REPO_BRANCH",run(["git","-C",str(REPO),"branch","--show-current"]))
    print("REPO_COMMIT",run(["git","-C",str(REPO),"rev-parse","--short","HEAD"]))
    print("REPO_CHANGES",run(["git","-C",str(REPO),"status","--short"]))
    print("DATA_SPACE",run(["df","-h",str(BASE)]))
    print("\\n===== ACTIVE LTD-C CONTAINERS =====")
    print(run(["docker","ps","--format","{{.Names}} | {{.Status}}"],timeout=20))
    print("\\n===== DATA-BASE TOP-LEVEL (NO DELETION) =====")
    if BASE.is_dir():
        for p in sorted(BASE.iterdir(),key=lambda x:x.name):
            if p.name.startswith("."):continue
            print(f"{du(p):>9}  {p.name}")
    else: print("WARNING: BASE unavailable")
    print("\\n===== CRON OUTPUT =====")
    logdir=BASE/"three_logs"
    if logdir.is_dir():
        logs=sorted(logdir.glob("run_*.log"),key=lambda p:p.stat().st_mtime,reverse=True)
        print("Automated logs:",len(logs))
        if logs:
            print("LATEST:",logs[0])
            # Print only known summary markers; no untrusted web content or Tor details.
            for line in logs[0].read_text(errors="replace").splitlines():
                if re.match(r"^(OUT=|INST [123] SLOTS|FIN THREE_EXIT|INTENTOS=|OK_PCAPP_VALID=|REPORT=|TOR_OK|ERROR:|STOP:)",line):
                    print(" ",line[:250])
    else:
        print("No three_logs directory yet; cron execution NOT confirmed")
    print("\\n===== LAST PARALLEL QA =====")
    if BASE.is_dir():
        items=sorted(BASE.glob("three_hourly_*"),key=lambda p:p.stat().st_mtime,reverse=True)
        if items:
            latest=items[0]
            print("LAST_RUN",latest)
            p=latest/"three_quality_summary.json"
            if p.is_file():
                import json
                dat=json.loads(p.read_text())
                print("LAST_QUALITY",json.dumps(dat.get("totals",{}),ensure_ascii=False))
                print("STATUS",json.dumps(dat.get("status_counts",{}),ensure_ascii=False))
            else: print("No post-run quality summary for latest run")
    sort_home()
    print("\\n===== EXPLICIT PILOT HARDCODING =====")
    for name in ("three_instance_qa.py","three_hourly_qa.sh"):
        path=REPO/"scripts"/"dataset_c"/name
        if not path.is_file(): continue
        text=path.read_text(errors="replace")
        for token in ("-2963","/data/ltd-c-qa-bsierra","ltd-collector-c-smoke-v22","0 * * * *","MIN_DISK_FREE_GB=120"):
            n=text.count(token)
            if n: print(name,"literal",repr(token),"count",n)
    print("RECOMMENDATIONS: maintain historical/Noise/model data. Verify host clock and first Cron run before scientific study.")
    if args.tidy: tidy()
    else: print("HOME_TIDY: dry run; to move ONLY named logs/backups use --tidy")
if __name__=="__main__":
    main()
