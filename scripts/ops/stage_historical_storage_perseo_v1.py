#!/usr/bin/env python3
"""STAGE, never cut over, Perseo LTD historical storage from HOME to /data.

Copies without modifying the source, then runs a full rsync --checksum
--dry-run comparison. No deletion, mount changes, symlinks, or crontab edits.
Avoids interference with Dataset C using low CPU/IO scheduling priority.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE=Path.home()/"ltd-storage"
DEST=Path("/data/ltd-attack/storage")
MARKER=".ltd_storage_stage_v1.json"
AUDIT=".ltd_storage_verify_v1.txt"
MIN_FREE_AFTER_BYTES=120*1024**3
MIN_FREE_BEFORE_BYTES=160*1024**3

def guard():
    if SOURCE.is_symlink() or not SOURCE.is_dir():
        raise RuntimeError("SOURCE_MUST_BE_REAL_EXISTING_HOME_LTD_STORAGE")
    if SOURCE.resolve()!=Path.home()/"ltd-storage":
        raise RuntimeError("UNEXPECTED_SOURCE")
    if not Path("/data").is_dir():
        raise RuntimeError("DATA_MOUNT_MISSING")
    if Path("/data").is_symlink() or Path("/data/ltd-attack").is_symlink() or DEST.is_symlink():
        raise RuntimeError("UNEXPECTED_DEST_SYMLINK")
    if os.stat(SOURCE).st_dev==os.stat("/data").st_dev:
        raise RuntimeError("DEST_NOT_SEPARATE_FILESYSTEM")
    if not shutil.which("rsync"):
        raise RuntimeError("RSYNC_MISSING")
    data_free=shutil.disk_usage("/data").free
    if data_free<MIN_FREE_BEFORE_BYTES:
        raise RuntimeError("DATA_FREE_BELOW_160_GIB")
    if DEST.exists():
        marker=DEST/MARKER
        if not marker.is_file():
            raise RuntimeError("DEST_EXISTS_WITHOUT_LTD_STAGE_MARKER_NO_TOUCH")
        d=json.loads(marker.read_text())
        if d.get("source")!=str(SOURCE) or d.get("dest")!=str(DEST):
            raise RuntimeError("DEST_STAGE_MARKER_INCONSISTENT")
    return data_free

def rsync_args(*,verify):
    args=["ionice","-c3","nice","-n","10","rsync","-aH",
          "--no-owner","--no-group",
          "--exclude=/"+MARKER,"--exclude=/"+AUDIT]
    if verify:
        args+=["--checksum","--dry-run","--delete","--itemize-changes"]
    else:
        args+=["--partial","--stats","--itemize-changes"]
    return args+[str(SOURCE)+"/",str(DEST)+"/"]

def stage():
    available=guard()
    if not DEST.exists():
        DEST.mkdir(parents=True,mode=0o700)
        (DEST/MARKER).write_text(json.dumps({
            "classification":"LTD_STORAGE_STAGE_V1_NO_CUTOVER",
            "source":str(SOURCE),"dest":str(DEST),
            "created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
            "original_source_preserved":True,
            "cron_unchanged":True,"config_unchanged":True
        },indent=2)+"\n")
        (DEST/MARKER).chmod(0o600)
    print("LTD_STAGE_SOURCE",SOURCE,flush=True)
    print("LTD_STAGE_DESTINATION",DEST,flush=True)
    print("LTD_STAGE_DISK_FREE_GIB",round(available/1024**3,1),flush=True)
    print("LTD_STAGE_COPY=START_LOW_PRIORITY",flush=True)
    completed=subprocess.run(rsync_args(verify=False),check=False)
    if completed.returncode:
        raise RuntimeError("LTD_STAGE_RSYNC_COPY_FAILED_"+str(completed.returncode))
    return verify()

def verify():
    available=guard()
    print("LTD_STAGE_VERIFY=CHECKSUM_DRY_RUN",flush=True)
    completed=subprocess.run(rsync_args(verify=True),
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                             text=True,check=False)
    # Store diagnostics in the staged data destination, not in HOME.
    (DEST/AUDIT).write_text(
        "returncode="+str(completed.returncode)+"\n"+
        "stdout:\n"+completed.stdout+"\nstderr:\n"+completed.stderr+"\n")
    (DEST/AUDIT).chmod(0o600)
    unexpected=[line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode or unexpected:
        print("LTD_STAGE_VERIFY=BLOCKED",
              "RSYNC_EXIT",completed.returncode,
              "DIFF_LINES",len(unexpected),flush=True)
        for line in unexpected[:15]:print("LTD_STAGE_DIFF",line[:300])
        raise RuntimeError("LTD_STAGE_NOT_IDENTICAL_DO_NOT_CUTOVER")
    if available<MIN_FREE_AFTER_BYTES:
        raise RuntimeError("DATA_RESERVE_BREACHED")
    print("LTD_STAGE_VERIFY=PASS_IDENTICAL_CHECKSUM_DRY_RUN")
    print("LTD_STAGE_NEW_DATA_FREE_GIB",round(available/1024**3,1))
    print("LTD_SOURCE_REMOVED=NO")
    print("LTD_CRON_CHANGED=NO")
    print("LTD_CODE_OR_VENV_MOVED=NO")
    print("LTD_STORAGE_CUTOVER=NOT_AUTHORIZED")
    print("LTD_STAGE_REPORT",DEST/AUDIT)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--stage-and-verify",action="store_true")
    p.add_argument("--verify-only",action="store_true")
    args=p.parse_args()
    if args.stage_and_verify==args.verify_only:
        p.error("select exactly one of --stage-and-verify or --verify-only")
    try:
        if args.stage_and_verify:stage()
        else:verify()
    except Exception as exc:
        print("LTD_STORAGE_STAGE_ABORTED",type(exc).__name__,str(exc),
              file=sys.stderr,flush=True)
        sys.exit(4)
