#!/usr/bin/env python3
"""Perseo LTD storage cutover, reversible and NO DELETE of historical data.

--preflight   Rechecks staged rsync checksum equivalence; no filesystem changes.
--activate    Rechecks then renames old folder within HOME and creates a symlink
              from ~/ltd-storage to /data/ltd-attack/storage. Keeps full backup.
--validate    Confirms activated link plus canonical scientific paths.
--rollback    Rechecks that stage and backup remain equivalent before reversing.
No cron edits, no copying or deleting datasets, no venv changes, no sudo.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE=Path.home()/"ltd-storage"
DEST=Path("/data/ltd-attack/storage")
STATE=Path("/data/ltd-attack/.ltd_storage_cutover_v1.json")
STAGE_MARKER=".ltd_storage_stage_v1.json"
AUDIT=".ltd_storage_verify_v1.txt"
MIN_DATA_FREE=120*1024**3

def require(value,message):
    if not value:raise RuntimeError("LTD_CUTOVER_BLOCKED_"+message)

def metadata_compatible(source=SOURCE,dest=DEST):
    require(not source.is_symlink() and source.is_dir(),"SOURCE_NOT_ORIGINAL_DIRECTORY")
    require(not dest.is_symlink() and dest.is_dir(),"STAGED_DEST_NOT_DIRECTORY")
    require(source.resolve()!=dest.resolve(),"SAME_SOURCE_AND_TARGET")
    require(os.stat(source).st_dev!=os.stat(dest).st_dev,
            "SOURCE_AND_TARGET_ON_SAME_DISK")
    marker=dest/STAGE_MARKER
    require(marker.is_file() and not marker.is_symlink(),"STAGE_MARKER_MISSING")
    payload=json.loads(marker.read_text(encoding="utf-8"))
    require(payload.get("source")==str(source) and payload.get("dest")==str(dest),
            "STAGE_MARKER_MISMATCH")
    require(shutil.which("rsync") is not None,"RSYNC_NOT_AVAILABLE")
    require(shutil.disk_usage(dest).free>=MIN_DATA_FREE,"DATA_DISK_RESERVE")
    return payload

def compare_trees(original,copied):
    """Metadata + contents, symlinks and missing/extra files; never transfers."""
    cmd=["ionice","-c3","nice","-n","10","rsync","-aH",
         "--no-owner","--no-group","--checksum","--dry-run",
         "--delete","--itemize-changes",
         "--exclude=/"+STAGE_MARKER,"--exclude=/"+AUDIT,
         str(original)+"/",str(copied)+"/"]
    r=subprocess.run(cmd,text=True,capture_output=True,check=False)
    differences=[x for x in r.stdout.splitlines() if x.strip()]
    if r.returncode or differences:
        for line in differences[:12]:print("LTD_CUTOVER_DIFFERENCE",line[:240],flush=True)
        print("LTD_CUTOVER_RSYNC_EXIT",r.returncode,flush=True)
        print("LTD_CUTOVER_RSYNC_ERROR",r.stderr[-350:],flush=True)
        raise RuntimeError("LTD_CUTOVER_BLOCKED_NOT_CHECKSUM_IDENTICAL")
    return True

def scientific_paths_ok(path):
    for name in ("datasets","artifacts","results","audits"):
        require((path/name).is_dir(),"MISSING_"+name)
    for name in ("datasets/historical/CLEAN_final_vectors_sites.csv",
                 "datasets/historical/CLEAN_final_features_sites.csv"):
        file=path/name
        require(file.is_file() and file.stat().st_size>0,"MISSING_"+name)
    return True

def backup_name():
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return SOURCE.with_name(SOURCE.name+".precutover-"+stamp)

def activate_paths(source,dest,backup):
    """Small, tested, reversible filesystem switch; no data deletion."""
    require(not source.is_symlink() and source.is_dir(),"BAD_SOURCE_BEFORE_SWITCH")
    require(not backup.exists() and not backup.is_symlink(),"BACKUP_ALREADY_EXISTS")
    source.rename(backup)
    try:
        source.symlink_to(dest,target_is_directory=True)
        require(source.is_symlink() and source.resolve()==dest.resolve(),
                "SYMLINK_NOT_RESOLVING")
    except BaseException:
        if source.is_symlink():
            source.unlink()
        backup.rename(source)
        raise

def rollback_paths(source,dest,backup):
    require(source.is_symlink() and source.resolve()==dest.resolve(),
            "UNEXPECTED_ACTIVE_SYMLINK")
    require(backup.is_dir() and not backup.is_symlink(),"BACKUP_NOT_REAL_DIR")
    source.unlink()
    try:
        backup.rename(source)
    except BaseException:
        source.symlink_to(dest,target_is_directory=True)
        raise
    require(source.is_dir() and not source.is_symlink(),"ROLLBACK_DID_NOT_RESTORE")

def assert_no_source_users():
    # Snapshot cannot prove no unlisted future access, but prevents switching
    # when an obvious active program references the original directory.
    p=subprocess.run(["ps","-eo","pid=,args="],
                     capture_output=True,text=True,check=True,timeout=12)
    blockers=[]
    needle=str(SOURCE)
    for line in p.stdout.splitlines():
        line=line.strip()
        if not line:continue
        pid,_,args=line.partition(" ")
        try:
            if int(pid)==os.getpid():continue
        except ValueError:continue
        if needle in args or "ltd-storage/datasets" in args:
            blockers.append(pid)
    require(not blockers,"ACTIVE_PROCESS_REFERENCES_SOURCE_"+",".join(blockers[:12]))

def preflight():
    require(not STATE.exists(),"STATE_ALREADY_EXISTS_CHECK_VALIDATE_OR_ROLLBACK")
    metadata_compatible()
    scientific_paths_ok(SOURCE)
    scientific_paths_ok(DEST)
    assert_no_source_users()
    compare_trees(SOURCE,DEST)
    print("LTD_CUTOVER_PREFLIGHT=PASS_CHECKSUM_IDENTICAL",flush=True)
    print("LTD_CUTOVER_BACKUP_PRESERVED=PLANNED",flush=True)

def activate():
    preflight()
    backup=backup_name()
    require(not backup.exists(),"TIMESTAMP_BACKUP_COLLISION")
    print("LTD_CUTOVER_BACKUP",backup,flush=True)
    activate_paths(SOURCE,DEST,backup)
    try:
        scientific_paths_ok(SOURCE)
        record={"classification":"LTD_STORAGE_REVERSIBLE_CUTOVER_V1",
                "state":"ACTIVE","source":str(SOURCE),"dest":str(DEST),
                "backup":str(backup),
                "activated_utc":datetime.now(timezone.utc).isoformat(),
                "source_data_deleted":False,"cron_changed":False,
                "config_changed":False,"venv_changed":False}
        STATE.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
        STATE.chmod(0o600)
    except BaseException:
        rollback_paths(SOURCE,DEST,backup)
        raise
    print("LTD_CUTOVER=ACTIVE_REVERSIBLE",flush=True)
    print("LTD_CUTOVER_LIVE_PATH",SOURCE,"=>",SOURCE.resolve(),flush=True)
    print("LTD_CUTOVER_ORIGINAL_BACKUP",backup,flush=True)
    print("LTD_CUTOVER_ROOT_SPACE_FREED=NO",flush=True)
    print("LTD_CUTOVER_CRON_CHANGED=NO",flush=True)

def state():
    require(STATE.is_file() and not STATE.is_symlink(),"STATE_MISSING")
    d=json.loads(STATE.read_text(encoding="utf-8"))
    require(d.get("classification")=="LTD_STORAGE_REVERSIBLE_CUTOVER_V1",
            "UNKNOWN_STATE")
    require(d.get("source")==str(SOURCE) and d.get("dest")==str(DEST),
            "STATE_PATH_MISMATCH")
    backup=Path(d.get("backup",""))
    require(backup.parent==SOURCE.parent and
            backup.name.startswith("ltd-storage.precutover-"),
            "UNEXPECTED_BACKUP_PATH")
    return d,backup

def validate():
    d,backup=state()
    require(d.get("state")=="ACTIVE","STATE_NOT_ACTIVE")
    require(SOURCE.is_symlink() and SOURCE.resolve()==DEST.resolve(),
            "SOURCE_LINK_INVALID")
    require(backup.is_dir() and not backup.is_symlink(),"BACKUP_MISSING")
    scientific_paths_ok(SOURCE)
    print("LTD_CUTOVER_VALIDATE=PASS",flush=True)
    print("LTD_CUTOVER_ORIGINAL_BACKUP",backup,flush=True)
    print("LTD_CUTOVER_DATA_ROOT",SOURCE.resolve(),flush=True)
    print("LTD_CUTOVER_CONFIG_CHANGED=NO",flush=True)
    print("LTD_CUTOVER_ORIGINAL_DELETED=NO",flush=True)

def rollback():
    d,backup=state()
    require(d.get("state")=="ACTIVE","STATE_NOT_ACTIVE")
    require(SOURCE.is_symlink() and SOURCE.resolve()==DEST.resolve(),
            "SOURCE_LINK_INVALID")
    assert_no_source_users()
    # Block rollback if anyone wrote new data to /data since cutover: no loss.
    compare_trees(backup,DEST)
    rollback_paths(SOURCE,DEST,backup)
    d["state"]="ROLLED_BACK"
    d["rolled_back_utc"]=datetime.now(timezone.utc).isoformat()
    STATE.write_text(json.dumps(d,indent=2)+"\n",encoding="utf-8")
    print("LTD_CUTOVER_ROLLBACK=PASS_ORIGINAL_RESTORED",flush=True)
    print("LTD_CUTOVER_DATA_REMOVED=NO",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight",action="store_true")
    group.add_argument("--activate",action="store_true")
    group.add_argument("--validate",action="store_true")
    group.add_argument("--rollback",action="store_true")
    args=p.parse_args()
    try:
        if args.preflight:preflight()
        elif args.activate:activate()
        elif args.validate:validate()
        else:rollback()
    except Exception as exc:
        print("LTD_CUTOVER_ABORTED",type(exc).__name__,str(exc)[:500],
              file=sys.stderr,flush=True)
        sys.exit(4)
