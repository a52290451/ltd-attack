#!/usr/bin/env python3
"""Safe housekeeping for P16C operational QA.

Only one named, verified 12-slot disposable smoke run may be purged.
Archives a SHA-256-verified copy of its text/JSON/TSV/log metadata first.
Never deletes full hourly captures, valid PCAP, Tor state, historical datasets,
models, backups, or any unknown folder. Does NOT change Cron.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE=Path("/data/ltd-c-qa-bsierra")
SMOKE="three_smoke_20261012T092123Z_520442"
ARCHIVE_NAME="three_smoke_20261012T092123Z_520442"
TEXT_EXT={".json",".tsv",".csv",".log",".md",".txt"}

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda:f.read(1024*1024),b""):
            h.update(buf)
    return h.hexdigest()

def active_browsers():
    p=subprocess.run(["docker","ps","--format","{{.Names}}"],
                     text=True,capture_output=True,timeout=20)
    if p.returncode:
        raise RuntimeError("STOP: Docker unavailable; cannot prove collectors are idle")
    return [n for n in p.stdout.splitlines() if
            n.startswith(("ltd-c-capture-","ltd-c-recovery-","ltd-c-audit120_"))]

def check_smoke(src):
    if src.is_symlink() or not src.is_dir() or src.resolve()!=BASE/SMOKE:
        raise RuntimeError("STOP: smoke folder not a direct real BASE child")
    files=list(src.rglob("attempts_*.tsv"))
    slots=0
    for p in files:
        if p.is_symlink():
            raise RuntimeError("STOP: unexpected symlinked TSV")
        with p.open(newline="",encoding="utf-8-sig") as f:
            slots+=sum(1 for r in csv.DictReader(f,delimiter="\t") if r.get("uid"))
    if slots!=12:
        raise RuntimeError(f"STOP: smoke expected 12 attempted slots, found {slots}")
    return slots

def cleanup():
    if not BASE.is_dir() or BASE.resolve()!=BASE or BASE.stat().st_uid!=os.getuid():
        raise RuntimeError("STOP: BASE path/owner not as expected")
    src=BASE/SMOKE
    if not src.exists():
        print("CLEANUP_SMOKE_ALREADY_REMOVED")
        return
    count=check_smoke(src)
    busy=active_browsers()
    if busy:
        print("CLEANUP_DEFERRED_ACTIVE_BROWSERS",len(busy))
        return
    archivedir=BASE/"qa_archive"/ARCHIVE_NAME
    if archivedir.is_symlink():
        raise RuntimeError("STOP: QA archive cannot be symlink")
    archivedir.mkdir(mode=0o700,parents=True,exist_ok=True)
    archivedir.chmod(0o700)
    candidates=[]
    for p in src.rglob("*"):
        if p.is_symlink(): raise RuntimeError("STOP: unexpected symlink within smoke")
        if p.is_file() and p.suffix.lower() in TEXT_EXT:
            candidates.append(p)
    if not candidates:raise RuntimeError("STOP: no metadata available for archive")
    for p in candidates:
        dest=archivedir/p.relative_to(src)
        dest.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        if dest.exists():
            if digest(p)!=digest(dest):
                raise RuntimeError("STOP: conflicting archived metadata "+str(dest))
        else:
            shutil.copy2(p,dest)
            if digest(p)!=digest(dest):
                raise RuntimeError("STOP: failed archive hash verification")
    manifest=archivedir/"ARCHIVE_MANIFEST.txt"
    manifest.write_text("P16C operational QA smoke only\n"
        f"Slots={count}\nMetadataFilesSHA256Verified={len(candidates)}\n"
        "Raw smoke PCAP intentionally discarded; original hourly raw QA and scientific datasets untouched.\n")
    manifest.chmod(0o600)
    # Reread all archived SHA256 hashes before the one tightly scoped delete.
    for p in candidates:
        if digest(p)!=digest(archivedir/p.relative_to(src)):
            raise RuntimeError("STOP: final archive verification failed")
    shutil.rmtree(src)
    print("CLEANUP_SMOKE_DISCARDED",src)
    print("CLEANUP_METADATA_ARCHIVED",archivedir)
    print("CLEANUP_FULL_RUNS_PRESERVED")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--apply",action="store_true")
    a=p.parse_args()
    print("BASE",BASE)
    if not BASE.is_dir() or BASE.resolve()!=BASE or BASE.stat().st_uid!=os.getuid():
        raise RuntimeError("STOP: unexpected storage path")
    smoke=BASE/SMOKE
    print("SMOKE_EXISTS",smoke.is_dir())
    if smoke.is_dir():
        print("SMOKE_SLOTS",check_smoke(smoke))
    free=os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize/(1024**3)
    print("FREE_GIB",round(free,1))
    if free<150: print("DISK_WARNING: below 150GiB; retain all scientific and OK capture evidence")
    if a.apply:cleanup()
    else: print("DRY_RUN: nothing deleted; use --apply for the known 12-slot smoke ONLY")
if __name__=="__main__":
    try:main()
    except Exception as ex:
        print("HOUSEKEEPING_ERROR",str(ex),file=sys.stderr)
        sys.exit(2)
