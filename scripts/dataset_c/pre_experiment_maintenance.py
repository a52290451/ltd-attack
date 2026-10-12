#!/usr/bin/env python3
"""P16C maintenance: preserve evidence, pause poor-quality hourly QA,
prune only verified redundant smoke PCAPs, and inventory large home folders.
No deletion of historical A/B, Noise, model weights, backups, or Tor identities.
"""
from __future__ import annotations
import csv,datetime,hashlib,os,pathlib,shutil,subprocess,sys

HOME=pathlib.Path.home()
BASE=pathlib.Path("/data/ltd-c-qa-bsierra")
OPS=HOME/"ltd-ops"
CRON_BAK=OPS/"cron-backups"
ARCHIVE=BASE/"qa_archive"
SMOKE_NAME="three_smoke_20261012T092123Z_520442"
FULL_NAME="three_hourly_20261012T092246Z_523162"

def output(*args,check=True):
    return subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=check)
def valid_base():
    if not BASE.is_dir() or BASE.resolve()!=BASE or BASE.stat().st_uid!=os.getuid():
        raise RuntimeError("STOP: unexpected data path or owner")
    if not (BASE/FULL_NAME/"VALIDATED_THREE_FULL.txt").is_file():
        raise RuntimeError("STOP: complete three-instance reference missing")
def active_capture():
    names=output("docker","ps","--format","{{.Names}}").stdout.splitlines()
    return [n for n in names if n.startswith(("ltd-c-capture-","ltd-c-recovery-","ltd-c-audit120_"))]
def pause_cron():
    CRON_BAK.mkdir(parents=True,exist_ok=True)
    CRON_BAK.chmod(0o700)
    cmd=output("crontab","-l",check=False)
    if cmd.returncode not in (0,1):raise RuntimeError("STOP: cannot read crontab")
    lines=cmd.stdout.splitlines()
    tagged=[line for line in lines if "LTD_C_THREE_QA" in line]
    if len(tagged)>1:raise RuntimeError("STOP: more than one three-instance cron entry")
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target=CRON_BAK/f"before-quality-maintenance-{stamp}.txt"
    target.write_text(cmd.stdout)
    target.chmod(0o600)
    if tagged:
        new_text="\n".join([line for line in lines if "LTD_C_THREE_QA" not in line])+"\n"
        changed=subprocess.run(["crontab","-"],input=new_text,text=True,capture_output=True)
        if changed.returncode:raise RuntimeError("STOP: crontab not changed: "+changed.stderr[:150])
        print("CRON_QA_PAUSED",tagged[0])
    else:
        print("CRON_QA_ALREADY_PAUSED")
    print("CRON_BACKUP",target)
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.digest()
def prune_one_smoke():
    src=BASE/SMOKE_NAME
    if not src.is_dir():
        print("SMOKE_ALREADY_ABSENT",src)
        return
    if src.is_symlink() or src.resolve().parent!=BASE:
        raise RuntimeError("STOP: unusual smoke path")
    files=list(src.rglob("attempts_*.tsv"))
    count=0
    for p in files:
        with p.open(encoding="utf-8-sig",newline="") as f:
            count+=sum(1 for _ in csv.DictReader(f,delimiter="\t"))
    if count!=12:raise RuntimeError(f"STOP: expected 12 smoke attempts, got {count}")
    if active_capture():raise RuntimeError("STOP: collector active")
    dst=ARCHIVE/SMOKE_NAME
    if dst.exists():raise RuntimeError("STOP: smoke archive destination already exists")
    dst.mkdir(parents=True,mode=0o700)
    allowed={".csv",".tsv",".json",".md",".txt",".log"}
    originals=[p for p in src.rglob("*") if p.is_file() and not p.is_symlink() and p.suffix.lower() in allowed]
    for p in originals:
        relative=p.relative_to(src)
        out=dst/relative
        out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,out)
        if digest(p)!=digest(out):raise RuntimeError("STOP: archive hash mismatch")
    if not originals:raise RuntimeError("STOP: no smoke metadata copied")
    evidence=dst/"ARCHIVE_MANIFEST.txt"
    evidence.write_text(f"Source: {src}\nSmoke attempts: {count}\nMetadata files SHA256 verified: {len(originals)}\nPCAP intentionally not archived; scientific datasets untouched.\n")
    shutil.rmtree(src)
    print("SMOKE_PRUNED",src,"META_ARCHIVED",dst,"FILES",len(originals))
def du(p):
    try:
        x=output("du","-sh","--",str(p))
        return x.stdout.split("\t")[0]
    except Exception: return "unknown"
def inventory():
    print("\n===== HOME LARGE DIRECTORIES: NO DELETION =====")
    for n in ("ltd-storage","ltd-dataset-c-recovery","modelos",
              "dataset-c-container-audit-20261006","ltd-perseo-backup-20261004","ltd-collector-c-smoke-v22"):
        p=HOME/n
        if not p.exists():continue
        print("TOP",n,du(p))
        if p.is_dir() and not p.is_symlink():
            for q in sorted(p.iterdir(),key=lambda x:x.name):
                if q.is_symlink():
                    print(" ",q.name,"SYMLINK =>",os.readlink(q))
                else:
                    print(" ",q.name,du(q))
    print("\n===== ROOT DATA QA =====")
    for p in sorted(BASE.iterdir(),key=lambda x:x.name):
        if p.name.startswith("."):continue
        print(p.name,du(p))
    print("SPACE",output("df","-h",str(BASE)).stdout.strip())
def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--apply",action="store_true",help="pause three QA cron and prune verified one-off smoke only")
    a=parser.parse_args()
    valid_base()
    print("SNAPSHOT_TIME_HOST_UNSYNC",datetime.datetime.now(datetime.timezone.utc).isoformat())
    print("CLOCK",output("timedatectl","show","-p","NTPSynchronized","-p","NTP").stdout.strip())
    print("CRON",output("systemctl","is-active","cron",check=False).stdout.strip())
    print("COLLECTORS_RUNNING",active_capture())
    if a.apply:
        pause_cron()
        if active_capture():
            print("SMOKE_PRUNE_DEFERRED: a capture is running; hourly QA remains paused")
        else:
            prune_one_smoke()
    else:
        print("DRY_RUN: no changes; use --apply to pause hourly QA and archive/prune 12-slot smoke")
    inventory()
if __name__=="__main__":
    try:main()
    except Exception as exc: print("ERROR",exc,file=sys.stderr);sys.exit(2)
