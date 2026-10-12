#!/usr/bin/env python3
"""Read-only P16C home/data inventory and B/C hostname comparison.
No deletion, moving, container control, installation or cron edits.
"""
import csv, hashlib, json, os, subprocess, sys
from pathlib import Path
from urllib.parse import urlsplit
from collections import Counter

HOME=Path.home()
BASE=Path("/data/ltd-c-qa-bsierra")
FUT=HOME/"ltd-storage/datasets/future/CLEAN_final_features_sites_concept_drift.csv"
HIST=HOME/"ltd-storage/datasets/historical/CLEAN_final_features_sites.csv"
NOTES={
"ltd-attack":"ACTIVE_REPOSITORY_KEEP","ltd-storage":"SCIENTIFIC_DATA_KEEP",
"modelos":"MODEL_AND_PYTHON_ENV_KEEP","ltd-venv":"RESEARCH_ENV_KEEP",
"ltd-collector-c-smoke-v22":"ACTIVE_COLLECTOR_SOURCE_KEEP",
"ltd-ops":"CURRENT_OPERATIONS_AND_CRON_BACKUPS_KEEP",
"ltd-dataset-c-recovery":"REVIEW_RECOVERY_BEFORE_ARCHIVE",
"ltd-battery-logs":"REVIEW_RESEARCH_LOGS",
"ltd-perseo-backup-20261004":"REVIEW_BACKUP",
"dataset-c-container-audit-20261006":"REVIEW_EARLY_AUDIT",
"codigo":"REVIEW_UNKNOWN_CODE",
"LTD_ATTACK_P13B_P13C_P14A_ANALYSIS.tar.gz":"RESEARCH_CHECKPOINT_ARCHIVE_KEEP",
"LTD_Attack_P16C_Recolector_V2_2_SMOKE.zip":"REVIEW_VERSIONED_SOURCE",
"LTD_C_CAPTURE_SOURCE_REVIEW.tar.gz":"REVIEW_VERSIONED_SOURCE",
"snap":"SYSTEM_APPLICATION_DATA_KEEP"
}
def shell(*args):
    p=subprocess.run(args,text=True,capture_output=True,timeout=50)
    return p.stdout.strip() if p.returncode==0 else "ERROR:"+p.stderr.strip()[:100]
def size(path):
    return shell("du","-sh","--",str(path)).split("\t")[0]
def norm(v):
    s=str(v or "").strip().lower()
    if not s or s.isdigit():return ""
    try:
        h=(urlsplit(s if "://" in s else "https://"+s).hostname or "").strip(".")
        return h[4:] if h.startswith("www.") else h
    except ValueError:return ""
def csvsites(path):
    if not path.is_file():raise RuntimeError("MISSING "+str(path))
    sitecounts=Counter();fields=[]; rows=0
    with path.open(newline="",encoding="utf-8-sig") as fd:
        reader=csv.DictReader(fd)
        fields=reader.fieldnames or []
        if "site" not in fields:raise RuntimeError("NO_SITE_COL "+str(path))
        for row in reader:
            rows+=1
            host=norm(row.get("site"))
            if host:sitecounts[host]+=1
    return sitecounts,fields,rows
def cinfo(root):
    counts=Counter(); statuses=Counter(); ok=Counter();validated=[]
    for i,cc in ((1,("belleza","turismo")),(2,("bancos","salud")),(3,("deportes","politica"))):
        for cat in cc:
            for f in (root/f"inst{i}"/cat).glob("attempts_*.tsv"):
                with f.open(newline="",encoding="utf-8-sig") as fd:
                    for r in csv.DictReader(fd,delimiter="\t"):
                        site=norm(r.get("url"))
                        if site:counts[site]+=1
                        state=r.get("status","")
                        statuses[state]+=1
                        if state=="OK" and str(r.get("pcap_valid","")).lower() in ("1","true","yes"):
                            pc=Path(r.get("pcap",""))
                            if pc.is_absolute() and pc.parts[1:2] in (("audit",),("audit120",)):
                                pc=root/Path(*pc.parts[2:])
                            else:pc=root/pc
                            if pc.resolve().is_relative_to(root.resolve()) and pc.is_file() and pc.stat().st_size>24:
                                if site:ok[site]+=1
                                validated.append(r.get("uid",""))
    return counts,ok,statuses,len(validated)
def main():
    print("=== HOME: NO DELETE NO MOVE ===")
    for p in sorted(HOME.iterdir(),key=lambda q:q.name.lower()):
        if p.name.startswith("."):continue
        if p.name in NOTES or p.name.startswith(("ltd-","LTD_","dataset-c")):
            print(size(p),p.name,NOTES.get(p.name,"REVIEW_UNCLASSIFIED"))
    print("=== QA BASE: NO DELETE NO MOVE ===")
    for p in sorted(BASE.iterdir()):
        if p.name.startswith("."):continue
        name=p.name
        if name.startswith("three_hourly_"):tag="HOURLY_RAW_PCAP_KEEP"
        elif name=="three_tor_config":tag="SECRET_BRIDGE_CONFIG_KEEP"
        elif name in ("three_logs","cron_logs"):tag="CURRENT_LOGS_KEEP"
        elif name=="qa_archive":tag="ARCHIVED_AUDIT_KEEP"
        elif name.startswith(("three_targeted_recovery_","recovery_v23_","audit120_")):tag="ONE_OFF_QA_REVIEW"
        elif name in ("inst1","inst2","inst3"):tag="LEGACY_INSTANCE_DATA_REVIEW"
        else:tag="OPS_METADATA_REVIEW"
        print(size(p),name,tag)
    print("=== B/C COMPARISON: RAW HOMEPAGE HOSTS, NOT CANONICAL CLASS LABELS ===")
    b,bc,br=csvsites(FUT)
    a,ac,ar=csvsites(HIST)
    print("HIST_RAW_ROWS",ar,"HIST_RAW_DOMAINS",len(a),"HIST_COLUMNS",",".join(ac[:20]))
    print("FUTURE_RAW_ROWS",br,"FUTURE_RAW_DOMAINS",len(b),"FUTURE_COLUMNS",",".join(bc[:20]))
    rounds=sorted((r for r in BASE.glob("three_hourly_*") if (r/"three_quality_summary.json").is_file()
                   and (r/"VALIDATED_THREE_FULL.txt").is_file()),key=lambda r:r.stat().st_mtime)
    if not rounds:
        print("C_NO_VALIDATED_ROUNDS");return
    for root in rounds[-4:]:
        c,ok,statuses,n=cinfo(root)
        common_raw=set(c)&set(b)
        common_ok=set(ok)&set(b)
        print("RUN",root.name,"C_ATTEMPT_DOMAINS",len(c),"C_OK_DOMAINS",len(ok),
              "C_OK_VALID_CAPTURES",n,"B_RAW_IN_C_ATTEMPT",len(common_raw),
              "B_RAW_IN_C_OK",len(common_ok),"B_RAW_MISSING_C_ATTEMPT",len(set(b)-set(c)),
              "B_RAW_MISSING_C_OK",len(set(b)-set(ok)))
        print("B_RAW_MISSING_FROM_C_ATTEMPT",",".join(sorted(set(b)-set(c))))
        print("B_RAW_WITHOUT_OK_IN_C",",".join(sorted(set(b)-set(ok))))
        print("C_DOMAINS_NOT_IN_B_RAW",",".join(sorted(set(c)-set(b))))
        print("STATES",json.dumps(statuses,sort_keys=True))
    print("CAUTION: These are 69 raw B-domain comparisons, NOT the frozen 65-class crosswalk.")
    print("CAUTION: Output is diagnostic, not scientific validation; NTP still unsynchronized.")
if __name__=="__main__":
    try:main()
    except Exception as e:
        print("AUDIT_ERROR",str(e),file=sys.stderr)
        sys.exit(2)
