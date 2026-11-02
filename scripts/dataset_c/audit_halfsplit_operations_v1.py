#!/usr/bin/env python3
"""Read-only Dataset C operational audit: host cron, :00/:30 batches, storage.

No captures, Docker changes, crontab writes, file cleanup, or scientific claims.
Reports summarize existing completed half60 summaries and logs only.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys

import run_direct120_halfsplit_qa_v2 as h
import run_direct120_recurrent_qa_v1 as legacy

MIN_FREE=120_000_000_000
STORAGE=legacy.BASE
LOGS=legacy.OPERATION
MAPPING=h.BATCHES

def try_cmd(args,timeout=8):
    try:
        p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         text=True,timeout=timeout)
        return {"return_code":p.returncode,"output":p.stdout.strip(),
                "stderr":p.stderr.strip()[:200]}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {"return_code":None,"error":type(exc).__name__}

def parse_batch(path,batch,canonical=None,check_files=False):
    obj=json.loads(path.read_text(encoding="utf-8"))
    if obj.get("classification")!="C_DIRECT_120_HALFHOUR_QA_ONLY":
        raise ValueError("unexpected batch classification")
    if obj.get("batch")!=batch or obj.get("attempts")!=60:
        raise ValueError("batch type or 60 slots mismatch")
    rows=obj.get("records",[])
    if len(rows)!=60:raise ValueError("60-row evidence missing")
    expected=MAPPING[batch]
    counter=Counter()
    ids=set()
    counts=Counter()
    pcaps_valid=0
    for r in rows:
        inst=r.get("instance")
        cat=r.get("category")
        slot=r.get("slot")
        try:slot=int(slot)
        except (ValueError,TypeError):raise ValueError("invalid slot")
        if inst not in expected or expected[inst]!=cat or not 1<=slot<=20:
            raise ValueError("category/instance/position mismatch")
        uid=(inst,cat,slot)
        if uid in ids:raise ValueError("duplicate site slot")
        if canonical is not None and r.get("url")!=canonical.get(uid):
            raise ValueError("site URL differs from original 120-slot inventory")
        if check_files and r.get("qa_valid"):
            stem=r.get("stem")
            if not stem or not isinstance(stem,str) or "/" in stem or ".." in stem:
                raise ValueError("missing/unsafe successful capture stem")
            outputs=legacy.CAPTURES/obj["run_id"]/"outputs"
            for ext in (".json",".pcap",".html"):
                file=outputs/(stem+ext)
                if not file.is_file() or file.stat().st_size<=0:
                    raise ValueError("valid PCAP evidence missing "+file.name)
        ids.add(uid)
        counter[inst]+=1
        counts[r.get("status","UNKNOWN")]+=1
        pcaps_valid+=bool(r.get("qa_valid"))
    if counter!=Counter({"inst1":20,"inst2":20,"inst3":20}):
        raise ValueError("not 20 per Tor instance")
    if pcaps_valid!=obj.get("valid_pcap_qa"):
        raise ValueError("PCAP QA total differs from summary")
    return {"path":str(path),"batch":batch,"run_id":obj.get("run_id"),
            "valid":pcaps_valid,"slots":len(ids),"slot_ids":ids,
            "status_counts":dict(counts),
            "source_dir":str(path.parent),
            "unverified_clock":obj.get("host_time_unverified")}

def audit():
    cron=try_cmd(["crontab","-l"])
    lines=cron.get("output","").splitlines()
    expected=h.schedule_rows()
    active=[x for x in lines if x.strip() and not x.lstrip().startswith("#")]
    exact={b:expected[i] in active for i,b in enumerate(("first","second"))}
    older=[x for x in active if legacy.MARKER in x and h.MARKER not in x]
    other_c=[x for x in active if any(k in x for k in
             ("LTD_C_THREE_QA","LTD_C_QA_HOURLY","run_hourly_qa.sh",
              "three_hourly_qa.sh"))]
    service=try_cmd(["systemctl","is-active","cron"])
    ntp=legacy.clock_info()
    disk=shutil.disk_usage(STORAGE)
    enough=disk.free>=MIN_FREE
    # Anchor every completed site against the same immutable local 120-slot
    # manifest used by the scheduler, not merely the expected category names.
    canonical={legacy.key(row):row["url"] for row in legacy.inventory()}
    if len(canonical)!=120:raise ValueError("canonical 120-slot inventory mismatch")
    reports={}
    errors=[]
    latest=[]
    for b in ("first","second"):
        matches=sorted(legacy.REPORTS.glob(
            f"direct120_half_{b}_*/half60_summary.json"),
            key=lambda p:p.stat().st_mtime,reverse=True)
        parsed=[]
        for p in matches[:24]:
            try:parsed.append(parse_batch(p,b,canonical=canonical,check_files=True))
            except (ValueError,KeyError,TypeError,OSError) as e:
                errors.append({"path":str(p),"error":str(e)[:300]})
        reports[b]=parsed
        if parsed:latest.append(parsed[0])
    if len(latest)==2:
        combined=set().union(*(x["slot_ids"] for x in latest))
        if len(combined)!=120:
            errors.append({"combined_last_batches":len(combined),
                           "reason":"not the full 120 unique canonical slots"})
    logs={}
    for b in ("first","second"):
        path=LOGS/("scheduler_split_"+b+".log")
        if path.is_file():
            lines=path.read_text(errors="replace").splitlines()
            logs[b]={
                "exists":True,"lines":len(lines),
                "batch_runs":sum("HALFSPLIT_RUN " in x for x in lines),
                "completed_60":sum("HALFSPLIT_ATTEMPTS 60/60" in x for x in lines),
                "overlaps_skipped":sum("HALFSPLIT_SKIPPED_OVERLAP" in x for x in lines),
                "blocked":sum("HALFSPLIT_BLOCKED" in x for x in lines),
                "tail":lines[-12:]}
        else:logs[b]={"exists":False}
    last_full=sorted(legacy.REPORTS.glob("direct120_qa_*/direct120_summary.json"),
                     key=lambda p:p.stat().st_mtime,reverse=True)
    status="OBSERVING"
    if any(errors) or not all(exact.values()) or older or other_c or not enough:
        status="ACTION_REQUIRED"
    elif all(reports[b] for b in reports) and all(logs[b].get("completed_60",0)>0 for b in reports):
        status="TWO_BATCHES_CONFIRMED_QA_ONLY"
    # A pair of arbitrary last batches has 120 slots but not necessarily a same-hour run.
    # Report separate half-batches; do not call a pair same-hour verified.
    warnings=[]
    if not all(reports[b] for b in reports):warnings.append("HALF60_BATCHES_NOT_BOTH_COMPLETED")
    if not all(logs[b].get("exists") for b in reports):warnings.append("CRON_LOGS_NOT_BOTH_FOUND")
    if any(logs[b].get("overlaps_skipped",0)>0 for b in reports):warnings.append("SKIPPED_CRON_OVERLAP")
    if ntp!="yes":warnings.append("NTP_NOT_SYNCHRONIZED")
    if service.get("output")!="active":warnings.append("CRON_DAEMON_NOT_ACTIVE")
    if not enough:warnings.append("DISK_RESERVE_BELOW_120GB")
    if older or other_c:warnings.append("OTHER_C_CRON_STILL_ACTIVE")
    data={
        "classification":"DATASET_C_HALFSPLIT_OPERATIONAL_AUDIT_ONLY",
        "host_time":dt.datetime.now().astimezone().isoformat(),
        "status":status,"warnings":warnings,"errors":errors,
        "cron":{"daemon":service,"exact_entries":exact,"old_entries":older,
                "other_c_entries":other_c},
        "clock_ntp":ntp,
        "storage":{"mount":str(STORAGE),"total_bytes":disk.total,
                   "used_bytes":disk.used,"free_bytes":disk.free,
                   "reserve_bytes":MIN_FREE,"reserve_healthy":enough},
        "latest_half_batches":[
            {k:v for k,v in x.items() if k!="slot_ids"} for x in latest],
        "completed_reports_found":{b:len(reports[b]) for b in reports},
        "scheduler_logs":logs,
        "initial120_report":str(last_full[0]) if last_full else None,
        "files_deleted":False,"captured_sites":False,
        "scientific_capture_authorized":False
    }
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--save",action="store_true",
                   help="Save a new versioned operational audit in ~/ltd-ops")
    args=p.parse_args()
    d=audit()
    print("C_HALFSPLIT_STATUS",d["status"])
    print("C_CRON_EXACT",d["cron"]["exact_entries"])
    print("C_CRON_DAEMON",d["cron"]["daemon"].get("output"))
    print("C_NTP_SYNCHRONIZED",d["clock_ntp"])
    for b in ("first","second"):
        print("C_COMPLETED_REPORTS",b,d["completed_reports_found"][b])
        print("C_SCHEDULER_LOG",b,
              {k:v for k,v in d["scheduler_logs"][b].items() if k!="tail"})
        for line in d["scheduler_logs"][b].get("tail",[]):
            print("C_LOG_TAIL",b,line[:240])
    for x in d["latest_half_batches"]:
        print("C_LATEST_BATCH",x["batch"],"VALID",x["valid"],"/60",
              "RUN",x["run_id"],"STATUSES",x["status_counts"])
    print("C_DISK_FREE_GIB",round(d["storage"]["free_bytes"]/1024**3,1))
    print("C_DISK_RESERVE_HEALTHY",d["storage"]["reserve_healthy"])
    print("C_WARNINGS",d["warnings"])
    print("C_ERRORS",d["errors"][:5])
    if args.save:
        directory=LOGS
        directory.mkdir(parents=True,exist_ok=True)
        name="audit_halfsplit_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".json"
        path=directory/name
        path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
        path.chmod(0o600)
        print("C_AUDIT_REPORT",path)
    print("CAPTURES_PERFORMED_BY_AUDIT=0")
    print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED")
if __name__=="__main__":
    try:main()
    except Exception as exc:
        print("C_HALFSPLIT_AUDIT_ABORTED",type(exc).__name__,str(exc)[:500],
              file=sys.stderr,flush=True)
        sys.exit(4)
