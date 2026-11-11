#!/usr/bin/env python3
"""Read-only detailed progress of original SOTA models and frozen external evaluation.

No CUDA allocations. Reads only stage manifests, SHA256 checkpoints, background
PID, and last matching training events from logs, without modifying artifacts.
"""
from __future__ import annotations
import json,os,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.sota import original_sota_orchestrator_v2 as v2
from scripts.sota.sota_paths_v1 import ARTIFACT
TRACK=ARTIFACT/"background_battery_v1"
def latest_log(model,stage):
    if model=="NetCLR":
        glob=ARTIFACT/"s9_netclr_author_original_v1/logs"
    elif model in ("DF","VarCNN","TikTok","RoFiRe"):
        glob=ARTIFACT/"authors_keras_logs"
    else:
        glob=v2.ROOTS[model]/"logs"
    pattern="*_"+model+"-"+stage+".log" if model in ("DF","VarCNN","TikTok","RoFiRe") else "*_"+stage+".log"
    matches=sorted(glob.glob(pattern))
    return matches[-1] if matches else None
def last_event(p):
    if not p or not p.is_file():return ""
    # Read only final 32 KiB of a log, rather than huge epoch output.
    with p.open("rb") as f:
        f.seek(0,2);f.seek(max(0,f.tell()-32768))
        raw=f.read().decode("utf-8",errors="replace")
    candidates=[s.strip() for s in raw.splitlines() if
        ("_EPOCH" in s or "_PASS" in s or "_FAILED" in s or
         "Epoch " in s or "step " in s)]
    return candidates[-1][:240] if candidates else ""
def run_status():
    p=TRACK/"latest_run"
    if not p.exists():return "NOT_STARTED"
    d=TRACK/p.read_text().strip()
    return (d/"status").read_text().strip() if (d/"status").exists() else "STARTING"
def main():
    report={"classification":"LTD_SOTA_MODEL_READONLY_PROGRESS_V1",
            "host":os.uname().nodename,
            "background_status":run_status(),"models":{}}
    for model in v2.MODELS:
        records={}
        for stage,folder,klass,cohort in v2.stages(model,"all"):
            try:
                st=v2.state(model,stage,folder,klass,cohort)
            except (RuntimeError,ValueError,KeyError,FileNotFoundError) as e:
                st="BLOCKED_INVALID_EVIDENCE: "+str(e)[:150]
            p=latest_log(model,stage)
            records[stage]={"status":st,
                            "last_event":last_event(p),
                            "log":str(p) if p else None}
        report["models"][model]=records
    if "--json" in sys.argv:
        print(json.dumps(report,indent=2))
    else:
        print("HOST",report["host"],"BATTERY",report["background_status"])
        for model,records in report["models"].items():
            print("\nMODEL",model)
            for stage,x in records.items():
                print(" ",stage.ljust(12),x["status"])
                if x["last_event"]:print("    "+x["last_event"])
                if x["log"] and x["status"]!="PASS_VERIFIED":
                    print("    log:",x["log"])
if __name__=="__main__":main()
