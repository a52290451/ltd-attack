#!/usr/bin/env python3
"""Read-only compute capacity preflight while Dataset C QA cron runs.

Do not install packages, initialize GPU, or run models. Record machine
resources/driver and current interpreter for a later isolated PyTorch plan.
Does not access Historical/09B/Future-B/C data or trained checkpoints.
"""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

OPS=Path.home()/"ltd-ops"/"sota"
CAPTURE_BASE=Path("/data/ltd-c-qa-bsierra/C_DIRECT_V1")
DATA_MOUNT=Path("/data")
MIN_FREE=120_000_000_000

def call(args,timeout=12):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
        return {"returncode":p.returncode,
                "stdout":p.stdout.strip()[:4000],
                "stderr":p.stderr.strip()[:500]}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {"returncode":None,"error":type(exc).__name__}

def meminfo():
    fields={}
    p=Path("/proc/meminfo")
    if p.is_file():
        for line in p.read_text().splitlines():
            k,sep,v=line.partition(":")
            if sep and k in ("MemTotal","MemAvailable","SwapTotal","SwapFree"):
                fields[k+"_KiB"]=int(v.strip().split()[0])
    return fields

def runtime():
    usage=shutil.disk_usage(DATA_MOUNT)
    info={
        "classification":"SOTA_S3_READONLY_RESOURCE_PREFLIGHT",
        "captured_local":dt.datetime.now().astimezone().isoformat(),
        "python":sys.executable,
        "python_version":sys.version.split()[0],
        "cpu_logical":os.cpu_count(),
        "cpu_memory":meminfo(),
        "gpu_driver_query":call(["nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits"],timeout=15),
        "gpu_driver_summary":call(["nvidia-smi","-L"],timeout=15),
        "package_availability":{
            name:importlib.util.find_spec(name) is not None
            for name in ("torch","numpy","pandas","scipy","sklearn")
        },
        "python_install_isolation":sys.prefix!=sys.base_prefix,
        "disk_data":{
            "mount":str(DATA_MOUNT),
            "total_bytes":usage.total,
            "free_bytes":usage.free,
            "reserve_bytes":MIN_FREE,
            "reserve_healthy":usage.free>=MIN_FREE
        },
        "dataset_c_qa_root_exists":CAPTURE_BASE.is_dir(),
        "capture_scheduler_processes":call(["pgrep","-af",
            r"run_direct120_halfsplit_qa_v2.py|run_direct120_recurrent_qa_v1.py"]),
        "training_executed":False,"future_b_opened":False,
        "dataset_c_changed":False,
        "gpu_training_approved":False,
    }
    return info

def main(args):
    report=runtime()
    print("SOTA_RESOURCE_PYTHON",report["python"])
    print("SOTA_RESOURCE_IS_VENV",report["python_install_isolation"])
    print("SOTA_RESOURCE_CPU_COUNT",report["cpu_logical"])
    print("SOTA_RESOURCE_MEMORY",report["cpu_memory"])
    print("SOTA_RESOURCE_PACKAGES",report["package_availability"])
    print("SOTA_RESOURCE_NVIDIA",report["gpu_driver_query"])
    print("SOTA_RESOURCE_DISK_FREE_GIB",
          round(report["disk_data"]["free_bytes"]/1024**3,1))
    print("SOTA_RESOURCE_DISK_RESERVE_OK",report["disk_data"]["reserve_healthy"])
    print("SOTA_RESOURCE_C_ACTIVE",report["capture_scheduler_processes"]["stdout"][:500])
    if args.save:
        OPS.mkdir(mode=0o700,parents=True,exist_ok=True)
        output=OPS/("s3_runtime_resources_"+dt.datetime.now(dt.timezone.utc).
                    strftime("%Y%m%dT%H%M%SZ")+".json")
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        output.chmod(0o600)
        print("SOTA_RESOURCE_REPORT",output)
    print("SOTA_TRAINING_GATE=PENDING")
    print("FUTURE_B_UNTOUCHED=YES")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--save",action="store_true")
    a=ap.parse_args()
    try:main(a)
    except Exception as exc:
        print("SOTA_RESOURCE_ABORTED",type(exc).__name__,str(exc)[:500],
              file=sys.stderr,flush=True)
        sys.exit(4)
