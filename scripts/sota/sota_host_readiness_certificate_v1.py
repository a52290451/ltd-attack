#!/usr/bin/env python3
"""Per-host immutable CPU readiness certificates for LTD original SOTA.

Only records certificates after the enclosing shell stages all succeeded.
A changed Git commit or modified source invalidates the certificates.
Host identity prevents a PASS from Perseo being reused on Zeus.
Certificates are operational metadata, not experiment result artifacts.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.sota.sota_paths_v1 import ARTIFACT

STAGES=("cpu-preflight","cpu-integration","cpu-prepare")
def cmd(*args):
    p=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,check=True)
    return p.stdout.strip()
def require(ok,msg):
    if not ok:raise RuntimeError("SOTA_READINESS_CERTIFICATE_BLOCKED "+msg)
def reference():
    commit=cmd("git","rev-parse","HEAD")
    dirty=cmd("git","status","--porcelain","--untracked-files=no",
              "--","scripts/sota","src/models/sota")
    require(not dirty,"tracked SOTA code modified outside commit")
    return {"commit":commit,"host":os.uname().nodename,
            "stage_set":list(STAGES),"protocol":"57,916 Historical DEV, 65 classes",
            "external_eval":["09B","10B_DEV_FROZEN"],
            "dataset_c_used":False}
def path(stage,meta):
    return ARTIFACT/"original_sota_readiness_v1"/(
        meta["host"]+"_"+stage+".json")
def record(stage):
    meta=reference()
    file=path(stage,meta)
    data=dict(meta,stage=stage,status="PASS",
              created_utc=datetime.now(timezone.utc).isoformat())
    file.parent.mkdir(parents=True,exist_ok=True)
    # Certificate is replaceable operational metadata; frozen evidence isn't.
    temp=file.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data,indent=2)+"\n")
    temp.replace(file)
    print("SOTA_HOST_STAGE_CERTIFIED",stage,str(file),flush=True)
def inspect(required):
    meta=reference()
    for stage in required:
        file=path(stage,meta)
        require(file.is_file(),"missing "+stage+" certificate on this host")
        data=json.loads(file.read_text())
        require(data.get("stage")==stage and data.get("status")=="PASS"
                and data.get("commit")==meta["commit"]
                and data.get("host")==meta["host"]
                and data.get("dataset_c_used") is False,
                "invalid or stale "+stage+" certificate")
        print("SOTA_HOST_STAGE_CERTIFICATE_PASS",stage,meta["commit"][:12])
    print("SOTA_HOST_READINESS_CERTIFICATES_PASS",len(required),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("record","inspect"),required=True)
    ap.add_argument("--name",choices=STAGES)
    a=ap.parse_args()
    if a.stage=="record":
        require(a.name is not None,"--name required")
        record(a.name)
    else:inspect(STAGES if a.name is None else (a.name,))
if __name__=="__main__":main()
