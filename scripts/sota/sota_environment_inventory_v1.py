#!/usr/bin/env python3
"""Portable package inventory, no installs, no secrets, no GPU allocation.

Capture exact package versions for each independent Python environment on
Perseo, then compare Zeus to that reference. Uses stdlib importlib.metadata,
does not import torch or tensorflow and never touches GPU. It intentionally
does not export PIP_INDEX_URL/credentials, environment variables or pip VCS
URLs that might contain access tokens.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.sota.sota_paths_v1 import ARTIFACT
ENV={
    "pytorch":os.environ.get("LTD_SOTA_PYTORCH_PY",
            "/data/ltd-sota-bsierra/venv-cu128/bin/python"),
    "keras":os.environ.get("LTD_SOTA_KERAS_PY",
            "/data/ltd-sota-bsierra/venv-keras-authors/bin/python")}
CODE='''import json,sys,importlib.metadata as m
ds={}
for d in m.distributions():
    name=(d.metadata.get("Name") or "").strip().lower().replace("_","-")
    if name: ds[name]=d.version
print(json.dumps({"python":sys.version.split()[0],"packages":ds},sort_keys=True))
'''
CRITICAL={
    "pytorch":["torch","torchvision","numpy","scikit-learn"],
    "keras":["tensorflow","keras","numpy","scikit-learn"],
}
def read_env():
    result={}
    for name,python in ENV.items():
        p=subprocess.run([python,"-c",CODE],capture_output=True,
                         text=True,timeout=60)
        if p.returncode:
            raise RuntimeError("SOTA_ENV_BLOCKED "+name+": "+p.stderr[-400:])
        x=json.loads(p.stdout.strip())
        missing=[d for d in CRITICAL[name] if d not in x["packages"]]
        if missing:raise RuntimeError("SOTA_ENV_BLOCKED missing "+
                                      name+": "+repr(missing))
        result[name]=x
    return result
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("capture","compare"),required=True)
    ap.add_argument("--reference",type=Path)
    ap.add_argument("--output",type=Path)
    opts=ap.parse_args()
    inventory=read_env()
    host=os.uname().nodename
    if opts.stage=="capture":
        output=opts.output or (
            ARTIFACT/"portability"/(host+"_sota_environments_v1.json"))
        if output.exists():raise RuntimeError("inventory exists; no overwrite")
        output.parent.mkdir(parents=True,exist_ok=True)
        obj={"classification":"SOTA_AUTHOR_ENV_INVENTORY_V1",
             "host":host,"created_utc":datetime.now(timezone.utc).isoformat(),
             "environments":inventory}
        output.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
        print("SOTA_ENV_CAPTURE_PASS",str(output))
    else:
        if not opts.reference:
            raise RuntimeError("--reference is required to compare")
        old=json.loads(opts.reference.read_text())
        for env in CRITICAL:
            for pkg in CRITICAL[env]:
                actual=inventory[env]["packages"][pkg]
                expected=old["environments"][env]["packages"][pkg]
                if actual!=expected:
                    raise RuntimeError("SOTA_ENV_INCOMPATIBLE "+env+" "+
                                       pkg+" "+actual+" expected "+expected)
        print("SOTA_ENV_CRITICAL_VERSIONS_PASS",host)
if __name__=="__main__":main()
