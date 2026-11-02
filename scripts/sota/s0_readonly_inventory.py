#!/usr/bin/env python3
"""LTD-Attack SOTA S0: read-only scientific/reproducibility inventory.

This is not a training job. Does not open Future-B data, read secrets, run
Docker, install dependencies, or change datasets/models/branches. Only reads
tracked source-code filenames and conservative environment metadata. Saves a
small JSON report in the user's personal operational directory.
"""
from __future__ import annotations
import collections
import datetime as dt
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
TERMS={
    "frozen":["frozen","freeze","manifest","immutable","hash","sha256"],
    "historical_dev":["historical","history","hist","dev","training"],
    "future_b":["future","future_b","future-b","posthoc","post_hoc"],
    "modeling":["xgboost","hybrid","macro","micro","train","classifier","predict"],
    "evaluation":["mrr","metric","evaluation","bootstrap","pairwise","ablation","statistics"],
    "data_pipeline":["feature","extract","preprocess","trace","pcap","direction","tor"],
}
EXCLUDE_PARTS={".git",".venv","venv","node_modules",".agent_results","__pycache__"}
MODULES={
    "python":"python",
    "numpy":"numpy",
    "pandas":"pandas",
    "sklearn":"scikit-learn",
    "torch":"torch",
    "tensorflow":"tensorflow",
    "xgboost":"xgboost",
    "scipy":"scipy",
}

def run(command,timeout=20):
    return subprocess.run(command,cwd=ROOT,check=True,stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,timeout=timeout)

def tracked_files():
    raw=run(["git","ls-files","-z"]).stdout
    return [p.decode("utf-8","replace") for p in raw.split(b"\0") if p]

def inspect_environment():
    found={}
    for name,dist in MODULES.items():
        try:
            exists=importlib.util.find_spec(name) is not None
            try:version=importlib.metadata.version(dist) if exists else None
            except importlib.metadata.PackageNotFoundError:version=None
            found[name]={"installed":exists,"version":version}
        except (ValueError,ImportError) as exc:
            found[name]={"installed":False,"reason":type(exc).__name__}
    return {"python":sys.version.split()[0],
            "platform":sys.platform,
            "packages":found,
            "gpu_cli_present":bool(shutil.which("nvidia-smi"))}

def inspect_root_data_locations():
    # Strict nonrecursive stat only: never read trace bytes, labels, or datasets.
    candidates=[
        ROOT/"data"/"local",
        ROOT/"experiments",
        Path.home()/"ltd-storage",
        Path("/data"),
    ]
    values=[]
    for p in candidates:
        try:
            st=p.stat()
            values.append({"path":str(p),"exists":True,"directory":p.is_dir(),
                           "size_of_directory_entry_bytes":st.st_size})
        except OSError:
            values.append({"path":str(p),"exists":False})
    return values

def text_matches(paths):
    result={}
    for section,terms in TERMS.items():
        entries=[p for p in paths if any(term in p.lower() for term in terms)]
        result[section]={"count":len(entries),"file_paths":entries[:80],
                         "truncated":len(entries)>80}
    return result

def classify(paths):
    categories=collections.Counter(p.split("/")[0] for p in paths)
    code_suffixes={".py",".sh",".yaml",".yml",".json",".toml",".md"}
    # Static name-only inventory. Future-B content is never accessed.
    candidates=[]
    for p in paths:
        name=Path(p).name.lower()
        if Path(p).suffix.lower() not in code_suffixes:continue
        if ("10c" in name or "14a" in name or "ablation" in name or
            "pairwise" in name or "bootstrap" in name or "train" in name or
            "evaluate" in name or "sota" in name or "extract" in name or
            "frozen" in name):
            candidates.append(p)
    return {"tracked_file_count":len(paths),
            "tracked_top_level_counts":dict(categories),
            "candidate_entrypoints":candidates[:130],
            "candidate_entrypoints_truncated":len(candidates)>130}

def main():
    if not (ROOT/".git").exists():
        raise RuntimeError("Run on the checked-out LTD-Attack repository")
    branch=run(["git","branch","--show-current"]).stdout.decode().strip()
    if branch!="feature/macro-v2-historical-only":
        raise RuntimeError("Unexpected scientific branch "+branch)
    commit=run(["git","rev-parse","HEAD"]).stdout.decode().strip()
    files=tracked_files()
    if not files or not any(p.startswith("src/") for p in files):
        raise RuntimeError("Research code or git tracked inventory missing")
    report={
      "classification":"SOTA_S0_READ_ONLY_SOURCE_DISCOVERY",
      "timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
      "repo":str(ROOT),"branch":branch,"git_head":commit,
      "future_b_data_opened":False,"historical_data_modified":False,
      "model_training_started":False,
      "inventory":classify(files),
      "groups":text_matches(files),
      "environment":inspect_environment(),
      "data_directories_metadata_only":inspect_root_data_locations(),
      "methods_shortlist":["Deep Fingerprinting (DF)","Var-CNN","NetCLR",
                           "Proteus (adaptive protocol separately)"],
      "next_gate":"Confirm frozen Historical DEV sample IDs, the A/B extractor, labels, "
                  "dates, and fixed Future-B evaluator before writing tensor adapters"
    }
    destination=Path.home()/"ltd-ops"/"sota"
    destination.mkdir(parents=True,exist_ok=True)
    outfile=destination/("s0_inventory_"+dt.datetime.now(dt.timezone.utc)
                        .strftime("%Y%m%dT%H%M%SZ")+".json")
    outfile.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("SOTA_S0_STATUS=PASS")
    print("SOTA_S0_BRANCH",branch)
    print("SOTA_S0_GIT_HEAD",commit)
    print("SOTA_S0_TRACKED_FILES",len(files))
    print("SOTA_S0_TOP_LEVEL",report["inventory"]["tracked_top_level_counts"])
    for section,items in report["groups"].items():
        print("SOTA_S0_GROUP",section,items["count"])
        for p in items["file_paths"][:12]:
            print("SOTA_S0_FILE",section,p)
    print("SOTA_S0_ENVIRONMENT",report["environment"])
    print("SOTA_S0_REPORT",outfile)
    print("FUTURE_B_UNTOUCHED=YES")
    print("SOTA_TRAINING_NOT_STARTED=YES")

if __name__=="__main__":
    try:main()
    except Exception as exc:
        print("SOTA_S0_BLOCKED",type(exc).__name__,str(exc),file=sys.stderr)
        sys.exit(4)
