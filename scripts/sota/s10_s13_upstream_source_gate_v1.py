#!/usr/bin/env python3
"""Pin and audit original S10-S12 authors' code without installing or training.

--stage inspect: read-only verify cloned sources (no downloads).
--stage fetch: clone upstream to isolated ops path and checkout exact commit.
Never modifies WFlib, Historical DEV, A/B, C or Git repository tracked files.
No automatic dependency installs, data access or model training.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.sota.sota_paths_v1 import MODERN
ROOT=MODERN
REPOS={
 "S10_RoFiRe":{"url":"https://github.com/AISec-EWHA/RoFiRe.git",
   "commit":"7ef45ee7c9af9ac04048456992d3d96398791bbc",
   "paths":["model/DF.py","model/main.py","feature/features.py",
            "auto_aug.py","requirements.txt"],
   "notes":"Authors' TensorFlow/Keras DFVarNet with augmentation and normalization; original loaders use fixed NPZ and 8:1:1 random split; not LTD-ready."},
 "S11_Swallow":{"url":"https://github.com/wujinhe0814/Swallow.git",
   "commit":"c6dc304bca6e61b10b36b00178ee84e38c17961b",
   "paths":["PreTrain.py","fineTuning.py","ProcessData.py",
            "models/resnet_base_network.py","config.yaml"],
   "notes":"Authors' ResNet and representation/pretraining; source expects its own scenarios and config; requires chronological 65-class loader integration."},
 "S12_WFCAT":{"url":"https://github.com/websitefingerprinting/wfcat.git",
   "commit":"cd382d30f7d33714fa4d025cddb3daaf2ffb0f09",
   "paths":["src/models/wfcat.py","src/features.py","src/train.py",
            "src/data.py","requirements.txt"],
   "notes":"Authors' PyTorch WFCAT attention/IAT; original training uses stratified folds over .cell files, not Historical DEV split."},
 "S13_Proteus_Reference":{"url":"https://github.com/Xinhao-Deng/Adaptive-WF-Attack.git",
   "commit":"4cdab4163bf3de7036a2498dac533c888b97664d",
   "paths":["README.md","scripts/TemporalDrift/Proteus.sh"],
   "notes":"Explicit future/unlabeled adaptation; related work only unless separate leakage-safe protocol is designed."},
}
def run(args):
    return subprocess.run(args,check=True,capture_output=True,text=True,timeout=180).stdout.strip()
def inspect(name,spec):
    target=ROOT/name
    if not (target/".git").exists():
        return {"name":name,"status":"MISSING","path":str(target)}
    try:
        head=run(["git","-C",str(target),"rev-parse","HEAD"])
        dirty=run(["git","-C",str(target),"status","--porcelain",
                   "--untracked-files=no"])
        absent=[p for p in spec["paths"] if not (target/p).is_file()]
        return {"name":name,"path":str(target),"commit":head,
                "commit_match":head==spec["commit"],
                "tracked_dirty":bool(dirty),"missing_paths":absent,
                "status":("PINNED_SOURCE_PASS" if head==spec["commit"]
                          and not dirty and not absent else "BLOCKED"),
                "notes":spec["notes"]}
    except Exception as e:
        return {"name":name,"status":"ERROR","detail":str(e)[:400]}
def fetch(name,spec):
    target=ROOT/name
    if target.exists():
        print("EXISTING_CHECKOUT_NO_OVERWRITE",name)
        return
    ROOT.mkdir(parents=True,exist_ok=True)
    run(["git","clone","--no-checkout",spec["url"],str(target)])
    run(["git","-C",str(target),"checkout","--detach",spec["commit"]])
    print("PINNED_UPSTREAM_FETCHED",name,spec["commit"])
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["inspect","fetch"],required=True)
    args=ap.parse_args()
    if args.stage=="fetch":
        for name,spec in REPOS.items():
            fetch(name,spec)
    results=[inspect(k,v) for k,v in REPOS.items()]
    print("S10_S13_ORIGINAL_SOURCE_AUDIT",
          json.dumps({"date_utc":datetime.now(timezone.utc).isoformat(),
                      "results":results},indent=2))
    if not all(r["status"]=="PINNED_SOURCE_PASS" for r in results):
        sys.exit(3)
    print("S10_S13_SOURCE_PIN_PASS")
    print("NEXT_GATE: Chronological dataset adapter, original model interface,"
          " 65-class head, source-equivalent augmentation and UID-aligned export."
          " Source pin alone does NOT authorize training.")
if __name__=="__main__":main()
