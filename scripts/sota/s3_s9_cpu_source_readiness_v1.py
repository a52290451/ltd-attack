#!/usr/bin/env python3
"""CPU/std-lib source audit for already-evaluated SOTA author/public checkouts.

No dynamic imports of authors' code, no GPU, no training or dataset access.
Reports original framework and what must be adapted before chronological runs.
This is a SOURCE audit, NOT an architecture parity proof or benchmark result.
"""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

ROOT=Path.home()/"ltd-ops/sota/upstream_original_2026"
SOURCE={
 "S3_DF_Authors":{
   "paths":["src/Model_NoDef.py","src/ClosedWorld_DF_NoDef.py"],
   "framework":"TensorFlow/Keras original",
   "required_integration":"original Keras topology, 65-class labels, Historical DEV-only chronological fit and frozen inference"},
 "S4_VarCNN_Authors":{
   "paths":["var_cnn.py","preprocess_data.py","run_model.py"],
   "framework":"TensorFlow/Keras original",
   "required_integration":"original residual direction/time/metadata ensemble, bypass random splits, export calibrated 65-class predictions"},
 "S7_CUMUL_PublicReproduction":{
   "paths":["utils/data.py","utils/struct.py"],
   "framework":"signed-size engineering and SVM via third-party public reproduction",
   "required_integration":"verify signed payload/packet bytes and 100 cumulative interpolants, training-only scaling and RBF-SVM fit"},
 "S8_TikTok_Authors":{
   "paths":["DL_Experiments/cw_attack.py",
            "DL_Experiments/data_utils.py",
            "Timing_Features/Tik_Tok_timing_features.py"],
   "framework":"original TikTok legacy Keras code",
   "required_integration":"exact directional timing representation, original model topology and optimizer, chronological split"},
 "S9_NetCLR_Authors":{
   "paths":["artifacts/src/NetCLR/pre-training.ipynb",
            "artifacts/src/NetCLR/fine-tuning-cw.ipynb"],
   "framework":"authors' PyTorch notebooks, NetAugment + DFsimCLR",
   "required_integration":"resolve NetAugment CDF index, preserve original augmentation, full Historical DEV pretrain/finetune, 65-class frozen inference"},
}
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def run(command):
    r=subprocess.run(command,text=True,capture_output=True,timeout=15)
    return r.stdout.strip() if r.returncode==0 else ""
def main():
    report=[]
    for key,spec in SOURCE.items():
        base=ROOT/key
        files={}
        for rel in spec["paths"]:
            p=base/rel
            files[rel]={"exists":p.is_file(),
                        "sha256":sha(p) if p.is_file() else None,
                        "bytes":p.stat().st_size if p.is_file() else None}
        clean=not run(["git","-C",str(base),"status","--porcelain",
                       "--untracked-files=no"])
        head=run(["git","-C",str(base),"rev-parse","HEAD"])
        ok=bool(head) and clean and all(x["exists"] for x in files.values())
        report.append({"id":key,"status":"ORIGINAL_SOURCE_INVENTORY_PASS" if ok
                      else "BLOCKED","commit":head,
                      "tracked_clean":clean,"files":files,
                      "framework":spec["framework"],
                      "required_integration":spec["required_integration"],
                      "gpu_used":False,"training_started":False})
    print("S3_S9_CPU_SOURCE_INVENTORY",json.dumps(report,indent=2))
    if not all(r["status"]=="ORIGINAL_SOURCE_INVENTORY_PASS" for r in report):
        raise SystemExit(3)
    print("S3_S9_CPU_SOURCE_AUDIT_PASS; NO_TRAINING_AUTHORIZED")
if __name__=="__main__":main()
