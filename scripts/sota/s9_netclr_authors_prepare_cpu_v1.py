#!/usr/bin/env python3
"""S9 author-source CPU-only preflight and Historical DEV cache verification.

Steps (all without CUDA):
1) verify pinned NetCLR authors' notebook and instantiate its original model;
2) verify frozen Historical DEV index and generate/verify 57,916 direction
   traces in the existing provenance-controlled cache;
3) check exact chronological counts and 65-class support;
4) verify the author's source pretraining adapter is not yet substituted
   with the WFlib model, and report remaining augmentation/protocol gates.

Read/write only the S9 supervised DEV cache if missing; never access 09B,
Future-B or Dataset C. No training/checkpoint writes.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.sota import s9_netclr_supervised_b_v1 as B

def main():
    gate=ROOT/"scripts/sota/s9_netclr_authors_cpu_gate_v1.py"
    result=subprocess.run([sys.executable,"-u",str(gate),
                           "--stage","smoke"],check=True)
    X,y,s,manifest=B.prepare_cache()
    assert X.shape==(57916,5000),X.shape
    assert y.shape==(57916,),y.shape
    assert s.shape==(57916,),s.shape
    import numpy as np
    hist=dict(zip(*np.unique(s,return_counts=True)))
    assert hist=={0:19098,1:20160,2:18658},hist
    assert set(map(int,np.unique(y)))==set(range(65))
    assert manifest["external_data_opened"] is False
    assert manifest["dataset_c_opened"] is False
    print("S9_AUTHOR_HISTORICAL_DEV_CPU_PREFLIGHT_PASS",
          json.dumps({"population":int(len(y)),"classes":65,
            "chronological_split_counts":hist,
            "authors_original_model_cpu_smoke":True,
            "authors_original_pretraining_completed":False,
            "authors_original_augmentation_runtime_approved":False,
            "dataset_cache_path":str(B.CACHE),
            "cache_manifest_sha256":B.P.sha(B.CACHE/"manifest.json")},
            indent=2,default=str),flush=True)
    print("NEXT_GATE_AUTHOR_AUGMENT_FIDELITY_AND_HISTORICAL_PRETRAINING",
          flush=True)

if __name__=="__main__":
    main()
