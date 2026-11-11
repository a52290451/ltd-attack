#!/usr/bin/env python3
"""SOTA legacy original-source pinning and audit (2016-2023).

Do NOT execute original training scripts: they often assume random split,
historical TensorFlow/CUDA and public datasets with different labels. This
gate only fetches exact Git objects into isolated checkouts; it does NOT
replace any frozen LTD predictions or authorize model training.

Use: python3 scripts/sota/s3_s9_original_sources_gate_v1.py --stage fetch
     python3 scripts/sota/s3_s9_original_sources_gate_v1.py --stage inspect
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path.home()/"ltd-ops"/"sota"/"upstream_original_2026"
REPOS={
 "S3_DF_Authors":{
    "url":"https://github.com/deep-fingerprinting/df.git",
    "commit":"38df0c15a089f13228e4df06bd5269c8a37340fa",
    "paths":["src/Model_NoDef.py","src/ClosedWorld_DF_NoDef.py","README.md"],
    "kind":"authors_original",
    "note":"TensorFlow/Keras original; original directory loaders and epoch settings require Historical DEV adapters"},
 "S4_VarCNN_Authors":{
    "url":"https://github.com/sanjit-bhat/Var-CNN.git",
    "commit":"e5db76a86fcbf8839f764b9aca241749d1c1f700",
    "paths":["var_cnn.py","run_model.py","README.md","config.json"],
    "kind":"authors_original",
    "note":"Archived TensorFlow/Keras with direction/timing/metadata ensemble. Original random-split pipeline MUST be bypassed"},
 "S7_CUMUL_PublicReproduction":{
    "url":"https://github.com/lsvih/CUMUL.git",
    "commit":"da1297a5b3bd8ba0a138e666f55686c109ff69c1",
    "paths":["main.py","utils/utils.py","README.md"],
    "kind":"public_reimplementation_NOT_verified_authors_original",
    "note":"Public implementation claiming derivation from authors' earlier code; inspect feature extraction and packet semantics before replacing CUMUL-style"},
 "S8_TikTok_Authors":{
    "url":"https://github.com/msrocean/Tik_Tok.git",
    "commit":"0aa56660b435d88f3542939b3f08c0ba1409dbf5",
    "paths":["DL_Experiments/cw_attack.py","DL_Experiments/data_utils.py",
             "README.md"],
    "kind":"authors_original",
    "note":"TensorFlow 1.14 / Keras 2.3 and legacy CUDA. Official DT direction*timestamps representation"},
 "S9_NetCLR_Authors":{
    "url":"https://github.com/SPIN-UMass/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces.git",
    "commit":"c53ca10df0cad6bd5c8b27301a10dcc464278187",
    "paths":["artifacts/src/NetCLR/pre-training.ipynb",
             "artifacts/src/NetCLR/fine-tuning-cw.ipynb",
             "artifacts/src/NetCLR/README.md"],
    "kind":"authors_original",
    "note":"Original notebooks include augmentation and contrastive training; must audit structure and supervision against frozen Historical DEV, no external adaptation"},
}

def command(args,timeout=180):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if p.returncode:
        raise RuntimeError("command failed: "+" ".join(args[:3])+" "+p.stderr[-300:])
    return p.stdout.strip()

def inspect(key,record):
    path=ROOT/key
    if not (path/".git").is_dir():
        return {"id":key,"status":"MISSING","path":str(path)}
    try:
        sha=command(["git","-C",str(path),"rev-parse","HEAD"])
        dirty=command(["git","-C",str(path),"status","--porcelain",
                       "--untracked-files=no"])
        absent=[rel for rel in record["paths"] if not (path/rel).is_file()]
        status=("PINNED_PUBLIC_SOURCE_PASS" if sha==record["commit"]
                and not dirty and not absent else "BLOCKED")
        return {"id":key,"path":str(path),"commit":sha,
                "source_class":record["kind"],"status":status,
                "missing_paths":absent,"dirty":bool(dirty),"note":record["note"]}
    except Exception as e:
        return {"id":key,"status":"ERROR","detail":str(e)[:450]}

def fetch(key,record):
    dst=ROOT/key
    if dst.exists():
        print("EXISTING_SOURCE_REFUSE_OVERWRITE",key,flush=True)
        return
    ROOT.mkdir(parents=True,exist_ok=True)
    command(["git","clone","--no-checkout",record["url"],str(dst)],timeout=300)
    command(["git","-C",str(dst),"checkout","--detach",record["commit"]])
    print("SOURCE_FETCHED",key,record["commit"],flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("inspect","fetch"),required=True)
    args=parser.parse_args()
    if args.stage=="fetch":
        for key,record in REPOS.items():fetch(key,record)
    results=[inspect(k,v) for k,v in REPOS.items()]
    print("S3_S9_ORIGINAL_PROVENANCE_AUDIT",
          json.dumps(results,indent=2,ensure_ascii=False),flush=True)
    if not all(x["status"]=="PINNED_PUBLIC_SOURCE_PASS" for x in results):
        raise SystemExit(3)
    print("S3_S9_ORIGINAL_SOURCES_PIN_PASS",flush=True)
    print("NO_HISTORICAL_BENCHMARK_EXECUTED; ORIGINAL_DATA_ADAPTER_GATE_REQUIRED",
          flush=True)

if __name__=="__main__":main()
