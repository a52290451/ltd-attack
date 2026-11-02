#!/usr/bin/env python3
"""S4 no-training preflight: freeze configuration, S1 chronology, and files.

Opens ONLY verified S1 DEV metadata and versioned S4 JSON. Never loads
Historical feature payload, INTERNAL_TEST, Future-B, Dataset C, GPU, or
checkpoints. Produces one small report in /data/ltd-attack/sota/protocol.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.data_adapter import load_s1_dev_index
from src.models.sota.v1.s4_protocol import (
    validate_config,examine_s1_index,sha256_file,canonical_json
)

CONFIG=ROOT/"configs/sota/sota_s4_frozen_protocol_v1.json"
S1_ROOT=Path.home()/"ltd-ops"/"sota"
REPORT_ROOT=Path("/data/ltd-attack/sota/protocol")

def insist(value,message):
    if not value:raise RuntimeError("SOTA_S4_PREFLIGHT_BLOCKED "+message)

def repo_head():
    p=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                     capture_output=True,text=True,check=True)
    return p.stdout.strip()

def plan():
    insist(CONFIG.is_file() and not CONFIG.is_symlink(),
           "frozen S4 protocol missing or linked")
    content=json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(content)
    paths=sorted(S1_ROOT.glob("s1_dev_frozen_*/s1_manifest.json"),
                 key=lambda p:p.stat().st_mtime,reverse=True)
    insist(bool(paths),"verified S1 DEV manifest missing")
    by_uid,labels,s1=load_s1_dev_index(paths[0])
    report=examine_s1_index(by_uid)
    insist(len(labels)==65,"S1 label count changed")
    insist(s1["historical_checksums"]["historical_vectors"]["sha256"]==
           content["population"]["source_vector_sha256"],
           "Historical vector SHA differs from S1")
    insist(s1["historical_checksums"]["historical_macro"]["sha256"]==
           content["population"]["source_macro_sha256"],
           "Historical macro SHA differs from S1")
    insist(s1["label_order_verified_against_frozen_score_arrays"] is False,
           "S1 unexpectedly claims external label order confirmation")
    result={
        "classification":"LTD_SOTA_S4_EXANTE_PREFLIGHT_V1",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"PASS_PLAN_NO_TRAINING",
        "git_commit":repo_head(),
        "protocol_path":str(CONFIG),
        "protocol_sha256":sha256_file(CONFIG),
        "protocol_canonical_sha256":__import__("hashlib").sha256(
            canonical_json(content)).hexdigest(),
        "s1_manifest":str(paths[0]),
        "s1_dev_index_sha256":s1["dev_index_sha256"],
        "s1_label_order_sha256":s1["label_order_sha256"],
        "s1_temporal_assignment_method":
            s1["temporal_assignments_provenance"]["method"],
        "original_s1_assignments_verified":
            s1["temporal_assignments_provenance"].get("original_assignment_hash_verified"),
        "s1_chronology":report,
        "predeclared_model_names":sorted(content["model_configs"]),
        "predeclared_stages":[s["id"] for s in content["schedule"]],
        "primary_seeds":content["seeds"]["primary"],
        "external_score_class_order_verified":False,
        "ready_to_train":False,
        "ready_to_evaluate_external":False,
        "datasets_opened":["S1_DEV_METADATA_ONLY"],
        "future_b_opened":False,
        "dataset_c_opened":False,
        "training_steps":0,
    }
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--save",action="store_true",
                    help="Write report JSON under /data/ltd-attack/sota")
    args=ap.parse_args()
    info=plan()
    if args.save:
        REPORT_ROOT.mkdir(mode=0o700,parents=True,exist_ok=True)
        path=REPORT_ROOT/("s4_preflight_"+datetime.now(timezone.utc).
                          strftime("%Y%m%dT%H%M%SZ")+".json")
        insist(not path.exists(),"report collision")
        path.write_text(json.dumps(info,indent=2,ensure_ascii=False)+"\n",
                        encoding="utf-8")
        path.chmod(0o600)
        print("SOTA_S4_PREFLIGHT_REPORT",path)
    print("SOTA_S4_PROTOCOL_SHA256",info["protocol_sha256"])
    print("SOTA_S4_DEV_SPLITS",info["s1_chronology"]["split_counts"])
    print("SOTA_S4_DEV_DAYS",info["s1_chronology"]["days_per_split"])
    print("SOTA_S4_UPSTREAM_BASELINES",info["predeclared_model_names"])
    print("SOTA_S4_PRIMARY_SEEDS",info["primary_seeds"])
    print("SOTA_S4_PREFLIGHT=PASS_PLAN_NO_TRAINING")
    print("SOTA_S4_TRAINING=NOT_STARTED")
    print("FUTURE_B_UNTOUCHED=YES")

if __name__=="__main__":
    try:main()
    except Exception as exc:
        print("SOTA_S4_PREFLIGHT_ABORTED",type(exc).__name__,str(exc)[:600],
              file=sys.stderr,flush=True)
        sys.exit(4)
