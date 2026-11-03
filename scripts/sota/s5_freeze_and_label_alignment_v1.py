#!/usr/bin/env python3
"""Freeze S4 refits and establish 65-label crosswalk for S5 without inference.

Reads verified S1 label metadata, both local C checkpoint files and frozen
canonical 09B/10B NPZ candidate label arrays ONLY. No training, no scores
for hyperparameter selection, no raw Future-B, no Dataset C.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.data_adapter import load_s1_dev_index
from src.models.sota.v1.s4_protocol import sha256_file
import numpy as np

BASE=Path("/data/ltd-attack/sota/training/s4_wflib_v1_seed42")
S1=Path.home()/"ltd-ops/sota/s1_dev_frozen_20261102T130232Z/s1_manifest.json"
PROTOCOL=ROOT/"configs/sota/sota_s4_frozen_protocol_v1.json"
ROOT_SCORES=Path.home()/"ltd-storage/results/final"
OUT=Path("/data/ltd-attack/sota/s5")
EXPECTED={
 "DF":{"epochs":28,"sha":"0bb25cb4aa245af4ac545936b39bef73986cb4c034c9c31e853ca527fc3ceab3"},
 "VarCNN":{"epochs":30,"sha":"642a944c8d751d5a9e4f81714cd2cc957dab6a1badfb0ed4e23627199ee8a54f"}
}
SCORES={
 "09B":ROOT_SCORES/"INTERNAL-TEST/09B_internal_scores.npz",
 "10B_DEV_FROZEN":ROOT_SCORES/"FUTURE-B/10B_scores_DEV_FROZEN.npz"
}

def demand(ok,reason):
    if not ok:raise RuntimeError("S5_FREEZE_BLOCKED "+reason)

def inspect():
    by_uid,class_to_id,s1=load_s1_dev_index(S1)
    labels=[x for x,i in sorted(class_to_id.items(),key=lambda kv:kv[1])]
    demand(len(labels)==65 and len(by_uid)==57916,"S1 schema changed")
    selected={}
    proto_sha=sha256_file(PROTOCOL)
    for model,expected in EXPECTED.items():
        folder=BASE/model/"C"
        p=folder/"manifest.json"
        demand(p.is_file(),"missing refit manifest "+model)
        meta=json.loads(p.read_text(encoding="utf-8"))
        ck=folder/"final_model_state.pt"
        demand(ck.is_file(),"missing final model "+model)
        digest=sha256_file(ck)
        demand(digest==expected["sha"] and meta["checkpoint_sha256"]==digest,
               "checkpoint SHA mismatch "+model)
        demand(meta["epochs_completed"]==expected["epochs"] and
               meta["protocol_sha256"]==proto_sha and
               meta["S1_label_sha256"]==s1["label_order_sha256"],
               "training provenance mismatch "+model)
        demand(meta["train_splits"]==["DEV_EARLY","DEV_MIDDLE","DEV_LATE"]
               and meta["valid_splits"]==[] and
               meta["verified_counts"]=={"DEV_EARLY":19098,"DEV_MIDDLE":20160,"DEV_LATE":18658},
               "refit split counts invalid "+model)
        demand(meta["future_b_opened"] is False and
               meta["internal_test_opened"] is False and
               meta["dataset_c_opened"] is False,"external dataset used during fit "+model)
        selected[model]={"checkpoint":str(ck),"sha256":digest,
                         "epochs":expected["epochs"]}
        print("S5_CHECKPOINT_FROZEN",model,digest,flush=True)
    canonical={}
    for name,path in SCORES.items():
        demand(path.is_file(),"canonical score file missing "+name)
        with np.load(path,allow_pickle=True) as data:
            demand("candidate_labels" in data.files,
                   "canonical candidate_labels missing "+name)
            arr=data["candidate_labels"].astype(str).tolist()
        demand(len(arr)==65 and len(set(arr))==65,"bad canonical labels "+name)
        demand(set(arr)==set(labels),"candidate label set differs "+name)
        canonical[name]={"source":str(path),"source_sha256":sha256_file(path),
                         "candidate_labels_sha256":hashlib.sha256(
                             json.dumps(arr,ensure_ascii=False).encode()).hexdigest(),
                         "s1_to_canonical_indices":[arr.index(x) for x in labels],
                         "canonical_to_s1_indices":[class_to_id[x] for x in arr],
                         "identity_order":arr==labels}
        print("S5_LABEL_ALIGNMENT",name,
              "IDENTICAL" if arr==labels else "EXPLICIT_PERMUTATION",flush=True)
    same_order=(canonical["09B"]["candidate_labels_sha256"]==
                canonical["10B_DEV_FROZEN"]["candidate_labels_sha256"])
    print("S5_COHORT_CANONICAL_ORDER_MATCH",same_order,flush=True)
    # Cohorts may use different canonical orders; explicit mappings are
    # saved independently and evaluation must apply the matching permutation.
    result={"classification":"SOTA_S5_FROZEN_REFITS_AND_LABEL_CROSSWALK_V1",
            "created_utc":datetime.now(timezone.utc).isoformat(),
            "s1_label_order_sha256":s1["label_order_sha256"],
            "protocol_sha256":proto_sha,"models":selected,"label_alignment":canonical,
            "population":{"09B":14687,"10B_DEV_FROZEN":18543},
            "training_performed":False,"evaluation_performed":False,
            "external_scores_read_for_model_selection":False,
            "external_raw_features_opened":False,"dataset_c_opened":False}
    OUT.mkdir(parents=True,exist_ok=True,mode=0o700)
    target=OUT/"frozen_s5_refits_and_labels_v1.json"
    if target.exists():
        demand(target.is_file() and not target.is_symlink(),
               "existing freeze evidence is not a regular file")
        existing=json.loads(target.read_text(encoding="utf-8"))
        # Creation timestamp does not affect scientific provenance.
        comparable=lambda item:{k:v for k,v in item.items()
                                if k!="created_utc"}
        demand(comparable(existing)==comparable(result),
               "existing freeze evidence conflicts; preserved, no overwrite")
        print("S5_FREEZE_EVIDENCE=REUSED_VERIFIED_IDENTICAL",flush=True)
    else:
        with target.open("x",encoding="utf-8") as handle:
            handle.write(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
        target.chmod(0o600)
        print("S5_FREEZE_EVIDENCE=CREATED",flush=True)
    print("S5_FREEZE_LABEL_GATE=PASS_NO_INFERENCE")
    print("S5_MANIFEST",target)

if __name__=="__main__":
    try:inspect()
    except Exception as e:
        print("S5_FREEZE_ERROR",type(e).__name__,str(e)[:500],
              file=sys.stderr,flush=True)
        sys.exit(4)
