#!/usr/bin/env python3
"""Frozen SOTA S5 inference on 09B and 10B DEV_FROZEN.

No gradient/training, no calibration, no model selection, no Dataset C. Read
only historical / future frozen vector CSV, S4 SHA-verified checkpoints, and
original canonical score cohort IDs/labels. Never changes frozen LTD scores.
Output new aligned probabilities and provenance under /data.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.data_adapter import parse_direction,fixed_length
from src.models.sota.v1.timing_adapter import dt2_from_text
from src.models.sota.v1.s4_protocol import sha256_file
from src.models.sota.v1.upstream_models import get_model,verify_checkout

STAGE=Path("/data/ltd-attack/sota")
EVIDENCE=STAGE/"s5/frozen_s5_refits_and_labels_v1.json"
OUT=STAGE/"s5/inference"
HIST=Path.home()/"ltd-storage/datasets/historical/CLEAN_final_vectors_sites.csv"
FUTURE=Path.home()/"ltd-storage/datasets/future/CLEAN_final_vectors_sites_concept_drift.csv"
EXPECTED_HIST="a8a88041c44e5627e126e2c29c7e068b17c8598a0b01c63a5ff1cd3105c2c7c9"
EXPECTED_FUTURE="eecf3cef8079aba6b52cfaa4ba125a6fcf3212e54bb009f2977c24e71ae418a4"
N={"09B":14687,"10B_DEV_FROZEN":18543}
MODELS=("DF","VarCNN")

def check(ok,why):
    if not ok:raise RuntimeError("S5_INFERENCE_BLOCKED "+why)

def read_source_subset(source,need,labels,model,np):
    """Scan one ORIGINAL CSV once, keep only exact canonical UIDs."""
    arrays={}
    names=set(need)
    csv.field_size_limit(sys.maxsize)
    with source.open("r",newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        check({"pcap_uid","site_label","direction_vector"}<=set(reader.fieldnames or []),
              "source schema differs")
        if model=="VarCNN":
            check("time_vector" in (reader.fieldnames or []),
                  "VarCNN requires real observed time_vector")
        for row in reader:
            uid=row["pcap_uid"]
            if uid not in names:continue
            check(uid not in arrays,"duplicate requested UID in vector CSV")
            check(row["site_label"]==labels[uid],
                  "external UID site_label not canonical "+uid)
            if model=="DF":
                dirs=fixed_length(parse_direction(row["direction_vector"]))
                x=np.frombuffer(dirs.tobytes(),dtype=np.int8).astype(np.float32)
                x=x.reshape(1,5000)
            else:
                dt=dt2_from_text(row["direction_vector"],row["time_vector"])
                x=np.stack([np.frombuffer(dt.direction.tobytes(),dtype=np.float32),
                            np.frombuffer(dt.interarrival.tobytes(),dtype=np.float32)])
            arrays[uid]=x
            if len(arrays)%5000==0:
                print("S5_VECTOR_PROGRESS",model,len(arrays),len(need),flush=True)
    check(len(arrays)==len(need) and set(arrays)==names,
          "missing or duplicated external UID vectors")
    # Preserve EXACT frozen score UID ordering, never sort, shuffle or sample.
    x=np.stack([arrays[uid] for uid in need])
    return x

def inspect_existing(target,cohort,model,checkpoint_sha,protocol_sha,source_sha,
                     original_sha,uid,day,true,labels,np):
    """Resume safely: reuse a COMPLETE scientifically identical result.

    An empty legacy directory may be filled; any incomplete nonempty
    directory blocks rather than deleting/overwriting evidence.
    """
    if not target.exists() and not target.is_symlink():
        return "NEW"
    check(target.is_dir() and not target.is_symlink(),
          "existing target is not a real directory")
    contents={p.name for p in target.iterdir()}
    if not contents:
        print("S5_OUTPUT_RESUME=EXISTING_EMPTY_DIRECTORY",target,flush=True)
        return "EMPTY"
    check(contents=={"scores_sota_aligned.npz","manifest.json"},
          "existing output has incomplete/unknown files; preserve and inspect manually")
    manifest_path=target/"manifest.json"
    score_path=target/"scores_sota_aligned.npz"
    check(not manifest_path.is_symlink() and not score_path.is_symlink(),
          "existing output contains symlinks")
    m=json.loads(manifest_path.read_text(encoding="utf-8"))
    required={
        "classification":"SOTA_S5_EXTERNAL_FROZEN_INFERENCE_V1",
        "cohort":cohort,"model":model,"population":len(uid),
        "checkpoint_sha256":checkpoint_sha,"protocol_sha256":protocol_sha,
        "source_sha256":source_sha,"canonical_score_sha256":original_sha,
        "future_b_used_for_training":False,
        "external_scores_used_for_selection":False,
        "dataset_c_opened":False,"training_steps":0
    }
    check(all(m.get(k)==v for k,v in required.items()),
          "existing inference metadata conflicts with frozen science")
    check(sha256_file(score_path)==m.get("sota_score_sha256"),
          "existing inference score digest changed")
    with np.load(score_path,allow_pickle=False) as saved:
        check(set(saved.files)=={"pcap_uid","query_date","y_true",
                                "candidate_labels","sota_probs"},
              "existing SOTA output schema changed")
        same=(np.array_equal(saved["pcap_uid"].astype(str),uid)
              and np.array_equal(saved["query_date"].astype(str),day)
              and np.array_equal(saved["y_true"].astype(np.int64),true)
              and saved["candidate_labels"].astype(str).tolist()==labels)
        check(same,"existing inference UID/date/class alignment mismatch")
        probs=saved["sota_probs"]
        check(probs.shape==(len(uid),65) and np.isfinite(probs).all()
              and (probs>=0).all()
              and np.allclose(probs.sum(axis=1),1.0,atol=2e-4,rtol=0),
              "existing inference probabilities invalid")
    print("S5_INFERENCE_REUSED_VERIFIED",cohort,model,
          "ACCURACY",m["accuracy"],"MACRO_F1",m["macro_f1"],
          "TOP5",m["top5"],"MRR",m["mrr"],flush=True)
    print("S5_INFERENCE_MANIFEST",manifest_path,flush=True)
    return "COMPLETE"

def run(cohort,model):
    import numpy as np
    import torch
    check(EVIDENCE.is_file() and not EVIDENCE.is_symlink(),
          "S5 immutable freeze evidence missing")
    frozen=json.loads(EVIDENCE.read_text(encoding="utf-8"))
    check(frozen.get("classification")==
          "SOTA_S5_FROZEN_REFITS_AND_LABEL_CROSSWALK_V1",
          "S5 freeze evidence classification invalid")
    check(model in frozen["models"] and cohort in frozen["label_alignment"],
          "unknown model/cohort in frozen evidence")
    check(frozen["population"][cohort]==N[cohort],"cohort size not frozen")
    checkpoint=Path(frozen["models"][model]["checkpoint"])
    check(checkpoint.is_file(),"frozen checkpoint missing")
    cksha=sha256_file(checkpoint)
    check(cksha==frozen["models"][model]["sha256"],"checkpoint SHA changed")
    original=Path(frozen["label_alignment"][cohort]["source"])
    check(original.is_file() and sha256_file(original)==
          frozen["label_alignment"][cohort]["source_sha256"],
          "canonical original score SHA changed")
    if cohort=="09B":
        recorded=json.loads((ROOT/"docs/evidence/FINAL/09B_manifest_internal_test.json")
                            .read_text())
        check(original.name=="09B_internal_scores.npz" and
              recorded["outputs"]["scores"]==sha256_file(original),
              "09B original score not backed by frozen manifest")
        source,source_sha=HIST,EXPECTED_HIST
    else:
        recorded=json.loads((ROOT/"docs/evidence/FINAL/FUTURE-B/10B_manifest_future_b.json")
                            .read_text())
        source,source_sha=FUTURE,recorded["future_inputs"]["micro"]["sha256"]
        check(source_sha==EXPECTED_FUTURE,"future vector SHA not canonical")
        check(original.name=="10B_scores_DEV_FROZEN.npz",
              "not the canonical 10B DEV_FROZEN score")
    check(source.is_file() and sha256_file(source)==source_sha,
          "raw vector source SHA differs from frozen research input")
    with np.load(original,allow_pickle=True) as scores:
        for name in ("pcap_uid","query_date","candidate_labels","y_true"):
            check(name in scores.files,"canonical score missing "+name)
        uid=scores["pcap_uid"].astype(str)
        labels=scores["candidate_labels"].astype(str).tolist()
        true=scores["y_true"].astype(np.int64)
        day=scores["query_date"].astype("datetime64[D]").astype(str)
    check(len(uid)==N[cohort] and len(set(uid))==N[cohort],
          "UID count or uniqueness wrong")
    check(len(labels)==65 and len(set(labels))==65,"canonical 65 classes wrong")
    check(np.all((true>=0)&(true<65)),"out-of-range canonical true labels")
    target=OUT/cohort/model
    status=inspect_existing(target,cohort,model,cksha,
                            frozen["protocol_sha256"],source_sha,
                            frozen["label_alignment"][cohort]["source_sha256"],
                            uid,day,true,labels,np)
    if status=="COMPLETE":
        return
    lookup={str(u):labels[int(y)] for u,y in zip(uid,true)}
    print("S5_EXTERNAL_DATA_READING",cohort,model,
          "UIDS",len(uid),flush=True)
    X=read_source_subset(source,uid,lookup,model,np)
    check(tuple(X.shape)==(N[cohort],1 if model=="DF" else 2,5000),
          "input tensor shape mismatch")
    check(torch.cuda.is_available(),"no CUDA")
    check(shutil.disk_usage("/data").free>=120*1024**3,
          "data reserve less than 120GiB")
    check(torch.cuda.mem_get_info()[0]>=5*1024**3,"insufficient free GPU memory")
    verified=verify_checkout()
    check(verified["commit"]=="537ec8e03c805540b04b95b144fa158ce50aa255",
          "upstream source changed")
    torch.set_num_threads(2)
    net=get_model(model,65)
    weights=torch.load(checkpoint,map_location="cpu",weights_only=True)
    net.load_state_dict(weights,strict=True)
    net=net.to("cuda:0").eval()
    canonical_to_s1=frozen["label_alignment"][cohort]["canonical_to_s1_indices"]
    check(len(canonical_to_s1)==65 and sorted(canonical_to_s1)==list(range(65)),
          "invalid label permutation")
    batch=128 if model=="DF" else 50
    probs=np.empty((len(uid),65),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,len(uid),batch):
            xb=torch.from_numpy(X[start:start+batch]).to("cuda:0")
            logits=net(xb)
            mapped=logits[:,canonical_to_s1]
            p=torch.softmax(mapped,dim=1).cpu().numpy()
            probs[start:start+len(p)]=p
    check(np.isfinite(probs).all() and (probs>=0).all() and
          np.allclose(probs.sum(axis=1),1.0,atol=2e-4,rtol=0),
          "invalid probabilities")
    ranks=np.argsort(-probs,axis=1)
    top1=float(np.mean(ranks[:,0]==true))
    r=(ranks==true[:,None]).argmax(axis=1)+1
    top5=float(np.mean(r<=5))
    mrr=float(np.mean(1.0/r))
    tp=np.bincount(true[(ranks[:,0]==true)],minlength=65)
    support=np.bincount(true,minlength=65)
    predicted=np.bincount(ranks[:,0],minlength=65)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(65,dtype=float),
                 where=(support+predicted)>0).mean()
    if status=="NEW":
        target.mkdir(parents=True,mode=0o700,exist_ok=False)
    else:
        check(status=="EMPTY" and target.is_dir()
              and not any(target.iterdir()),
              "previously empty directory changed while computing")
    exported=target/"scores_sota_aligned.npz"
    check(not exported.exists() and
          not (target/"manifest.json").exists(),
          "refuse to overwrite existing files")
    np.savez_compressed(exported,pcap_uid=uid,query_date=day,
                        y_true=true,candidate_labels=np.array(labels),
                        sota_probs=probs)
    manifest={
        "classification":"SOTA_S5_EXTERNAL_FROZEN_INFERENCE_V1",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "cohort":cohort,"model":model,"population":len(uid),
        "checkpoint_sha256":cksha,
        "protocol_sha256":frozen["protocol_sha256"],
        "source_sha256":source_sha,
        "canonical_score_sha256":sha256_file(original),
        "sota_score_sha256":sha256_file(exported),
        "accuracy":top1,"macro_f1":float(f1),"top5":top5,"mrr":mrr,
        "future_b_used_for_training":False,
        "external_scores_used_for_selection":False,
        "dataset_c_opened":False,"training_steps":0
    }
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("S5_INFERENCE_PASS",cohort,model,
          "ACCURACY",round(top1,6),"MACRO_F1",round(float(f1),6),
          "TOP5",round(top5,6),"MRR",round(mrr,6),flush=True)
    print("S5_INFERENCE_MANIFEST",target/"manifest.json",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cohort",required=True,choices=list(N))
    parser.add_argument("--model",required=True,choices=MODELS)
    args=parser.parse_args()
    try:run(args.cohort,args.model)
    except Exception as e:
        print("S5_INFERENCE_ERROR",type(e).__name__,str(e)[:700],
              file=sys.stderr,flush=True)
        sys.exit(4)
