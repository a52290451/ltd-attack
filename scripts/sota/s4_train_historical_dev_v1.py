#!/usr/bin/env python3
"""REAL Historical DEV DF/VarCNN training, chronological stages A/B/C.

Never reads 09B, Future-B, Dataset C, or previous external predictions.
Only S1 SHA-verified DEV UID membership and Historical vector rows.
Uses one existing CUDA venv, single GPU, WFlib pinned code, full 65-class
macro-F1 selection on stage B; stage C refit uses stage B epoch count.
No dynamic hyperparameter search and no synthetic-smoke dependency.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.data_adapter import (
    load_s1_dev_index,parse_direction,fixed_length
)
from src.models.sota.v1.timing_adapter import dt2_from_text
from src.models.sota.v1.s4_protocol import (
    validate_config,score_macro_f1,select_epoch,sha256_file
)
from src.models.sota.v1.upstream_models import get_model,verify_checkout

CONFIG=ROOT/"configs/sota/sota_s4_frozen_protocol_v1.json"
S1_PATH=Path.home()/"ltd-ops/sota/s1_dev_frozen_20261102T130232Z/s1_manifest.json"
OUTPUT=Path("/data/ltd-attack/sota/training")
RUN_ID="s4_wflib_v1_seed42"

def require(test,message):
    if not test:raise RuntimeError("SOTA_S4_TRAIN_ABORTED "+message)

def get_stage(cfg,stage):
    return next(s for s in cfg["schedule"] if s["id"]==stage)

def git_head():
    p=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                     capture_output=True,text=True,check=True)
    return p.stdout.strip()

def data_arrays(source,by_uid,classes,model,stage_spec,np):
    """One streaming pass, exact UID set, preallocated float32 arrays."""
    splits=tuple(stage_spec["train"]+stage_spec["validate"])
    expected=stage_spec["train_count"]+stage_spec["val_count"]
    channels=1 if model=="DF" else 2
    X=np.empty((expected,channels,5000),dtype=np.float32)
    y=np.empty(expected,dtype=np.int64)
    parts=[]
    counts=Counter()
    seen=set()
    csv.field_size_limit(sys.maxsize)
    with source.open("r",encoding="utf-8",newline="") as f:
        rows=csv.DictReader(f)
        require({"pcap_uid","site_label","direction_vector"}<=set(rows.fieldnames or []),
                "wrong Historical CSV schema")
        if model=="VarCNN":
            require("time_vector" in (rows.fieldnames or []),"time_vector absent")
        for row in rows:
            uid=row["pcap_uid"]
            meta=by_uid.get(uid)
            if meta is None or meta[2] not in splits:continue
            require(uid not in seen,"duplicate historical DEV UID")
            seen.add(uid)
            label,date,split=meta
            require(row["site_label"]==label,"UID label mismatch")
            i=len(parts)
            require(i<expected,"more Historical rows than S1")
            if model=="DF":
                d=fixed_length(parse_direction(row["direction_vector"]))
                X[i,0]=np.frombuffer(d.tobytes(),dtype=np.int8).astype(np.float32)
            else:
                sample=dt2_from_text(row["direction_vector"],row["time_vector"])
                X[i,0]=np.frombuffer(sample.direction.tobytes(),dtype=np.float32)
                X[i,1]=np.frombuffer(sample.interarrival.tobytes(),dtype=np.float32)
            y[i]=classes[label]
            parts.append(split)
            counts[split]+=1
            if i and i%5000==0:
                print("SOTA_S4_DATA_PROGRESS",i,expected,flush=True)
    require(len(parts)==expected,"incomplete stage Historical UID population")
    require(len(seen)==expected,"duplicate historical UID")
    for split in splits:
        require(counts[split]==sum(s["train_count"] for s in [] ) if False else True,
                "unreachable")
    require(counts==Counter({s:sum(1 for uid,m in by_uid.items() if m[2]==s)
                            for s in splits}),"split count changed")
    train=np.array([i for i,s in enumerate(parts) if s in stage_spec["train"]],dtype=np.int64)
    valid=np.array([i for i,s in enumerate(parts) if s in stage_spec["validate"]],dtype=np.int64)
    require(len(train)==stage_spec["train_count"],"training cohort wrong")
    require(len(valid)==stage_spec["val_count"],"validation cohort wrong")
    require(set(train).isdisjoint(set(valid)),"train/validation overlap")
    return X,y,train,valid,dict(counts)

def run_one(model_name,stage,epoch_override):
    import numpy as np
    import torch
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(cfg)
    require(model_name in cfg["model_configs"],"unknown model")
    settings=cfg["model_configs"][model_name]
    spec=get_stage(cfg,stage)
    require(torch.cuda.is_available(),"CUDA not available")
    require(torch.__version__.startswith("2.11.0+cu128"),
            "wrong torch wheel; no environment changes allowed")
    require(shutil.disk_usage("/data").free>=120*1024**3,
            "less than 120 GiB free on data volume")
    require(torch.cuda.mem_get_info()[0]>=5*1024**3,
            "less than 5GiB free VRAM")
    provenance=verify_checkout()
    require(provenance["commit"]==cfg["model_source"]["commit"],
            "wrong WFlib commit")
    require(S1_PATH.is_file(),"frozen S1 index missing")
    by_uid,classes,s1=load_s1_dev_index(S1_PATH)
    require(len(by_uid)==57916 and len(classes)==65,"invalid S1 population")
    source=Path(s1["historical_checksums"]["historical_vectors"]["path"])
    require(source.resolve().is_file(),"Historical vector source missing")
    require(s1["historical_checksums"]["historical_vectors"]["sha256"]==
            cfg["population"]["source_vector_sha256"],
            "Historical source SHA differs from frozen S1")
    require(source.resolve().is_relative_to(Path("/data").resolve())
            or source.resolve().is_relative_to(Path.home()),
            "unapproved source mount")
    run=OUTPUT/RUN_ID/model_name/stage
    require(not run.exists(),"refuse to overwrite an existing scientific run")
    require(stage!="C" or epoch_override is not None,
            "stage C requires frozen epoch count from stage B")
    require(stage!="C" or 1<=epoch_override<=30,
            "invalid stage B selected epoch")
    require(stage!="C" or (OUTPUT/RUN_ID/model_name/"B"/"selected_epoch.json").is_file(),
            "no stage B selection evidence")
    if stage=="C":
        chosen=json.loads((OUTPUT/RUN_ID/model_name/"B"/"selected_epoch.json").read_text())
        require(chosen["selected_epoch"]==epoch_override,
                "stage C epoch does not match stage B")
        require(chosen["protocol_sha256"]==sha256_file(CONFIG),
                "stage B protocol SHA changed")
    print("SOTA_S4_REAL_TRAIN_BEGIN",model_name,stage,flush=True)
    print("SOTA_S4_PROTOCOL_SHA",sha256_file(CONFIG),flush=True)
    print("SOTA_S4_DATA_SOURCE",source,flush=True)
    X,y,train_idx,val_idx,counts=data_arrays(source,by_uid,classes,model_name,spec,np)
    # Only after complete UID verification do we initialize and update a model.
    run.mkdir(parents=True,mode=0o700)
    torch.set_num_threads(2)
    random.seed(42);np.random.seed(42)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    device=torch.device("cuda:0")
    net=get_model(model_name,65).to(device)
    net.train()
    optimizer=getattr(torch.optim,settings["optimizer"])(
        net.parameters(),lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"])
    epochs=epoch_override if stage=="C" else settings["epochs_max"]
    batch=settings["batch_size"]
    metrics={}
    started=time.time()
    history=[]
    for epoch in range(1,epochs+1):
        net.train()
        rng=np.random.default_rng(42+epoch)
        shuffled=rng.permutation(train_idx)
        loss_sum=0.0
        seen_samples=0
        for start in range(0,len(shuffled),batch):
            indices=shuffled[start:start+batch]
            xb=torch.from_numpy(X[indices]).to(device)
            yb=torch.from_numpy(y[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=net(xb)
            loss=torch.nn.functional.cross_entropy(logits,yb)
            require(bool(torch.isfinite(loss)),"nonfinite real training loss")
            loss.backward()
            optimizer.step()
            loss_sum+=float(loss.item())*len(indices)
            seen_samples+=len(indices)
        train_loss=loss_sum/seen_samples
        line={"epoch":epoch,"training_loss":train_loss,"training_samples":seen_samples}
        if len(val_idx):
            net.eval()
            predicted=[]
            with torch.inference_mode():
                for start in range(0,len(val_idx),batch):
                    indices=val_idx[start:start+batch]
                    xb=torch.from_numpy(X[indices]).to(device)
                    logits=net(xb)
                    predicted.extend(logits.argmax(dim=1).cpu().tolist())
            truth=y[val_idx].tolist()
            macro=score_macro_f1(truth,predicted)
            accuracy=sum(a==b for a,b in zip(truth,predicted))/len(truth)
            line.update({"valid_macro_f1":macro,"valid_accuracy":accuracy,
                         "validation_samples":len(truth)})
            metrics[epoch]=macro
        history.append(line)
        print("SOTA_S4_EPOCH",model_name,stage,json.dumps(line,sort_keys=True),
              flush=True)
        (run/"history.json").write_text(json.dumps(history,indent=2)+"\n")
    selected=select_epoch(metrics) if metrics and stage=="B" else None
    checkpoint=run/"final_model_state.pt"
    torch.save(net.cpu().state_dict(),checkpoint)
    artifact_hash=sha256_file(checkpoint)
    record={"classification":"SOTA_S4_REAL_HISTORICAL_DEV_TRAIN_V1",
            "model":model_name,"stage":stage,
            "train_splits":spec["train"],"valid_splits":spec["validate"],
            "verified_counts":counts,"epochs_completed":epochs,
            "selected_epoch_from_DEV_LATE":selected if stage=="B" else epoch_override,
            "protocol_sha256":sha256_file(CONFIG),
            "S1_manifest":str(S1_PATH),"S1_label_sha256":s1["label_order_sha256"],
            "upstream_commit":provenance["commit"],"torch":torch.__version__,
            "checkpoint":str(checkpoint),"checkpoint_sha256":artifact_hash,
            "history":history,"elapsed_seconds":time.time()-started,
            "future_b_opened":False,"internal_test_opened":False,
            "dataset_c_opened":False}
    (run/"manifest.json").write_text(json.dumps(record,indent=2)+"\n")
    if stage=="B":
        (run/"selected_epoch.json").write_text(json.dumps({
            "selected_epoch":selected,
            "protocol_sha256":sha256_file(CONFIG),
            "metric":"macro_f1","tie_break":"earliest_epoch"
        },indent=2)+"\n")
    print("SOTA_S4_REAL_TRAIN=PASS",model_name,stage,
          "CHECKPOINT_SHA256",artifact_hash,flush=True)
    print("SOTA_S4_REAL_TRAIN_MANIFEST",run/"manifest.json",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=["DF","VarCNN"],required=True)
    p.add_argument("--stage",choices=["A","B","C"],required=True)
    p.add_argument("--epochs-from-B",type=int)
    args=p.parse_args()
    try:run_one(args.model,args.stage,args.epochs_from_B)
    except Exception as exc:
        print("SOTA_S4_REAL_TRAIN_ERROR",type(exc).__name__,str(exc)[:700],
              file=sys.stderr,flush=True)
        sys.exit(4)
