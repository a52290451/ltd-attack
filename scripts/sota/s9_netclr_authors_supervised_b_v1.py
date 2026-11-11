#!/usr/bin/env python3
"""S9-O authors' NetCLR supervised B: original-source model + Historical DEV protocol.

smoke: CPU checkpoint/state/forward/backward test (no experiment writes).
train-B: Historical DEV_EARLY+MIDDLE only (39258), selection DEV_LATE (18658).
Does not read 09B, Future-B or Dataset C. No checkpoint overwrite across runs.
Original authors' model loaded from pinned notebook; adaptations declared, not paper-exact.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.sota.v1.data_adapter import (
    load_s1_dev_index, iter_dev_vectors, digest,
)

import importlib.util
PRETRAIN = ROOT / "scripts/sota/s9_netclr_authors_pretrain_b_v1.py"
SPEC = importlib.util.spec_from_file_location("ltd_s9_pretraining_pinned", PRETRAIN)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)
OUT = P.OUT
S1 = P.S1
CHECKPOINT = OUT / "pretrain_B_v1/checkpoint_epoch_100.pt"
PRETRAIN_MANIFEST = OUT / "pretrain_B_v1/manifest.json"
CACHE = P.DATA.CACHE
TARGET = OUT / "supervised_B_v1"
CONFIG = {
    "stage": "S9_NETCLR_AUTHORS_SUPERVISED_B_V1",
    "seed": 42,
    "classes": 65,
    "input": "DIR_5000",
    "optimizer": "Adam",
    "lr": 0.0003,
    "batch_size": 64,
    "max_epochs": 30,
    "selection": "max_DEV_LATE_macro_F1_earliest_tie",
    "initialization": "pretrained_DFsimCLR_backbone_except_projection_head",
    "supervised_classifier": "SPIN_UMass_DFNet_from_authors_notebook",
    "method_status": "authors_source_adapted_not_paper_exact",
    "no_external_fit_or_selection": True,
}

def check(ok, message):
    if not ok:
        raise RuntimeError("S9_SUPERVISED_BLOCKED " + message)

def source():
    uids, class_to_id, s1 = load_s1_dev_index(S1)
    info = s1["historical_checksums"]["historical_vectors"]
    p = Path(info["path"])
    check(digest(p) == info["sha256"], "historical source checksum differs")
    return uids, class_to_id, s1, p

def checkpoint_evidence():
    check(CHECKPOINT.is_file() and PRETRAIN_MANIFEST.is_file(),
          "missing author-original pretrain checkpoint and manifest")
    m=json.loads(PRETRAIN_MANIFEST.read_text())
    check(m.get("classification")==P.CONFIG["classification"],
          "authors source B provenance mismatch")
    check(m.get("config")==P.CONFIG and
          m.get("training_n")==39258,"pretrain B config/population mismatch")
    check(m.get("checkpoint_sha256")==P.sha(CHECKPOINT),
          "pretrain authors checkpoint SHA mismatch")
    check(m.get("external_data_opened") is False and
          m.get("dataset_c_opened") is False,"pretraining leakage assertion failed")
    ck=torch.load(CHECKPOINT,map_location="cpu",weights_only=True)
    check(ck.get("epoch")==100 and ck.get("config")==P.CONFIG,
          "author pretraining tensor metadata mismatch")
    return ck,m

def classifier():
    ns,evidence=P.original()
    ck,m=checkpoint_evidence()
    pre=ns["DFsimCLR"](ns["DFNet"](out_dim=512),out_dim=128)
    pre.load_state_dict(ck["model"],strict=True)
    clf=ns["DFNet"](out_dim=65)
    original=pre.backbone.state_dict()
    current=clf.state_dict()
    retained={k:v for k,v in original.items()
              if not k.startswith("fc.") and k in current}
    expected={k for k in current if not k.startswith("fc.")}
    check(set(retained)==expected,"authors pretraining backbone coverage")
    for k,v in retained.items():
        check(v.shape==current[k].shape,"tensor mismatch "+k)
    current.update(retained)
    clf.load_state_dict(current,strict=True)
    return clf

def prepare_cache():
    by_uid, classes, s1, p = source()
    manifest = CACHE / "manifest.json"
    matrix = CACHE / "directions.npy"
    labels = CACHE / "labels.npy"
    split = CACHE / "split.npy"
    paths = [matrix,labels,split]
    if manifest.exists():
        m = json.loads(manifest.read_text())
        check(m.get("source_sha256") == s1["historical_checksums"]
              ["historical_vectors"]["sha256"], "cache source altered")
        check(m.get("label_sha256") == s1["label_order_sha256"],
              "cache label order altered")
        for f in paths:
            check(f.is_file() and P.sha(f) == m["sha256"][f.name],
                  "cache checksum: "+f.name)
        x = np.load(matrix,mmap_mode="r")
        y = np.load(labels,mmap_mode="r")
        s = np.load(split,mmap_mode="r")
        check(x.shape == (57916,5000) and
              y.shape == (57916,) and s.shape == (57916,),
              "cached tensor shapes")
        check(dict(zip(*np.unique(s,return_counts=True))) ==
              {0:19098,1:20160,2:18658},"cached chronological counts")
        return x,y,s,m
    check(not CACHE.exists(), "partial supervised cache; inspect, do not overwrite")
    CACHE.mkdir(mode=0o700, parents=True)
    temp=CACHE/"directions.npy.partial"
    X=np.lib.format.open_memmap(temp,mode="w+",dtype=np.int8,
                                shape=(57916,5000))
    Y=np.empty(57916,dtype=np.int64)
    S=np.empty(57916,dtype=np.uint8)
    split_id={"DEV_EARLY":0,"DEV_MIDDLE":1,"DEV_LATE":2}
    seen=set()
    counts=Counter()
    n=0
    for item in iter_dev_vectors(p,by_uid,classes):
        check(item.pcap_uid not in seen,"duplicate UID in vectors")
        seen.add(item.pcap_uid)
        X[n]=np.asarray(item.directions,dtype=np.int8)
        Y[n]=item.label_id
        S[n]=split_id[item.temporal_split]
        counts[item.temporal_split]+=1
        n+=1
        if n%10000==0:
            print("S9_SUPERVISED_CACHE_PROGRESS",n,"57916",flush=True)
    check(n==57916 and seen==set(by_uid) and
          dict(counts)=={"DEV_EARLY":19098,"DEV_MIDDLE":20160,
                         "DEV_LATE":18658},"historical DEV coverage")
    X.flush()
    del X
    os.replace(temp,matrix)
    np.save(labels,Y,allow_pickle=False)
    np.save(split,S,allow_pickle=False)
    checks={f.name:P.sha(f) for f in paths}
    m={"classification":"S9_SUPERVISED_HISTORICAL_DEV_CACHE_V1",
       "n":57916,"splits":dict(counts),
       "source_sha256":s1["historical_checksums"]
              ["historical_vectors"]["sha256"],
       "label_sha256":s1["label_order_sha256"],
       "sha256":checks,"external_data_opened":False,
       "dataset_c_opened":False}
    manifest.write_text(json.dumps(m,indent=2)+"\n")
    print("S9_SUPERVISED_CACHE_PASS",n,flush=True)
    return (np.load(matrix,mmap_mode="r"),
            np.load(labels,mmap_mode="r"),
            np.load(split,mmap_mode="r"),m)

class Subset(Dataset):
    def __init__(self,X,y,indices):
        self.X=X
        self.y=y
        self.indices=np.asarray(indices,dtype=np.int64)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,i):
        j=int(self.indices[i])
        a=np.array(self.X[j],dtype=np.float32,copy=True)
        return torch.from_numpy(a[None,:]),int(self.y[j])

def gpu_idle():
    r=subprocess.run(["nvidia-smi",
          "--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],
          capture_output=True,text=True,timeout=20)
    check(r.returncode==0,"nvidia-smi could not check other GPU processes")
    jobs=[s.strip() for s in r.stdout.splitlines() if s.strip()]
    check(not jobs,"GPU occupied by other compute processes: "+repr(jobs))

def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def smoke():
    seed()
    m=classifier().cpu().train()
    z=torch.randint(-1,2,(2,1,5000),dtype=torch.float32)
    scores=m(z)
    check(scores.shape==(2,65) and bool(torch.isfinite(scores).all().item()),
          "smoke forward output")
    loss=nn.CrossEntropyLoss()(scores,torch.tensor([0,1]))
    loss.backward()
    check(bool(torch.isfinite(loss).item()),"smoke backward loss")
    print("S9_AUTHOR_SUPERVISED_B_SMOKE_PASS",
          "input",(2,1,5000),"classes",65,
          "pretrain_sha",P.sha(CHECKPOINT),flush=True)

def train_b():
    check(not TARGET.exists(),"supervised B already exists; no overwrite")
    check(torch.cuda.is_available(),"CUDA unavailable")
    gpu_idle()
    seed()
    X,y,s,cache=prepare_cache()
    tr=np.flatnonzero(s!=2)
    va=np.flatnonzero(s==2)
    check(len(tr)==39258 and len(va)==18658,"bad B partition")
    check(set(int(v) for v in y)==set(range(65)),"65 class coverage")
    train=DataLoader(Subset(X,y,tr),batch_size=64,
                     shuffle=True,drop_last=True,num_workers=0,
                     pin_memory=True)
    valid=DataLoader(Subset(X,y,va),batch_size=128,
                     shuffle=False,drop_last=False,num_workers=0,
                     pin_memory=True)
    model=classifier().to("cuda:0")
    opt=torch.optim.Adam(model.parameters(),lr=0.0003)
    crit=nn.CrossEntropyLoss()
    TARGET.mkdir(parents=True,mode=0o700)
    selected=-1.
    selected_epoch=0
    history=[]
    start=time.monotonic()
    print("S9_SUPERVISED_B_BEGIN","train",len(tr),"valid",len(va),
          "epochs",30,"gpu",torch.cuda.get_device_name(0),flush=True)
    for epoch in range(1,31):
        e0=time.monotonic()
        model.train()
        sum_loss=0.
        steps=0
        for bx,by in train:
            bx=bx.to("cuda:0",non_blocking=True)
            by=by.to("cuda:0",non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits=model(bx)
            loss=crit(logits,by)
            check(bool(torch.isfinite(loss).item()),"train loss not finite")
            loss.backward()
            opt.step()
            sum_loss+=float(loss.item())
            steps+=1
            if steps%100==0:
                print("S9_SUPERVISED_B_PROGRESS",json.dumps({
                    "epoch":epoch,"batch":steps,"batches":len(train),
                    "elapsed_s":round(time.monotonic()-start,1)}),flush=True)
        model.eval()
        gt=[]
        pd=[]
        with torch.inference_mode():
            for bx,by in valid:
                logits=model(bx.to("cuda:0",non_blocking=True))
                check(bool(torch.isfinite(logits).all().item()),
                      "nonfinite val logits")
                pd.append(logits.argmax(1).cpu().numpy())
                gt.append(by.numpy())
        gt=np.concatenate(gt)
        pd=np.concatenate(pd)
        acc=float((pd==gt).mean())
        f1=float(f1_score(gt,pd,labels=np.arange(65),average="macro"))
        record={"epoch":epoch,"loss":sum_loss/steps,"val_accuracy":acc,
                "val_macro_f1":f1,
                "epoch_elapsed_s":round(time.monotonic()-e0,1),
                "total_elapsed_s":round(time.monotonic()-start,1)}
        history.append(record)
        print("S9_SUPERVISED_B_EPOCH",json.dumps(record),flush=True)
        if f1>selected:
            selected=f1
            selected_epoch=epoch
            # A checkpoint inside this uniquely reserved experiment folder;
            # replacement only within this single run on a better validation epoch.
            torch.save({"model":model.cpu().state_dict(),
                        "epoch":epoch,"config":CONFIG,
                        "pretrain_sha256":P.sha(CHECKPOINT)},
                       TARGET/"best_checkpoint.pt")
            model.to("cuda:0")
    ck=TARGET/"best_checkpoint.pt"
    report={"classification":"S9_NETCLR_AUTHORS_SUPERVISED_B_V1",
            "config":CONFIG,"train_n":39258,"valid_n":18658,
            "selected_epoch":selected_epoch,
            "validation":history[selected_epoch-1],
            "history":history,
            "checkpoint":str(ck),"checkpoint_sha256":P.sha(ck),
            "pretrain_sha256":P.sha(CHECKPOINT),
            "pretrain_manifest_sha256":P.sha(PRETRAIN_MANIFEST),
            "cache_manifest_sha256":P.sha(CACHE/"manifest.json"),
            "external_data_opened":False,"dataset_c_opened":False,
            "created_utc":datetime.now(timezone.utc).isoformat()}
    (TARGET/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S9_SUPERVISED_B_PASS","epoch",selected_epoch,
          "val",report["validation"],
          "checkpoint_sha256",report["checkpoint_sha256"],flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=("smoke","train-B"),required=True)
    a=p.parse_args()
    if a.stage=="smoke":smoke()
    else:train_b()

if __name__=="__main__":
    try:main()
    except Exception as e:
        print("S9_SUPERVISED_B_ERROR",type(e).__name__,str(e)[:1200],
              file=sys.stderr,flush=True)
        raise
