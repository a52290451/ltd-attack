#!/usr/bin/env python3
"""S9 NetCLR real Historical DEV-only SimCLR pretraining (WFlib-adapted).

Stages:
 smoke: Historical DEV_EARLY+MIDDLE subset, source/augmentation/forward/backward
        checks without creating experiment artifacts.
 pretrain-B: 100 epochs, ONLY DEV_EARLY+MIDDLE (39,258 examples).
 No Future-B, 09B, DEV_LATE, Dataset C, network or external training.
 Uses WFlib pinned NetCLR, DFsimCLR and NetAugment-like WFlib augmentor.
 Adaptation from upstream: batch 64 (RTX 5080), dynamic NCE batches,
 reliable current-torch optimization loop. DO NOT claim paper-exact NetCLR.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, random, subprocess, sys, time, types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.data_adapter import load_s1_dev_index,parse_direction,digest,fixed_length
from src.models.sota.v1.upstream_models import resolve_checkout,verify_checkout,WFLIB_COMMIT

S1=Path.home()/"ltd-ops/sota/s1_dev_frozen_20261102T130232Z/s1_manifest.json"
OUT=Path("/data/ltd-attack/sota/s9_netclr_wflib_adapted_seed42")
SOURCE_BLOBS={
 "model":("WFlib/models/NetCLR.py","f63cdf80046220baf0544e967ce7a6077a7096e4"),
 "augment":("WFlib/tools/netclr_augmentor.py","e28a587c6caa31b39397354316a7d22edeb4b414")
}
CONFIG={"model":"WFlib_NetCLR_DFsimCLR","upstream_commit":WFLIB_COMMIT,
 "upstream_blobs":{k:v[1] for k,v in SOURCE_BLOBS.items()},
 "representation":"DIR_5000","classes":65,"seed":42,
 "contrastive":"NT_Xent_2views","embedding":128,"temperature":0.5,
 "optimizer":"Adam","lr":0.0003,"batch_size":64,"epochs":100,
 "dataset":"Historical_DEV_EARLY_MIDDLE_only",
 "augmentor":"pinned_WFlib_NetAugment_like","drop_last":True,
 "method_status":"WFlib_adapted_not_original_paper_exact",
 "reduced_batch_vs_upstream_256":True}

def check(ok,msg):
    if not ok:raise RuntimeError("S9_BLOCKED "+msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()
def pinned():
    checkout=resolve_checkout();verify_checkout(checkout)
    modules={}
    for key,(rel,expected) in SOURCE_BLOBS.items():
        r=subprocess.run(["git","-C",str(checkout),"hash-object",rel],
                         capture_output=True,text=True,check=True)
        check(r.stdout.strip()==expected,"upstream source SHA mismatch "+key)
        path=checkout/rel
        mod=types.ModuleType("ltd_netclr_"+key);mod.__file__=str(path)
        exec(compile(path.read_bytes(),str(path),"exec"),mod.__dict__)
        modules[key]=mod
    return modules

def build_cache():
    uids,labels,s1=load_s1_dev_index(S1)
    source=Path(s1["historical_checksums"]["historical_vectors"]["path"])
    check(digest(source)==s1["historical_checksums"]["historical_vectors"]["sha256"],
          "Historical source hash mismatch")
    # Cache contains training split only, to prevent pretraining from
    # implicitly observing validation or any external cohort.
    base=OUT/"dev_train_cache"
    p=base/"directions.npy"
    info=base/"manifest.json"
    if info.exists():
        m=json.loads(info.read_text())
        check(m["source_sha256"]==digest(source) and
              m["label_sha256"]==s1["label_order_sha256"],"cache source changed")
        check(p.is_file() and sha(p)==m["array_sha256"],"cache SHA mismatch")
        check(m["n"]==39258,"cache training population changed")
        return np.load(p,mmap_mode="r"),m
    check(not base.exists(),"partial S9 cache exists; inspect; no overwrites")
    base.mkdir(parents=True,mode=0o700)
    tmp=base/"directions.npy.partial"
    X=np.lib.format.open_memmap(tmp,mode="w+",dtype=np.int8,shape=(39258,5000))
    seen=set();counts=Counter();n=0
    csv.field_size_limit(sys.maxsize)
    with source.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        check({"pcap_uid","site_label","direction_vector"}<=set(reader.fieldnames or []),"missing columns")
        for row in reader:
            uid=row["pcap_uid"]
            if uid not in uids:continue
            name,date,split=uids[uid]
            if split=="DEV_LATE":continue
            check(uid not in seen,"duplicate train UID")
            check(row["site_label"]==name,"UID class mismatch")
            d=parse_direction(row["direction_vector"])
            check(all(v in (-1,1) for v in d),"observed zero/invalid direction")
            X[n]=np.asarray(fixed_length(d),dtype=np.int8)
            seen.add(uid);n+=1;counts[split]+=1
            if n%10000==0:print("S9_TRAIN_CACHE_PROGRESS",n,flush=True)
    check(n==39258 and counts==Counter({"DEV_EARLY":19098,"DEV_MIDDLE":20160}),
          "Historical train count mismatch")
    expected={k for k,v in uids.items() if v[2]!="DEV_LATE"}
    check(seen==expected,"Historical train UID coverage mismatch")
    X.flush();del X
    os.replace(tmp,p)
    m={"classification":"S9_NETCLR_HISTORICAL_TRAIN_ONLY_CACHE",
       "n":n,"counts":dict(counts),"source_sha256":digest(source),
       "label_sha256":s1["label_order_sha256"],"array_sha256":sha(p),
       "dev_late_opened":False,"external_data_opened":False,"dataset_c_opened":False}
    info.write_text(json.dumps(m,indent=2)+"\n")
    return np.load(p,mmap_mode="r"),m

def burst_distribution(X):
    # Estimate outgoing burst-size distribution exclusively from training.
    # Max unique size cap at 5000; input is already 5000-clip representation.
    counts=Counter()
    for x in X:
        a=np.asarray(x)
        valid=a[a!=0]
        if not len(valid):continue
        changes=np.flatnonzero(np.diff(valid)!=0)+1
        starts=np.r_[0,changes];ends=np.r_[changes,len(valid)]
        for start,end in zip(starts,ends):
            if valid[start]==1:counts[int(end-start)]+=1
    check(bool(counts),"no outgoing burst sizes")
    values=np.asarray(sorted(counts),dtype=np.int32)
    weights=np.asarray([counts[int(k)] for k in values],dtype=np.float64)
    cdf=np.cumsum(weights)/weights.sum()
    check(np.isclose(cdf[-1],1.0) and np.all(np.diff(cdf)>=0),"CDF invalid")
    # Upstream constructor internally indexes range(max_outgoing_burst_size),
    # which does not preserve actual burst size values; record this limitation.
    max_outgoing=int(max(values.max()+1,len(values)+1))
    return max_outgoing,values.tolist(),cdf

class Views(Dataset):
    def __init__(self,X,augmentor):self.X=X;self.augmentor=augmentor
    def __len__(self):return len(self.X)
    def __getitem__(self,i):
        x=np.array(self.X[i],dtype=np.float32,copy=True)
        views=[]
        for _ in range(2):
            augmented=np.asarray(self.augmentor.augment(x.copy()),dtype=np.float32)
            check(augmented.shape==(5000,),"augment shape")
            check(np.isfinite(augmented).all(),"augment nonfinite")
            check(np.all(np.isin(augmented,(-1.,0.,1.))),"augment direction invalid")
            views.append(torch.from_numpy(augmented[None,:]))
        return views

def info_nce(emb,temperature):
    check(emb.ndim==2 and emb.shape[0]%2==0,"contrastive batch dimension")
    n=emb.shape[0]//2
    check(n>=2,"too few contrastive pairs")
    z=F.normalize(emb,dim=1)
    logits=z@z.T/temperature
    logits.fill_diagonal_(-1e9)
    targets=(torch.arange(n*2,device=emb.device)+n)%(2*n)
    return F.cross_entropy(logits,targets)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["smoke","pretrain-B"],required=True)
    args=ap.parse_args()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    modules=pinned()
    X,cache=build_cache()
    maximum,values,cdf=burst_distribution(X)
    augmentor=modules["augment"].Augmentor(maximum,values,cdf)
    train=Views(X,augmentor)
    if args.stage=="smoke":
        # Detect upstream augmentation edge cases before launching a full run.
        for i in np.linspace(0,len(train)-1,128,dtype=int):
            views=train[int(i)]
            check(len(views)==2 and all(v.shape==(1,5000) for v in views),
                  "invalid augmented pair")
        backbone=modules["model"].NetCLR(out_dim=128)
        model=modules["model"].DFsimCLR(backbone,out_dim=128).cpu().train()
        a=torch.stack([train[i][0][0] for i in range(4)])[:,None,:]
        b=torch.stack([train[i][1][0] for i in range(4)])[:,None,:]
        emb=model(torch.cat([a,b],dim=0))
        loss=info_nce(emb,0.5)
        check(bool(torch.isfinite(loss).item()),"loss nonfinite")
        loss.backward()
        print("S9_NETCLR_PRETRAIN_SMOKE_PASS","samples",len(X),
              "augmented_pairs_checked",128,"loss",float(loss.item()),
              "upstream_burst_values",len(values),flush=True)
        return
    check(torch.cuda.is_available(),"GPU required")
    target=OUT/"pretrain_B"
    check(not target.exists(),"pretrain B already exists; no overwrite")
    target.mkdir(parents=True,mode=0o700)
    torch.cuda.manual_seed_all(42)
    model=modules["model"].DFsimCLR(
        modules["model"].NetCLR(out_dim=128),out_dim=128).to("cuda:0")
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0003)
    loader=DataLoader(train,batch_size=64,shuffle=True,drop_last=True,
                      num_workers=0,pin_memory=True)
    beginning=time.monotonic()
    print("S9_NETCLR_PRETRAIN_B_BEGIN",len(X),"epochs",100,
          "batch_size",64,"GPU",torch.cuda.get_device_name(0),flush=True)
    losses=[]
    for epoch in range(1,101):
        model.train();running=[]
        for (views,) in loader:
            q=torch.cat(views,dim=0).to("cuda:0",non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            emb=model(q)
            loss=info_nce(emb,0.5)
            check(bool(torch.isfinite(loss).item()),"nonfinite contrastive loss")
            loss.backward();optimizer.step()
            running.append(float(loss.item()))
        record={"epoch":epoch,"contrastive_loss":float(np.mean(running))}
        losses.append(record)
        print("S9_NETCLR_PRETRAIN_B_EPOCH",json.dumps(record),flush=True)
        if epoch%10==0 or epoch==100:
            # Per-run checkpoint for resilience. No resumed scientific run
            # is claimed without independently preserving full RNG/optimizer state.
            ck=target/f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save({"model":model.cpu().state_dict(),"epoch":epoch,
                        "config":CONFIG},ck)
            model.to("cuda:0")
    last=target/"checkpoint_epoch_100.pt"
    m={"classification":"S9_NETCLR_WFLIB_ADAPTED_PRETRAIN_B",
       "config":CONFIG,"n":len(X),"checkpoint":str(last),
       "checkpoint_sha256":sha(last),
       "cache_manifest_sha256":sha(OUT/"dev_train_cache"/"manifest.json"),
       "outgoing_burst_max":maximum,"outgoing_burst_unique":len(values),
       "loss_history":losses,"elapsed_seconds":time.monotonic()-beginning,
       "train_splits":["DEV_EARLY","DEV_MIDDLE"],
       "dev_late_used_for_pretraining":False,"external_data_opened":False,
       "dataset_c_opened":False,
       "created_utc":datetime.now(timezone.utc).isoformat()}
    (target/"manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print("S9_NETCLR_PRETRAIN_B_PASS",m["checkpoint_sha256"],flush=True)

if __name__=="__main__":
    try:main()
    except Exception as exc:
        print("S9_NETCLR_ERROR",type(exc).__name__,str(exc)[:900],file=sys.stderr,flush=True)
        sys.exit(4)
