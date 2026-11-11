#!/usr/bin/env python3
"""S8 Tik-Tok WFlib fixed architecture: historical-only chronological B training.

B: train on DEV_EARLY+DEV_MIDDLE, select best Macro-F1 epoch on DEV_LATE.
No 09B, Future-B, Dataset C, third-party file modifications or network.
All artifact writes are exclusive. Run C only after reviewing B evidence.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, random, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.data_adapter import load_s1_dev_index, parse_direction, digest
from src.models.sota.v1.timing_adapter import parse_times
from src.models.sota.v1.tiktok_adapter import tiktok_dt_from_sequences
from src.models.sota.v1.upstream_models import resolve_checkout, verify_checkout, WFLIB_COMMIT

from scripts.sota.sota_paths_v1 import S1 as FROZEN_S1, ARTIFACT
S1=FROZEN_S1
OUT=ARTIFACT/"s8_tiktok_wflib_dt_seed42"
REL="WFlib/models/TikTok.py"
BLOB="8736ef4d7d4c00bb1559def532f511cb8e142625"
COUNTS={"DEV_EARLY":19098,"DEV_MIDDLE":20160,"DEV_LATE":18658}
CONFIG={"arch":"WFlib_TikTok","upstream_commit":WFLIB_COMMIT,
        "upstream_blob":BLOB,"feature":"DT_signed_relative_seconds",
        "length":5000,"num_classes":65,"seed":42,
        "optimizer":"Adamax","lr":0.002,"batch_size":128,"epochs":30,
        "selection":"DEV_LATE_macro_f1_best_epoch_first_tie",
        "no_external_tuning":True,"torch_precision":"float32",
        "training":"Historical_DEV_EARLY_MIDDLE_only"}

def must(ok,detail):
    if not ok:raise RuntimeError("S8_BLOCKED "+detail)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()

def original_model():
    checkout=resolve_checkout()
    verify_checkout(checkout)
    p=checkout/REL
    import subprocess, types
    r=subprocess.run(["git","-C",str(checkout),"hash-object",REL],
                     text=True,capture_output=True,check=True)
    must(r.stdout.strip()==BLOB,"upstream TikTok blob changed")
    module=types.ModuleType("ltd_pinned_tiktok")
    module.__file__=str(p)
    exec(compile(p.read_bytes(),str(p),"exec"),module.__dict__)
    return module.TikTok(num_classes=65)

def load_data():
    uids,classes,s1=load_s1_dev_index(S1)
    spec=s1["historical_checksums"]["historical_vectors"]
    source=Path(spec["path"])
    must(digest(source)==spec["sha256"],"historical source SHA mismatch")
    # A cache is never considered validated unless its manifest and arrays
    # have the expected checksums. It is intentionally reusable by B only.
    cache=OUT/"historical_cache_v1"
    manifest=cache/"manifest.json"
    matrix=cache/"dt_float32.npy"
    labels=cache/"labels.npy"
    splits=cache/"split.npy"
    if manifest.exists():
        m=json.loads(manifest.read_text())
        must(m["source_sha256"]==spec["sha256"] and m["label_sha256"]==s1["label_order_sha256"],"cache provenance mismatch")
        for p in (matrix,labels,splits):
            must(p.exists() and sha(p)==m["sha256"][p.name],"cache bytes mismatch "+p.name)
        X=np.load(matrix,mmap_mode="r")
        y=np.load(labels)
        s=np.load(splits)
        must(X.shape==(57916,5000) and y.shape==(57916,) and s.shape==(57916,),"cached shapes mismatch")
        return X,y,s,s1,m
    must(not cache.exists(),"partial cache exists; inspect before retry")
    cache.mkdir(parents=True,mode=0o700)
    # tmp outputs exist only during cache construction; don't overwrite.
    tmp=cache/"dt_float32.npy.partial"
    X=np.lib.format.open_memmap(tmp,mode="w+",dtype=np.float32,shape=(57916,5000))
    y=np.empty(57916,dtype=np.int64)
    s=np.empty(57916,dtype=np.uint8)
    seen=set();counts=Counter();negative_steps=0
    split_id={"DEV_EARLY":0,"DEV_MIDDLE":1,"DEV_LATE":2}
    csv.field_size_limit(sys.maxsize)
    with source.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        must({"pcap_uid","site_label","direction_vector","time_vector"}<=set(reader.fieldnames or []),
             "missing historical DT columns")
        for row in reader:
            uid=row["pcap_uid"]
            if uid not in uids:continue
            must(uid not in seen,"duplicate historical UID")
            label,date,split=uids[uid]
            must(row["site_label"]==label,"UID/label mismatch")
            directions=parse_direction(row["direction_vector"])
            times=parse_times(row["time_vector"])
            must(len(directions)==len(times) and all(d in (-1,1) for d in directions),
                 "bad DT sequences")
            negative_steps+=sum(times[j+1]<times[j] for j in range(len(times)-1))
            i=len(seen)
            X[i]=np.asarray(tiktok_dt_from_sequences(directions,times),dtype=np.float32)
            y[i]=classes[label];s[i]=split_id[split]
            seen.add(uid);counts[split]+=1
            if len(seen)%10000==0:
                print("S8_DT_CACHE_PROGRESS",len(seen),57916,flush=True)
    must(seen==set(uids) and dict(counts)==COUNTS,
         "historical split population mismatch")
    X.flush();del X
    os.replace(tmp,matrix)
    np.save(labels,y,allow_pickle=False)
    np.save(splits,s,allow_pickle=False)
    m={"classification":"S8_TIKTOK_HISTORICAL_CACHE_V1",
       "source_sha256":spec["sha256"],"label_sha256":s1["label_order_sha256"],
       "n":57916,"splits":dict(counts),"negative_timestamp_steps":negative_steps,
       "sha256":{p.name:sha(p) for p in (matrix,labels,splits)},
       "dataset_c_opened":False,"future_b_opened":False}
    manifest.write_text(json.dumps(m,indent=2)+"\n")
    print("S8_DT_CACHE_PASS",len(seen),"negative_steps",negative_steps,flush=True)
    return np.load(matrix,mmap_mode="r"),y,s,s1,m

class NpyDataset(Dataset):
    def __init__(self,X,y,idx):self.X=X;self.y=y;self.idx=np.asarray(idx)
    def __len__(self):return len(self.idx)
    def __getitem__(self,i):
        j=int(self.idx[i])
        # np.array makes an owned writable batch sample
        return torch.from_numpy(np.array(self.X[j],copy=True)[None,:]),int(self.y[j])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["B"],required=True)
    args=parser.parse_args()
    target=OUT/args.stage
    must(not target.exists(),"refuse overwrite prior S8 B run")
    must(torch.cuda.is_available(),"CUDA GPU required; CPU-only training not authorized")
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device("cuda:0")
    X,y,s,s1,cache=load_data()
    tr=np.flatnonzero(s!=2)
    va=np.flatnonzero(s==2)
    must(len(tr)==39258 and len(va)==18658,"wrong train/validation count")
    must(set(y)==set(range(65)),"65 classes incomplete")
    train=DataLoader(NpyDataset(X,y,tr),batch_size=128,shuffle=True,
                     drop_last=True,num_workers=0,pin_memory=True)
    valid=DataLoader(NpyDataset(X,y,va),batch_size=256,shuffle=False,
                     drop_last=False,num_workers=0,pin_memory=True)
    model=original_model().to(device)
    opt=torch.optim.Adamax(model.parameters(),lr=0.002)
    criterion=nn.CrossEntropyLoss()
    print("S8_TIKTOK_B_TRAIN_BEGIN",len(tr),len(va),"cuda",torch.cuda.get_device_name(0),flush=True)
    t0=time.monotonic()
    # Reserve exclusive directory at start, never silently overwrite.
    target.mkdir(parents=True,mode=0o700)
    best=-1.0;best_epoch=None;history=[]
    for epoch in range(1,31):
        model.train()
        losses=[]
        for bx,by in train:
            bx=bx.to(device,non_blocking=True)
            by=by.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits=model(bx)
            loss=criterion(logits,by)
            must(bool(torch.isfinite(loss).item()),"non-finite training loss")
            loss.backward();opt.step()
            losses.append(float(loss.item()))
        model.eval()
        preds=[];truth=[]
        with torch.no_grad():
            for bx,by in valid:
                logits=model(bx.to(device,non_blocking=True))
                must(bool(torch.isfinite(logits).all().item()),"nonfinite validation logits")
                preds.append(logits.argmax(1).cpu().numpy())
                truth.append(by.numpy())
        pred=np.concatenate(preds)
        gold=np.concatenate(truth)
        acc=float(np.mean(pred==gold))
        f1=float(f1_score(gold,pred,labels=np.arange(65),average="macro"))
        rec={"epoch":epoch,"train_loss":float(np.mean(losses)),
             "val_accuracy":acc,"val_macro_f1":f1}
        history.append(rec)
        print("S8_TIKTOK_B_EPOCH",json.dumps(rec),flush=True)
        if f1>best:
            best=f1;best_epoch=epoch
            # temporary single selected checkpoint, replaced on genuine best epoch;
            # no previous run checkpoint is touched.
            torch.save({"model":model.state_dict(),"epoch":epoch,
                        "config":CONFIG},target/"best_checkpoint.pt")
    must(best_epoch is not None and best>=0,"no epoch selected")
    ck=target/"best_checkpoint.pt"
    manifest={"classification":"S8_TIKTOK_WFLIB_HISTORICAL_B_V1",
              "stage":"B","config":CONFIG,"selected_epoch":best_epoch,
              "validation":history[best_epoch-1],
              "history":history,"checkpoint":str(ck),
              "checkpoint_sha256":sha(ck),
              "cache_manifest_sha256":sha(OUT/"historical_cache_v1"/"manifest.json"),
              "label_sha256":s1["label_order_sha256"],
              "upstream_commit":WFLIB_COMMIT,"upstream_blob":BLOB,
              "elapsed_seconds":time.monotonic()-t0,
              "used_external_samples_for_fit":False,"dataset_c_opened":False,
              "created_utc":datetime.now(timezone.utc).isoformat()}
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("S8_TIKTOK_B_PASS",best_epoch,manifest["validation"],manifest["checkpoint_sha256"],flush=True)

if __name__=="__main__":
    try:main()
    except Exception as exc:
        print("S8_TIKTOK_B_ERROR",type(exc).__name__,str(exc)[:700],file=sys.stderr,flush=True)
        sys.exit(4)
