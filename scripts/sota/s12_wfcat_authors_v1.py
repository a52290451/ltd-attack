#!/usr/bin/env python3
"""S12 WFCAT original-author model and IAT feature extraction; LTD chronological protocol.

Loads the EXACT Python code at the pinned official WFCAT Git commit.
No WFlib architecture or recreated attention layer. Original .cell parser
sorts timestamps; we preserve that behavior on Historical CSV time/direction
pairs. Outputs isolated B/C/frozen 09B/Future-B evidence with 65 labels.

--stage prepare: CPU cache Historical DEV only (uint16 IAT histograms).
--stage smoke: CPU original model/feature smoke (no artifacts).
--stage B: train Early+Middle, select epoch by Late macro-F1.
--stage C: fresh fit all Historical DEV with selected B epochs.
--stage evaluate: only frozen inference, never adaptive fit.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,random,subprocess,sys,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.data_adapter import (
    load_s1_dev_index,parse_direction,digest)
from src.models.sota.v1.timing_adapter import parse_times

SOURCE=Path.home()/"ltd-ops/sota/upstream_2026/S12_WFCAT"
COMMIT="cd382d30f7d33714fa4d025cddb3daaf2ffb0f09"
S1=Path.home()/"ltd-ops/sota/s1_dev_frozen_20261102T130232Z/s1_manifest.json"
OUT=Path("/data/ltd-attack/sota/s12_wfcat_authors_iat_seed42")
CACHE=OUT/"historical_cache"
COHORTS={"09B":14687,"10B_DEV_FROZEN":18543}
S5=Path("/data/ltd-attack/sota/s5/frozen_s5_refits_and_labels_v1.json")
CFG={"model":"authors_WFCAT","upstream_commit":COMMIT,
     "feature":"iat_histogram_9_bins_per_direction","seq_length":1800,
     "time_window_seconds":0.044,"num_classes":65,
     "batch_size":16,"lr":0.001,"weight_decay":0.0005,
     "epochs":30,"seed":42,"num_kernels":4,
     "selection":"DEV_LATE_macro_F1_earliest_tie",
     "no_external_selection":True,"dataset_c_opened":False}
def must(ok,msg):
    if not ok:raise RuntimeError("S12_WFCAT_BLOCKED "+msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def upstream():
    r=subprocess.run(["git","-C",str(SOURCE),"rev-parse","HEAD"],
                     capture_output=True,text=True,check=True)
    must(r.stdout.strip()==COMMIT,"WFCAT original source changed")
    r=subprocess.run(["git","-C",str(SOURCE),"status","--porcelain",
                      "--untracked-files=no"],capture_output=True,text=True,check=True)
    must(not r.stdout.strip(),"original authors' source modified")
    sys.path.insert(0,str(SOURCE/"src"))
    from features import feature_transform
    from models.wfcat import WFCAT
    return feature_transform,WFCAT
def create_feature(direction,time_values,transform):
    must(len(direction)==len(time_values)>0,"unaligned direction/time")
    times=np.asarray(time_values,dtype=np.float64)
    direct=np.asarray(direction,dtype=np.float64)
    must(np.isfinite(times).all() and np.isin(direct,(-1,1)).all(),
         "invalid trace time/direction")
    # WFCAT official parse_trace sorts by time before feature extraction.
    ids=np.argsort(times,kind="stable")
    trace=np.stack((times[ids],direct[ids]),axis=1)
    z=transform(trace,"iat",1800,granularity=9,time_window=0.044)
    must(z.shape==(18,1800) and np.isfinite(z).all() and
         np.all(z>=0) and np.all(z<=65535) and
         np.all(z==np.floor(z)),"original WFCAT IAT count shape")
    return z.astype(np.uint16)
def cache():
    mpath=CACHE/"manifest.json"
    xs=CACHE/"iat_uint16.npy"
    ys=CACHE/"labels.npy"
    ss=CACHE/"split.npy"
    paths=(xs,ys,ss)
    if mpath.exists():
        m=json.loads(mpath.read_text())
        for p in paths:
            must(p.is_file() and sha(p)==m["sha256"][p.name],
                 "cache SHA changed "+p.name)
        X=np.load(xs,mmap_mode="r")
        y=np.load(ys)
        s=np.load(ss)
        must(X.shape==(57916,18,1800) and len(y)==len(s)==57916,
             "WFCAT cache shapes invalid")
        must(dict(zip(*np.unique(s,return_counts=True)))==
             {0:19098,1:20160,2:18658},"WFCAT split counts")
        return X,y,s,m
    if CACHE.exists():
        known=sorted(p.name for p in CACHE.iterdir())
        must(known in ([],["iat_uint16.npy.partial"]),
             "incomplete WFCAT cache has unexpected files: "+str(known))
        if known:
            (CACHE/"iat_uint16.npy.partial").unlink()
            print("S12_WFCAT_RESTART_INCOMPLETE_CPU_MEMMAP_ONLY",flush=True)
        else:
            print("S12_WFCAT_RESUME_EMPTY_CPU_CACHE_DIR",flush=True)
    transform,_=upstream()
    by_uid,cl,s1=load_s1_dev_index(S1)
    spec=s1["historical_checksums"]["historical_vectors"]
    source=Path(spec["path"])
    must(digest(source)==spec["sha256"],"frozen Historical vector digest")
    CACHE.mkdir(parents=True,mode=0o700,exist_ok=True)
    tmp=CACHE/"iat_uint16.npy.partial"
    X=np.lib.format.open_memmap(tmp,mode="w+",dtype=np.uint16,
                                shape=(57916,18,1800))
    Y=np.empty(57916,dtype=np.int64);S=np.empty(57916,dtype=np.uint8)
    n=0;seen=set();counts=Counter()
    split_id={"DEV_EARLY":0,"DEV_MIDDLE":1,"DEV_LATE":2}
    csv.field_size_limit(sys.maxsize)
    with source.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        must({"pcap_uid","site_label","direction_vector","time_vector"}<=
             set(reader.fieldnames or []),"missing source columns")
        for row in reader:
            uid=row["pcap_uid"]
            if uid not in by_uid:continue
            must(uid not in seen,"duplicate Historical UID")
            label,date,sp=by_uid[uid]
            must(row["site_label"]==label,"Historical class disagreement")
            direction=parse_direction(row["direction_vector"])
            times=parse_times(row["time_vector"])
            X[n]=create_feature(direction,times,transform)
            Y[n]=cl[label];S[n]=split_id[sp]
            seen.add(uid);counts[sp]+=1;n+=1
            if n%5000==0:
                print("S12_WFCAT_CPU_CACHE_PROGRESS",n,57916,flush=True)
    must(n==57916 and seen==set(by_uid) and
         dict(counts)=={"DEV_EARLY":19098,"DEV_MIDDLE":20160,
                        "DEV_LATE":18658},"WFCAT historical coverage")
    X.flush();del X
    os.replace(tmp,xs)
    np.save(ys,Y,allow_pickle=False);np.save(ss,S,allow_pickle=False)
    m={"classification":"S12_WFCAT_HISTORICAL_IAT_CACHE_V1",
       "n":n,"split_counts":dict(counts),
       "historical_source_sha256":spec["sha256"],
       "label_sha256":s1["label_order_sha256"],
       "source_commit":COMMIT,
       "sha256":{p.name:sha(p) for p in paths},
       "external_data_opened":False,"dataset_c_opened":False}
    mpath.write_text(json.dumps(m,indent=2)+"\n")
    return np.load(xs,mmap_mode="r"),Y,S,m
class Set(Dataset):
    def __init__(self,X,y,idx):self.X=X;self.y=y;self.idx=np.asarray(idx)
    def __len__(self):return len(self.idx)
    def __getitem__(self,i):
        j=int(self.idx[i])
        return torch.from_numpy(np.asarray(self.X[j],dtype=np.float32).copy()),int(self.y[j])
def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
def guard():
    p=subprocess.run(["nvidia-smi","--query-compute-apps=pid",
                      "--format=csv,noheader,nounits"],
                     capture_output=True,text=True,timeout=15)
    must(p.returncode==0 and not p.stdout.strip(),"GPU occupied "+p.stdout.strip())
def train(stage):
    must(stage in ("B","C"),"invalid fit")
    guard();must(torch.cuda.is_available(),"GPU unavailable")
    seed()
    X,y,s,cm=cache()
    target=OUT/stage
    must(not target.exists(),"existing result; refusing overwrite")
    _,Model=upstream()
    epochs=CFG["epochs"]
    if stage=="C":
        b=OUT/"B/manifest.json"
        must(b.is_file(),"B selection incomplete")
        info=json.loads(b.read_text())
        must(info["classification"]=="S12_WFCAT_CHRONOLOGICAL_B" and
             info["config"]==CFG,"B configuration changed")
        epochs=info["selected_epoch"]
        must(1<=epochs<=30,"selected epoch invalid")
    tr=np.flatnonzero(s!=2) if stage=="B" else np.arange(57916)
    va=np.flatnonzero(s==2)
    loader=DataLoader(Set(X,y,tr),batch_size=16,shuffle=True,
                      drop_last=True,num_workers=0,pin_memory=True)
    validation=DataLoader(Set(X,y,va),batch_size=32,shuffle=False,
                          num_workers=0,pin_memory=True)
    model=Model(num_classes=65,iat_bins=9,num_kernels_1d=4,
                num_kernels_2d=4).to("cuda:0")
    opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0005)
    criterion=nn.CrossEntropyLoss()
    target.mkdir(parents=True,mode=0o700)
    history=[];best=-1.;selected=0;started=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();total=0.;count=0
        for x,labels in loader:
            opt.zero_grad(set_to_none=True)
            logits=model(x.to("cuda:0",non_blocking=True))
            loss=criterion(logits,labels.to("cuda:0",non_blocking=True))
            must(bool(torch.isfinite(loss).item()),"non-finite fit loss")
            loss.backward();opt.step()
            total+=float(loss.item());count+=1
            if count%200==0:
                print("S12_WFCAT_FIT_PROGRESS",
                      json.dumps({"stage":stage,"epoch":epoch,"step":count,
                                  "of":len(loader),"elapsed_s":round(time.monotonic()-started,1)}),
                      flush=True)
        rec={"epoch":epoch,"train_loss":total/count}
        if stage=="B":
            model.eval();truth=[];pred=[]
            with torch.inference_mode():
                for x,labels in validation:
                    scores=model(x.to("cuda:0",non_blocking=True))
                    pred.append(scores.argmax(1).cpu().numpy());truth.append(labels.numpy())
            gold=np.concatenate(truth);p=np.concatenate(pred)
            rec["val_accuracy"]=float((gold==p).mean())
            rec["val_macro_f1"]=float(f1_score(gold,p,labels=np.arange(65),average="macro"))
            if rec["val_macro_f1"]>best:
                best=rec["val_macro_f1"];selected=epoch
                torch.save({"model":model.state_dict(),"config":CFG,"epoch":epoch},
                           target/"best.pt")
        history.append(rec)
        print("S12_WFCAT_EPOCH",stage,json.dumps(rec),flush=True)
    if stage=="C":
        ck=target/"model.pt";selected=epochs
        torch.save({"model":model.cpu().state_dict(),"config":CFG,"epoch":epochs},ck)
    else:ck=target/"best.pt"
    m={"classification":"S12_WFCAT_CHRONOLOGICAL_"+stage,
       "config":CFG,"train_n":len(tr),
       "selected_epoch":selected,"history":history,
       "checkpoint":str(ck),"checkpoint_sha256":sha(ck),
       "cache_manifest_sha256":sha(CACHE/"manifest.json"),
       "external_data_opened":False,"dataset_c_opened":False,
       "created_utc":datetime.now(timezone.utc).isoformat()}
    (target/"manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print("S12_WFCAT_FIT_PASS",stage,sha(ck),flush=True)

def evaluate(cohort):
    must(cohort in COHORTS,"unknown cohort")
    target=OUT/"evaluation"/cohort
    must(not target.exists(),"evaluation evidence exists")
    mp=OUT/"C/manifest.json"
    must(mp.is_file(),"C checkpoint missing")
    c=json.loads(mp.read_text())
    must(c["classification"]=="S12_WFCAT_CHRONOLOGICAL_C"
         and c["config"]==CFG,"C method mismatch")
    ck=Path(c["checkpoint"])
    must(sha(ck)==c["checkpoint_sha256"],"C checkpoint SHA mismatch")
    frozen=json.loads(S5.read_text())
    ref=frozen["label_alignment"][cohort]
    must(ref["identity_order"] is True,"canonical class order invalid")
    canonical=Path(ref["source"])
    must(sha(canonical)==ref["source_sha256"],"canonical score hash differs")
    with np.load(canonical,allow_pickle=True) as z:
        uid=z["pcap_uid"].astype(str);y=z["y_true"].astype(int)
        dates=z["query_date"].astype("datetime64[D]").astype(str)
        names=z["candidate_labels"].astype(str)
    n=COHORTS[cohort]
    must(len(uid)==n and len(set(uid))==n and len(names)==65,"cohort UID/labels")
    true_by_uid={str(k):str(names[v]) for k,v in zip(uid,y)}
    src=(Path.home()/"ltd-storage/datasets/historical/CLEAN_final_vectors_sites.csv"
         if cohort=="09B" else
         Path.home()/"ltd-storage/datasets/future/CLEAN_final_vectors_sites_concept_drift.csv")
    if cohort=="09B":
        s1=json.loads(S1.read_text())
        expected=s1["historical_checksums"]["historical_vectors"]["sha256"]
    else:
        fm=json.loads((ROOT/"docs/evidence/FINAL/FUTURE-B/10B_manifest_future_b.json").read_text())
        expected=fm["future_inputs"]["micro"]["sha256"]
    must(sha(src)==expected,"external CSV digest mismatch")
    transform,Model=upstream()
    got={}
    csv.field_size_limit(sys.maxsize)
    with src.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        for row in reader:
            k=row["pcap_uid"]
            if k not in true_by_uid:continue
            must(k not in got and row["site_label"]==true_by_uid[k],
                 "external UID class mismatch")
            got[k]=create_feature(parse_direction(row["direction_vector"]),
                                  parse_times(row["time_vector"]),transform)
    must(len(got)==n,"external sample coverage")
    model=Model(num_classes=65,iat_bins=9,num_kernels_1d=4,num_kernels_2d=4)
    chk=torch.load(ck,map_location="cpu",weights_only=True)
    must(chk["config"]==CFG and chk["epoch"]==c["selected_epoch"],
         "C checkpoint config mismatch")
    model.load_state_dict(chk["model"],strict=True)
    model.eval()
    probs=np.empty((n,65),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,n,16):
            end=min(start+16,n)
            x=np.stack([got[k] for k in uid[start:end]]).astype(np.float32)
            logits=model(torch.from_numpy(x))
            probs[start:end]=torch.softmax(logits,dim=1).numpy()
            if start%1600==0:
                print("S12_WFCAT_EVALUATION_PROGRESS",cohort,end,n,flush=True)
    must(np.isfinite(probs).all() and
         np.allclose(probs.sum(1),1.,atol=2e-4,rtol=0),"bad probabilities")
    order=np.argsort(-probs,axis=1)
    rank=(order==y[:,None]).argmax(1)+1
    top=order[:,0]
    tp=np.bincount(y[top==y],minlength=65)
    support=np.bincount(y,minlength=65)
    pred=np.bincount(top,minlength=65)
    f1=float(np.divide(2*tp,support+pred,out=np.zeros(65),
                       where=support+pred>0).mean())
    metrics={"accuracy":float((top==y).mean()),"macro_f1":f1,
             "top5":float((rank<=5).mean()),"mrr":float((1/rank).mean())}
    target.mkdir(parents=True,mode=0o700)
    score=target/"scores_sota_aligned.npz"
    np.savez_compressed(score,pcap_uid=uid,query_date=dates,y_true=y,
                        candidate_labels=names,sota_probs=probs)
    m={"classification":"S12_WFCAT_FROZEN_EXTERNAL_V1",
       "cohort":cohort,"n":n,"metrics":metrics,
       "score_sha256":sha(score),"checkpoint_sha256":sha(ck),
       "no_external_tuning":True,"dataset_c_opened":False}
    (target/"manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print("S12_WFCAT_EXTERNAL_PASS",cohort,json.dumps(metrics),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["smoke","prepare","B","C","evaluate"],required=True)
    parser.add_argument("--cohort",choices=list(COHORTS))
    a=parser.parse_args()
    if a.stage=="smoke":
        transform,Model=upstream()
        sample=np.array([[0.,1.],[0.01,-1.],[0.02,1.]])
        x=create_feature(sample[:,1],sample[:,0],transform)
        model=Model(num_classes=65,iat_bins=9,num_kernels_1d=4,num_kernels_2d=4)
        model.eval()
        with torch.inference_mode():
            out=model(torch.tensor(x,dtype=torch.float32)[None])
        must(out.shape==(1,65),"WFCAT CPU model smoke output")
        print("S12_WFCAT_AUTHORS_SOURCE_SMOKE_PASS",COMMIT,flush=True)
    elif a.stage=="prepare":cache();print("S12_WFCAT_CPU_CACHE_PASS",flush=True)
    elif a.stage in ("B","C"):train(a.stage)
    else:
        must(a.cohort is not None,"cohort required")
        evaluate(a.cohort)
if __name__=="__main__":main()
