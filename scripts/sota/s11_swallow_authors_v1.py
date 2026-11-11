#!/usr/bin/env python3
"""S11 Swallow original authors' CIF, ResNet18, MLPHead, NetAugment, BYOL.

Original-source files loaded from immutable Swallow git checkout. One explicit
safety fix corrects the upstream trace_aggregation while loop which otherwise
never increments i during the first 20 packets. No replacement architecture.
Historical DEV chronology enforced; 09B/Future-B never used for fit/selection.

Stages: smoke / prepare / pretrain-B / supervised-B / C / evaluate.
This is author-source-adapted, NOT an exact paper/5-shot reproduction.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,inspect,json,random,subprocess,sys,textwrap,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.data_adapter import load_s1_dev_index,parse_direction,digest
from src.models.sota.v1.timing_adapter import parse_times

SOURCE=Path.home()/"ltd-ops/sota/upstream_2026/S11_Swallow"
COMMIT="c6dc304bca6e61b10b36b00178ee84e38c17961b"
S1=Path.home()/"ltd-ops/sota/s1_dev_frozen_20261102T130232Z/s1_manifest.json"
OUT=Path("/data/ltd-attack/sota/s11_swallow_authors_seed42")
CACHE=OUT/"historical_cif_cache"
S5=Path("/data/ltd-attack/sota/s5/frozen_s5_refits_and_labels_v1.json")
COHORTS={"09B":14687,"10B_DEV_FROZEN":18543}
CONFIG={"method":"Swallow_BBYOL_ResNet18_authors_source_adapted",
        "commit":COMMIT,"seed":42,"classes":65,"representation":"authors_CIF_2x1000",
        "pretrain_epochs":100,"pretrain_batch":32,"m":0.996,
        "optimizer":"SGD","lr":0.03,"momentum":0.9,"weight_decay":0.0004,
        "supervised_epochs_max":30,"fine_tune_batch":32,
        "fine_tune_lr":0.0001,"augmentation":"authors_3methods_with_aggregation_loop_fix",
        "author_original_pretrain_epochs":200,
        "author_original_batch_size":512,
        "selection":"DEV_LATE_macro_f1_earliest_tie",
        "external_data_used_for_fit":False,"dataset_c_opened":False,
        "paper_exact":False}
def must(ok,reason):
    if not ok:raise RuntimeError("S11_SWALLOW_BLOCKED "+reason)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def origin():
    r=subprocess.run(["git","-C",str(SOURCE),"rev-parse","HEAD"],
                     text=True,capture_output=True,check=True)
    must(r.stdout.strip()==COMMIT,"Swallow checkout changed")
    d=subprocess.run(["git","-C",str(SOURCE),"status",
                      "--porcelain","--untracked-files=no"],
                     text=True,capture_output=True,check=True)
    must(not d.stdout.strip(),"author tracked files modified")
    sys.path.insert(0,str(SOURCE))
    from CIF import fun
    from models.resnet_base_network import ResNet18,MLPHead
    from loader import dataAugmentation as source_aug
    # Upstream trace_aggregation has an unconditional continue before i += 1
    # for <=20 handshake packets, causing infinite loops. Explicit patch
    # scoped to the ORIGINAL function's AST; source checkout is never modified.
    # Original source remains unchanged on disk. Only patch the specific
    # handshake branch of the original trace_aggregation method, preserving
    # the indentation as published. The adapter calls origin() more than once.
    if not getattr(source_aug.Augmentor, "_ltd_aggregation_patched", False):
        import re
        method=textwrap.dedent(
            inspect.getsource(source_aug.Augmentor.trace_aggregation))
        pattern=r"(?m)^([ \t]*)if packet_sum <= self\.handshake_packet_sum:\n([ \t]*)continue[ \t]*$"
        matches=list(re.finditer(pattern,method))
        must(len(matches)==1,
             "expected one original trace_aggregation handshake branch")
        def insert_increment(m):
            return (m.group(1)+"if packet_sum <= self.handshake_packet_sum:\n"
                    +m.group(2)+"i += 1\n"+m.group(2)+"continue")
        patched=re.sub(pattern,insert_increment,method)
        namespace=dict(source_aug.__dict__)
        exec(compile(patched,"swallow_authors_loop_fix","exec"),namespace)
        source_aug.Augmentor.trace_aggregation=namespace["trace_aggregation"]
        source_aug.Augmentor._ltd_aggregation_patched=True
    return fun,ResNet18,MLPHead,source_aug.Augmentor

def cif(direction,times,fun):
    must(len(direction)==len(times)>0,"CIF trace mismatch")
    arr=np.asarray(direction,dtype=np.int64)
    t=np.asarray(times,dtype=np.float64)
    must(np.isfinite(t).all() and (t>=0).all(),"CIF nonfinite/negative times")
    # Original ProcessData.py retains the incoming packet order and calls
    # CIF.fun directly. Some captured timestamps regress; do not sort packets
    # (sorting would change the feature distribution and topology). The
    # original CIF computes each slot index from each packet's timestamp.
    # Preserve that upstream behavior; negative gaps are counted separately.
    must(float(t[-1])>0 and len(t)>=2,"original CIF needs positive final load time")
    # Original CIF.fun consumes signs, not packet lengths. This is the
    # same input semantics as its own ProcessData.py for +/-1 traces.
    matrix,load,duration=fun(t,arr)
    out=np.asarray(matrix,dtype=np.int32)
    must(out.shape==(2,1000) and np.all(out>=0) and
         out.max()<=65535 and duration>0,"CIF source format mismatch")
    return out.astype(np.uint16),float(load),float(duration)

def cache():
    manifest=CACHE/"manifest.json"
    files=(CACHE/"cif_uint16.npy",CACHE/"labels.npy",CACHE/"split.npy",
           CACHE/"load_time.npy",CACHE/"slot_duration.npy")
    if manifest.exists():
        m=json.loads(manifest.read_text())
        for p in files:
            must(p.is_file() and sha(p)==m["sha256"][p.name],
                 "cached Swallow CIF digest mismatch "+p.name)
        X=np.load(files[0],mmap_mode="r")
        y=np.load(files[1]);s=np.load(files[2])
        t=np.load(files[3]);dur=np.load(files[4])
        must(X.shape==(57916,2,1000) and len(y)==len(s)==len(t)==len(dur)==57916,
             "CIF cached shapes")
        must(dict(zip(*np.unique(s,return_counts=True)))==
             {0:19098,1:20160,2:18658},"CIF chrono split")
        return X,y,s,t,dur,m
    if CACHE.exists():
        # Prior failing CPU run creates only a partially written local .npy
        # memmap. Never touch complete/frozen files, and never delete a
        # directory containing artifacts from another process.
        known=sorted(p.name for p in CACHE.iterdir())
        must(known==["cif_uint16.npy.partial"],
             "partial/nonstandard Swallow CIF cache; inspect manually: "+str(known))
        (CACHE/"cif_uint16.npy.partial").unlink()
        print("S11_SWALLOW_RESTART_INCOMPLETE_CPU_MEMMAP_ONLY",flush=True)
    fun,_,_,_=origin()
    by_uid,classes,s1=load_s1_dev_index(S1)
    spec=s1["historical_checksums"]["historical_vectors"]
    src=Path(spec["path"])
    must(digest(src)==spec["sha256"],"frozen DEV changed")
    CACHE.mkdir(parents=True,mode=0o700,exist_ok=True)
    temp=CACHE/"cif_uint16.npy.partial"
    X=np.lib.format.open_memmap(temp,mode="w+",dtype=np.uint16,
                                shape=(57916,2,1000))
    y=np.empty(57916,dtype=np.int64);s=np.empty(57916,dtype=np.uint8)
    load_time=np.empty(57916,dtype=np.float64)
    duration=np.empty(57916,dtype=np.float64)
    mapping={"DEV_EARLY":0,"DEV_MIDDLE":1,"DEV_LATE":2}
    seen=set();counts=Counter();n=0
    regressions=0
    traces_with_regressions=0
    csv.field_size_limit(sys.maxsize)
    with src.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        must({"pcap_uid","site_label","direction_vector","time_vector"}<=
             set(reader.fieldnames or []),"Historical CIF columns missing")
        for row in reader:
            uid=row["pcap_uid"]
            if uid not in by_uid:continue
            must(uid not in seen,"duplicate DEV UID")
            label,date,sp=by_uid[uid]
            must(label==row["site_label"],"DEV UID class mismatch")
            dirs=parse_direction(row["direction_vector"])
            times=parse_times(row["time_vector"])
            backward=int(np.count_nonzero(np.diff(times)<0))
            regressions+=backward
            traces_with_regressions+=int(backward>0)
            x,t,dt=cif(dirs,times,fun)
            X[n]=x;y[n]=classes[label];s[n]=mapping[sp]
            load_time[n]=t;duration[n]=dt
            seen.add(uid);counts[sp]+=1;n+=1
            if n%5000==0:print("S11_CIF_CPU_CACHE_PROGRESS",n,57916,flush=True)
    must(n==57916 and seen==set(by_uid) and
         dict(counts)=={"DEV_EARLY":19098,"DEV_MIDDLE":20160,
                        "DEV_LATE":18658},"CIF canonical source coverage")
    X.flush();del X
    import os
    os.replace(temp,files[0])
    np.save(files[1],y,allow_pickle=False)
    np.save(files[2],s,allow_pickle=False)
    np.save(files[3],load_time,allow_pickle=False)
    np.save(files[4],duration,allow_pickle=False)
    m={"classification":"S11_SWALLOW_CIF_SOURCE_HISTORICAL_CACHE_V1",
       "source_commit":COMMIT,"original_CIF_sha256":sha(SOURCE/"CIF.py"),
       "timestamp_regressions":regressions,
       "traces_with_timestamp_regressions":traces_with_regressions,
       "timestamp_policy":"preserve_packet_order_use_original_CIF_fun_no_sort",
       "n":n,"counts":dict(counts),
       "historical_sha256":spec["sha256"],
       "sha256":{p.name:sha(p) for p in files},
       "external_data_opened":False,"dataset_c_opened":False}
    manifest.write_text(json.dumps(m,indent=2)+"\n")
    return np.load(files[0],mmap_mode="r"),y,s,load_time,duration,m

class Data(Dataset):
    def __init__(self,X,y,indices,load_time=None,duration=None,aug=None):
        self.X=X;self.y=y;self.idx=np.asarray(indices)
        self.times=load_time;self.duration=duration;self.aug=aug
    def __len__(self):return len(self.idx)
    def __getitem__(self,i):
        j=int(self.idx[i])
        x=np.asarray(self.X[j],dtype=np.float32).copy()
        if self.aug is not None:
            x1=np.asarray(self.aug.augment(x.copy(),float(self.times[j]),
                                           float(self.duration[j])),dtype=np.float32)
            x2=np.asarray(self.aug.augment(x.copy(),float(self.times[j]),
                                           float(self.duration[j])),dtype=np.float32)
            must(x1.shape==(2,1000) and x2.shape==(2,1000) and
                 np.isfinite(x1).all() and np.isfinite(x2).all(),
                 "original augment output invalid")
            return torch.from_numpy(x1[None,:,:]),torch.from_numpy(x2[None,:,:])
        return torch.from_numpy(x[None,:,:]),int(self.y[j])

def network():
    fun,Model,MLP,Augmentor=origin()
    kw={"name":"resnet18","projection_head":
        {"mlp_hidden_size":512,"projection_size":128}}
    return Model(**kw),MLP,Augmentor

def nce_byol(pred,target):
    return (2-2*(nn.functional.normalize(pred,dim=1)*
                 nn.functional.normalize(target,dim=1)).sum(dim=1)).mean()
def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def guard():
    r=subprocess.run(["nvidia-smi","--query-compute-apps=pid",
                      "--format=csv,noheader,nounits"],capture_output=True,
                     text=True,timeout=15)
    must(r.returncode==0 and not r.stdout.strip(),"GPU occupied: "+r.stdout.strip())

def pretrain():
    must(not (OUT/"pretrain_B").exists(),"S11 B pretrain exists")
    guard();must(torch.cuda.is_available(),"CUDA required")
    seed();X,y,s,t,d,cm=cache()
    tr=np.flatnonzero(s!=2)
    must(len(tr)==39258,"train excludes DEV_LATE")
    online,MLP,Aug=network()
    target,_,_=network()
    target.load_state_dict(online.state_dict(),strict=True)
    for p in target.parameters():p.requires_grad_(False)
    predictor=MLP(in_channels=128,mlp_hidden_size=512,
                  projection_size=128)
    online=online.to("cuda:0");target=target.to("cuda:0")
    predictor=predictor.to("cuda:0")
    opt=torch.optim.SGD(list(online.parameters())+list(predictor.parameters()),
                       lr=0.03,momentum=0.9,weight_decay=0.0004)
    loader=DataLoader(Data(X,y,tr,t,d,Aug()),batch_size=32,shuffle=True,
                      drop_last=True,num_workers=0,pin_memory=True)
    dest=OUT/"pretrain_B";dest.mkdir(parents=True,mode=0o700)
    started=time.monotonic();hist=[]
    for epoch in range(1,101):
        online.train();predictor.train()
        losses=[]
        for step,(v1,v2) in enumerate(loader,1):
            v1=v1.to("cuda:0",non_blocking=True)
            v2=v2.to("cuda:0",non_blocking=True)
            q1=predictor(online(v1));q2=predictor(online(v2))
            with torch.no_grad():
                target.eval();t1=target(v1);t2=target(v2)
            loss=nce_byol(q1,t2)+nce_byol(q2,t1)
            must(bool(torch.isfinite(loss).item()),"BYOL loss nonfinite")
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            with torch.no_grad():
                for p,q in zip(online.parameters(),target.parameters()):
                    q.mul_(0.996).add_(p,alpha=0.004)
            losses.append(float(loss.item()))
            if step%100==0:
                print("S11_SWALLOW_PRETRAIN_PROGRESS",
                      json.dumps({"epoch":epoch,"step":step,"of":len(loader),
                                  "elapsed_s":round(time.monotonic()-started,1)}),
                      flush=True)
        rec={"epoch":epoch,"BYOL_loss":float(np.mean(losses))}
        hist.append(rec)
        print("S11_SWALLOW_PRETRAIN_EPOCH",json.dumps(rec),flush=True)
        if epoch%20==0 or epoch==100:
            ck=dest/f"epoch_{epoch:03d}.pt"
            torch.save({"model":online.cpu().state_dict(),"epoch":epoch,
                        "config":CONFIG},ck)
            online.to("cuda:0")
    ck=dest/"epoch_100.pt"
    m={"classification":"S11_SWALLOW_AUTHORS_B_PRETRAIN_V1",
       "config":CONFIG,"training_n":39258,"checkpoint":str(ck),
       "checkpoint_sha256":sha(ck),
       "cache_sha256":sha(CACHE/"manifest.json"),
       "history":hist,"external_data_opened":False,
       "dataset_c_opened":False}
    (dest/"manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print("S11_SWALLOW_PRETRAIN_PASS",sha(ck),flush=True)

def supervised(stage):
    must(stage in ("B","C"),"invalid supervised stage")
    guard();must(torch.cuda.is_available(),"GPU required")
    X,y,s,t,d,cm=cache()
    pre=OUT/"pretrain_B/manifest.json"
    must(pre.is_file(),"original authors BYOL B pretraining missing")
    meta=json.loads(pre.read_text())
    must(meta.get("config")==CONFIG and
         meta.get("classification")=="S11_SWALLOW_AUTHORS_B_PRETRAIN_V1",
         "pretrain provenance changed")
    ck=Path(meta["checkpoint"])
    must(sha(ck)==meta["checkpoint_sha256"],"pretrain checkpoint changed")
    dest=OUT/("supervised_"+stage)
    must(not dest.exists(),"supervised evidence exists")
    epochs=CONFIG["supervised_epochs_max"]
    if stage=="C":
        prior=OUT/"supervised_B/manifest.json"
        must(prior.is_file(),"selected supervised B missing")
        bm=json.loads(prior.read_text())
        must(bm["classification"]=="S11_SWALLOW_SUPERVISED_B_V1",
             "B evidence changed")
        epochs=bm["selected_epoch"]
    seed()
    full,_,_=network()
    pretrained=torch.load(ck,map_location="cpu",weights_only=True)
    must(pretrained["config"]==CONFIG and pretrained["epoch"]==100,
         "pretrained model contract")
    full.load_state_dict(pretrained["model"],strict=True)
    # Original author's fine-tuning removes the projection head.
    encoder=nn.Sequential(*list(full.children())[:-1])
    classifier=nn.Linear(512,65)
    encoder=encoder.to("cuda:0");classifier=classifier.to("cuda:0")
    opt=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),
                         lr=0.0001)
    crit=nn.CrossEntropyLoss()
    tr=np.flatnonzero(s!=2) if stage=="B" else np.arange(57916)
    va=np.flatnonzero(s==2)
    loader=DataLoader(Data(X,y,tr),batch_size=32,shuffle=True,
                      drop_last=True,num_workers=0,pin_memory=True)
    valid=DataLoader(Data(X,y,va),batch_size=64,shuffle=False,
                     num_workers=0,pin_memory=True)
    dest.mkdir(parents=True,mode=0o700)
    hist=[];best=-1.;selected=0
    for epoch in range(1,epochs+1):
        encoder.train();classifier.train();losses=[]
        for x,labels in loader:
            x=x.to("cuda:0",non_blocking=True)
            labels=labels.to("cuda:0",non_blocking=True)
            z=classifier(encoder(x).flatten(1))
            loss=crit(z,labels)
            must(bool(torch.isfinite(loss).item()),"supervised nonfinite")
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            losses.append(float(loss.item()))
        rec={"epoch":epoch,"loss":float(np.mean(losses))}
        if stage=="B":
            encoder.eval();classifier.eval();p=[];gold=[]
            with torch.inference_mode():
                for x,labels in valid:
                    z=classifier(encoder(x.to("cuda:0",non_blocking=True)).flatten(1))
                    p.append(z.argmax(1).cpu().numpy());gold.append(labels.numpy())
            pred=np.concatenate(p);truth=np.concatenate(gold)
            rec["val_accuracy"]=float((pred==truth).mean())
            rec["val_macro_f1"]=float(f1_score(truth,pred,
                                                 labels=np.arange(65),average="macro"))
            if rec["val_macro_f1"]>best:
                best=rec["val_macro_f1"];selected=epoch
                torch.save({"encoder":encoder.state_dict(),
                            "classifier":classifier.state_dict(),
                            "config":CONFIG,"epoch":epoch},dest/"best.pt")
        hist.append(rec)
        print("S11_SWALLOW_SUPERVISED_EPOCH",stage,json.dumps(rec),flush=True)
    if stage=="C":
        selected=epochs
        torch.save({"encoder":encoder.cpu().state_dict(),
                    "classifier":classifier.cpu().state_dict(),
                    "config":CONFIG,"epoch":epochs},dest/"model.pt")
    model=dest/("best.pt" if stage=="B" else "model.pt")
    m={"classification":"S11_SWALLOW_SUPERVISED_"+stage+"_V1",
       "config":CONFIG,"training_n":len(tr),"selected_epoch":selected,
       "checkpoint":str(model),"checkpoint_sha256":sha(model),
       "pretrain_manifest_sha256":sha(pre),"history":hist,
       "external_data_opened":False,"dataset_c_opened":False}
    (dest/"manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print("S11_SWALLOW_SUPERVISED_PASS",stage,sha(model),flush=True)

def evaluate(cohort):
    must(cohort in COHORTS,"unsupported cohort")
    dest=OUT/"evaluation"/cohort
    must(not dest.exists(),"existing evaluation output")
    mpath=OUT/"supervised_C/manifest.json"
    must(mpath.is_file(),"C checkpoint missing")
    m=json.loads(mpath.read_text())
    must(m["classification"]=="S11_SWALLOW_SUPERVISED_C_V1" and
         m["config"]==CONFIG,"C frozen provenance mismatch")
    ck=Path(m["checkpoint"])
    must(sha(ck)==m["checkpoint_sha256"],"C model SHA mismatch")
    frozen=json.loads(S5.read_text())
    ref=frozen["label_alignment"][cohort]
    must(ref["identity_order"] is True,"class order differs")
    p=Path(ref["source"])
    must(sha(p)==ref["source_sha256"],"external cohort label index digest")
    with np.load(p,allow_pickle=True) as z:
        uid=z["pcap_uid"].astype(str);y=z["y_true"].astype(int)
        names=z["candidate_labels"].astype(str)
        dates=z["query_date"].astype("datetime64[D]").astype(str)
    n=COHORTS[cohort]
    must(len(uid)==n and len(set(uid))==n and len(names)==65,"cohort contract")
    lookup={str(k):str(names[v]) for k,v in zip(uid,y)}
    src=(Path.home()/"ltd-storage/datasets/historical/CLEAN_final_vectors_sites.csv"
         if cohort=="09B" else
         Path.home()/"ltd-storage/datasets/future/CLEAN_final_vectors_sites_concept_drift.csv")
    if cohort=="09B":
        m1=json.loads(S1.read_text())
        expected=m1["historical_checksums"]["historical_vectors"]["sha256"]
    else:
        fm=json.loads((ROOT/"docs/evidence/FINAL/FUTURE-B/10B_manifest_future_b.json").read_text())
        expected=fm["future_inputs"]["micro"]["sha256"]
    must(sha(src)==expected,"external vectors changed")
    fun,_,_,_=origin()
    matrix={}
    csv.field_size_limit(sys.maxsize)
    with src.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        for row in reader:
            k=row["pcap_uid"]
            if k not in lookup:continue
            must(k not in matrix and row["site_label"]==lookup[k],
                 "UID/label mismatch")
            x,load,duration=cif(parse_direction(row["direction_vector"]),
                                 parse_times(row["time_vector"]),fun)
            matrix[k]=x
    must(len(matrix)==n,"external UID coverage")
    full,_,_=network()
    encoder=nn.Sequential(*list(full.children())[:-1])
    classifier=nn.Linear(512,65)
    checkpoint=torch.load(ck,map_location="cpu",weights_only=True)
    must(checkpoint["config"]==CONFIG and
         checkpoint["epoch"]==m["selected_epoch"],"C checkpoint metadata")
    encoder.load_state_dict(checkpoint["encoder"],strict=True)
    classifier.load_state_dict(checkpoint["classifier"],strict=True)
    encoder.eval();classifier.eval()
    probs=np.empty((n,65),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,n,32):
            end=min(start+32,n)
            x=np.stack([matrix[k] for k in uid[start:end]]).astype(np.float32)
            logits=classifier(encoder(torch.from_numpy(x[:,None,:,:])).flatten(1))
            probs[start:end]=torch.softmax(logits,dim=1).numpy()
    must(np.isfinite(probs).all() and
         np.allclose(probs.sum(1),1.,atol=2e-4,rtol=0),"probabilities invalid")
    order=np.argsort(-probs,axis=1);top=order[:,0]
    pos=(order==y[:,None]).argmax(1)+1
    tp=np.bincount(y[top==y],minlength=65)
    sp=np.bincount(y,minlength=65)
    pp=np.bincount(top,minlength=65)
    macro=float(np.divide(2*tp,sp+pp,out=np.zeros(65),where=sp+pp>0).mean())
    metrics={"accuracy":float((top==y).mean()),"macro_f1":macro,
             "top5":float((pos<=5).mean()),"mrr":float((1/pos).mean())}
    dest.mkdir(parents=True,mode=0o700)
    score=dest/"scores_sota_aligned.npz"
    np.savez_compressed(score,pcap_uid=uid,query_date=dates,
                        y_true=y,candidate_labels=names,sota_probs=probs)
    info={"classification":"S11_SWALLOW_EXTERNAL_FROZEN_V1",
          "cohort":cohort,"n":n,"metrics":metrics,
          "checkpoint_sha256":m["checkpoint_sha256"],
          "score_sha256":sha(score),"no_external_tuning":True,
          "dataset_c_opened":False}
    (dest/"manifest.json").write_text(json.dumps(info,indent=2)+"\n")
    print("S11_SWALLOW_EXTERNAL_PASS",cohort,json.dumps(metrics),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["smoke","prepare","pretrain-B",
                                       "supervised-B","C","evaluate"],required=True)
    p.add_argument("--cohort",choices=list(COHORTS))
    args=p.parse_args()
    if args.stage=="smoke":
        seed();fun,Resnet,MLP,Aug=origin()
        net,_,_=network();net.cpu().eval()
        x,t,d=cif([1,-1,1,-1],[0.,0.1,0.2,0.3],fun)
        with torch.inference_mode():
            z=net(torch.tensor(x,dtype=torch.float32)[None,None])
        must(z.shape==(1,128),"original Swallow ResNet output shape")
        print("S11_SWALLOW_AUTHORS_SOURCE_CPU_SMOKE_PASS",COMMIT,flush=True)
    elif args.stage=="prepare":
        cache();print("S11_SWALLOW_CIF_CPU_PREPARE_PASS",flush=True)
    elif args.stage=="pretrain-B":pretrain()
    elif args.stage=="supervised-B":supervised("B")
    elif args.stage=="C":supervised("C")
    else:
        must(args.cohort is not None,"cohort required")
        evaluate(args.cohort)
if __name__=="__main__":main()
