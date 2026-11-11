#!/usr/bin/env python3
"""S9-O NetCLR: authors' original model/NetAugment with Historical DEV adaptation.

Author source is loaded from pinned SPIN-UMass notebook AST at runtime.
No WFlib model or augmentation is used. Only explicit source fixes:
  (1) modern torch.nn.init.xavier_uniform_
  (2) outgoing burst CDF maps observed positive sizes to probability bins
LTD-specific batch/epoch values are declared; not paper-identical.
--stage smoke is CPU-only, --stage B trains contrastively on DEV_EARLY/MIDDLE.
"""
from __future__ import annotations
import argparse,ast,bisect,hashlib,importlib.util,json,random,sys,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
GATE=ROOT/"scripts/sota/s9_netclr_authors_cpu_gate_v1.py"
BPATH=ROOT/"scripts/sota/s9_netclr_supervised_b_v1.py"

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
G=load(GATE,"s9_authors_source_gate")
DATA=load(BPATH,"s9_historical_cache")
OUT=Path("/data/ltd-attack/sota/s9_netclr_author_original_v1")
TARGET=OUT/"pretrain_B_v1"
CONFIG={"classification":"S9_NETCLR_AUTHORS_MODEL_AUGMENT_HISTORICAL_B",
        "source_commit":G.COMMIT,"train":["DEV_EARLY","DEV_MIDDLE"],
        "training_n":39258,"seed":42,"input":"DIR_5000",
        "batch_size":64,"contrastive_views":2,"epochs":100,
        "optimizer":"Adam","learning_rate":0.0003,"temperature":0.5,
        "projection_dim":128,"original_notebook_batch_size":256,
        "original_notebook_actual_epoch_parameter":401,
        "deviations":["historical-only input instead of AWF-PT-sup",
                      "lower batch (64) for available 16GB GPU",
                      "100 epochs instead of notebook trainer parameter 401",
                      "corrected outgoing burst CDF bin index to observed positive size",
                      "deprecated Xavier API spelling"],
        "paper_exact":False,"Dataset_C_used":False,
        "09B_or_Future_B_used_for_fit":False}
def must(ok,why):
    if not ok:raise RuntimeError("S9_AUTHORS_B_BLOCKED "+why)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def original():
    evidence,model_src,aug_src=G.inspect()
    ns={"np":np,"torch":torch,"nn":nn,"F":F,"random":random,"bisect":bisect}
    tree=ast.parse(model_src)
    must(all(isinstance(x,ast.ClassDef) for x in tree.body),
         "unexpected original model statements")
    exec(compile(tree,"netclr_authors_original_model","exec"),ns)
    t=ast.parse(aug_src)
    must(len(t.body)==1 and isinstance(t.body[0],ast.ClassDef),
         "unexpected authors augmentation statements")
    exec(compile(t,"netclr_authors_original_augment","exec"),ns)
    return ns,evidence

def distribution(X,indices):
    # Notebook samples 1000 training traces to estimate outgoing burst sizes.
    # Its original helper omits the last burst; preserve that exact behavior.
    rng=np.random.RandomState(42)
    chosen=rng.choice(indices,size=1000,replace=False)
    collected=[]
    for i in chosen:
        x=np.asarray(X[int(i)],dtype=np.int32)
        direction=x[0];start=0;burst=int(x[0])
        for j in range(1,len(x)):
            if x[j]==0:break
            if x[j]==direction:burst+=int(x[j])
            else:
                if burst>0:collected.append(burst)
                start=j;burst=int(x[j]);direction*=-1
    must(bool(collected),"empty author burst distribution")
    counts=Counter(collected)
    sizes=np.asarray(sorted(counts),dtype=np.int32)
    weights=np.asarray([counts[int(q)] for q in sizes],dtype=np.float64)
    cdf=np.cumsum(weights)/weights.sum()
    must(len(cdf)>0 and np.isclose(cdf[-1],1.) and np.all(sizes>0),
         "burst probability normalization")
    return sizes,cdf,hashlib.sha256(chosen.astype(np.int64).tobytes()).hexdigest()

class Views(Dataset):
    def __init__(self,X,ids,augment):self.X=X;self.ids=ids;self.augment=augment
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        x=np.array(self.X[int(self.ids[i])],dtype=np.int32,copy=True)
        # Original notebook's TrainData calls augment(x) twice, not _aug().
        z=[]
        for _ in range(2):
            view=np.asarray(self.augment.augment(x.copy()),dtype=np.float32)
            must(view.shape==(5000,) and np.isin(view,(-1,0,1)).all(),
                 "original NetAugment output invalid")
            z.append(torch.from_numpy(view[None,:]))
        return z

def make_augment(ns,X,ids):
    sizes,cdf,sample_sha=distribution(X,ids)
    ns["max_outgoing_burst_size"]=int(max(sizes))
    ns["outgoing_burst_sizes"]=sizes.tolist()
    ns["OUTGOING_BURST_SIZE_CDF"]=cdf
    augment=ns["Augmentor"]()
    # The original notebook couples its sampled CDF to range(max_size);
    # these indices do not in general map to actual positive burst sizes.
    # Explicitly correct only that pairing without modifying the upstream code.
    augment.outgoing_burst_sizes=sizes.tolist()
    must(len(augment.outgoing_burst_sizes)==len(cdf),
         "cdf must be aligned to actually observed burst sizes")
    return augment,{"samples":1000,"sample_indices_sha256":sample_sha,
                    "positive_burst_size_count":len(sizes),
                    "cdf_last":float(cdf[-1]),
                    "correction":"CDF indexed by observed positive sizes (explicit deviation)"}

def nce(embedding):
    must(embedding.ndim==2 and embedding.shape[0]%2==0,"contrastive shape")
    n=embedding.shape[0]//2
    z=F.normalize(embedding,dim=1)
    logits=z@z.T/0.5
    logits.fill_diagonal_(-1e4)
    targets=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return F.cross_entropy(logits,targets)

def gpu_guard():
    import subprocess
    p=subprocess.run(["nvidia-smi","--query-compute-apps=pid",
                      "--format=csv,noheader,nounits"],
                     capture_output=True,text=True,timeout=20)
    must(p.returncode==0 and not p.stdout.strip(),
         "other compute process occupying the GPU: "+p.stdout.strip())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("smoke","B"),required=True)
    a=parser.parse_args()
    seed()
    ns,evidence=original()
    X,y,s,cache=DATA.prepare_cache()
    ids=np.flatnonzero(s!=2)
    must(len(ids)==39258 and int((s==2).sum())==18658,
         "Historical DEV-only split")
    augment,dist=make_augment(ns,X,ids)
    dataset=Views(X,ids,augment)
    print("S9_AUTHOR_SOURCE_AND_DISTRIBUTION_PASS",
          json.dumps({"commit":G.COMMIT,"data_n":len(ids),"dist":dist}),flush=True)
    samples=next(iter(DataLoader(dataset,batch_size=4,num_workers=0)))
    q=torch.cat(samples,dim=0)
    must(q.shape==(8,1,5000),"2-view smoke batch")
    model=ns["DFsimCLR"](ns["DFNet"](out_dim=512),out_dim=128).cpu()
    if a.stage=="smoke":
        # Tiny CPU contrastive forward/backward using EXACT AUTHORS' architecture
        model.train()
        z=model(q)
        must(z.shape==(8,128),"embedding dimension mismatch")
        loss=nce(z)
        must(bool(torch.isfinite(loss).item()),"nonfinite loss")
        loss.backward()
        print("S9_AUTHOR_B_CPU_SMOKE_PASS",
              json.dumps({"embedding":list(z.shape),"loss":float(loss.item()),
                          "source":"SPIN-UMass authors notebook",
                          "training_run":False}),flush=True)
        return
    must(not TARGET.exists(),"existing B run, no overwrite")
    must(torch.cuda.is_available(),"CUDA unavailable")
    gpu_guard()
    seed()
    # Reconstruct from clean seed so smoke does not alter train initialization.
    model=ns["DFsimCLR"](ns["DFNet"](out_dim=512),out_dim=128).to("cuda:0")
    opt=torch.optim.Adam(model.parameters(),lr=0.0003)
    loader=DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True,
                      num_workers=0,pin_memory=True)
    TARGET.mkdir(parents=True,mode=0o700)
    history=[]
    started=time.monotonic()
    print("S9_AUTHOR_B_BEGIN",len(ids),"epochs",100,flush=True)
    for epoch in range(1,101):
        model.train()
        loss_total=0.
        for step,views in enumerate(loader,1):
            batch=torch.cat(views,dim=0).to("cuda:0",non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss=nce(model(batch))
            must(bool(torch.isfinite(loss).item()),"nonfinite contrastive loss")
            loss.backward();opt.step()
            loss_total+=float(loss.item())
            if step%100==0:
                print("S9_AUTHOR_B_PROGRESS",
                      json.dumps({"epoch":epoch,"step":step,"of":len(loader),
                                  "elapsed_s":round(time.monotonic()-started,1)}),
                      flush=True)
        record={"epoch":epoch,"contrastive_loss":loss_total/len(loader)}
        history.append(record)
        print("S9_AUTHOR_B_EPOCH",json.dumps(record),flush=True)
        if epoch%10==0 or epoch==100:
            ck=TARGET/f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save({"model":model.cpu().state_dict(),
                        "epoch":epoch,"config":CONFIG},ck)
            model.to("cuda:0")
    ck=TARGET/"checkpoint_epoch_100.pt"
    manifest={"classification":CONFIG["classification"],"config":CONFIG,
              "checkpoint":str(ck),"checkpoint_sha256":sha(ck),
              "authors_pretrain_notebook_sha256":evidence["pretrain_notebook_sha256"],
              "cache_manifest_sha256":sha(DATA.CACHE/"manifest.json"),
              "burst_distribution":dist,"training_n":39258,
              "history":history,"elapsed_s":time.monotonic()-started,
              "external_data_opened":False,"dataset_c_opened":False,
              "created_utc":datetime.now(timezone.utc).isoformat()}
    (TARGET/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("S9_AUTHOR_B_PASS",sha(ck),flush=True)
if __name__=="__main__":
    try:main()
    except Exception as e:
        print("S9_AUTHOR_B_ERROR",type(e).__name__,str(e)[:1000],
              file=sys.stderr,flush=True)
        raise
