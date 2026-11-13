#!/usr/bin/env python3
"""Swallow historical-only controlled BYOL root-cause intervention.

Two *new* small-scale experiments, never resume or alter collapsed Swallow:
A: authors' default target in train() (BN batch statistics),
B: prior LTD adaptation target.eval() (BN running statistics).
Same original source model, augmented views, Historical DEV Early+Middle,
seed, steps, effective optimizer, cache. Fixed 24 independent historical
probe traces. Read-only existing files. No 09B/Future-B, no model selection.
This is a causal pilot, NOT an adequate full Swallow reproduction.
"""
from __future__ import annotations
import argparse,hashlib,json,os,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

def must(v,msg):
    if not v:raise RuntimeError("SWALLOW_BYOL_PILOT_BLOCKED "+msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def statistics(net,x):
    net.eval()
    with torch.inference_mode():
        emb=net(x).float().cpu().numpy().reshape(x.shape[0],-1)
    std=emb.std(axis=0)
    return {"mean_inter_capture_feature_std":float(std.mean()),
            "max_feature_std":float(std.max()),
            "unique_embeddings_rounded_6":int(len(np.unique(emb.round(6),axis=0))),
            "embedding_rms":float(np.sqrt(np.mean(emb.astype(np.float64)**2)))}
def arm(name,target_training,S,cache,subset,probe,steps,batch,report_dir):
    S.seed()
    torch.set_num_threads(2)
    online,MLP,Aug=S.network()
    target,_,_=S.network()
    target.load_state_dict(online.state_dict(),strict=True)
    for par in target.parameters():par.requires_grad_(False)
    predictor=MLP(in_channels=128,mlp_hidden_size=512,projection_size=128)
    online=online.cuda();target=target.cuda();predictor=predictor.cuda()
    opt=torch.optim.SGD(list(online.parameters())+list(predictor.parameters()),
        lr=.03,momentum=.9,weight_decay=.0004)
    X,y,s,load,duration,_=cache
    ds=S.Data(X,y,subset,load,duration,Aug())
    generator=torch.Generator().manual_seed(42)
    loader=DataLoader(ds,batch_size=batch,shuffle=True,drop_last=True,
        num_workers=0,pin_memory=True,generator=generator)
    it=iter(loader);losses=[];probes=[]
    initial=statistics(online,probe)
    for step in range(1,steps+1):
        try: v1,v2=next(it)
        except StopIteration:
            it=iter(loader);v1,v2=next(it)
        online.train();predictor.train()
        target.train(target_training)
        v1=v1.cuda(non_blocking=True);v2=v2.cuda(non_blocking=True)
        q1=predictor(online(v1));q2=predictor(online(v2))
        with torch.no_grad():
            t1=target(v1);t2=target(v2)
        loss=S.nce_byol(q1,t2)+S.nce_byol(q2,t1)
        must(bool(torch.isfinite(loss).item()),name+" BYOL nonfinite")
        opt.zero_grad(set_to_none=True);loss.backward()
        gradmax=max((p.grad.detach().abs().max().item()
             for p in online.parameters() if p.grad is not None),default=0.)
        must(np.isfinite(gradmax),name+" gradients nonfinite")
        opt.step()
        with torch.no_grad():
            for p,q in zip(online.parameters(),target.parameters()):
                q.mul_(.996).add_(p,alpha=.004)
        losses.append(float(loss.detach().item()))
        if step in (1,10,25,50,100,steps):
            s=statistics(online,probe)
            s["step"]=step;s["loss"]=losses[-1];s["max_online_gradient"]=gradmax
            probes.append(s)
            print("SWALLOW_PILOT",name,"STEP",step,
                "LOSS",round(losses[-1],4),
                "EMBED_STD",s["mean_inter_capture_feature_std"],
                "UNIQUE",s["unique_embeddings_rounded_6"],flush=True)
        del v1,v2,q1,q2,t1,t2,loss
    return {"mode":name,"target_training_mode":target_training,
      "steps":steps,"batch_size":batch,"initial_probe":initial,
      "checkpoints_in_memory_only":True,"probes":probes,
      "first_loss":losses[0],"final_loss":losses[-1],
      "mean_loss":float(np.mean(losses)),
      "target_mode_difference_is_the_only_controlled_factor":True}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--steps",type=int,default=120)
    p.add_argument("--batch",type=int,default=32)
    a=p.parse_args()
    must(25<=a.steps<=500,"pilot 25..500 steps")
    must(16<=a.batch<=64,"pilot batch 16..64")
    must(not a.output_dir.exists(),"refuse overwrite")
    must(torch.cuda.is_available(),"pilot requires free CUDA GPU")
    import subprocess
    r=subprocess.run(["nvidia-smi","--query-compute-apps=pid",
       "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15)
    must(r.returncode==0 and not r.stdout.strip(),"GPU occupied: "+r.stdout.strip())
    sys.path.insert(0,str(a.repo))
    from scripts.sota import s11_swallow_authors_v1 as S
    cache=S.cache()
    X,y,s,load,duration,m=cache
    must(len(X)==57916 and np.count_nonzero(s!=2)==39258,
         "frozen Historical DEV split differs")
    rng=np.random.default_rng(20261113)
    pool=np.flatnonzero(s!=2)
    subset=rng.choice(pool,size=min(2048,len(pool)),replace=False).astype(int)
    # Separate fixed pilot probe, never in the pilot optimizer batch pool.
    probe_ids=rng.choice(np.flatnonzero(s==2),size=24,replace=False)
    x=torch.from_numpy(np.asarray(X[probe_ids],dtype=np.float32).copy())[:,None].cuda()
    result={"classification":"LTD_SWALLOW_ORIGINAL_BYOL_TARGET_BN_CONTROLLED_PILOT_V1",
            "timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "source_v1_sha256":sha(a.repo/"scripts/sota/s11_swallow_authors_v1.py"),
            "original_frozen_cache_manifest_sha256":sha(S.CACHE/"manifest.json"),
            "data_used":"ONLY Historical DEV; optimizer EARLY+MIDDLE, probe LATE",
            "dataset_C_used":False,"09B_or_FutureB_used":False,
            "same_seed_per_arm":42,"training_subset_n":len(subset),
            "probe_count":len(probe_ids),"arms":{},"limitations":[
              "This 120-step pilot cannot establish that a 100-200 epoch run remains noncollapsed.",
              "It isolates target BN mode, not 6x data augmentation or different batch sizes.",
              "The probe is DEV_LATE and cannot be used for model training.",
              "An identical positive result for both arms does not rule out late-stage instability.",
              "A successful pilot must precede a separate full versioned historical-only training."]}
    a.output_dir.mkdir(parents=True)
    for name,mode in (("AUTHOR_TARGET_TRAIN",True),("ADAPTED_TARGET_EVAL",False)):
        result["arms"][name]=arm(name,mode,S,cache,subset,x,a.steps,
                                 a.batch,a.output_dir)
        (a.output_dir/"progress.json").write_text(json.dumps(result,indent=2)+"\n")
        torch.cuda.empty_cache()
    stats={k:v["probes"][-1]["mean_inter_capture_feature_std"]
           for k,v in result["arms"].items()}
    result["status"]="CONTROLLED_PILOT_COMPLETED_NOT_FULL_SWALLOW_CERTIFICATION"
    result["final_embedding_variation"]=stats
    (a.output_dir/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print("SWALLOW_BYOL_PILOT_PASS",result["status"])
    print("TARGET_MODE_COMPARISON",json.dumps(stats),flush=True)
    print("SWALLOW_BYOL_PILOT_REPORT",a.output_dir/"result.json")
if __name__=="__main__":main()
