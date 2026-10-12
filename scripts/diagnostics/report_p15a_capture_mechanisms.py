#!/usr/bin/env python3
"""15A Historical-only capture diagnostic from existing frozen 13B/13C caches.
Never imports Future-B, opens Dataset-C, trains, calibrates, or writes to frozen artifact folders.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path.home()/"ltd-storage"
CACHE=BASE/"artifacts/experiments/PHASE13_BATTERY_V1"
RESULT=BASE/"results"
EPS=1e-12
EXPERTS=[("UNIFORM",11),("UNIFORM",42),("UNIFORM",73),
         ("EARLY",42),("LATE",42)]
WINDOWS=("NEAR","MID","FAR")
ORIGINS=("ORIGIN14","ORIGIN28")

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()

def verify_manifest(stage,directory):
    m=json.loads((directory/(stage+"_manifest.json")).read_text())
    assert m["stage"]==stage
    for name,expected in m["outputs_sha256"].items():
        f=directory/name
        if not f.is_file() or sha(f)!=expected:
            raise RuntimeError(f"CORRUPT_RESULTS: {f}")

def read_npz(path,fields):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in fields}

def geometric(values):
    p=np.mean([np.log(np.clip(np.asarray(v,dtype=np.float64),EPS,1.0)) for v in values],axis=0)
    p-=p.max(axis=1,keepdims=True)
    a=np.exp(p)
    return a/a.sum(axis=1,keepdims=True)

def ranks(p,y):
    return np.argmax(np.argsort(-p,axis=1,kind="stable")==y[:,None],axis=1)+1

def components(p,y):
    rank=ranks(p,y)
    confidence=np.sort(p,axis=1)
    top1=confidence[:,-1]
    margin=top1-confidence[:,-2]
    entropy=-np.sum(np.clip(p,EPS,1)*np.log(np.clip(p,EPS,1)),axis=1)
    return rank,top1,margin,entropy

def diagnostics(name,p,y,summary_row):
    rank,conf,margin,ent=components(p,y)
    vals={"TOP1_CORRECT":rank==1,"TOP5_RECOVERABLE":(rank>=2)&(rank<=5),"TOP5_MISSED":rank>5}
    acc=np.mean(rank==1)
    top5=np.mean(rank<=5)
    if abs(acc-float(summary_row["accuracy"]))>1e-5 or abs(top5-float(summary_row["top5_accuracy"]))>1e-5:
        raise RuntimeError(f"RECONSTRUCTION_MISMATCH {name}: acc={acc}, top5={top5}")
    print("VALIDATED",name,"N",len(y),"ACC",f"{acc:.6f}","TOP5",f"{top5:.6f}")
    for cat,mask in vals.items():
        if mask.sum():
            print("  GROUP",cat,"count",mask.sum(),"fraction",f"{mask.mean():.6f}",
                  "mean_conf",f"{conf[mask].mean():.6f}",
                  "mean_margin",f"{margin[mask].mean():.6f}",
                  "mean_entropy",f"{ent[mask].mean():.6f}")
    return rank,p.argmax(axis=1)

def main():
    result_b=RESULT/"micro/MICRO-ROBUSTNESS-PHASE13-BATTERY"
    result_c=RESULT/"hybrid/LTD-ROBUSTNESS-PHASE13-BATTERY"
    verify_manifest("13B",result_b)
    verify_manifest("13C",result_c)
    b=pd.read_csv(result_b/"13B_model_summary.csv").set_index(["origin","window","candidate"])
    c=pd.read_csv(result_c/"13C_factorial_summary.csv").set_index(["origin","window","micro_candidate","macro_candidate"])
    print("HASH_GUARD 13B/13C VERIFIED; HISTORICAL ONLY")
    for origin in ORIGINS:
        macro_file=CACHE/"macro"/(origin+".npz")
        # Ignore macro_failed_* and macro_pre_* copies; canonical path only.
        for window in WINDOWS:
            raw={}
            ids=dates=y=labels=None
            for mode,seed in EXPERTS:
                path=CACHE/origin/mode/str(seed)/(window+".npz")
                d=read_npz(path,["pcap_uid","query_date","y_true","candidate_labels","probs","origin","window","seed","temporal_mode"])
                if str(d["origin"])!=origin or str(d["window"])!=window or int(d["seed"])!=seed or str(d["temporal_mode"])!=mode:
                    raise RuntimeError(f"EXPERT_IDENTITY_MISMATCH: {path}")
                if ids is None:
                    ids,dates,y,labels=(d[k] for k in ["pcap_uid","query_date","y_true","candidate_labels"])
                    if len(np.unique(ids))!=len(ids):raise RuntimeError("NONUNIQUE_UID")
                    if len(labels)!=65 or d["probs"].shape!=(len(y),65):raise RuntimeError("BAD_65_COHORT")
                elif any(not np.array_equal(d[k],v) for k,v in
                         [("pcap_uid",ids),("query_date",dates),("y_true",y),("candidate_labels",labels)]):
                    raise RuntimeError(f"MICRO_CAPTURE_MISALIGNMENT: {path}")
                raw[(mode,seed)]=d["probs"]
            with np.load(macro_file,allow_pickle=False) as mz:
                for key,ref in [(f"pcap_uid_{window}",ids),(f"dates_{window}",dates),
                                (f"y_{window}",y),("candidate_labels",labels)]:
                    if not np.array_equal(mz[key],ref):
                        raise RuntimeError(f"MACRO_ALIGNMENT_FAILED {origin} {window} {key}")
                macro=mz[f"M0_{window}"].copy()
            micro={
                "U1":raw[("UNIFORM",42)],
                "U3":geometric([raw[("UNIFORM",s)] for s in (11,42,73)]),
                "T3_S42":geometric([raw[(m,42)] for m in ("EARLY","UNIFORM","LATE")])
            }
            for name,p in micro.items():
                mr,_=diagnostics(f"{origin}/{window}/MICRO/{name}",p,y,b.loc[(origin,window,name)])
                h=np.exp(.45*np.log(np.clip(p,EPS,1.))+ .55*np.log(np.clip(macro,EPS,1.)))
                h/=h.sum(axis=1,keepdims=True)
                hr,hpred=diagnostics(f"{origin}/{window}/HYBRID/{name}xM0",h,y,c.loc[(origin,window,name,"M0")])
                rescued=np.sum((mr!=1)&(hr==1))
                harmed=np.sum((mr==1)&(hr!=1))
                print("  PAIRED_MICRO_TO_HYBRID","rescue",int(rescued),"new_errors",int(harmed),"net",int(rescued-harmed),
                      "micro_vs_hybrid_top1_disagreement",f"{np.mean(p.argmax(axis=1)!=hpred):.6f}")
    print("P15A_CAPTURE_ANALYSIS_OK: no fitting, no Future-B access, no writes")
if __name__=="__main__":
    main()
