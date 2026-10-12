#!/usr/bin/env python3
"""P15B: fixed, Historical-only frozen-score fusion ablation; no training."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from report_p15a_capture_mechanisms import BASE,CACHE,RESULT,EPS,ORIGINS,WINDOWS,geometric,ranks,read_npz,verify_manifest,sha

ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/"configs/experiments/P15B_FROZEN_FUSION_ABLATION_V1.yaml"
OUT=RESULT/"diagnostics/P15B-FROZEN-FUSION-V1"
EXPERTS=[("UNIFORM",s) for s in (11,42,73)]

def normalize(a):
    a=np.clip(np.asarray(a,dtype=np.float64),EPS,None)
    return a/a.sum(axis=1,keepdims=True)

def rrf(p,m):
    # k=60 fixed, 65 classes; stable sort/ties, no labels used
    n,c=p.shape
    pr=np.empty((n,c),dtype=np.int64);mr=np.empty((n,c),dtype=np.int64)
    rows=np.arange(n)[:,None]
    pr[rows,np.argsort(-p,axis=1,kind="stable")]=np.arange(1,c+1)
    mr[rows,np.argsort(-m,axis=1,kind="stable")]=np.arange(1,c+1)
    return normalize(.45/(60+pr)+.55/(60+mr))

def metrics(scores,y):
    rank=ranks(scores,y)
    return dict(accuracy=float(np.mean(rank==1)),
       macro_f1=float(f1_score(y,np.argmax(scores,axis=1),labels=np.arange(65),average="macro",zero_division=0)),
       top5_accuracy=float(np.mean(rank<=5)),mrr=float(np.mean(1/rank)))

def day_bootstrap(y,dates,base,other,seed,n=2000):
    # paired day-block resampling; evaluate metrics on concatenated indices, no fitted quantities
    days=np.unique(dates)
    if len(days)<2:raise RuntimeError("NOT_ENOUGH_DAYS")
    idx=[np.flatnonzero(dates==d) for d in days]
    rng=np.random.default_rng(seed)
    values=np.empty((n,2))
    for i in range(n):
        sample=np.concatenate([idx[j] for j in rng.integers(len(idx),size=len(idx))])
        z=metrics(base[sample],y[sample]);q=metrics(other[sample],y[sample])
        values[i]=[q["macro_f1"]-z["macro_f1"],q["accuracy"]-z["accuracy"]]
    return dict(f1_ci_low=float(np.quantile(values[:,0],.025)),
      f1_ci_high=float(np.quantile(values[:,0],.975)),
      f1_fraction_positive=float(np.mean(values[:,0]>0)),
      accuracy_ci_low=float(np.quantile(values[:,1],.025)),
      accuracy_ci_high=float(np.quantile(values[:,1],.975)))

def main():
    cfg=yaml.safe_load(CFG.read_text())
    if not cfg["historical_only"] or cfg["training"]!="forbidden" or cfg["future_b"]!="forbidden" or cfg["dataset_c"]!="forbidden":
        raise RuntimeError("GUARDRAIL")
    if cfg["variants"]!=["B0_geometric","B1_arithmetic","B2_rank_rrf_k60","Bneg_temp2"]:
        raise RuntimeError("UNDECLARED_VARIANTS")
    verify_manifest("13B",RESULT/"micro/MICRO-ROBUSTNESS-PHASE13-BATTERY")
    verify_manifest("13C",RESULT/"hybrid/LTD-ROBUSTNESS-PHASE13-BATTERY")
    ref=pd.read_csv(RESULT/"hybrid/LTD-ROBUSTNESS-PHASE13-BATTERY/13C_factorial_summary.csv").set_index(["origin","window","micro_candidate","macro_candidate"])
    rows=[]; ci=[]; hashes={}
    for oi,origin in enumerate(ORIGINS):
        macro_file=CACHE/"macro"/(origin+".npz");hashes[str(macro_file)]=sha(macro_file)
        with np.load(macro_file,allow_pickle=False) as z:
            macro={k:z[k].copy() for k in z.files}
        for wi,window in enumerate(WINDOWS):
            ds=[];identity=None
            for mode,seed in EXPERTS:
                path=CACHE/origin/mode/str(seed)/(window+".npz")
                hashes[str(path)]=sha(path)
                z=read_npz(path,["pcap_uid","query_date","y_true","candidate_labels","probs","origin","window","seed","temporal_mode"])
                if (str(z["origin"]),str(z["window"]),int(z["seed"]),str(z["temporal_mode"]))!=(origin,window,seed,mode):
                    raise RuntimeError("IDENTITY_ERROR")
                id_now=(z["pcap_uid"],z["query_date"],z["y_true"],z["candidate_labels"])
                if identity is None:identity=id_now
                elif not all(np.array_equal(x,t) for x,t in zip(identity,id_now)):raise RuntimeError("UNALIGNED_MICRO")
                ds.append(z["probs"])
            uid,dates,y,labels=identity
            if len(labels)!=65 or len(set(uid.tolist()))!=len(uid):raise RuntimeError("INVALID_COHORT")
            for field,expect in [(f"pcap_uid_{window}",uid),(f"dates_{window}",dates),(f"y_{window}",y),("candidate_labels",labels)]:
                if field not in macro or not np.array_equal(macro[field],expect):raise RuntimeError("UNALIGNED_MACRO "+field)
            p=geometric(ds); m=macro[f"M0_{window}"]
            baseline=normalize(np.exp(.45*np.log(np.clip(p,EPS,1.))+.55*np.log(np.clip(m,EPS,1.))))
            variants={
                "B0_geometric":baseline,
                "B1_arithmetic":normalize(.45*p+.55*m),
                "B2_rank_rrf_k60":rrf(p,m),
                "Bneg_temp2":normalize(np.sqrt(baseline))
            }
            bmetrics=metrics(baseline,y)
            source=ref.loc[(origin,window,"U3","M0")]
            for field in ("accuracy","macro_f1","top5_accuracy","mrr"):
                if abs(bmetrics[field]-float(source[field]))>1e-5:
                    raise RuntimeError("BASELINE_NOT_REPRODUCED "+origin+window+field)
            if not np.array_equal(np.argsort(-baseline,axis=1,kind="stable"),np.argsort(-variants["Bneg_temp2"],axis=1,kind="stable")):
                raise RuntimeError("NEGATIVE_CONTROL_CHANGED_RANK")
            for vi,(name,score) in enumerate(variants.items()):
                v=metrics(score,y)
                rows.append({"origin":origin,"window":window,"variant":name,"n":len(y),
                    **v,**{"delta_"+k:v[k]-bmetrics[k] for k in bmetrics}})
                if name!="B0_geometric":
                    bs=day_bootstrap(y,dates,baseline,score,20261012+100*oi+10*wi+vi)
                    ci.append({"origin":origin,"window":window,"variant":name,**bs})
                print("VALIDATED",origin,window,name,"accuracy",f"{v['accuracy']:.6f}",
                      "f1",f"{v['macro_f1']:.6f}","top5",f"{v['top5_accuracy']:.6f}",
                      "delta_f1",f"{v['macro_f1']-bmetrics['macro_f1']:+.6f}")
    summary=pd.DataFrame(rows); boot=pd.DataFrame(ci)
    gate={}
    for variant in ("B1_arithmetic","B2_rank_rrf_k60"):
        far=summary[(summary.window=="FAR")&(summary.variant==variant)]
        farboot=boot[(boot.window=="FAR")&(boot.variant==variant)]
        passed=(len(far)==2 and len(farboot)==2 and
          far.delta_macro_f1.mean()>=.005 and
          (far.delta_macro_f1>=-.0025).all() and
          (far.delta_accuracy>0).all() and
          (far.delta_top5_accuracy>=-.005).all() and
          (farboot.f1_fraction_positive>.90).all())
        gate[variant]={"passed":bool(passed),"far_mean_delta_macro_f1":float(far.delta_macro_f1.mean())}
    # Gate has a predeclared criterion, but comparing 2 candidates on the same folds is exploratory.
    if OUT.exists():raise RuntimeError("OUTPUT_DIRECTORY_EXISTS: refusing overwrite "+str(OUT))
    OUT.mkdir(parents=True)
    artifacts={"P15B_summary.csv":summary,"P15B_bootstrap.csv":boot}
    oh={}
    for name,frame in artifacts.items():
        path=OUT/name;frame.to_csv(path,index=False);oh[name]=sha(path)
    manifest={"stage":cfg["stage"],"status":"EXPLORATORY_HISTORICAL_ONLY",
        "no_training":True,"future_b_opened":False,"dataset_c_opened":False,
        "config_sha256":sha(CFG),"script_sha256":sha(Path(__file__)),
        "inputs_sha256":hashes,"outputs_sha256":oh,"gate":gate}
    (OUT/"P15B_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True))
    print("P15B_GATE",json.dumps(gate,sort_keys=True))
    print("P15B_COMPLETE",len(summary),"rows",OUT)
if __name__=="__main__":main()
