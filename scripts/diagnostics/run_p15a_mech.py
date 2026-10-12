#!/usr/bin/env python3
"""P15A-MECH. Historical-only paired fusion mechanism diagnostics; no fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from report_p15a_capture_mechanisms import (
    BASE, CACHE, RESULT, EPS, EXPERTS, WINDOWS, ORIGINS,
    sha, verify_manifest, read_npz, geometric, ranks
)
HERE=Path(__file__).resolve().parents[2]
CONFIG=HERE/"configs/experiments/P15A_MECH_V1.yaml"
OUT=RESULT/"diagnostics/P15A-MECH"

def stats(probs):
    v=np.asarray(probs,dtype=np.float64)
    top=np.sort(v,axis=1)
    return {
        "confidence":top[:,-1],
        "margin":top[:,-1]-top[:,-2],
        "entropy":-np.sum(np.clip(v,EPS,1.0)*np.log(np.clip(v,EPS,1.0)),axis=1),
    }

def validate(p,y,row,name):
    r=ranks(p,y)
    if abs(np.mean(r==1)-float(row["accuracy"]))>1e-5 or abs(np.mean(r<=5)-float(row["top5_accuracy"]))>1e-5:
        raise RuntimeError("FROZEN_METRIC_MISMATCH "+name)
    return r

def summarize(frame,origin,window,candidate):
    counts=frame.outcome.value_counts()
    n=len(frame)
    a=int(counts.get("rescued_by_hybrid",0))
    b=int(counts.get("harmed_by_hybrid",0))
    result={"origin":origin,"window":window,"candidate":candidate,"n":n,
        "rescued":a,"harmed":b,"net":a-b,"delta_top1":(a-b)/n,
        "delta_top5":float(frame.h_top5.mean()-frame.m_top5.mean()),
        "micro_top1":float(frame.m_correct.mean()),"hybrid_top1":float(frame.h_correct.mean()),
        "micro_top5":float(frame.m_top5.mean()),"hybrid_top5":float(frame.h_top5.mean()),
        "disagreement_top1":float(frame.disagree.mean())}
    for outcome in ("both_correct","rescued_by_hybrid","harmed_by_hybrid","both_wrong"):
        mask=frame.outcome==outcome
        result[outcome+"_count"]=int(mask.sum())
        for col in ("m_confidence","h_confidence","m_margin","h_margin","m_entropy","h_entropy"):
            result[outcome+"_"+col]=float(frame.loc[mask,col].mean()) if mask.any() else float("nan")
    # Descriptive only: fixed condition defined without any fitted threshold.
    for prefix in ("m","h"):
        mask=(~frame[prefix+"_correct"]) & (frame[prefix+"_confidence"]>=.8)
        result[prefix+"_wrong_conf_ge_08_count"]=int(mask.sum())
        result[prefix+"_wrong_conf_ge_08_fraction"]=float(mask.mean())
    return result

def bootstrap(frame,iterations,seed):
    per_day=frame.groupby("date",sort=True).agg(
        n=("outcome","size"),
        rescued=("outcome",lambda x:int((x=="rescued_by_hybrid").sum())),
        harmed=("outcome",lambda x:int((x=="harmed_by_hybrid").sum())),
        m_top5=("m_top5","sum"),h_top5=("h_top5","sum"))
    if len(per_day)<2:
        raise RuntimeError("INSUFFICIENT_DISTINCT_DAYS")
    d=per_day.to_numpy(dtype=np.float64)
    rng=np.random.default_rng(seed)
    samples=np.empty((iterations,2))
    for k in range(iterations):
        sampled=d[rng.integers(0,len(d),size=len(d))].sum(axis=0)
        samples[k,0]=(sampled[1]-sampled[2])/sampled[0]
        samples[k,1]=(sampled[4]-sampled[3])/sampled[0]
    return {
        "days":len(d),
        "top1_ci_low":float(np.quantile(samples[:,0],.025)),
        "top1_ci_high":float(np.quantile(samples[:,0],.975)),
        "top1_fraction_positive":float(np.mean(samples[:,0]>0)),
        "top5_ci_low":float(np.quantile(samples[:,1],.025)),
        "top5_ci_high":float(np.quantile(samples[:,1],.975))
    },per_day

def main():
    import yaml
    conf=yaml.safe_load(CONFIG.read_text())
    if conf["guardrails"]["future_b_access"] or conf["guardrails"]["dataset_c_access"] or conf["guardrails"]["train"]:
        raise RuntimeError("PROTOCOL_VIOLATION")
    verify_manifest("13B",RESULT/"micro/MICRO-ROBUSTNESS-PHASE13-BATTERY")
    verify_manifest("13C",RESULT/"hybrid/LTD-ROBUSTNESS-PHASE13-BATTERY")
    B=pd.read_csv(RESULT/"micro/MICRO-ROBUSTNESS-PHASE13-BATTERY/13B_model_summary.csv").set_index(["origin","window","candidate"])
    C=pd.read_csv(RESULT/"hybrid/LTD-ROBUSTNESS-PHASE13-BATTERY/13C_factorial_summary.csv").set_index(["origin","window","micro_candidate","macro_candidate"])
    summaries=[]; days=[]; sites=[]; boots=[]; input_hashes={}
    for origin in ORIGINS:
        mf=CACHE/"macro"/(origin+".npz")
        input_hashes[str(mf)]=sha(mf)
        with np.load(mf,allow_pickle=False) as mz:
            macro_data={k:mz[k].copy() for k in mz.files}
        for window in WINDOWS:
            raw={}; ids=dates=y=labels=None
            for mode,seed in EXPERTS:
                path=CACHE/origin/mode/str(seed)/(window+".npz")
                data=read_npz(path,["pcap_uid","query_date","y_true","candidate_labels","probs","origin","window","seed","temporal_mode"])
                input_hashes[str(path)]=sha(path)
                if (str(data["origin"]),str(data["window"]),int(data["seed"]),str(data["temporal_mode"]))!=(origin,window,seed,mode):
                    raise RuntimeError("EXPERT_IDENTITY_MISMATCH: "+str(path))
                if ids is None:
                    ids,dates,y,labels=[data[k] for k in ("pcap_uid","query_date","y_true","candidate_labels")]
                    if len(np.unique(ids))!=len(ids) or len(labels)!=65:raise RuntimeError("INVALID_COHORT")
                elif any(not np.array_equal(data[k],v) for k,v in [("pcap_uid",ids),("query_date",dates),("y_true",y),("candidate_labels",labels)]):
                    raise RuntimeError("UNALIGNED_MICRO")
                if data["probs"].shape!=(len(y),65):raise RuntimeError("BAD_PROBABILITY_MATRIX")
                raw[(mode,seed)]=data["probs"]
            for key,ref in [(f"pcap_uid_{window}",ids),(f"dates_{window}",dates),(f"y_{window}",y),("candidate_labels",labels)]:
                if key not in macro_data or not np.array_equal(macro_data[key],ref):
                    raise RuntimeError("UNALIGNED_MACRO: "+key)
            M=macro_data[f"M0_{window}"]
            models={
                "U1":raw[("UNIFORM",42)],
                "U3":geometric([raw[("UNIFORM",k)] for k in (11,42,73)]),
                "T3_S42":geometric([raw[(m,42)] for m in ("EARLY","UNIFORM","LATE")])
            }
            for name,p in models.items():
                r=validate(p,y,B.loc[(origin,window,name)],origin+window+name)
                h=np.exp(.45*np.log(np.clip(p,EPS,1.))+.55*np.log(np.clip(M,EPS,1.)))
                h=h/h.sum(axis=1,keepdims=True)
                hr=validate(h,y,C.loc[(origin,window,name,"M0")],origin+window+name+"xM0")
                ms=stats(p);hs=stats(h)
                mcorrect=r==1;hcorrect=hr==1
                outcome=np.select(
                    [mcorrect&hcorrect,(~mcorrect)&hcorrect,mcorrect&(~hcorrect)],
                    ["both_correct","rescued_by_hybrid","harmed_by_hybrid"],
                    default="both_wrong")
                f=pd.DataFrame({"uid":ids.astype(str),"date":dates.astype("datetime64[D]").astype(str),
                    "site":labels[y].astype(str),"outcome":outcome,
                    "m_correct":mcorrect,"h_correct":hcorrect,"m_top5":r<=5,"h_top5":hr<=5,
                    "disagree":p.argmax(axis=1)!=h.argmax(axis=1),
                    **{"m_"+k:v for k,v in ms.items()},**{"h_"+k:v for k,v in hs.items()}})
                summary=summarize(f,origin,window,name)
                summaries.append(summary)
                boot,d=bootstrap(f,conf["bootstrap"]["iterations"],
                    conf["bootstrap"]["seed"]+ORIGINS.index(origin)*100+WINDOWS.index(window)*10+list(models).index(name))
                boots.append({"origin":origin,"window":window,"candidate":name,**boot})
                for date,row in d.iterrows():
                    days.append({"origin":origin,"window":window,"candidate":name,"date":date,
                                 "n":int(row["n"]),"rescued":int(row["rescued"]),"harmed":int(row["harmed"]),
                                 "micro_top5_count":int(row["m_top5"]),"hybrid_top5_count":int(row["h_top5"])})
                for site,g in f.groupby("site",sort=True):
                    sites.append({"origin":origin,"window":window,"candidate":name,"site_label":site,"n":len(g),
                                  "rescued":int((g.outcome=="rescued_by_hybrid").sum()),
                                  "harmed":int((g.outcome=="harmed_by_hybrid").sum()),
                                  "micro_top1":float(g.m_correct.mean()),"hybrid_top1":float(g.h_correct.mean()),
                                  "micro_top5":float(g.m_top5.mean()),"hybrid_top5":float(g.h_top5.mean())})
                print("P15A_PAIR_OK",origin,window,name,"N",summary["n"],
                      "rescue",summary["rescued"],"harm",summary["harmed"],
                      "net",summary["net"],"ci95",f"[{boot['top1_ci_low']:.5f},{boot['top1_ci_high']:.5f}]")
    OUT.mkdir(parents=True,exist_ok=True)
    files={
        "P15A_mechanisms_summary.csv":pd.DataFrame(summaries),
        "P15A_day_counts.csv":pd.DataFrame(days),
        "P15A_site_counts.csv":pd.DataFrame(sites),
        "P15A_day_block_bootstrap.csv":pd.DataFrame(boots)
    }
    hashes={}
    for name,df in files.items():
        path=OUT/name
        if path.exists():raise RuntimeError("OUTPUT_EXISTS_NO_OVERWRITE: "+str(path))
        df.to_csv(path,index=False)
        hashes[name]=sha(path)
    manifest={"stage":"P15A_MECH","status":"DESCRIPTIVE_ONLY",
              "training_allowed":False,"selection_allowed":False,"future_b_opened":False,
              "inputs_sha256":input_hashes,"outputs_sha256":hashes,"config_sha256":sha(CONFIG),
              "script_sha256":sha(Path(__file__)),"records":len(summaries)}
    mp=OUT/"P15A_manifest.json"
    if mp.exists():raise RuntimeError("MANIFEST_ALREADY_EXISTS_NO_OVERWRITE")
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True))
    print("P15A_MECH_COMPLETE",len(summaries),"comparisons, output",OUT)
    print(pd.DataFrame(summaries)[["origin","window","candidate","rescued","harmed","net","delta_top1","delta_top5"]].to_string(index=False))
if __name__=="__main__":main()
